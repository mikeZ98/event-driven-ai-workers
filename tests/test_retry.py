"""Retry policy: failure classification, Retry-After parsing and jittered backoff."""

from datetime import UTC, datetime

import httpx
import pytest
from pydantic import BaseModel, ValidationError
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from event_driven_ai_workers.core import (
    BackoffPolicy,
    ErrorClass,
    PoisonPillError,
    UnsupportedTaskTypeError,
    UpstreamHTTPError,
    WorkerSettings,
    classify,
    classify_status,
    parse_retry_after,
)

NOW = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)


def _status_error(status: int, headers: dict[str, str] | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://llm.test/v1/summarize")
    response = httpx.Response(status, headers=headers, request=request)
    return httpx.HTTPStatusError("upstream error", request=request, response=response)


def _validation_error() -> ValidationError:
    class Model(BaseModel):
        n: int

    try:
        Model.model_validate({"n": "x"})
    except ValidationError as exc:
        return exc
    raise AssertionError("unreachable")


class TestClassifyStatus:
    @pytest.mark.parametrize(
        ("status", "error_class"),
        [
            (429, ErrorClass.RATE_LIMITED),
            (500, ErrorClass.UPSTREAM_UNAVAILABLE),
            (502, ErrorClass.UPSTREAM_UNAVAILABLE),
            (503, ErrorClass.UPSTREAM_UNAVAILABLE),
            (504, ErrorClass.UPSTREAM_UNAVAILABLE),
            (529, ErrorClass.UPSTREAM_UNAVAILABLE),
            (408, ErrorClass.TIMEOUT),
        ],
    )
    def test_transient_statuses_are_retryable(self, status: int, error_class: ErrorClass) -> None:
        failure = classify_status(status, retry_after_s=7.0)
        assert failure.retryable
        assert failure.error_class is error_class
        assert failure.status_code == status
        assert failure.retry_after_s == 7.0

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 413, 422])
    def test_client_errors_are_poison_pills(self, status: int) -> None:
        failure = classify_status(status, retry_after_s=7.0)
        assert not failure.retryable
        assert failure.error_class is ErrorClass.CLIENT_ERROR
        assert failure.retry_after_s is None

    @pytest.mark.parametrize("status", [501, 505, 511, 302])
    def test_other_statuses_are_permanent_upstream_errors(self, status: int) -> None:
        failure = classify_status(status)
        assert (failure.error_class, failure.retryable) == (ErrorClass.UPSTREAM_ERROR, False)


class TestClassify:
    def test_upstream_http_error_carries_retry_after(self) -> None:
        failure = classify(UpstreamHTTPError(429, "slow down", retry_after_s=12.0))
        assert (failure.error_class, failure.retryable, failure.retry_after_s) == (
            ErrorClass.RATE_LIMITED,
            True,
            12.0,
        )

    def test_httpx_status_error_reads_retry_after_header(self) -> None:
        failure = classify(_status_error(429, {"Retry-After": "30"}))
        assert failure.error_class is ErrorClass.RATE_LIMITED
        assert failure.retry_after_s == 30.0

    def test_httpx_status_error_poison_pill(self) -> None:
        failure = classify(_status_error(401))
        assert (failure.error_class, failure.retryable) == (ErrorClass.CLIENT_ERROR, False)

    @pytest.mark.parametrize(
        "exc",
        [
            httpx.ReadTimeout("read timed out"),
            httpx.ConnectTimeout("connect timed out"),
            httpx.PoolTimeout("pool exhausted"),
            TimeoutError(),
            RedisTimeoutError(),
        ],
    )
    def test_timeouts_are_retryable(self, exc: Exception) -> None:
        failure = classify(exc)
        assert (failure.error_class, failure.retryable) == (ErrorClass.TIMEOUT, True)

    @pytest.mark.parametrize(
        "exc",
        [
            httpx.ConnectError("refused"),
            httpx.ReadError("reset"),
            httpx.RemoteProtocolError("server hung up"),
            ConnectionResetError(),
            RedisConnectionError(),
        ],
    )
    def test_connection_failures_are_retryable(self, exc: Exception) -> None:
        failure = classify(exc)
        assert (failure.error_class, failure.retryable) == (ErrorClass.CONNECTION, True)

    @pytest.mark.parametrize(
        "exc",
        [_validation_error(), PoisonPillError("bad"), UnsupportedTaskTypeError("video")],
    )
    def test_validation_failures_are_poison_pills(self, exc: Exception) -> None:
        failure = classify(exc)
        assert (failure.error_class, failure.retryable) == (ErrorClass.VALIDATION, False)

    @pytest.mark.parametrize(
        "exc", [KeyError("x"), RuntimeError("bug"), httpx.UnsupportedProtocol("ftp://")]
    )
    def test_unknown_failures_fail_fast(self, exc: Exception) -> None:
        failure = classify(exc)
        assert (failure.error_class, failure.retryable) == (ErrorClass.UNEXPECTED, False)

    def test_upstream_error_message(self) -> None:
        assert str(UpstreamHTTPError(503)) == "upstream HTTP 503"
        assert str(UpstreamHTTPError(503, "maintenance")) == "upstream HTTP 503: maintenance"


class TestParseRetryAfter:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("120", 120.0),
            (" 5 ", 5.0),
            ("0", 0.0),
            ("Wed, 28 Oct 2026 12:00:30 GMT", 30.0),
            ("Wed, 28 Oct 2026 11:59:00 GMT", 0.0),
            ("Wed, 28 Oct 2026 12:00:30", 30.0),
            (None, None),
            ("", None),
            ("-5", None),
            ("1.5", None),
            ("soon", None),
        ],
    )
    def test_parse(self, value: str | None, expected: float | None) -> None:
        assert parse_retry_after(value, now=NOW) == expected

    def test_defaults_to_current_time(self) -> None:
        assert parse_retry_after("Thu, 01 Jan 1970 00:00:00 GMT") == 0.0


class TestBackoffPolicy:
    def test_ceiling_grows_exponentially_and_caps(self) -> None:
        policy = BackoffPolicy(base_s=1.0, max_s=10.0, multiplier=2.0)
        assert [policy.ceiling(n) for n in range(1, 7)] == [1.0, 2.0, 4.0, 8.0, 10.0, 10.0]

    def test_huge_retry_numbers_do_not_overflow(self) -> None:
        assert BackoffPolicy(base_s=1.0, max_s=60.0).ceiling(10_000) == 60.0

    def test_constant_backoff_when_multiplier_is_one(self) -> None:
        assert BackoffPolicy(base_s=3.0, max_s=60.0, multiplier=1.0).ceiling(9) == 3.0

    def test_equal_jitter_bounds(self) -> None:
        policy = BackoffPolicy(base_s=1.0, max_s=60.0)
        calls: list[tuple[float, float]] = []

        def low(a: float, b: float) -> float:
            calls.append((a, b))
            return a

        assert policy.delay(4, uniform=low) == 4.0
        assert calls == [(4.0, 8.0)]
        assert policy.delay(4, uniform=lambda _a, b: b) == 8.0

    def test_real_jitter_stays_in_range_and_varies(self) -> None:
        policy = BackoffPolicy(base_s=1.0, max_s=60.0)
        delays = {policy.delay(3) for _ in range(200)}
        assert all(2.0 <= d <= 4.0 for d in delays)
        assert len(delays) > 1

    def test_retry_after_raises_delay(self) -> None:
        policy = BackoffPolicy(base_s=1.0, max_s=60.0, retry_after_cap_s=300.0)
        assert policy.delay(1, retry_after_s=45.0, uniform=lambda a, _b: a) == 45.0

    def test_retry_after_never_shortens_backoff(self) -> None:
        policy = BackoffPolicy(base_s=1.0, max_s=60.0)
        assert policy.delay(5, retry_after_s=1.0, uniform=lambda a, _b: a) == 8.0

    def test_retry_after_is_capped(self) -> None:
        policy = BackoffPolicy(base_s=1.0, max_s=60.0, retry_after_cap_s=120.0)
        assert policy.delay(1, retry_after_s=86_400.0, uniform=lambda a, _b: a) == 120.0

    def test_from_settings(self) -> None:
        settings = WorkerSettings(
            backoff_base_s=0.5, backoff_max_s=30, backoff_multiplier=3, retry_after_cap_s=90
        )
        assert BackoffPolicy.from_settings(settings) == BackoffPolicy(0.5, 30.0, 3.0, 90.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"base_s": 0.0},
            {"base_s": 5.0, "max_s": 1.0},
            {"multiplier": 0.5},
            {"retry_after_cap_s": 0.0},
        ],
    )
    def test_rejects_invalid_parameters(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):  # noqa: PT011
            BackoffPolicy(**kwargs)

    def test_retry_number_is_one_based(self) -> None:
        with pytest.raises(ValueError, match="1-based"):
            BackoffPolicy().ceiling(0)
