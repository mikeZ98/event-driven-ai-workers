"""Ingestion guardrails: bounded body reads, size and content screening, queue backpressure."""

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from arq.connections import ArqRedis
from pydantic import ValidationError

from event_driven_ai_workers.core import WorkerSettings, enqueue_task
from event_driven_ai_workers.guardrails import (
    Backpressure,
    DocumentTooLargeError,
    GuardrailSettings,
    PayloadGuard,
    PayloadTooLargeError,
    QueueSaturatedError,
    RejectedContentError,
)
from event_driven_ai_workers.schemas import MAX_DOCUMENT_CHARS
from tests.conftest import ApiFactory, ApiHarness
from tests.support import make_task
from tests.test_api import BODY, SUMMARIZE, _problem, _submit

SMALL = GuardrailSettings(max_payload_bytes=1_024, max_document_chars=100)


async def _chunks(*parts: bytes, then_fail: bool = False) -> AsyncIterator[bytes]:
    for part in parts:
        yield part
    if then_fail:
        raise AssertionError("the guard read past the limit")


def _request(text: str, **fields: Any) -> bytes:
    return json.dumps({"text": text, **fields}).encode()


async def _fill(pool: ArqRedis, settings: WorkerSettings, count: int, start: int = 0) -> None:
    for i in range(start, start + count):
        task = make_task(idempotency_key=f"fill-{i}")
        assert await enqueue_task(pool, task, queue_name=settings.queue_name) is not None


async def _depth(pool: ArqRedis, settings: WorkerSettings) -> int:
    depth: int = await pool.zcard(settings.queue_name)
    return depth


class TestGuardrailSettings:
    def test_defaults_follow_schema_ceiling(self) -> None:
        settings = GuardrailSettings()

        assert settings.max_document_chars == MAX_DOCUMENT_CHARS
        assert settings.max_payload_bytes == 1_048_576
        assert (settings.max_queue_depth, settings.backpressure_retry_after_s) == (10_000, 5)

    def test_reads_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EDAW_GUARDRAILS_MAX_PAYLOAD_BYTES", "4096")
        monkeypatch.setenv("EDAW_GUARDRAILS_MAX_DOCUMENT_CHARS", "500")
        monkeypatch.setenv("EDAW_GUARDRAILS_MAX_QUEUE_DEPTH", "50")
        monkeypatch.setenv("EDAW_GUARDRAILS_BACKPRESSURE_RETRY_AFTER_S", "30")

        settings = GuardrailSettings()

        assert settings.model_dump() == {
            "max_payload_bytes": 4096,
            "max_document_chars": 500,
            "max_queue_depth": 50,
            "backpressure_retry_after_s": 30,
        }

    @pytest.mark.parametrize(
        "fields",
        [
            {"max_document_chars": MAX_DOCUMENT_CHARS + 1},
            {"max_payload_bytes": 1_023},
            {"max_queue_depth": 0},
            {"backpressure_retry_after_s": 0},
            {"unknown": 1},
        ],
    )
    def test_rejects_out_of_range_limits(self, fields: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            GuardrailSettings(**fields)


class TestBoundedRead:
    async def test_body_at_the_limit_is_read(self) -> None:
        guard = PayloadGuard(SMALL)

        body = await guard.read(_chunks(b"a" * 512, b"b" * 512), content_length="1024")

        assert body == b"a" * 512 + b"b" * 512

    async def test_declared_oversize_is_refused_unread(self) -> None:
        guard = PayloadGuard(SMALL)

        with pytest.raises(PayloadTooLargeError, match="1024 bytes"):
            await guard.read(_chunks(then_fail=True), content_length="1025")

    async def test_undeclared_stream_is_cut_off_at_the_limit(self) -> None:
        guard = PayloadGuard(SMALL)

        with pytest.raises(PayloadTooLargeError):
            await guard.read(_chunks(b"a" * 1_000, b"b" * 25, then_fail=True), content_length=None)

    async def test_understated_length_is_still_enforced(self) -> None:
        guard = PayloadGuard(SMALL)

        with pytest.raises(PayloadTooLargeError):
            await guard.read(_chunks(b"a" * 2_048, then_fail=True), content_length="10")

    async def test_unparseable_length_falls_back_to_counting(self) -> None:
        guard = PayloadGuard(SMALL)

        assert await guard.read(_chunks(b"{}"), content_length="ten") == b"{}"


class TestPayloadScreening:
    def test_valid_request_passes(self) -> None:
        text = "Line one.\n\tLine two.\r\nZażółć gęślą jaźń 🚀"

        request = PayloadGuard(SMALL).summarization(_request(text, focus_areas=["queues"]))

        assert request.text == text
        assert request.focus_areas == ("queues",)

    def test_document_over_configured_limit(self) -> None:
        guard = PayloadGuard(SMALL)

        assert guard.summarization(_request("x" * 100)).text == "x" * 100
        with pytest.raises(DocumentTooLargeError, match="100 characters"):
            guard.summarization(_request("x" * 101))

    def test_document_over_schema_ceiling_is_also_too_large(self) -> None:
        with pytest.raises(DocumentTooLargeError, match=str(MAX_DOCUMENT_CHARS)):
            PayloadGuard(GuardrailSettings()).summarization(
                _request("x" * (MAX_DOCUMENT_CHARS + 1))
            )

    def test_limit_counts_code_points_not_bytes(self) -> None:
        assert PayloadGuard(SMALL).summarization(_request("ż" * 100)).text == "ż" * 100

    @pytest.mark.parametrize(
        ("text", "focus_areas", "field", "reason"),
        [
            ("   \n\t ", [], "text", "is blank"),
            ("NUL\x00byte", [], "text", "contains control characters"),
            ("bell\x07", [], "text", "contains control characters"),
            ("\x1b[31mansi", [], "text", "contains control characters"),
            ("delete\x7f", [], "text", "contains control characters"),
            ("c1 next line\x85", [], "text", "contains control characters"),
            ("fine", ["ok", "bad\x00"], "focus_areas.1", "contains control characters"),
        ],
    )
    def test_poison_content_is_rejected(
        self, text: str, focus_areas: list[str], field: str, reason: str
    ) -> None:
        with pytest.raises(RejectedContentError) as rejected:
            PayloadGuard(SMALL).summarization(_request(text, focus_areas=focus_areas))

        assert (rejected.value.field, rejected.value.reason) == (field, reason)

    @pytest.mark.parametrize("raw", [b'{"text": ""}', b'{"text": "x", "extra": 1}', b"[1]", b"{"])
    def test_other_defects_raise_validation_error(self, raw: bytes) -> None:
        with pytest.raises(ValidationError):
            PayloadGuard(SMALL).summarization(raw)


class TestBackpressure:
    async def test_admits_below_and_refuses_at_threshold(
        self, arq_pool: ArqRedis, fast_settings: WorkerSettings
    ) -> None:
        settings = GuardrailSettings(max_queue_depth=2, backpressure_retry_after_s=9)
        backpressure = Backpressure(
            arq_pool, queue_name=fast_settings.queue_name, settings=settings
        )

        await backpressure.admit()
        await _fill(arq_pool, fast_settings, 1)
        await backpressure.admit()
        await _fill(arq_pool, fast_settings, 1, start=1)

        with pytest.raises(QueueSaturatedError) as saturated:
            await backpressure.admit()
        assert (saturated.value.depth, saturated.value.limit) == (2, 2)
        assert saturated.value.retry_after_s == 9


class TestSizeGuardrailsOverHttp:
    async def test_oversized_body_is_413_before_redis(self, api_factory: ApiFactory) -> None:
        async with api_factory(guardrails=SMALL) as api:
            response = await _submit(api, {"text": "x" * 2_000})

            problem = _problem(response, 413, "payload-too-large")
            assert problem["detail"] == "The request body exceeds 1024 bytes."
            assert await api.redis.keys("*") == []
            assert api.submissions("too_large") == 1

    async def test_chunked_oversized_body_is_413(self, api_factory: ApiFactory) -> None:
        async with api_factory(guardrails=SMALL) as api:
            response = await api.client.post(
                SUMMARIZE,
                content=_chunks(b'{"text": "', b"x" * 2_000, b'"}'),
                headers={"idempotency-key": "doc-1"},
            )

            assert "content-length" not in response.request.headers
            _problem(response, 413, "payload-too-large")
            assert await api.redis.keys("*") == []

    async def test_oversized_document_is_413_before_redis(self, api_factory: ApiFactory) -> None:
        async with api_factory(guardrails=SMALL) as api:
            response = await _submit(api, {"text": "x" * 101})

            problem = _problem(response, 413, "document-too-large")
            assert "x" * 101 not in response.text
            assert problem["detail"] == "The document text exceeds 100 characters."
            assert await api.redis.keys("*") == []
            assert api.submissions("too_large") == 1

    async def test_document_over_schema_ceiling_is_413(self, api: ApiHarness) -> None:
        response = await _submit(api, {"text": "x" * (MAX_DOCUMENT_CHARS + 1)})

        _problem(response, 413, "document-too-large")
        assert await api.redis.keys("*") == []

    async def test_poison_pill_is_422_before_redis(self, api: ApiHarness) -> None:
        response = await _submit(api, {"text": "binary\x00\x01payload"})

        problem = _problem(response, 422, "rejected-content")
        assert problem["detail"] == "Rejected content: text: contains control characters"
        assert "binary" not in response.text
        assert await api.redis.keys("*") == []
        assert api.submissions("invalid") == 1

    async def test_openapi_documents_413(self, api: ApiHarness) -> None:
        spec = (await api.client.get("/openapi.json")).json()

        assert "413" in spec["paths"][SUMMARIZE]["post"]["responses"]


class TestBackpressureOverHttp:
    LIMITS = GuardrailSettings(max_queue_depth=3, backpressure_retry_after_s=7)

    async def test_saturated_queue_is_503_with_retry_after(self, api_factory: ApiFactory) -> None:
        async with api_factory(guardrails=self.LIMITS) as api:
            await _fill(api.redis, api.worker, 3)

            response = await _submit(api)

            problem = _problem(response, 503, "queue-saturated")
            assert response.headers["retry-after"] == "7"
            assert problem["detail"] == "The task queue is at capacity; retry later."
            assert await _depth(api.redis, api.worker) == 3
            assert await api.redis.keys("edaw:idem:*") == []
            assert api.submissions("saturated") == 1

    async def test_refused_key_stays_free_for_the_retry(self, api_factory: ApiFactory) -> None:
        async with api_factory(guardrails=self.LIMITS) as api:
            await _fill(api.redis, api.worker, 3)
            _problem(await _submit(api), 503, "queue-saturated")

            await api.redis.zrem(api.worker.queue_name, "fill-0")  # a worker drained one job
            retried = await _submit(api)

            assert retried.status_code == 202, retried.text
            assert "idempotent-replayed" not in retried.headers
            assert api.submissions("accepted") == 1

    async def test_submission_within_bounds_is_accepted(self, api_factory: ApiFactory) -> None:
        async with api_factory(guardrails=self.LIMITS) as api:
            await _fill(api.redis, api.worker, 2)

            response = await _submit(api)

            assert response.status_code == 202, response.text
            assert response.json()["status"] == "PENDING"
            assert await _depth(api.redis, api.worker) == 3

    async def test_replays_are_served_while_saturated(self, api_factory: ApiFactory) -> None:
        async with api_factory(guardrails=GuardrailSettings(max_queue_depth=1)) as api:
            first = await _submit(api)
            assert first.status_code == 202

            replay = await _submit(api, BODY)

            assert replay.status_code == 202, replay.text
            assert replay.headers["idempotent-replayed"] == "true"
            assert replay.json()["task_id"] == first.json()["task_id"]
            _problem(await _submit(api, key="doc-2"), 503, "queue-saturated")
