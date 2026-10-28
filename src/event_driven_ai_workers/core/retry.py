"""Failure classification and exponential backoff with jitter.

Transient failures (HTTP 408/429/500/502/503/504/529, timeouts, dropped connections) are
retried; everything else is a poison pill routed straight to the dead-letter queue.
Unknown exceptions are deliberately *not* retried: a bug fails fast into the DLQ,
where the original task is preserved for replay, instead of burning the retry budget.
"""

import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import Self

import httpx
from pydantic import ValidationError
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from event_driven_ai_workers.core.config import WorkerSettings
from event_driven_ai_workers.core.errors import PoisonPillError, UpstreamHTTPError

Uniform = Callable[[float, float], float]

_SYSTEM_RANDOM = random.SystemRandom()


class ErrorClass(StrEnum):
    """Bounded failure vocabulary, safe to use as a metric label."""

    RATE_LIMITED = "rate_limited"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    CLIENT_ERROR = "client_error"
    UPSTREAM_ERROR = "upstream_error"
    VALIDATION = "validation"
    UNEXPECTED = "unexpected"


@dataclass(frozen=True, slots=True)
class FailureClassification:
    """How the engine should treat one failed attempt."""

    error_class: ErrorClass
    retryable: bool
    status_code: int | None = None
    retry_after_s: float | None = None


# 529 is the LLM provider's ``overloaded_error``: capacity, not a bad request.
_UNAVAILABLE_STATUSES = frozenset({500, 502, 503, 504, 529})


def classify_status(status_code: int, retry_after_s: float | None = None) -> FailureClassification:
    """Classify an upstream HTTP status code."""
    if status_code == httpx.codes.TOO_MANY_REQUESTS:
        error_class, retryable = ErrorClass.RATE_LIMITED, True
    elif status_code in _UNAVAILABLE_STATUSES:
        error_class, retryable = ErrorClass.UPSTREAM_UNAVAILABLE, True
    elif status_code == httpx.codes.REQUEST_TIMEOUT:
        error_class, retryable = ErrorClass.TIMEOUT, True
    elif 400 <= status_code < 500:  # noqa: PLR2004
        error_class, retryable = ErrorClass.CLIENT_ERROR, False
    else:
        error_class, retryable = ErrorClass.UPSTREAM_ERROR, False
    return FailureClassification(
        error_class, retryable, status_code, retry_after_s if retryable else None
    )


def classify(exc: BaseException) -> FailureClassification:
    """Map an exception raised by a task handler to a retry decision."""
    match exc:
        case UpstreamHTTPError():
            return classify_status(exc.status_code, exc.retry_after_s)
        case httpx.HTTPStatusError():
            retry_after = parse_retry_after(exc.response.headers.get("retry-after"))
            return classify_status(exc.response.status_code, retry_after)
        case httpx.TimeoutException() | RedisTimeoutError() | TimeoutError():
            return FailureClassification(ErrorClass.TIMEOUT, retryable=True)
        case (
            httpx.NetworkError()
            | httpx.RemoteProtocolError()
            | RedisConnectionError()
            | ConnectionError()
        ):
            return FailureClassification(ErrorClass.CONNECTION, retryable=True)
        case ValidationError() | PoisonPillError():
            return FailureClassification(ErrorClass.VALIDATION, retryable=False)
        case _:
            return FailureClassification(ErrorClass.UNEXPECTED, retryable=False)


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    """Parse an HTTP ``Retry-After`` header (delta-seconds or HTTP-date) into seconds."""
    if value is None:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - (now or datetime.now(UTC))).total_seconds())


@dataclass(frozen=True, slots=True)
class BackoffPolicy:
    """Capped exponential backoff with *equal jitter*.

    Retry ``n`` waits a uniform random time in ``[c/2, c]`` where
    ``c = min(max_s, base_s * multiplier ** (n - 1))``: the curve keeps growing
    while concurrent retries are decorrelated (no thundering herd on 429s).
    An upstream ``Retry-After`` raises the delay, bounded by ``retry_after_cap_s``.
    """

    base_s: float = 1.0
    max_s: float = 60.0
    multiplier: float = 2.0
    retry_after_cap_s: float = 300.0

    def __post_init__(self) -> None:
        if not 0 < self.base_s <= self.max_s:
            raise ValueError("require 0 < base_s <= max_s")
        if self.multiplier < 1:
            raise ValueError("multiplier must be >= 1")
        if self.retry_after_cap_s <= 0:
            raise ValueError("retry_after_cap_s must be > 0")

    @classmethod
    def from_settings(cls, settings: WorkerSettings) -> Self:
        return cls(
            base_s=settings.backoff_base_s,
            max_s=settings.backoff_max_s,
            multiplier=settings.backoff_multiplier,
            retry_after_cap_s=settings.retry_after_cap_s,
        )

    def ceiling(self, retry_number: int) -> float:
        """Upper bound of the delay before retry ``retry_number`` (1-based)."""
        if retry_number < 1:
            raise ValueError("retry_number is 1-based")
        if self.multiplier == 1:
            return self.base_s
        # Compare exponents in log space so huge retry numbers cannot overflow.
        if retry_number - 1 >= math.log(self.max_s / self.base_s, self.multiplier):
            return self.max_s
        return min(self.max_s, self.base_s * self.multiplier ** (retry_number - 1))

    def delay(
        self,
        retry_number: int,
        *,
        retry_after_s: float | None = None,
        uniform: Uniform = _SYSTEM_RANDOM.uniform,
    ) -> float:
        """Jittered delay before retry ``retry_number``, honouring ``Retry-After``."""
        ceiling = self.ceiling(retry_number)
        jittered = uniform(ceiling / 2, ceiling)
        if retry_after_s is None:
            return jittered
        return max(jittered, min(retry_after_s, self.retry_after_cap_s))
