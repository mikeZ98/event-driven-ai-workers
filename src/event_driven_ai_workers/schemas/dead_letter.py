"""Diagnostic envelope for poison-pill tasks quarantined in the dead-letter queue."""

from pydantic import Field

from event_driven_ai_workers.schemas._base import UtcDatetime, WorkerModel, utc_now
from event_driven_ai_workers.schemas.task import TaskPayload


class DeadLetterEnvelope(WorkerModel):
    """The original task plus why and when it was quarantined.

    ``retry_attempts`` may be 0 when a non-retryable error routes the task straight
    to the DLQ. ``error_trace`` is bounded so a runaway traceback cannot bloat Redis;
    producers truncate before constructing the envelope.
    """

    original_task: TaskPayload
    failure_reason: str = Field(min_length=1, max_length=1024)
    retry_attempts: int = Field(ge=0)
    error_trace: str | None = Field(default=None, max_length=32_768)
    quarantined_at: UtcDatetime = Field(default_factory=utc_now)
