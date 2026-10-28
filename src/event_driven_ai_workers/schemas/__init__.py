"""Immutable, strictly validated data contracts shared by the producer and workers."""

from event_driven_ai_workers.schemas._base import UtcDatetime, WorkerModel, utc_now
from event_driven_ai_workers.schemas.dead_letter import DeadLetterEnvelope
from event_driven_ai_workers.schemas.summarization import (
    FocusArea,
    SummarizationRequest,
    SummarizationResult,
)
from event_driven_ai_workers.schemas.task import (
    IdempotencyKey,
    TaskPayload,
    TaskResult,
    TaskStatus,
    TaskType,
)

__all__ = [
    "DeadLetterEnvelope",
    "FocusArea",
    "IdempotencyKey",
    "SummarizationRequest",
    "SummarizationResult",
    "TaskPayload",
    "TaskResult",
    "TaskStatus",
    "TaskType",
    "UtcDatetime",
    "WorkerModel",
    "utc_now",
]
