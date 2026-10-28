"""Task envelope submitted by the producer and the result reported by workers."""

from enum import StrEnum
from typing import Annotated, Self
from uuid import UUID, uuid4

from pydantic import Field, JsonValue, StringConstraints, model_validator

from event_driven_ai_workers.schemas._base import UtcDatetime, WorkerModel, utc_now

IdempotencyKey = Annotated[
    str, StringConstraints(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
]


class TaskType(StrEnum):
    """Workloads the worker pool knows how to execute."""

    SUMMARIZATION = "summarization"
    AUDIO_TRANSCRIPTION = "audio_transcription"


class TaskStatus(StrEnum):
    """Task lifecycle: ``PENDING → RUNNING → COMPLETED | FAILED | DEAD_LETTER``."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    DEAD_LETTER = "DEAD_LETTER"

    @property
    def is_terminal(self) -> bool:
        """Whether the task has finished and will not be picked up again."""
        return self not in {TaskStatus.PENDING, TaskStatus.RUNNING}


class TaskPayload(WorkerModel):
    """A unit of work enqueued for a worker; deduplicated by ``idempotency_key``."""

    task_id: UUID = Field(default_factory=uuid4)
    idempotency_key: IdempotencyKey
    task_type: TaskType
    payload: dict[str, JsonValue]
    created_at: UtcDatetime = Field(default_factory=utc_now)


class TaskResult(WorkerModel):
    """Observable state of a task.

    Pending and running tasks carry no outcome fields. Terminal tasks record
    ``completed_at`` and ``duration_ms``; only ``COMPLETED`` tasks carry a ``result``.
    """

    task_id: UUID
    status: TaskStatus
    result: dict[str, JsonValue] | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    completed_at: UtcDatetime | None = None

    @model_validator(mode="after")
    def _check_lifecycle(self) -> Self:
        if not self.status.is_terminal:
            if (self.result, self.duration_ms, self.completed_at) != (None, None, None):
                raise ValueError(f"{self.status} task must not carry outcome fields")
            return self
        if self.completed_at is None or self.duration_ms is None:
            raise ValueError(f"{self.status} task requires completed_at and duration_ms")
        if (self.status is TaskStatus.COMPLETED) != (self.result is not None):
            raise ValueError("result is required for COMPLETED tasks and forbidden otherwise")
        return self
