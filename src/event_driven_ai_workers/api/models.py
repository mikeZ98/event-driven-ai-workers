"""Response bodies of the producer API, serialized with ``model_dump_json``."""

from typing import Literal, Self
from uuid import UUID

from event_driven_ai_workers.schemas import (
    IdempotencyKey,
    TaskPayload,
    TaskStatus,
    TaskType,
    UtcDatetime,
    WorkerModel,
)


class TaskAccepted(WorkerModel):
    """Answer to a submission: where to poll and the task's current status.

    A replayed submission (same ``Idempotency-Key``) returns the original task.
    """

    task_id: UUID
    idempotency_key: IdempotencyKey
    task_type: TaskType
    status: TaskStatus
    created_at: UtcDatetime
    status_url: str

    @classmethod
    def for_task(cls, task: TaskPayload, status: TaskStatus, status_url: str) -> Self:
        return cls(
            task_id=task.task_id,
            idempotency_key=task.idempotency_key,
            task_type=task.task_type,
            status=status,
            created_at=task.created_at,
            status_url=status_url,
        )


class ComponentHealth(WorkerModel):
    """Availability of one dependency."""

    name: str
    available: bool
    detail: str


class HealthReport(WorkerModel):
    """``unavailable`` (HTTP 503) when Redis is unreachable; ``degraded`` when no worker
    heartbeat is visible: tasks are still accepted and wait in the queue."""

    status: Literal["ok", "degraded", "unavailable"]
    version: str
    components: tuple[ComponentHealth, ...]
