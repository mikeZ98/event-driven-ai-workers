"""Test doubles shared across the suite (no Redis server, no network)."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from event_driven_ai_workers.core import DeadLetterReason, ErrorClass, FailureClassification
from event_driven_ai_workers.schemas import (
    DeadLetterEnvelope,
    TaskPayload,
    TaskResult,
    TaskStatus,
    TaskType,
)

NOW = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)


def make_task(**overrides: Any) -> TaskPayload:
    fields: dict[str, Any] = {
        "idempotency_key": "doc-1",
        "task_type": TaskType.SUMMARIZATION,
        "payload": {"text": "A long document about queues.", "max_length": 64},
        "created_at": NOW,
    } | overrides
    return TaskPayload(**fields)


class FakeClock:
    """Monotonic clock advanced explicitly by tests."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class InMemoryDeadLetters:
    envelopes: list[DeadLetterEnvelope] = field(default_factory=list)
    error: Exception | None = None

    async def publish(self, envelope: DeadLetterEnvelope) -> None:
        if self.error is not None:
            raise self.error
        self.envelopes.append(envelope)


@dataclass
class InMemoryStates:
    history: list[TaskResult] = field(default_factory=list)
    error: Exception | None = None

    async def save(self, result: TaskResult) -> None:
        if self.error is not None:
            raise self.error
        self.history.append(result)

    @property
    def statuses(self) -> list[TaskStatus]:
        return [r.status for r in self.history]


@dataclass
class RecordingObserver:
    events: list[tuple[Any, ...]] = field(default_factory=list)

    def task_started(self, task: TaskPayload) -> None:
        self.events.append(("started", task.task_id))

    def task_finished(self, task: TaskPayload, status: TaskStatus, duration_s: float) -> None:
        self.events.append(("finished", task.task_id, status, duration_s))

    def retry_scheduled(
        self, task: TaskPayload, failure: FailureClassification, delay_s: float
    ) -> None:
        self.events.append(("retry", task.task_id, failure.error_class, delay_s))

    def dead_lettered(
        self, task: TaskPayload, reason: DeadLetterReason, error_class: ErrorClass
    ) -> None:
        self.events.append(("dead_lettered", task.task_id, reason, error_class))

    def queue_depth(self, queue_name: str, depth: int) -> None:
        self.events.append(("queue_depth", queue_name, depth))

    def kinds(self) -> list[str]:
        return [str(e[0]) for e in self.events]
