"""Ports the engine depends on; adapters live in ``dlq``/``tracking`` and ``tools.metrics``."""

import inspect
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Protocol, cast

from pydantic import JsonValue

from event_driven_ai_workers.core.retry import ErrorClass, FailureClassification
from event_driven_ai_workers.schemas import DeadLetterEnvelope, TaskPayload, TaskResult, TaskStatus

TaskOutput = dict[str, JsonValue]
TaskHandler = Callable[[TaskPayload], Awaitable[TaskOutput]]
"""Executes one task and returns its JSON-serialisable result."""


class DeadLetterReason(StrEnum):
    """Why a task was quarantined."""

    NON_RETRYABLE = "non_retryable"
    RETRIES_EXHAUSTED = "retries_exhausted"


class DeadLetterSink(Protocol):
    """Durable destination for poison pills."""

    async def publish(self, envelope: DeadLetterEnvelope) -> None: ...


class TaskStateStore(Protocol):
    """Records each task's latest observable :class:`TaskResult`."""

    async def save(self, result: TaskResult) -> None: ...


class EngineObserver(Protocol):
    """Receives engine lifecycle events (metrics, tracing). Must never raise."""

    def task_started(self, task: TaskPayload) -> None: ...

    def task_finished(self, task: TaskPayload, status: TaskStatus, duration_s: float) -> None: ...

    def retry_scheduled(
        self, task: TaskPayload, failure: FailureClassification, delay_s: float
    ) -> None: ...

    def dead_lettered(
        self, task: TaskPayload, reason: DeadLetterReason, error_class: ErrorClass
    ) -> None: ...

    def queue_depth(self, queue_name: str, depth: int) -> None: ...


class NullObserver:
    """Observer that ignores every event."""

    def task_started(self, task: TaskPayload) -> None:
        pass

    def task_finished(self, task: TaskPayload, status: TaskStatus, duration_s: float) -> None:
        pass

    def retry_scheduled(
        self, task: TaskPayload, failure: FailureClassification, delay_s: float
    ) -> None:
        pass

    def dead_lettered(
        self, task: TaskPayload, reason: DeadLetterReason, error_class: ErrorClass
    ) -> None:
        pass

    def queue_depth(self, queue_name: str, depth: int) -> None:
        pass


async def resolve[T](value: Awaitable[T] | T) -> T:
    """Await redis-py command results, which are typed ``Awaitable[T] | T``."""
    if inspect.isawaitable(value):
        return await cast("Awaitable[T]", value)
    return value
