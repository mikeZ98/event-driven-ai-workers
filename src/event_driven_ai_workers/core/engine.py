"""Task execution engine: dispatch, timing, state transitions, retry and DLQ routing.

The engine is queue-agnostic. It turns one raw ``TaskPayload`` JSON message plus the
attempt number into either a terminal :class:`TaskResult` or a :class:`RetryScheduled`
decision; :mod:`event_driven_ai_workers.core.worker` maps that onto arq.
"""

import asyncio
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

import structlog
from pydantic import ValidationError

from event_driven_ai_workers.core.config import WorkerSettings
from event_driven_ai_workers.core.errors import (
    DeadLetterPublishError,
    MalformedTaskError,
    UnsupportedTaskTypeError,
)
from event_driven_ai_workers.core.ports import (
    DeadLetterReason,
    DeadLetterSink,
    EngineObserver,
    NullObserver,
    TaskHandler,
    TaskStateStore,
)
from event_driven_ai_workers.core.retry import BackoffPolicy, FailureClassification, classify
from event_driven_ai_workers.schemas import (
    DeadLetterEnvelope,
    TaskPayload,
    TaskResult,
    TaskStatus,
    TaskType,
    utc_now,
)

MAX_FAILURE_REASON_CHARS = 1024
MAX_ERROR_TRACE_CHARS = 32_768

_log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RetryScheduled:
    """The attempt failed transiently; run the task again after ``delay_s``."""

    task_id: UUID
    delay_s: float
    failure: FailureClassification


ExecutionOutcome = TaskResult | RetryScheduled


class TaskEngine:
    """Executes tasks through registered handlers and routes failures."""

    def __init__(
        self,
        handlers: Mapping[TaskType, TaskHandler],
        *,
        dead_letters: DeadLetterSink,
        states: TaskStateStore,
        settings: WorkerSettings,
        backoff: BackoffPolicy | None = None,
        observer: EngineObserver | None = None,
        clock: Callable[[], float] = time.perf_counter,
        now: Callable[[], datetime] = utc_now,
    ) -> None:
        self._handlers = dict(handlers)
        self._dead_letters = dead_letters
        self._states = states
        self._settings = settings
        self._backoff = backoff or BackoffPolicy.from_settings(settings)
        self._observer: EngineObserver = observer or NullObserver()
        self._clock = clock
        self._now = now

    async def execute(self, raw_task: str | bytes, *, attempt: int) -> ExecutionOutcome:
        """Run attempt ``attempt`` (1-based) of the task encoded in ``raw_task``.

        Raises :class:`MalformedTaskError` when the message is not a ``TaskPayload``
        (there is no task to envelope) and :class:`DeadLetterPublishError` when a
        poison pill cannot be quarantined. Cancellation propagates untouched.
        """
        if attempt < 1:
            raise ValueError("attempt is 1-based")
        try:
            task = TaskPayload.model_validate_json(raw_task)
        except ValidationError as exc:
            _log.error("task.malformed", attempt=attempt, errors=exc.error_count())
            raise MalformedTaskError("queued message is not a valid TaskPayload") from exc

        log = _log.bind(task_id=str(task.task_id), task_type=task.task_type, attempt=attempt)
        self._observer.task_started(task)
        await self._save(TaskResult(task_id=task.task_id, status=TaskStatus.RUNNING))
        started = self._clock()
        try:
            handler = self._handler_for(task.task_type)
            async with asyncio.timeout(self._settings.task_timeout_s):
                output = await handler(task)
            result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result=output,
                duration_ms=self._elapsed_ms(started),
                completed_at=self._now(),
            )
        except Exception as exc:  # noqa: BLE001 — every failure is classified, none is lost
            return await self._handle_failure(task, exc, attempt, started, log)

        await self._save(result)
        self._observer.task_finished(task, TaskStatus.COMPLETED, self._clock() - started)
        log.info("task.completed", duration_ms=result.duration_ms)
        return result

    async def _handle_failure(
        self,
        task: TaskPayload,
        exc: Exception,
        attempt: int,
        started: float,
        log: structlog.typing.FilteringBoundLogger,
    ) -> ExecutionOutcome:
        failure = classify(exc)
        retries_done = attempt - 1
        log = log.bind(error_class=failure.error_class, error=type(exc).__name__)

        if failure.retryable and retries_done < self._settings.dlq_threshold:
            delay_s = self._backoff.delay(attempt, retry_after_s=failure.retry_after_s)
            log.warning("task.retry_scheduled", delay_s=round(delay_s, 3))
            self._observer.retry_scheduled(task, failure, delay_s)
            await self._save(TaskResult(task_id=task.task_id, status=TaskStatus.PENDING))
            return RetryScheduled(task.task_id, delay_s, failure)

        reason = (
            DeadLetterReason.RETRIES_EXHAUSTED
            if failure.retryable
            else DeadLetterReason.NON_RETRYABLE
        )
        envelope = DeadLetterEnvelope(
            original_task=task,
            failure_reason=_truncate_head(
                f"{reason}: {failure.error_class} ({type(exc).__name__}: {exc})",
                MAX_FAILURE_REASON_CHARS,
            ),
            retry_attempts=retries_done,
            error_trace=_format_trace(exc),
            quarantined_at=self._now(),
        )
        duration_s = self._clock() - started
        try:
            await self._dead_letters.publish(envelope)
        except Exception as publish_exc:
            log.critical("task.dead_letter_publish_failed", reason=reason)
            await self._save(self._terminal(task, TaskStatus.FAILED, started))
            self._observer.task_finished(task, TaskStatus.FAILED, duration_s)
            raise DeadLetterPublishError(task.task_id) from publish_exc

        result = self._terminal(task, TaskStatus.DEAD_LETTER, started)
        await self._save(result)
        self._observer.dead_lettered(task, reason, failure.error_class)
        self._observer.task_finished(task, TaskStatus.DEAD_LETTER, duration_s)
        log.error("task.dead_lettered", reason=reason, retry_attempts=retries_done)
        return result

    def _handler_for(self, task_type: TaskType) -> TaskHandler:
        handler = self._handlers.get(task_type)
        if handler is None:
            raise UnsupportedTaskTypeError(f"no handler for task type {task_type!r}")
        return handler

    def _terminal(self, task: TaskPayload, status: TaskStatus, started: float) -> TaskResult:
        return TaskResult(
            task_id=task.task_id,
            status=status,
            duration_ms=self._elapsed_ms(started),
            completed_at=self._now(),
        )

    def _elapsed_ms(self, started: float) -> int:
        return max(0, round((self._clock() - started) * 1000))

    async def _save(self, result: TaskResult) -> None:
        """Best-effort state tracking: a tracking outage must not fail the task itself."""
        try:
            await self._states.save(result)
        except Exception:  # noqa: BLE001
            _log.warning(
                "task.state_save_failed", task_id=str(result.task_id), status=result.status
            )


def _truncate_head(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _format_trace(exc: BaseException) -> str:
    """Formatted traceback, keeping the tail (where the root cause is) when too long."""
    trace = "".join(traceback.format_exception(exc))
    if len(trace) <= MAX_ERROR_TRACE_CHARS:
        return trace
    return "…" + trace[-(MAX_ERROR_TRACE_CHARS - 1) :]
