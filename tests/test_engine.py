"""Engine dispatch: timing, state transitions, retry scheduling and DLQ routing."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from event_driven_ai_workers.core import (
    BackoffPolicy,
    DeadLetterPublishError,
    DeadLetterReason,
    ErrorClass,
    MalformedTaskError,
    RetryScheduled,
    TaskEngine,
    TaskHandler,
    TaskOutput,
    UpstreamHTTPError,
    WorkerSettings,
)
from event_driven_ai_workers.core.engine import MAX_ERROR_TRACE_CHARS, MAX_FAILURE_REASON_CHARS
from event_driven_ai_workers.schemas import TaskPayload, TaskResult, TaskStatus, TaskType
from tests.support import (
    NOW,
    FakeClock,
    InMemoryDeadLetters,
    InMemoryStates,
    RecordingObserver,
    make_task,
)


@dataclass
class Harness:
    clock: FakeClock = field(default_factory=FakeClock)
    dead_letters: InMemoryDeadLetters = field(default_factory=InMemoryDeadLetters)
    states: InMemoryStates = field(default_factory=InMemoryStates)
    observer: RecordingObserver = field(default_factory=RecordingObserver)
    settings: WorkerSettings = field(default_factory=WorkerSettings)

    def engine(self, handler: TaskHandler | None) -> TaskEngine:
        handlers = {} if handler is None else {TaskType.SUMMARIZATION: handler}
        return TaskEngine(
            handlers,
            dead_letters=self.dead_letters,
            states=self.states,
            settings=self.settings,
            backoff=BackoffPolicy(base_s=1.0, max_s=60.0, retry_after_cap_s=300.0),
            observer=self.observer,
            clock=self.clock,
            now=lambda: NOW,
        )


def _handler(
    harness: Harness, *, output: TaskOutput | None = None, error: Exception | None = None
) -> Callable[[TaskPayload], Awaitable[TaskOutput]]:
    async def handle(task: TaskPayload) -> TaskOutput:
        harness.clock.advance(1.25)
        if error is not None:
            raise error
        return output if output is not None else {"summary": "ok", "task": str(task.task_id)}

    return handle


@pytest.fixture
def harness() -> Harness:
    return Harness()


@pytest.fixture
def task() -> TaskPayload:
    return make_task()


class TestSuccess:
    async def test_completed_result_with_timing(self, harness: Harness, task: TaskPayload) -> None:
        engine = harness.engine(_handler(harness, output={"summary": "short"}))
        outcome = await engine.execute(task.model_dump_json(), attempt=1)
        assert outcome == TaskResult(
            task_id=task.task_id,
            status=TaskStatus.COMPLETED,
            result={"summary": "short"},
            duration_ms=1250,
            completed_at=NOW,
        )
        assert harness.states.statuses == [TaskStatus.RUNNING, TaskStatus.COMPLETED]
        assert harness.dead_letters.envelopes == []
        assert harness.observer.events == [
            ("started", task.task_id),
            ("finished", task.task_id, TaskStatus.COMPLETED, 1.25),
        ]

    async def test_accepts_bytes(self, harness: Harness, task: TaskPayload) -> None:
        engine = harness.engine(_handler(harness))
        outcome = await engine.execute(task.model_dump_json().encode(), attempt=1)
        assert isinstance(outcome, TaskResult)
        assert outcome.status is TaskStatus.COMPLETED

    async def test_state_store_outage_does_not_fail_the_task(
        self, harness: Harness, task: TaskPayload
    ) -> None:
        harness.states.error = RedisConnectionError("down")
        outcome = await harness.engine(_handler(harness)).execute(task.model_dump_json(), attempt=1)
        assert isinstance(outcome, TaskResult)
        assert outcome.status is TaskStatus.COMPLETED


class TestRetry:
    async def test_rate_limit_schedules_retry_honouring_retry_after(
        self, harness: Harness, task: TaskPayload
    ) -> None:
        error = UpstreamHTTPError(429, retry_after_s=42.0)
        outcome = await harness.engine(_handler(harness, error=error)).execute(
            task.model_dump_json(), attempt=1
        )
        assert isinstance(outcome, RetryScheduled)
        assert outcome.task_id == task.task_id
        assert outcome.delay_s == 42.0
        assert outcome.failure.error_class is ErrorClass.RATE_LIMITED
        assert harness.states.statuses == [TaskStatus.RUNNING, TaskStatus.PENDING]
        assert harness.observer.kinds() == ["started", "retry"]
        assert harness.dead_letters.envelopes == []

    @pytest.mark.parametrize("attempt", [1, 2, 3, 4, 5])
    async def test_backoff_grows_with_attempts(
        self, harness: Harness, task: TaskPayload, attempt: int
    ) -> None:
        outcome = await harness.engine(_handler(harness, error=UpstreamHTTPError(503))).execute(
            task.model_dump_json(), attempt=attempt
        )
        assert isinstance(outcome, RetryScheduled)
        ceiling = 2.0 ** (attempt - 1)
        assert ceiling / 2 <= outcome.delay_s <= ceiling

    async def test_handler_timeout_is_retryable(self, task: TaskPayload) -> None:
        harness = Harness(settings=WorkerSettings(task_timeout_s=0.01))

        async def hang(_: TaskPayload) -> TaskOutput:
            await asyncio.sleep(5)
            return {}

        outcome = await harness.engine(hang).execute(task.model_dump_json(), attempt=1)
        assert isinstance(outcome, RetryScheduled)
        assert outcome.failure.error_class is ErrorClass.TIMEOUT


class TestDeadLetter:
    async def test_retries_exhausted_after_threshold(
        self, harness: Harness, task: TaskPayload
    ) -> None:
        engine = harness.engine(_handler(harness, error=UpstreamHTTPError(503, "overloaded")))
        outcome = await engine.execute(task.model_dump_json(), attempt=6)

        assert outcome == TaskResult(
            task_id=task.task_id,
            status=TaskStatus.DEAD_LETTER,
            duration_ms=1250,
            completed_at=NOW,
        )
        [envelope] = harness.dead_letters.envelopes
        assert envelope.original_task == task
        assert envelope.retry_attempts == 5
        assert envelope.quarantined_at == NOW
        assert envelope.failure_reason.startswith("retries_exhausted: upstream_unavailable")
        assert "upstream HTTP 503: overloaded" in envelope.failure_reason
        assert envelope.error_trace is not None
        assert "UpstreamHTTPError" in envelope.error_trace
        assert harness.states.statuses == [TaskStatus.RUNNING, TaskStatus.DEAD_LETTER]
        assert harness.observer.events[1:] == [
            (
                "dead_lettered",
                task.task_id,
                DeadLetterReason.RETRIES_EXHAUSTED,
                ErrorClass.UPSTREAM_UNAVAILABLE,
            ),
            ("finished", task.task_id, TaskStatus.DEAD_LETTER, 1.25),
        ]

    async def test_threshold_is_configurable(self, task: TaskPayload) -> None:
        harness = Harness(settings=WorkerSettings(dlq_threshold=0))
        outcome = await harness.engine(_handler(harness, error=UpstreamHTTPError(429))).execute(
            task.model_dump_json(), attempt=1
        )
        assert isinstance(outcome, TaskResult)
        assert outcome.status is TaskStatus.DEAD_LETTER
        assert harness.dead_letters.envelopes[0].retry_attempts == 0

    @pytest.mark.parametrize("status", [400, 401, 403, 422])
    async def test_client_errors_skip_retries(
        self, harness: Harness, task: TaskPayload, status: int
    ) -> None:
        outcome = await harness.engine(_handler(harness, error=UpstreamHTTPError(status))).execute(
            task.model_dump_json(), attempt=1
        )
        assert isinstance(outcome, TaskResult)
        assert outcome.status is TaskStatus.DEAD_LETTER
        [envelope] = harness.dead_letters.envelopes
        assert envelope.retry_attempts == 0
        assert envelope.failure_reason.startswith("non_retryable: client_error")

    async def test_unsupported_task_type(self, harness: Harness) -> None:
        task = make_task(task_type=TaskType.AUDIO_TRANSCRIPTION)
        outcome = await harness.engine(_handler(harness)).execute(task.model_dump_json(), attempt=1)
        assert isinstance(outcome, TaskResult)
        assert outcome.status is TaskStatus.DEAD_LETTER
        assert "UnsupportedTaskTypeError" in harness.dead_letters.envelopes[0].failure_reason

    async def test_no_handlers_registered(self, harness: Harness, task: TaskPayload) -> None:
        outcome = await harness.engine(None).execute(task.model_dump_json(), attempt=1)
        assert isinstance(outcome, TaskResult)
        assert outcome.status is TaskStatus.DEAD_LETTER

    async def test_non_json_handler_output_is_a_poison_pill(
        self, harness: Harness, task: TaskPayload
    ) -> None:
        async def bad_output(_: TaskPayload) -> TaskOutput:
            return {"handle": object()}  # type: ignore[dict-item]

        outcome = await harness.engine(bad_output).execute(task.model_dump_json(), attempt=1)
        assert isinstance(outcome, TaskResult)
        assert outcome.status is TaskStatus.DEAD_LETTER
        assert "validation" in harness.dead_letters.envelopes[0].failure_reason

    async def test_unexpected_bug_fails_fast(self, harness: Harness, task: TaskPayload) -> None:
        outcome = await harness.engine(_handler(harness, error=KeyError("text"))).execute(
            task.model_dump_json(), attempt=1
        )
        assert isinstance(outcome, TaskResult)
        assert outcome.status is TaskStatus.DEAD_LETTER
        assert harness.dead_letters.envelopes[0].failure_reason.startswith(
            "non_retryable: unexpected"
        )

    async def test_reason_and_trace_are_bounded(self, harness: Harness, task: TaskPayload) -> None:
        async def deep(_: TaskPayload) -> TaskOutput:
            raise RuntimeError("x" * 40_000 + " root cause")

        await harness.engine(deep).execute(task.model_dump_json(), attempt=1)
        [envelope] = harness.dead_letters.envelopes
        assert len(envelope.failure_reason) == MAX_FAILURE_REASON_CHARS
        assert envelope.failure_reason.endswith("…")
        assert envelope.error_trace is not None
        assert len(envelope.error_trace) == MAX_ERROR_TRACE_CHARS
        assert envelope.error_trace.startswith("…")
        assert "root cause" in envelope.error_trace

    async def test_dlq_outage_marks_task_failed_and_raises(
        self, harness: Harness, task: TaskPayload
    ) -> None:
        harness.dead_letters.error = RedisConnectionError("dlq down")
        engine = harness.engine(_handler(harness, error=UpstreamHTTPError(400)))
        with pytest.raises(DeadLetterPublishError) as excinfo:
            await engine.execute(task.model_dump_json(), attempt=1)
        assert excinfo.value.task_id == task.task_id
        assert isinstance(excinfo.value.__cause__, RedisConnectionError)
        assert harness.states.statuses == [TaskStatus.RUNNING, TaskStatus.FAILED]
        assert harness.observer.events[-1] == ("finished", task.task_id, TaskStatus.FAILED, 1.25)


class TestRejections:
    @pytest.mark.parametrize("raw", ["not json", "{}", '{"task_type": "summarization"}'])
    async def test_malformed_message(self, harness: Harness, raw: str) -> None:
        with pytest.raises(MalformedTaskError):
            await harness.engine(_handler(harness)).execute(raw, attempt=1)
        assert harness.states.history == []
        assert harness.observer.events == []

    async def test_python_mode_dump_is_rejected_by_strict_wire_contract(
        self, harness: Harness, task: TaskPayload
    ) -> None:
        raw = str(task.model_dump())
        with pytest.raises(MalformedTaskError):
            await harness.engine(_handler(harness)).execute(raw, attempt=1)

    async def test_attempt_is_one_based(self, harness: Harness, task: TaskPayload) -> None:
        with pytest.raises(ValueError, match="1-based"):
            await harness.engine(_handler(harness)).execute(task.model_dump_json(), attempt=0)

    async def test_cancellation_propagates(self, harness: Harness, task: TaskPayload) -> None:
        async def cancelled(_: TaskPayload) -> TaskOutput:
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await harness.engine(cancelled).execute(task.model_dump_json(), attempt=1)
        assert harness.dead_letters.envelopes == []
