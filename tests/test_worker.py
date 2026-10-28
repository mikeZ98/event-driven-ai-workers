"""arq integration: a real ``arq.worker.Worker`` against fakeredis (no sockets, no server)."""

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

import pytest
from arq.connections import ArqRedis
from arq.worker import Worker
from redis.exceptions import ConnectionError as RedisConnectionError

from event_driven_ai_workers.core import (
    TASK_FUNCTION,
    RedisDeadLetterQueue,
    RedisSettings,
    RedisTaskStateStore,
    TaskHandler,
    TaskOutput,
    UpstreamHTTPError,
    WorkerRuntime,
    WorkerSettings,
    create_worker_settings,
    enqueue_task,
)
from event_driven_ai_workers.core.ports import resolve
from event_driven_ai_workers.schemas import (
    DeadLetterEnvelope,
    TaskPayload,
    TaskResult,
    TaskStatus,
    TaskType,
)
from tests.support import RecordingObserver, make_task


class Script:
    """Handler that fails with the queued errors, then succeeds; tracks lifecycle."""

    def __init__(self, *errors: Exception) -> None:
        self.errors = list(errors)
        self.calls = 0
        self.opened = False
        self.closed = False

    async def handle(self, task: TaskPayload) -> TaskOutput:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return {"summary": f"summary of {task.idempotency_key}"}

    @asynccontextmanager
    async def factory(self) -> AsyncIterator[Mapping[TaskType, TaskHandler]]:
        self.opened = True
        try:
            yield {TaskType.SUMMARIZATION: self.handle}
        finally:
            self.closed = True


async def _run(
    pool: ArqRedis,
    script: Script,
    settings: WorkerSettings,
    observer: RecordingObserver | None = None,
) -> Worker:
    options: dict[str, Any] = create_worker_settings(
        script.factory, worker=settings, observer=observer
    )
    worker = Worker(**options, redis_pool=pool, burst=True, poll_delay=0.005, handle_signals=False)
    try:
        await worker.main()
    finally:
        await worker.close()
    return worker


def _store(pool: ArqRedis, settings: WorkerSettings) -> RedisTaskStateStore:
    return RedisTaskStateStore(pool, prefix=settings.state_key_prefix, ttl_s=settings.state_ttl_s)


async def test_completed_task_round_trips_as_json(
    arq_pool: ArqRedis, fast_settings: WorkerSettings
) -> None:
    task = make_task()
    job = await enqueue_task(arq_pool, task, queue_name=fast_settings.queue_name)
    assert job is not None
    script = Script()

    worker = await _run(arq_pool, script, fast_settings)

    result = TaskResult.model_validate_json(await job.result(timeout=1))
    assert result.status is TaskStatus.COMPLETED
    assert result.result == {"summary": "summary of doc-1"}
    assert await _store(arq_pool, fast_settings).get(task.task_id) == result
    # jobs_complete also counts the queue-depth cron run at startup.
    assert (worker.jobs_retried, worker.jobs_failed) == (0, 0)
    assert script.opened
    assert script.closed


async def test_transient_failures_are_retried_until_success(
    arq_pool: ArqRedis, fast_settings: WorkerSettings
) -> None:
    task = make_task()
    job = await enqueue_task(arq_pool, task, queue_name=fast_settings.queue_name)
    assert job is not None
    script = Script(UpstreamHTTPError(429, retry_after_s=0.0), UpstreamHTTPError(503))
    observer = RecordingObserver()

    worker = await _run(arq_pool, script, fast_settings, observer)

    assert script.calls == 3
    assert worker.jobs_retried == 2
    result = TaskResult.model_validate_json(await job.result(timeout=1))
    assert result.status is TaskStatus.COMPLETED
    assert observer.kinds().count("retry") == 2
    assert await RedisDeadLetterQueue(arq_pool, fast_settings.dlq_name).size() == 0


async def test_exhausted_retries_route_to_the_dlq(
    arq_pool: ArqRedis, fast_settings: WorkerSettings
) -> None:
    task = make_task()
    job = await enqueue_task(arq_pool, task, queue_name=fast_settings.queue_name)
    assert job is not None
    script = Script(*(UpstreamHTTPError(503) for _ in range(20)))

    worker = await _run(arq_pool, script, fast_settings)

    assert script.calls == 1 + fast_settings.dlq_threshold
    assert worker.jobs_retried == fast_settings.dlq_threshold
    result = TaskResult.model_validate_json(await job.result(timeout=1))
    assert result.status is TaskStatus.DEAD_LETTER
    [envelope] = await RedisDeadLetterQueue(arq_pool, fast_settings.dlq_name).peek()
    assert envelope.original_task == task
    assert envelope.retry_attempts == 5
    assert envelope.failure_reason.startswith("retries_exhausted")
    assert await _store(arq_pool, fast_settings).get(task.task_id) == result


async def test_poison_pill_is_quarantined_without_retry(
    arq_pool: ArqRedis, fast_settings: WorkerSettings
) -> None:
    task = make_task()
    await enqueue_task(arq_pool, task, queue_name=fast_settings.queue_name)
    script = Script(UpstreamHTTPError(422, "unprocessable"))

    worker = await _run(arq_pool, script, fast_settings)

    assert script.calls == 1
    assert worker.jobs_retried == 0
    [envelope] = await RedisDeadLetterQueue(arq_pool, fast_settings.dlq_name).peek()
    assert envelope.retry_attempts == 0
    assert envelope.failure_reason.startswith("non_retryable: client_error")


async def test_duplicate_idempotency_key_is_not_enqueued_twice(
    arq_pool: ArqRedis, fast_settings: WorkerSettings
) -> None:
    first = await enqueue_task(arq_pool, make_task(), queue_name=fast_settings.queue_name)
    second = await enqueue_task(arq_pool, make_task(), queue_name=fast_settings.queue_name)
    assert first is not None
    assert second is None
    script = Script()
    await _run(arq_pool, script, fast_settings)
    assert script.calls == 1


async def test_queue_depth_is_sampled_at_startup(
    arq_pool: ArqRedis, fast_settings: WorkerSettings
) -> None:
    for key in ("a", "b"):
        await enqueue_task(
            arq_pool, make_task(idempotency_key=key), queue_name=fast_settings.queue_name
        )
    observer = RecordingObserver()
    await _run(arq_pool, Script(), fast_settings, observer)
    depths = [e for e in observer.events if e[0] == "queue_depth"]
    assert depths
    assert depths[0][1] == fast_settings.queue_name


def test_worker_options_follow_settings() -> None:
    settings = WorkerSettings(queue_name="jobs", dlq_name="jobs:dlq", max_jobs=7)
    options = create_worker_settings(
        Script().factory, worker=settings, redis=RedisSettings.model_validate({"url": "redis://r"})
    )
    assert options["queue_name"] == "jobs"
    assert options["max_jobs"] == 7
    assert options["max_tries"] == settings.arq_max_tries
    assert options["redis_settings"].host == "r"
    [function] = options["functions"]
    assert function.name == TASK_FUNCTION
    assert function.max_tries == settings.arq_max_tries


def test_default_settings_come_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDAW_WORKER_QUEUE_NAME", "env-queue")
    assert create_worker_settings(Script().factory)["queue_name"] == "env-queue"


class FlakyRedis:
    """Wraps a pool, failing the first ``failures`` LPUSH calls."""

    def __init__(self, pool: ArqRedis, failures: int) -> None:
        self.pool = pool
        self.failures = failures
        self.attempts = 0

    async def lpush(self, key: str, *values: str) -> int:
        self.attempts += 1
        if self.attempts <= self.failures:
            raise RedisConnectionError("blip")
        return await resolve(self.pool.lpush(key, *values))


class TestRedisDeadLetterQueue:
    async def test_transient_redis_errors_are_retried(self, arq_pool: ArqRedis) -> None:
        flaky = FlakyRedis(arq_pool, failures=2)
        dlq = RedisDeadLetterQueue(flaky, "dlq", retry_base_s=0.0, retry_max_s=0.0)  # type: ignore[arg-type]
        envelope = DeadLetterEnvelope(
            original_task=make_task(), failure_reason="x", retry_attempts=0
        )
        await dlq.publish(envelope)
        assert flaky.attempts == 3
        assert await RedisDeadLetterQueue(arq_pool, "dlq").peek() == [envelope]

    async def test_gives_up_after_budget(self, arq_pool: ArqRedis) -> None:
        flaky = FlakyRedis(arq_pool, failures=10)
        dlq = RedisDeadLetterQueue(
            flaky,  # type: ignore[arg-type]
            "dlq",
            publish_attempts=3,
            retry_base_s=0.0,
            retry_max_s=0.0,
        )
        envelope = DeadLetterEnvelope(
            original_task=make_task(), failure_reason="x", retry_attempts=0
        )
        with pytest.raises(RedisConnectionError):
            await dlq.publish(envelope)
        assert flaky.attempts == 3

    async def test_peek_is_oldest_first_and_bounded(self, arq_pool: ArqRedis) -> None:
        dlq = RedisDeadLetterQueue(arq_pool, "dlq")
        envelopes = [
            DeadLetterEnvelope(
                original_task=make_task(idempotency_key=f"k{i}"),
                failure_reason=f"r{i}",
                retry_attempts=i,
            )
            for i in range(3)
        ]
        for envelope in envelopes:
            await dlq.publish(envelope)
        assert dlq.key == "dlq"
        assert await dlq.size() == 3
        assert await dlq.peek() == envelopes
        assert await dlq.peek(limit=2) == envelopes[:2]
        assert await dlq.peek(limit=0) == []

    def test_requires_at_least_one_attempt(self, arq_pool: ArqRedis) -> None:
        with pytest.raises(ValueError, match="publish_attempts"):
            RedisDeadLetterQueue(arq_pool, "dlq", publish_attempts=0)


class TestRedisTaskStateStore:
    async def test_roundtrip_and_ttl(self, arq_pool: ArqRedis) -> None:
        store = RedisTaskStateStore(arq_pool, prefix="state:", ttl_s=120)
        result = TaskResult(task_id=make_task().task_id, status=TaskStatus.RUNNING)
        assert await store.get(result.task_id) is None
        await store.save(result)
        assert await store.get(result.task_id) == result
        assert 0 < await arq_pool.ttl(store.key(result.task_id)) <= 120


async def test_shutdown_without_startup_is_a_no_op(fast_settings: WorkerSettings) -> None:
    script = Script()
    await WorkerRuntime(script.factory, settings=fast_settings).shutdown({})
    assert not script.closed


async def test_resolve_passes_through_synchronous_results() -> None:
    assert await resolve(5) == 5
