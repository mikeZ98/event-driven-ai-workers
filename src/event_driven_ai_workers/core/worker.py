"""arq wiring: the job function, worker lifecycle and the producer-side enqueue helper.

Tasks cross Redis as ``TaskPayload.model_dump_json()`` strings and results come back as
``TaskResult.model_dump_json()``, so strict schemas never meet lax Python-mode coercion.
"""

import asyncio
import contextlib
from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from typing import Any

import structlog
from arq.connections import ArqRedis
from arq.constants import health_check_key_suffix
from arq.jobs import Job
from arq.worker import Retry, func
from redis.exceptions import RedisError

from event_driven_ai_workers.core.config import RedisSettings, WorkerSettings
from event_driven_ai_workers.core.dlq import RedisDeadLetterQueue
from event_driven_ai_workers.core.engine import RetryScheduled, TaskEngine
from event_driven_ai_workers.core.ports import EngineObserver, NullObserver, TaskHandler, resolve
from event_driven_ai_workers.core.tracking import RedisTaskStateStore
from event_driven_ai_workers.schemas import TaskPayload, TaskType

TASK_FUNCTION = "run_task"
"""arq function name producers enqueue; part of the wire contract."""


def health_check_key(queue_name: str) -> str:
    """Heartbeat key every live worker on ``queue_name`` refreshes (arq's convention)."""
    return queue_name + health_check_key_suffix


HandlerFactory = Callable[[], AbstractAsyncContextManager[Mapping[TaskType, TaskHandler]]]
"""Opens the workload handlers (and their HTTP clients) for the worker's lifetime."""

WorkerContext = dict[Any, Any]

_ENGINE = "edaw.engine"
_EXIT_STACK = "edaw.exit_stack"

_log = structlog.get_logger(__name__)


class WorkerRuntime:
    """Owns the engine for one worker process; its bound methods are arq callbacks."""

    def __init__(
        self,
        handlers: HandlerFactory,
        *,
        settings: WorkerSettings,
        observer: EngineObserver | None = None,
    ) -> None:
        self._handlers = handlers
        self._settings = settings
        self._observer: EngineObserver = observer or NullObserver()

    async def startup(self, ctx: WorkerContext) -> None:
        redis: ArqRedis = ctx["redis"]
        stack = AsyncExitStack()
        handlers = await stack.enter_async_context(self._handlers())
        ctx[_EXIT_STACK] = stack
        ctx[_ENGINE] = TaskEngine(
            handlers,
            dead_letters=RedisDeadLetterQueue(redis, self._settings.dlq_name),
            states=RedisTaskStateStore(
                redis, prefix=self._settings.state_key_prefix, ttl_s=self._settings.state_ttl_s
            ),
            settings=self._settings,
            observer=self._observer,
        )
        sampler = asyncio.create_task(self._sample_queue_depth_forever(redis))
        stack.push_async_callback(_cancel, sampler)
        _log.info("worker.started", queue=self._settings.queue_name, task_types=sorted(handlers))

    async def shutdown(self, ctx: WorkerContext) -> None:
        stack: AsyncExitStack | None = ctx.pop(_EXIT_STACK, None)
        ctx.pop(_ENGINE, None)
        if stack is not None:
            await stack.aclose()
        _log.info("worker.stopped", queue=self._settings.queue_name)

    async def run_task(self, ctx: WorkerContext, raw_task: str) -> str:
        """arq job: execute one attempt; transient failures re-enqueue via ``Retry``."""
        engine: TaskEngine = ctx[_ENGINE]
        outcome = await engine.execute(raw_task, attempt=ctx["job_try"])
        if isinstance(outcome, RetryScheduled):
            raise Retry(defer=outcome.delay_s)
        return outcome.model_dump_json()

    async def sample_queue_depth(self, redis: ArqRedis) -> int | None:
        """Publish the queue's length (deferred retries and in-flight jobs included).

        Returns ``None`` when Redis is unreachable; the gauge keeps its last value.
        """
        try:
            depth: int = await resolve(redis.zcard(self._settings.queue_name))
        except (RedisError, OSError) as exc:
            _log.warning("worker.queue_depth_failed", error=type(exc).__name__)
            return None
        self._observer.queue_depth(self._settings.queue_name, depth)
        return depth

    async def _sample_queue_depth_forever(self, redis: ArqRedis) -> None:
        """Sample now, then every ``queue_depth_sample_s``.

        A plain task beside arq rather than an arq cron job: a job would count itself
        in the queue it measures, and could not run while every ``max_jobs`` slot is
        busy, which is exactly when the backlog matters.
        """
        while True:
            await self.sample_queue_depth(redis)
            await asyncio.sleep(self._settings.queue_depth_sample_s)


def create_worker_settings(
    handlers: HandlerFactory,
    *,
    worker: WorkerSettings | None = None,
    redis: RedisSettings | None = None,
    observer: EngineObserver | None = None,
) -> dict[str, Any]:
    """Keyword arguments for ``arq.worker.Worker`` (also accepted by ``arq`` CLI/``run_worker``)."""
    worker = worker or WorkerSettings()
    redis = redis or RedisSettings()
    runtime = WorkerRuntime(handlers, settings=worker, observer=observer)
    return {
        "functions": [
            func(
                runtime.run_task,
                name=TASK_FUNCTION,
                max_tries=worker.arq_max_tries,
                timeout=worker.arq_job_timeout_s,
            )
        ],
        "on_startup": runtime.startup,
        "on_shutdown": runtime.shutdown,
        "queue_name": worker.queue_name,
        "redis_settings": redis.to_arq(),
        "max_jobs": worker.max_jobs,
        "job_timeout": worker.arq_job_timeout_s,
        "max_tries": worker.arq_max_tries,
        "keep_result": worker.state_ttl_s,
        "health_check_interval": worker.health_check_interval_s,
        "health_check_key": health_check_key(worker.queue_name),
    }


async def _cancel(task: asyncio.Task[None]) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def enqueue_task(pool: ArqRedis, task: TaskPayload, *, queue_name: str) -> Job | None:
    """Enqueue ``task`` as JSON; returns ``None`` if its idempotency key is already queued.

    The arq job id is the idempotency key, so a duplicate submission is dropped
    while the original job (or its kept result) still exists.
    """
    return await pool.enqueue_job(
        TASK_FUNCTION,
        task.model_dump_json(),
        _job_id=task.idempotency_key,
        _queue_name=queue_name,
    )
