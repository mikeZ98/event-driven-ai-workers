"""arq wiring: the job function, worker lifecycle and the producer-side enqueue helper.

Tasks cross Redis as ``TaskPayload.model_dump_json()`` strings and results come back as
``TaskResult.model_dump_json()``, so strict schemas never meet lax Python-mode coercion.
"""

from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from typing import Any

import structlog
from arq import cron
from arq.connections import ArqRedis
from arq.jobs import Job
from arq.worker import Retry, func

from event_driven_ai_workers.core.config import RedisSettings, WorkerSettings
from event_driven_ai_workers.core.dlq import RedisDeadLetterQueue
from event_driven_ai_workers.core.engine import RetryScheduled, TaskEngine
from event_driven_ai_workers.core.ports import EngineObserver, NullObserver, TaskHandler, resolve
from event_driven_ai_workers.core.tracking import RedisTaskStateStore
from event_driven_ai_workers.schemas import TaskPayload, TaskType

TASK_FUNCTION = "run_task"
"""arq function name producers enqueue; part of the wire contract."""

QUEUE_DEPTH_FUNCTION = "sample_queue_depth"

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

    async def sample_queue_depth(self, ctx: WorkerContext) -> int:
        """arq cron: publish the number of queued (incl. deferred) jobs."""
        redis: ArqRedis = ctx["redis"]
        depth: int = await resolve(redis.zcard(self._settings.queue_name))
        self._observer.queue_depth(self._settings.queue_name, depth)
        return depth


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
    step = worker.queue_depth_sample_s
    return {
        "functions": [
            func(
                runtime.run_task,
                name=TASK_FUNCTION,
                max_tries=worker.arq_max_tries,
                timeout=worker.arq_job_timeout_s,
            )
        ],
        "cron_jobs": [
            cron(
                runtime.sample_queue_depth,
                name=QUEUE_DEPTH_FUNCTION,
                second=set(range(0, 60, step)),
                run_at_startup=True,
                unique=False,  # every worker refreshes its own gauge
                max_tries=1,
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
    }


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
