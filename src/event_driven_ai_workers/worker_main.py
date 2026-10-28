"""Worker process entrypoint: the arq worker loop plus a Prometheus ``/metrics`` listener.

Worker metrics live in the worker process, so each worker serves its own registry on
``EDAW_METRICS_HOST:EDAW_METRICS_PORT`` (or a Unix socket). The listener is an embedded
uvicorn server sharing the worker's event loop; the worker owns SIGINT/SIGTERM.
"""

import asyncio
import contextlib
import signal
import socket
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
import uvicorn
from arq.connections import ArqRedis
from arq.worker import Worker
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from event_driven_ai_workers.core import (
    HandlerFactory,
    RedisSettings,
    WorkerSettings,
    create_worker_settings,
)
from event_driven_ai_workers.tools import WorkerMetrics, create_registry, summarization_handlers

_STARTUP_POLL_S = 0.01

_log = structlog.get_logger(__name__)


class MetricsServerSettings(BaseSettings):
    """The worker's metrics listener, loaded from ``EDAW_METRICS_*`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="EDAW_METRICS_", extra="forbid", frozen=True, validate_default=True
    )

    enabled: bool = True
    host: str = Field(default="127.0.0.1", min_length=1)
    """Loopback by default; containers set ``0.0.0.0`` so Prometheus can scrape."""

    port: int = Field(default=9100, ge=1, le=65_535)
    uds: Path | None = None
    """Serve on this Unix socket instead of ``host:port``."""


def metrics_app(registry: CollectorRegistry) -> Starlette:
    """ASGI app exposing ``registry`` at ``GET /metrics``."""

    async def metrics(_request: Request) -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return Starlette(routes=[Route("/metrics", metrics, methods=["GET"])])


class _EmbeddedServer(uvicorn.Server):
    """uvicorn without its signal capture: the hosting worker decides when to stop."""

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


def _bind(settings: MetricsServerSettings) -> socket.socket:
    """Bind before handing the socket to uvicorn, so failures raise ``OSError`` here
    instead of uvicorn calling ``sys.exit`` inside a task."""
    if settings.uds is None:
        return socket.create_server((settings.host, settings.port))
    settings.uds.unlink(missing_ok=True)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(str(settings.uds))
    except OSError:
        sock.close()
        raise
    return sock


@asynccontextmanager
async def serve_metrics(
    registry: CollectorRegistry, settings: MetricsServerSettings
) -> AsyncIterator[None]:
    """Serve ``registry`` in the background for the duration of the block."""
    if not settings.enabled:
        yield
        return
    sock = _bind(settings)
    config = uvicorn.Config(
        metrics_app(registry), lifespan="off", access_log=False, log_level="warning"
    )
    server = _EmbeddedServer(config)
    serving = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        while not server.started:
            if serving.done():
                await serving
                raise RuntimeError("metrics server exited during startup")
            await asyncio.sleep(_STARTUP_POLL_S)
        _log.info("worker.metrics_listening", address=str(settings.uds or sock.getsockname()))
        yield
    finally:
        server.should_exit = True
        with contextlib.suppress(Exception):
            await serving
        sock.close()
        if settings.uds is not None:
            settings.uds.unlink(missing_ok=True)


@contextlib.contextmanager
def _stop_on_signals(stop: asyncio.Event) -> Iterator[None]:
    loop = asyncio.get_running_loop()
    signals = (signal.SIGINT, signal.SIGTERM)
    for sig in signals:
        loop.add_signal_handler(sig, stop.set)
    try:
        yield
    finally:
        for sig in signals:
            loop.remove_signal_handler(sig)


async def close_worker(worker: Worker) -> None:
    """``Worker.close()``, safe while jobs are still being finished.

    arq's ``close()`` cancels in-flight jobs, then gathers them without
    ``return_exceptions``: a job cancelled while arq records its outcome (a shielded
    ``finish_job``) re-raises ``CancelledError`` out of ``close()``, which then skips
    deleting the heartbeat key and ``on_shutdown`` (the LLM client, the queue-depth
    sampler). Cancelling and reaping the jobs first leaves ``close()`` nothing to gather.
    Cancelled jobs stay queued and are retried (the retry budget keeps headroom).
    """
    jobs = list(worker.tasks.values())
    for job in jobs:
        job.cancel()
    await asyncio.gather(*jobs, return_exceptions=True)
    worker.tasks.clear()
    await worker.close()


async def run_worker(
    handlers: HandlerFactory = summarization_handlers,
    *,
    worker: WorkerSettings | None = None,
    redis: RedisSettings | None = None,
    metrics: MetricsServerSettings | None = None,
    registry: CollectorRegistry | None = None,
    redis_pool: ArqRedis | None = None,
    burst: bool = False,
    stop: asyncio.Event | None = None,
) -> None:
    """Run one worker until SIGINT/SIGTERM or ``stop`` (or an empty queue with ``burst``).

    On stop, in-flight jobs are cancelled and arq re-queues them (the retry budget keeps
    headroom for this). ``redis_pool`` replaces the pool arq would open; the worker
    closes it on exit.
    """
    registry = create_registry() if registry is None else registry
    options = create_worker_settings(
        handlers, worker=worker, redis=redis, observer=WorkerMetrics(registry)
    )
    arq_worker = Worker(**options, redis_pool=redis_pool, burst=burst, handle_signals=False)
    stop = stop or asyncio.Event()
    with _stop_on_signals(stop):
        async with serve_metrics(registry, metrics or MetricsServerSettings()):
            main = asyncio.create_task(arq_worker.main())
            stopping = asyncio.create_task(stop.wait())
            try:
                await asyncio.wait({main, stopping}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                stopping.cancel()
                if not main.done():
                    main.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await main
                await close_worker(arq_worker)
            if not main.cancelled():
                main.result()  # surface a crash, e.g. Redis unreachable at startup
