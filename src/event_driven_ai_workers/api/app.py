"""Application factory: Redis pool lifespan, routes, CORS and problem handlers.

Run with ``edaw api`` or ``uvicorn --factory event_driven_ai_workers.api.app:create_app``.
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from functools import partial
from http import HTTPStatus

import structlog
from arq import create_pool
from arq.connections import ArqRedis
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from prometheus_client import CollectorRegistry
from redis.exceptions import RedisError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.cors import CORSMiddleware

from event_driven_ai_workers import __version__
from event_driven_ai_workers.api.config import ApiSettings
from event_driven_ai_workers.api.metrics import ApiMetrics
from event_driven_ai_workers.api.problems import ProblemError, problem_response
from event_driven_ai_workers.api.routes import (
    ApiServices,
    metrics_router,
    router,
    unavailable,
    validation_detail,
)
from event_driven_ai_workers.api.submission import TaskSubmitter
from event_driven_ai_workers.core import RedisSettings, WorkerSettings
from event_driven_ai_workers.guardrails import Backpressure, GuardrailSettings, PayloadGuard
from event_driven_ai_workers.tools.metrics import create_registry

PoolFactory = Callable[[], Awaitable[ArqRedis]]
"""Opens the app's Redis pool at startup; the app closes it at shutdown."""

_log = structlog.get_logger(__name__)


def _install_problem_handlers(app: FastAPI) -> None:
    @app.exception_handler(ProblemError)
    async def _problem(_request: Request, exc: ProblemError) -> Response:
        return problem_response(exc)

    @app.exception_handler(RequestValidationError)
    async def _invalid(_request: Request, exc: RequestValidationError) -> Response:
        return problem_response(
            ProblemError(422, "invalid-request", validation_detail(exc.errors()))
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(_request: Request, exc: StarletteHTTPException) -> Response:
        phrase = HTTPStatus(exc.status_code).phrase
        return problem_response(ProblemError(exc.status_code, "http-error", phrase))

    @app.exception_handler(RedisError)
    async def _redis(_request: Request, _exc: RedisError) -> Response:
        return problem_response(unavailable())

    @app.exception_handler(Exception)
    async def _unhandled(_request: Request, _exc: Exception) -> Response:
        return problem_response(
            ProblemError(500, "internal-error", "An unexpected error occurred.")
        )


def create_app(
    api: ApiSettings | None = None,
    worker: WorkerSettings | None = None,
    redis: RedisSettings | None = None,
    guardrails: GuardrailSettings | None = None,
    *,
    pool_factory: PoolFactory | None = None,
    registry: CollectorRegistry | None = None,
) -> FastAPI:
    """Build the producer API. Settings default to the ``EDAW_*`` environment.

    ``pool_factory`` replaces ``arq.create_pool`` (tests use fakeredis); ``registry``
    replaces the app's private Prometheus registry.
    """
    api = api or ApiSettings()
    worker = worker or WorkerSettings()
    redis = redis or RedisSettings()
    guardrails = guardrails or GuardrailSettings()
    registry = create_registry() if registry is None else registry
    metrics = ApiMetrics(registry)
    open_pool: PoolFactory = pool_factory or partial(
        create_pool, redis.to_arq(), default_queue_name=worker.queue_name
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        pool = await open_pool()
        backpressure = Backpressure(pool, queue_name=worker.queue_name, settings=guardrails)
        app.state.services = ApiServices(
            settings=api,
            worker=worker,
            pool=pool,
            submitter=TaskSubmitter(
                pool,
                worker=worker,
                key_prefix=api.idempotency_key_prefix,
                backpressure=backpressure,
            ),
            guard=PayloadGuard(guardrails),
            metrics=metrics,
            registry=registry,
        )
        _log.info("api.started", queue=worker.queue_name)
        try:
            yield
        finally:
            del app.state.services
            await pool.aclose(close_connection_pool=True)
            _log.info("api.stopped")

    app = FastAPI(title="event-driven-ai-workers", version=__version__, lifespan=lifespan)
    app.include_router(router)
    if api.metrics_enabled:
        app.include_router(metrics_router)
    if api.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(api.cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["content-type", "idempotency-key"],
            expose_headers=["location", "retry-after", "idempotent-replayed"],
        )
    _install_problem_handlers(app)
    return app
