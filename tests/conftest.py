"""Shared fixtures. The suite is hermetic: pytest-socket blocks every non-Unix socket and
Redis is replaced by fakeredis' in-process server."""

import os
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass

import arq.worker
import fakeredis
import httpx
import pytest
from arq.connections import ArqRedis
from fakeredis.aioredis import FakeAsyncRedisConnection
from fastapi import FastAPI
from prometheus_client import CollectorRegistry
from redis.asyncio import ConnectionPool

from event_driven_ai_workers.api import ApiSettings, create_app
from event_driven_ai_workers.core import WorkerSettings
from event_driven_ai_workers.guardrails import GuardrailSettings


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings read EDAW_* variables; never let the developer's shell leak in."""
    for key in list(os.environ):
        if key.startswith("EDAW_"):
            monkeypatch.delenv(key)


@pytest.fixture
def fast_settings() -> WorkerSettings:
    """Production topology with millisecond backoff so retry loops finish instantly."""
    return WorkerSettings(
        backoff_base_s=0.001, backoff_max_s=0.004, task_timeout_s=2.0, queue_depth_sample_s=30
    )


def make_pool(server: fakeredis.FakeServer) -> ArqRedis:
    """A new arq pool on ``server``; pools on one server share its data, like real clients."""
    return ArqRedis(
        connection_pool=ConnectionPool(connection_class=FakeAsyncRedisConnection, server=server)
    )


@pytest.fixture
def redis_server(monkeypatch: pytest.MonkeyPatch) -> fakeredis.FakeServer:
    """An in-process Redis server (fakeredis lacks INFO, which arq only logs)."""

    async def _skip_info(*_: object, **__: object) -> None:
        return None

    monkeypatch.setattr(arq.worker, "log_redis_info", _skip_info)
    return fakeredis.FakeServer()


@pytest.fixture
async def arq_pool(redis_server: fakeredis.FakeServer) -> AsyncIterator[ArqRedis]:
    """An arq pool backed by the test's fakeredis server."""
    pool = make_pool(redis_server)
    yield pool
    await pool.aclose()


@dataclass
class ApiHarness:
    """A producer app with its lifespan running, plus a client and a Redis inspection pool."""

    app: FastAPI
    client: httpx.AsyncClient
    redis: ArqRedis
    worker: WorkerSettings
    registry: CollectorRegistry

    def submissions(self, outcome: str, task_type: str = "summarization") -> float:
        value = self.registry.get_sample_value(
            "edaw_api_submissions_total", {"task_type": task_type, "outcome": outcome}
        )
        return value or 0.0


ApiFactory = Callable[..., AbstractAsyncContextManager[ApiHarness]]


@pytest.fixture
def api_factory(
    redis_server: fakeredis.FakeServer, arq_pool: ArqRedis, fast_settings: WorkerSettings
) -> ApiFactory:
    """Builds apps over ASGITransport (in-process, no sockets) sharing the test's Redis."""

    @asynccontextmanager
    async def build(
        api: ApiSettings | None = None,
        guardrails: GuardrailSettings | None = None,
        *,
        raise_app_exceptions: bool = True,
    ) -> AsyncIterator[ApiHarness]:
        registry = CollectorRegistry()

        async def open_pool() -> ArqRedis:
            return make_pool(redis_server)

        app = create_app(
            api or ApiSettings(),
            fast_settings,
            guardrails=guardrails,
            pool_factory=open_pool,
            registry=registry,
        )
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=raise_app_exceptions)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=transport, base_url="http://edaw.test") as client,
        ):
            yield ApiHarness(app, client, arq_pool, fast_settings, registry)

    return build


@pytest.fixture
async def api(api_factory: ApiFactory) -> AsyncIterator[ApiHarness]:
    async with api_factory() as harness:
        yield harness
