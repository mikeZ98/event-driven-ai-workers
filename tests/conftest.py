"""Shared fixtures. The suite is hermetic: pytest-socket blocks every non-Unix socket and
Redis is replaced by fakeredis' in-process server."""

import os
from collections.abc import AsyncIterator

import arq.worker
import fakeredis
import pytest
from arq.connections import ArqRedis
from fakeredis.aioredis import FakeAsyncRedisConnection
from redis.asyncio import ConnectionPool

from event_driven_ai_workers.core import WorkerSettings


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


@pytest.fixture
async def arq_pool(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[ArqRedis]:
    """An arq pool backed by fakeredis (fakeredis lacks INFO, which arq only logs)."""

    async def _skip_info(*_: object, **__: object) -> None:
        return None

    monkeypatch.setattr(arq.worker, "log_redis_info", _skip_info)
    server = fakeredis.FakeServer()
    pool = ArqRedis(
        connection_pool=ConnectionPool(connection_class=FakeAsyncRedisConnection, server=server)
    )
    yield pool
    await pool.aclose()
