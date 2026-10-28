"""Worker entrypoint: embedded /metrics listener (over a Unix socket), run loop and CLI."""

import asyncio
import os
import runpy
import shutil
import signal
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import fakeredis
import httpx
import pytest
from prometheus_client import CollectorRegistry, Counter
from redis.exceptions import ConnectionError as RedisConnectionError
from uvicorn.importer import import_from_string

from event_driven_ai_workers import cli, worker_main
from event_driven_ai_workers.api import create_app
from event_driven_ai_workers.core import (
    RedisTaskStateStore,
    WorkerSettings,
    enqueue_task,
    health_check_key,
)
from event_driven_ai_workers.schemas import TaskStatus
from event_driven_ai_workers.worker_main import MetricsServerSettings, run_worker, serve_metrics
from tests.conftest import make_pool
from tests.support import make_task


@pytest.fixture
def uds() -> Iterator[Path]:
    # AF_UNIX paths are capped at ~104 bytes on macOS; pytest's tmp_path is often longer.
    directory = Path(tempfile.mkdtemp(prefix="edaw-", dir="/tmp"))
    yield directory / "metrics.sock"
    shutil.rmtree(directory, ignore_errors=True)


def _absent(path: Path) -> bool:
    """A local stat: cheap enough to call from async tests."""
    return not path.exists()


async def _scrape(path: Path, url: str = "/metrics") -> httpx.Response:
    transport = httpx.AsyncHTTPTransport(uds=str(path))
    async with httpx.AsyncClient(transport=transport, base_url="http://worker") as client:
        return await client.get(url)


async def _scrape_when(path: Path, expected: str) -> str:
    """Scrape until ``expected`` appears (the listener and cron start asynchronously)."""
    for _ in range(500):
        try:
            text = (await _scrape(path)).text
        except httpx.TransportError:
            text = ""
        if expected in text:
            return text
        await asyncio.sleep(0.01)
    raise AssertionError(f"{expected!r} never appeared at {path}")


class TestMetricsServer:
    async def test_serves_registry_and_cleans_up(self, uds: Path) -> None:
        registry = CollectorRegistry()
        Counter("probe", "test counter", registry=registry).inc()

        async with serve_metrics(registry, MetricsServerSettings(uds=uds)):
            response = await _scrape(uds)
            missing = await _scrape(uds, "/other")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert "probe_total 1.0" in response.text
        assert missing.status_code == 404
        assert _absent(uds)

    async def test_disabled_serves_nothing(self, uds: Path) -> None:
        async with serve_metrics(
            CollectorRegistry(), MetricsServerSettings(enabled=False, uds=uds)
        ):
            assert _absent(uds)

    async def test_bind_failure_raises(self) -> None:
        settings = MetricsServerSettings(uds=Path("/nonexistent-edaw-dir/metrics.sock"))
        with pytest.raises(OSError, match="No such file"):
            async with serve_metrics(CollectorRegistry(), settings):
                pass  # pragma: no cover

    async def test_startup_failure_raises(self, uds: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _fail(*_: object, **__: object) -> None:
            raise OSError("listen failed")

        monkeypatch.setattr(worker_main._EmbeddedServer, "startup", _fail)
        with pytest.raises(OSError, match="listen failed"):
            async with serve_metrics(CollectorRegistry(), MetricsServerSettings(uds=uds)):
                pass  # pragma: no cover
        assert _absent(uds)

    async def test_exit_during_startup_raises(
        self, uds: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _give_up(self: worker_main._EmbeddedServer, **__: object) -> None:
            self.should_exit = True

        monkeypatch.setattr(worker_main._EmbeddedServer, "startup", _give_up)
        with pytest.raises(RuntimeError, match="exited during startup"):
            async with serve_metrics(CollectorRegistry(), MetricsServerSettings(uds=uds)):
                pass  # pragma: no cover

    def test_tcp_listener_binds_host_and_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        bound: list[Any] = []
        monkeypatch.setattr(worker_main.socket, "create_server", bound.append)

        worker_main._bind(MetricsServerSettings(host="0.0.0.0", port=9200))  # noqa: S104

        assert bound == [("0.0.0.0", 9200)]  # noqa: S104

    def test_settings_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EDAW_METRICS_PORT", "9300")
        monkeypatch.setenv("EDAW_METRICS_ENABLED", "false")

        settings = MetricsServerSettings()

        assert (settings.port, settings.enabled, settings.uds) == (9300, False, None)


class TestRunWorker:
    async def test_burst_processes_queue_and_records_metrics(
        self, redis_server: fakeredis.FakeServer, fast_settings: WorkerSettings
    ) -> None:
        producer = make_pool(redis_server)
        task = make_task()
        await enqueue_task(producer, task, queue_name=fast_settings.queue_name)
        registry = CollectorRegistry()

        await run_worker(
            worker=fast_settings,
            metrics=MetricsServerSettings(enabled=False),
            registry=registry,
            redis_pool=make_pool(redis_server),
            burst=True,
        )

        states = RedisTaskStateStore(producer, prefix=fast_settings.state_key_prefix, ttl_s=60)
        state = await states.get(task.task_id)
        assert state is not None
        assert state.status is TaskStatus.COMPLETED
        labels = {"task_type": "summarization", "status": "COMPLETED"}
        assert registry.get_sample_value("arq_task_latency_count", labels) == 1
        await producer.aclose()

    @pytest.mark.parametrize("trigger", ["event", "sigterm"])
    async def test_serves_metrics_and_heartbeat_until_stopped(
        self,
        redis_server: fakeredis.FakeServer,
        fast_settings: WorkerSettings,
        uds: Path,
        trigger: str,
    ) -> None:
        inspect = make_pool(redis_server)
        stop = asyncio.Event()
        runner = asyncio.create_task(
            run_worker(
                worker=fast_settings,
                metrics=MetricsServerSettings(uds=uds),
                redis_pool=make_pool(redis_server),
                stop=stop,
            )
        )

        try:
            text = await _scrape_when(uds, 'arq_queued_tasks{queue="edaw:queue"} 0.0')
            assert "process_" in text or "python_info" in text
            assert await inspect.exists(health_check_key(fast_settings.queue_name))

            if trigger == "event":
                stop.set()
            else:
                os.kill(os.getpid(), signal.SIGTERM)
        finally:
            stop.set()  # a failed assertion must not leak the listener's socket
            await asyncio.wait_for(runner, timeout=5)

        assert _absent(uds)
        assert not await inspect.exists(health_check_key(fast_settings.queue_name))
        await inspect.aclose()

    async def test_redis_outage_surfaces(
        self, redis_server: fakeredis.FakeServer, fast_settings: WorkerSettings
    ) -> None:
        redis_server.connected = False

        with pytest.raises(RedisConnectionError):
            await run_worker(
                worker=fast_settings,
                metrics=MetricsServerSettings(enabled=False),
                redis_pool=make_pool(redis_server),
            )


class TestCli:
    def test_api_command_runs_uvicorn_factory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[Any, ...]] = []
        monkeypatch.setattr(cli.uvicorn, "run", lambda *a, **k: calls.append((a, k)))
        monkeypatch.setenv("EDAW_API_PORT", "9000")

        assert cli.main(["api"]) == 0

        assert calls == [((cli.APP_FACTORY,), {"factory": True, "host": "127.0.0.1", "port": 9000})]
        assert import_from_string(cli.APP_FACTORY) is create_app

    @pytest.mark.parametrize(
        ("argv", "burst"), [(["worker"], False), (["worker", "--burst"], True)]
    )
    def test_worker_command_runs_worker(
        self, monkeypatch: pytest.MonkeyPatch, argv: list[str], burst: bool
    ) -> None:
        calls: list[bool] = []

        async def _run_worker(*, burst: bool) -> None:
            calls.append(burst)

        monkeypatch.setattr(cli, "run_worker", _run_worker)

        assert cli.main(argv) == 0
        assert calls == [burst]

    def test_requires_a_command(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exited:
            cli.main([])
        assert exited.value.code == 2
        assert "api" in capsys.readouterr().err

    def test_module_entrypoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cli, "main", lambda: 0)
        with pytest.raises(SystemExit) as exited:
            runpy.run_module("event_driven_ai_workers", run_name="__main__")
        assert exited.value.code == 0
