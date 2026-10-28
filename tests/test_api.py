"""Producer API over ``httpx.ASGITransport``: submission, idempotency, polling, health, metrics."""

import asyncio
from typing import Any
from uuid import UUID, uuid4

import fakeredis
import httpx
import pytest
from arq.connections import ArqRedis
from arq.worker import Worker
from redis.exceptions import ConnectionError as RedisConnectionError

from event_driven_ai_workers.api import (
    PROBLEM_MEDIA_TYPE,
    ApiSettings,
    IdempotencyKeyInUseError,
    IdempotencyRecord,
    TaskSubmitter,
    create_app,
    fingerprint,
)
from event_driven_ai_workers.api import app as app_module
from event_driven_ai_workers.api import submission as submission_module
from event_driven_ai_workers.core import (
    RedisSettings,
    WorkerSettings,
    create_worker_settings,
    enqueue_task,
    health_check_key,
)
from event_driven_ai_workers.schemas import (
    SummarizationRequest,
    TaskPayload,
    TaskResult,
    TaskStatus,
    TaskType,
)
from event_driven_ai_workers.tools import summarization_handlers
from tests.conftest import ApiFactory, ApiHarness, make_pool
from tests.support import make_task

SUMMARIZE = "/v1/tasks/summarize"
TEXT = "Queues decouple producers from consumers and absorb bursts of load. " * 4
BODY: dict[str, Any] = {"text": TEXT, "max_length": 64, "focus_areas": ["queues"]}


async def _submit(
    api: ApiHarness, body: dict[str, Any] | None = None, key: str | None = "doc-1"
) -> httpx.Response:
    headers = {} if key is None else {"idempotency-key": key}
    return await api.client.post(SUMMARIZE, json=BODY if body is None else body, headers=headers)


def _problem(response: httpx.Response, status: int, kind: str) -> dict[str, Any]:
    assert response.status_code == status, response.text
    assert response.headers["content-type"] == PROBLEM_MEDIA_TYPE
    assert response.headers["cache-control"] == "no-store"
    body: dict[str, Any] = response.json()
    assert body["type"] == f"urn:event-driven-ai-workers:problem:{kind}"
    assert body["status"] == status
    return body


async def _queued(pool: ArqRedis, settings: WorkerSettings) -> list[TaskPayload]:
    jobs = await pool.queued_jobs(queue_name=settings.queue_name)
    return [TaskPayload.model_validate_json(job.args[0]) for job in jobs]


async def _drain(server: fakeredis.FakeServer, settings: WorkerSettings) -> Worker:
    """Run a real arq worker (simulated LLM) until the queue is empty."""
    options = create_worker_settings(summarization_handlers, worker=settings)
    worker = Worker(
        **options, redis_pool=make_pool(server), burst=True, poll_delay=0.005, handle_signals=False
    )
    try:
        await worker.main()
    finally:
        await worker.close()
    return worker


class TestSubmit:
    async def test_accepts_and_enqueues_task(self, api: ApiHarness) -> None:
        response = await _submit(api)

        assert response.status_code == 202
        body = response.json()
        task_id = UUID(body["task_id"])
        assert response.headers["location"] == f"/v1/tasks/{task_id}" == body["status_url"]
        assert "idempotent-replayed" not in response.headers
        assert body["status"] == "PENDING"
        assert body["idempotency_key"] == "doc-1"
        assert body["task_type"] == "summarization"

        [queued] = await _queued(api.redis, api.worker)
        assert queued.task_id == task_id
        assert queued.payload == BODY
        assert api.submissions("accepted") == 1

    async def test_poll_returns_pending_state(self, api: ApiHarness) -> None:
        task_id = (await _submit(api)).json()["task_id"]

        response = await api.client.get(f"/v1/tasks/{task_id}")

        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        state = TaskResult.model_validate_json(response.content)
        assert state.status is TaskStatus.PENDING
        assert state.result is None

    async def test_payload_is_normalised_validated_request(self, api: ApiHarness) -> None:
        await _submit(api, {"text": TEXT, "focus_areas": ["  risk  "]})

        [queued] = await _queued(api.redis, api.worker)
        assert queued.payload == {"text": TEXT, "max_length": 256, "focus_areas": ["risk"]}


class TestIdempotency:
    async def test_repeated_key_queues_one_job(self, api: ApiHarness) -> None:
        first, second, third = [await _submit(api) for _ in range(3)]

        assert first.status_code == second.status_code == third.status_code == 202
        assert first.json() == second.json() == third.json()
        assert second.headers["idempotent-replayed"] == "true"
        assert second.headers["location"] == first.headers["location"]
        assert len(await _queued(api.redis, api.worker)) == 1
        assert api.submissions("accepted") == 1
        assert api.submissions("replayed") == 2

    async def test_concurrent_duplicates_share_one_task(self, api: ApiHarness) -> None:
        responses = await asyncio.gather(*(_submit(api) for _ in range(8)))

        assert {r.status_code for r in responses} == {202}
        assert len({r.json()["task_id"] for r in responses}) == 1
        assert len(await _queued(api.redis, api.worker)) == 1
        assert api.submissions("accepted") == 1
        assert api.submissions("replayed") == 7

    async def test_equivalent_body_is_a_replay(self, api: ApiHarness) -> None:
        first = await _submit(api, {"text": TEXT, "focus_areas": ["risk"]})
        again = await _submit(api, {"focus_areas": [" risk "], "text": TEXT, "max_length": 256})

        assert again.json()["task_id"] == first.json()["task_id"]
        assert again.headers["idempotent-replayed"] == "true"

    async def test_key_reused_for_different_request_is_rejected(self, api: ApiHarness) -> None:
        await _submit(api)

        response = await _submit(api, BODY | {"max_length": 128})

        _problem(response, 422, "idempotency-key-mismatch")
        assert len(await _queued(api.redis, api.worker)) == 1
        assert api.submissions("conflict") == 1

    async def test_replay_reports_current_status(self, api: ApiHarness) -> None:
        task_id = UUID((await _submit(api)).json()["task_id"])
        await api.app.state.services.submitter.states.save(
            TaskResult(task_id=task_id, status=TaskStatus.RUNNING)
        )

        replay = await _submit(api)

        assert replay.json()["status"] == "RUNNING"

    async def test_key_held_by_retained_job_conflicts(self, api: ApiHarness) -> None:
        # An earlier job with this key outlived its idempotency record.
        await enqueue_task(api.redis, make_task(), queue_name=api.worker.queue_name)

        response = await _submit(api)

        _problem(response, 409, "idempotency-key-in-use")
        assert response.headers["retry-after"] == "1"
        submitter: TaskSubmitter = api.app.state.services.submitter
        assert await submitter.lookup("doc-1") is None  # claim released for a later retry
        assert len(await _queued(api.redis, api.worker)) == 1

    @pytest.mark.parametrize(
        ("key", "kind"),
        [
            (None, "idempotency-key-required"),
            ("", "invalid-idempotency-key"),
            ("-leading-dash", "invalid-idempotency-key"),
            ("has space", "invalid-idempotency-key"),
            ("k" * 129, "invalid-idempotency-key"),
        ],
    )
    async def test_idempotency_key_header_is_validated(
        self, api: ApiHarness, key: str | None, kind: str
    ) -> None:
        _problem(await _submit(api, key=key), 400, kind)
        assert await _queued(api.redis, api.worker) == []


class TestSubmitter:
    async def test_replay_heals_interrupted_submission(
        self, arq_pool: ArqRedis, fast_settings: WorkerSettings
    ) -> None:
        submitter = TaskSubmitter(arq_pool, worker=fast_settings, key_prefix="idem:")
        task = make_task()
        request_fingerprint = fingerprint(task.task_type, SummarizationRequest(text="x"))
        record = IdempotencyRecord(fingerprint=request_fingerprint, task=task)
        # Crash after claiming the key: no state saved, nothing enqueued.
        await arq_pool.set("idem:doc-1", record.model_dump_json())

        submission = await submitter.replay(record, request_fingerprint)

        assert submission.replayed
        assert submission.status is TaskStatus.PENDING
        assert (await submitter.status(task.task_id)) == TaskResult(
            task_id=task.task_id, status=TaskStatus.PENDING
        )
        [queued] = await _queued(arq_pool, fast_settings)
        assert queued == task
        # Replaying again does not enqueue a second job.
        await submitter.replay(record, request_fingerprint)
        assert len(await _queued(arq_pool, fast_settings)) == 1

    async def test_lost_claim_replays_winner(
        self, arq_pool: ArqRedis, fast_settings: WorkerSettings
    ) -> None:
        submitter = TaskSubmitter(arq_pool, worker=fast_settings, key_prefix="idem:")
        winner, loser = make_task(), make_task()

        first = await submitter.submit(winner, "fp")
        second = await submitter.submit(loser, "fp")

        assert not first.replayed
        assert second.replayed
        assert second.task == winner

    async def test_vanished_claim_is_a_conflict(
        self,
        arq_pool: ArqRedis,
        fast_settings: WorkerSettings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        submitter = TaskSubmitter(arq_pool, worker=fast_settings, key_prefix="idem:")
        await submitter.submit(make_task(), "fp")

        async def _expired(_key: str) -> None:
            return None

        monkeypatch.setattr(submitter, "lookup", _expired)
        with pytest.raises(IdempotencyKeyInUseError):
            await submitter.submit(make_task(), "fp")

    async def test_failed_enqueue_releases_claim(
        self, api: ApiHarness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _broken(*_: object, **__: object) -> None:
            raise RedisConnectionError("connection reset")

        monkeypatch.setattr(submission_module, "enqueue_task", _broken)

        response = await _submit(api)

        _problem(response, 503, "queue-unavailable")
        assert response.headers["retry-after"] == "5"
        assert await api.redis.keys("*") == []  # neither the claim nor a PENDING state remain
        assert api.submissions("unavailable") == 1


class TestValidation:
    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"text": ""},
            {"text": TEXT, "max_length": 8},
            {"text": TEXT, "max_length": "64"},
            {"text": TEXT, "unexpected": True},
            {"text": TEXT, "focus_areas": ["x"] * 17},
            ["not", "an", "object"],
        ],
    )
    async def test_invalid_body_is_rejected_without_echo(self, api: ApiHarness, body: Any) -> None:
        response = await api.client.post(SUMMARIZE, json=body, headers={"idempotency-key": "doc-1"})

        problem = _problem(response, 422, "invalid-request")
        assert problem["detail"].startswith("Invalid request: ")
        assert TEXT not in response.text
        assert await _queued(api.redis, api.worker) == []
        assert api.submissions("invalid") == 1

    async def test_malformed_json_is_rejected(self, api: ApiHarness) -> None:
        response = await api.client.post(
            SUMMARIZE, content=b'{"text": "unterminated', headers={"idempotency-key": "doc-1"}
        )

        assert "json_invalid" in _problem(response, 422, "invalid-request")["detail"]


class TestPolling:
    async def test_lifecycle_pending_to_completed(
        self, api: ApiHarness, redis_server: fakeredis.FakeServer
    ) -> None:
        accepted = await _submit(api)
        location = accepted.headers["location"]
        assert (await api.client.get(location)).json()["status"] == "PENDING"

        await _drain(redis_server, api.worker)

        done = TaskResult.model_validate_json((await api.client.get(location)).content)
        assert done.status is TaskStatus.COMPLETED
        assert done.result is not None
        assert isinstance(done.result["summary"], str)
        assert done.result["summary"]
        assert done.duration_ms is not None
        # A replay after completion reports the terminal state, still without a new job.
        replay = await _submit(api)
        assert replay.json()["status"] == "COMPLETED"
        assert await _queued(api.redis, api.worker) == []

    async def test_unknown_task_is_404(self, api: ApiHarness) -> None:
        _problem(await api.client.get(f"/v1/tasks/{uuid4()}"), 404, "task-not-found")

    async def test_malformed_task_id_is_422(self, api: ApiHarness) -> None:
        problem = _problem(await api.client.get("/v1/tasks/not-a-uuid"), 422, "invalid-request")
        assert "task_id" in problem["detail"]


class TestHealth:
    @pytest.mark.parametrize("path", ["/health", "/healthz"])
    async def test_degraded_without_worker_heartbeat(self, api: ApiHarness, path: str) -> None:
        response = await api.client.get(path)

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "degraded"
        assert {c["name"]: c["available"] for c in body["components"]} == {
            "redis": True,
            "workers": False,
        }

    async def test_ok_with_worker_heartbeat(self, api: ApiHarness) -> None:
        await api.redis.set(health_check_key(api.worker.queue_name), b"alive", px=10_000)

        body = (await api.client.get("/health")).json()

        assert body["status"] == "ok"
        assert all(c["available"] for c in body["components"])

    async def test_unavailable_when_redis_is_down(
        self, api: ApiHarness, redis_server: fakeredis.FakeServer
    ) -> None:
        redis_server.connected = False

        response = await api.client.get("/health")

        assert response.status_code == 503
        assert response.json()["status"] == "unavailable"


class TestRedisOutage:
    async def test_submission_answers_503(
        self, api: ApiHarness, redis_server: fakeredis.FakeServer
    ) -> None:
        redis_server.connected = False

        response = await _submit(api)

        _problem(response, 503, "queue-unavailable")
        assert api.submissions("unavailable") == 1

    async def test_polling_answers_503(
        self, api: ApiHarness, redis_server: fakeredis.FakeServer
    ) -> None:
        redis_server.connected = False

        _problem(await api.client.get(f"/v1/tasks/{uuid4()}"), 503, "queue-unavailable")


class TestMetrics:
    async def test_exposes_registry(self, api: ApiHarness) -> None:
        await _submit(api)

        response = await api.client.get("/metrics")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert (
            'edaw_api_submissions_total{outcome="accepted",task_type="summarization"} 1.0'
            in response.text
        )

    async def test_can_be_disabled(self, api_factory: ApiFactory) -> None:
        async with api_factory(ApiSettings(metrics_enabled=False)) as api:
            _problem(await api.client.get("/metrics"), 404, "http-error")


class TestApplication:
    async def test_cors_preflight(self, api_factory: ApiFactory) -> None:
        settings = ApiSettings(cors_origins=("https://app.example",))
        async with api_factory(settings) as api:
            response = await api.client.options(
                SUMMARIZE,
                headers={
                    "origin": "https://app.example",
                    "access-control-request-method": "POST",
                    "access-control-request-headers": "idempotency-key,content-type",
                },
            )

        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "https://app.example"

    async def test_cors_disabled_by_default(self, api: ApiHarness) -> None:
        response = await api.client.get("/health", headers={"origin": "https://app.example"})
        assert "access-control-allow-origin" not in response.headers

    async def test_unhandled_error_is_opaque_500(
        self, api_factory: ApiFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _boom(*_: object) -> None:
            raise RuntimeError("secret internals")

        monkeypatch.setattr(TaskSubmitter, "status", _boom)
        async with api_factory(raise_app_exceptions=False) as api:
            response = await api.client.get(f"/v1/tasks/{uuid4()}")

        _problem(response, 500, "internal-error")
        assert "secret" not in response.text

    async def test_method_not_allowed_is_a_problem(self, api: ApiHarness) -> None:
        _problem(await api.client.delete(SUMMARIZE), 405, "http-error")

    async def test_openapi_documents_body_and_header(self, api: ApiHarness) -> None:
        spec = (await api.client.get("/openapi.json")).json()

        operation = spec["paths"][SUMMARIZE]["post"]
        schema = operation["requestBody"]["content"]["application/json"]["schema"]
        assert schema["required"] == ["text"]
        assert {p["name"] for p in operation["parameters"]} == {"idempotency-key"}
        assert "/healthz" not in spec["paths"]
        assert "/metrics" not in spec["paths"]

    async def test_lifespan_opens_and_closes_default_pool(
        self, redis_server: fakeredis.FakeServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        opened: list[Any] = []
        pool = make_pool(redis_server)
        closed: list[bool | None] = []

        async def _create_pool(settings: Any, **kwargs: Any) -> ArqRedis:
            opened.append((settings, kwargs))
            return pool

        async def _aclose(close_connection_pool: bool | None = None) -> None:
            closed.append(close_connection_pool)

        monkeypatch.setattr(app_module, "create_pool", _create_pool)
        monkeypatch.setattr(pool, "aclose", _aclose)
        worker = WorkerSettings(queue_name="jobs", dlq_name="jobs:dlq")
        redis = RedisSettings.model_validate({"url": "redis://cache:6380/2"})
        app = create_app(ApiSettings(), worker, redis)

        async with app.router.lifespan_context(app):
            assert app.state.services.pool is pool

        [(arq_settings, kwargs)] = opened
        assert (arq_settings.host, arq_settings.port, arq_settings.database) == ("cache", 6380, 2)
        assert kwargs == {"default_queue_name": "jobs"}
        assert closed == [True]
        assert not hasattr(app.state, "services")


class TestApiSettings:
    def test_reads_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EDAW_API_PORT", "9000")
        monkeypatch.setenv("EDAW_API_CORS_ORIGINS", '["https://a.example", "*"]')
        monkeypatch.setenv("EDAW_API_METRICS_ENABLED", "false")

        settings = ApiSettings()

        assert settings.port == 9000
        assert settings.cors_origins == ("https://a.example", "*")
        assert not settings.metrics_enabled
        assert settings.host == "127.0.0.1"

    def test_rejects_bad_cors_origin(self) -> None:
        with pytest.raises(ValueError, match="CORS origins"):
            ApiSettings(cors_origins=("app.example",))


async def test_task_type_is_part_of_fingerprint() -> None:
    request = SummarizationRequest(text="x")
    assert fingerprint(TaskType.SUMMARIZATION, request) != fingerprint(
        TaskType.AUDIO_TRANSCRIPTION, request
    )
