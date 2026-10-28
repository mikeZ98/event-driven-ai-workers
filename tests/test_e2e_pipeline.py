"""End-to-end pipeline: FastAPI producer -> Redis -> arq worker -> LLM, polled back by the client.

Hermetic and in one event loop: the client reaches the app through ``httpx.ASGITransport``,
producer and worker share one fakeredis server, and the worker's LLM client talks to a
scripted ``httpx.MockTransport``. Nothing between the HTTP request and the stored result
is stubbed: guardrails, idempotent submission, arq, the engine's retry and DLQ policy,
the state store and the worker's Prometheus registry all run as deployed.
"""

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import fakeredis
import httpx
import pytest
from arq.worker import Worker
from prometheus_client import CollectorRegistry
from prometheus_client.parser import text_string_to_metric_families

from event_driven_ai_workers.core import (
    RedisDeadLetterQueue,
    RedisTaskStateStore,
    create_worker_settings,
)
from event_driven_ai_workers.guardrails import GuardrailSettings
from event_driven_ai_workers.schemas import (
    DeadLetterEnvelope,
    SummarizationResult,
    TaskResult,
    TaskStatus,
)
from event_driven_ai_workers.tools import (
    SimulatedLLMProvider,
    WorkerMetrics,
    summarization_handlers,
)
from event_driven_ai_workers.worker_main import close_worker, metrics_app
from tests.conftest import ApiFactory, ApiHarness, make_pool
from tests.test_api import _problem, _submit

TERMINAL_TIMEOUT_S = 10.0
_POLL_S = 0.005
_LIFECYCLE = [TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.COMPLETED]


def _upstream_error(
    status: int, error_type: str, *, retry_after: str | None = None
) -> httpx.Response:
    return httpx.Response(
        status,
        headers={} if retry_after is None else {"retry-after": retry_after},
        json={"type": "error", "error": {"type": error_type, "message": f"scripted {status}"}},
    )


def _refusal() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "msg_refused",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": [],
            "stop_reason": "refusal",
            "usage": {"input_tokens": 12, "output_tokens": 0},
        },
    )


class ScriptedUpstream:
    """The LLM provider as the worker sees it: scripted responses first, then ``always``
    forever if given, else the deterministic simulated provider. Records call times."""

    def __init__(
        self, *script: httpx.Response, always: Callable[[], httpx.Response] | None = None
    ) -> None:
        self.script = list(script)
        self.always = always
        self.simulated = SimulatedLLMProvider()
        self.calls: list[float] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(time.monotonic())
        if self.script:
            return self.script.pop(0)
        if self.always is not None:
            return self.always()
        return self.simulated(request)

    @property
    def gaps(self) -> list[float]:
        return [later - earlier for earlier, later in zip(self.calls, self.calls[1:], strict=False)]


@dataclass
class Polled:
    """A task's terminal state as the client saw it, and every status it observed."""

    state: TaskResult
    seen: list[TaskStatus] = field(default_factory=list)


@dataclass
class Pipeline:
    api: ApiHarness
    server: fakeredis.FakeServer
    upstream: ScriptedUpstream
    registry: CollectorRegistry = field(default_factory=CollectorRegistry)

    async def submit(
        self, key: str = "doc-1", body: dict[str, Any] | None = None
    ) -> httpx.Response:
        return await _submit(self.api, body, key)

    @asynccontextmanager
    async def worker(self) -> AsyncIterator[Worker]:
        """A long-running arq worker: production wiring and shutdown, with a fast poll."""
        handlers = partial(summarization_handlers, transport=httpx.MockTransport(self.upstream))
        options = create_worker_settings(
            handlers, worker=self.api.worker, observer=WorkerMetrics(self.registry)
        )
        worker = Worker(
            **options, redis_pool=make_pool(self.server), poll_delay=_POLL_S, handle_signals=False
        )
        main = asyncio.create_task(worker.main())
        try:
            yield worker
        finally:
            main.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await main
            await close_worker(worker)

    async def await_terminal(self, accepted: httpx.Response) -> Polled:
        """Poll the status URL the API handed out until the task is terminal."""
        assert accepted.status_code == 202, accepted.text
        status_url = accepted.json()["status_url"]
        polled: list[TaskStatus] = []
        async with asyncio.timeout(TERMINAL_TIMEOUT_S):
            while True:
                response = await self.api.client.get(status_url)
                assert response.status_code == 200, response.text
                state = TaskResult.model_validate_json(response.content)
                if not polled or polled[-1] is not state.status:
                    polled.append(state.status)
                if state.status.is_terminal:
                    return Polled(state, polled)
                await asyncio.sleep(_POLL_S)

    async def stored(self, polled: Polled) -> TaskResult | None:
        settings = self.api.worker
        store = RedisTaskStateStore(
            self.api.redis, prefix=settings.state_key_prefix, ttl_s=settings.state_ttl_s
        )
        return await store.get(polled.state.task_id)

    async def dead_letters(self) -> list[DeadLetterEnvelope]:
        return await RedisDeadLetterQueue(self.api.redis, self.api.worker.dlq_name).peek()

    async def queue_depth(self) -> int:
        depth: int = await self.api.redis.zcard(self.api.worker.queue_name)
        return depth

    def sample(self, name: str, **labels: str) -> float | None:
        return self.registry.get_sample_value(name, {"task_type": "summarization", **labels})

    async def scrape(self) -> str:
        """The worker's exposition, as Prometheus would scrape it."""
        transport = httpx.ASGITransport(app=metrics_app(self.registry))
        async with httpx.AsyncClient(transport=transport, base_url="http://worker") as client:
            return (await client.get("/metrics")).text


PipelineFactory = Callable[..., contextlib.AbstractAsyncContextManager[Pipeline]]


@pytest.fixture
def pipeline(api_factory: ApiFactory, redis_server: fakeredis.FakeServer) -> PipelineFactory:
    @asynccontextmanager
    async def build(
        upstream: ScriptedUpstream | None = None, guardrails: GuardrailSettings | None = None
    ) -> AsyncIterator[Pipeline]:
        async with api_factory(guardrails=guardrails) as api:
            yield Pipeline(api, redis_server, upstream or ScriptedUpstream())

    return build


def _scraped(exposition: str, name: str, **labels: str) -> float | None:
    """A sample's value parsed from exposition text, as Prometheus would read it."""
    wanted = {"task_type": "summarization", **labels}
    for family in text_string_to_metric_families(exposition):
        for sample in family.samples:
            if sample.name == name and sample.labels == wanted:
                return sample.value
    return None


def _is_ordered_lifecycle(seen: list[TaskStatus]) -> bool:
    positions = [_LIFECYCLE.index(status) for status in seen if status in _LIFECYCLE]
    return positions == sorted(positions)


class TestHappyPath:
    async def test_submitted_task_is_summarized_and_polled_back(
        self, pipeline: PipelineFactory
    ) -> None:
        async with pipeline() as p, p.worker():
            accepted = await p.submit()
            ticket = accepted.json()
            assert (ticket["status"], accepted.headers["location"]) == (
                "PENDING",
                ticket["status_url"],
            )

            polled = await p.await_terminal(accepted)
            health = (await p.api.client.get("/health")).json()

        state = polled.state
        assert state.status is TaskStatus.COMPLETED
        assert str(state.task_id) == ticket["task_id"]
        assert state.completed_at is not None
        assert state.duration_ms is not None
        assert polled.seen[-1] is TaskStatus.COMPLETED
        assert _is_ordered_lifecycle(polled.seen)

        result = SummarizationResult.model_validate_json(json.dumps(state.result))
        assert result.summary == (
            "Queues decouple producers from consumers and absorb bursts of load. "
            "Queues decouple producers from consumers and absorb bursts of load."
        )
        assert result.tokens_used > 0

        assert await p.stored(polled) == state
        assert len(p.upstream.calls) == 1
        assert health["status"] == "ok"
        assert (await p.queue_depth(), await p.dead_letters()) == (0, [])
        assert p.sample("arq_task_latency_count", status="COMPLETED") == 1
        assert p.sample("arq_retry_count_total", error_class="rate_limited") is None
        assert p.api.submissions("accepted") == 1


class TestUpstreamRateLimits:
    async def test_429_and_529_are_retried_until_success(self, pipeline: PipelineFactory) -> None:
        upstream = ScriptedUpstream(
            _upstream_error(429, "rate_limit_error", retry_after="1"),
            _upstream_error(529, "overloaded_error"),
        )
        async with pipeline(upstream) as p, p.worker():
            polled = await p.await_terminal(await p.submit())
            exposition = await p.scrape()

        assert polled.state.status is TaskStatus.COMPLETED
        assert len(upstream.calls) == 3
        # The 429 asked for 1s and got it; the 529 carried no Retry-After, so the
        # millisecond backoff applies: the wait came from the header, not slow polling.
        after_429, after_529 = upstream.gaps
        assert after_429 >= 1.0
        assert after_529 < 1.0
        # Rescheduled tasks report PENDING again, so the client never saw a false outcome.
        assert polled.seen[0] is TaskStatus.PENDING
        assert TaskStatus.DEAD_LETTER not in polled.seen

        assert p.sample("arq_retry_count_total", error_class="rate_limited") == 1
        assert p.sample("arq_retry_count_total", error_class="upstream_unavailable") == 1
        assert p.sample("arq_task_latency_count", status="COMPLETED") == 1
        assert _scraped(exposition, "arq_retry_count_total", error_class="rate_limited") == 1
        assert await p.dead_letters() == []


class TestDeadLetters:
    async def test_persistent_overload_exhausts_retries_into_the_dlq(
        self, pipeline: PipelineFactory
    ) -> None:
        upstream = ScriptedUpstream(always=partial(_upstream_error, 529, "overloaded_error"))
        async with pipeline(upstream) as p, p.worker():
            accepted = await p.submit()
            polled = await p.await_terminal(accepted)
            exposition = await p.scrape()

        threshold = p.api.worker.dlq_threshold
        assert threshold == 5
        assert polled.state.status is TaskStatus.DEAD_LETTER
        assert polled.state.result is None
        assert polled.state.completed_at is not None
        assert await p.stored(polled) == polled.state
        assert len(upstream.calls) == 1 + threshold  # the first attempt plus five retries

        [envelope] = await p.dead_letters()
        assert str(envelope.original_task.task_id) == accepted.json()["task_id"]
        assert envelope.original_task.idempotency_key == "doc-1"
        assert envelope.retry_attempts == threshold
        assert envelope.failure_reason.startswith("retries_exhausted: upstream_unavailable")
        assert await p.queue_depth() == 0  # arq does not retry it behind the engine's back

        assert p.sample("arq_retry_count_total", error_class="upstream_unavailable") == threshold
        dlq = {"reason": "retries_exhausted", "error_class": "upstream_unavailable"}
        assert p.sample("arq_dlq_total", **dlq) == 1
        assert p.sample("arq_task_latency_count", status="DEAD_LETTER") == 1
        assert _scraped(exposition, "arq_dlq_total", **dlq) == 1

    @pytest.mark.parametrize(
        ("response", "error_class"),
        [
            (_upstream_error(400, "invalid_request_error"), "client_error"),
            (_refusal(), "validation"),
        ],
        ids=["http-400", "refusal"],
    )
    async def test_fatal_error_is_dead_lettered_without_retry(
        self, pipeline: PipelineFactory, response: httpx.Response, error_class: str
    ) -> None:
        upstream = ScriptedUpstream(response)
        async with pipeline(upstream) as p, p.worker():
            polled = await p.await_terminal(await p.submit())

        assert polled.state.status is TaskStatus.DEAD_LETTER
        assert len(upstream.calls) == 1
        [envelope] = await p.dead_letters()
        assert envelope.retry_attempts == 0
        assert envelope.failure_reason.startswith(f"non_retryable: {error_class}")
        assert p.sample("arq_dlq_total", reason="non_retryable", error_class=error_class) == 1
        assert p.sample("arq_retry_count_total", error_class=error_class) is None

    async def test_poison_payload_never_reaches_queue_worker_or_llm(
        self, pipeline: PipelineFactory
    ) -> None:
        async with pipeline() as p, p.worker():
            response = await p.submit(body={"text": "binary\x00\x01payload"})
            healthy = await p.await_terminal(await p.submit(key="doc-2"))

        _problem(response, 422, "rejected-content")
        assert healthy.state.status is TaskStatus.COMPLETED
        assert len(p.upstream.calls) == 1  # only the healthy task
        assert await p.dead_letters() == []


class TestBackpressureAndIdempotency:
    async def test_burst_is_shed_while_admitted_tasks_complete(
        self, pipeline: PipelineFactory
    ) -> None:
        limits = GuardrailSettings(max_queue_depth=5, backpressure_retry_after_s=2)
        async with pipeline(guardrails=limits) as p:
            keys = [f"burst-{i}" for i in range(30)]
            responses = dict(
                zip(keys, await asyncio.gather(*(p.submit(key) for key in keys)), strict=True)
            )
            accepted = {k: r for k, r in responses.items() if r.status_code == 202}
            shed = {k: r for k, r in responses.items() if r.status_code != 202}

            assert len(accepted) == 5
            for response in shed.values():
                _problem(response, 503, "queue-saturated")
                assert response.headers["retry-after"] == "2"
            assert await p.queue_depth() == 5
            assert p.api.submissions("saturated") == 25

            # A client resending an admitted task while saturated keeps its identity.
            original_key, original = next(iter(accepted.items()))
            replay = await p.submit(original_key)
            assert replay.status_code == 202
            assert replay.headers["idempotent-replayed"] == "true"
            assert replay.json()["task_id"] == original.json()["task_id"]

            async with p.worker():
                finished = [await p.await_terminal(r) for r in accepted.values()]
                assert all(f.state.status is TaskStatus.COMPLETED for f in finished)
                assert len(p.upstream.calls) == 5  # the replay added no work

                # A shed client retries after the drain with the same key: never burned.
                shed_key = next(iter(shed))
                retried = await p.submit(shed_key)
                assert "idempotent-replayed" not in retried.headers
                assert (await p.await_terminal(retried)).state.status is TaskStatus.COMPLETED

            # After completion, a duplicate answers with the original task and its outcome.
            late = await p.submit(original_key)
            assert late.json()["task_id"] == original.json()["task_id"]
            assert late.json()["status"] == "COMPLETED"
            assert len(p.upstream.calls) == 6

    async def test_concurrent_duplicates_run_once(self, pipeline: PipelineFactory) -> None:
        async with pipeline() as p, p.worker():
            responses = await asyncio.gather(*(p.submit("dup") for _ in range(10)))
            assert {r.status_code for r in responses} == {202}
            task_ids = {r.json()["task_id"] for r in responses}
            assert len(task_ids) == 1

            polled = await p.await_terminal(responses[0])

        assert polled.state.status is TaskStatus.COMPLETED
        assert str(polled.state.task_id) in task_ids
        assert len(p.upstream.calls) == 1
        assert sum(r.headers.get("idempotent-replayed") == "true" for r in responses) == 9
