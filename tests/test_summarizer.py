"""Summarization runner and Prometheus metrics, unit-level and through a real arq worker."""

from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from typing import Any

import httpx
import pytest
from arq.connections import ArqRedis
from arq.worker import Worker
from prometheus_client import CollectorRegistry, generate_latest
from pydantic import ValidationError

from event_driven_ai_workers.core import (
    DeadLetterReason,
    ErrorClass,
    FailureClassification,
    RedisDeadLetterQueue,
    TaskHandler,
    WorkerSettings,
    create_worker_settings,
    enqueue_task,
)
from event_driven_ai_workers.schemas import (
    SummarizationResult,
    TaskResult,
    TaskStatus,
    TaskType,
)
from event_driven_ai_workers.tools import (
    LLMSettings,
    SimulatedLLMProvider,
    SummarizationRunner,
    WorkerMetrics,
    open_llm_client,
    summarization_handlers,
)
from tests.support import make_task

DOCUMENT = "Workers pull jobs from Redis. Failed jobs are retried with backoff. Then more."


def _sample(registry: CollectorRegistry, name: str, **labels: str) -> float:
    return registry.get_sample_value(name, labels) or 0.0


class TestSummarizationRunner:
    async def test_payload_is_validated_and_result_is_json(self) -> None:
        task = make_task(payload={"text": DOCUMENT, "max_length": 64, "focus_areas": ["retries"]})
        async with open_llm_client() as client:
            output = await SummarizationRunner(client)(task)
        result = SummarizationResult.model_validate(output)
        assert result.summary.startswith("Workers pull jobs from Redis.")
        assert isinstance(output["tokens_used"], int)

    @pytest.mark.parametrize(
        "payload",
        [{}, {"text": ""}, {"text": "doc", "max_length": "64"}, {"text": "doc", "extra": 1}],
    )
    async def test_invalid_payload_is_a_validation_error(self, payload: dict[str, Any]) -> None:
        async with open_llm_client() as client:
            with pytest.raises(ValidationError):
                await SummarizationRunner(client)(make_task(payload=payload))

    async def test_handler_factory_registers_summarization(self) -> None:
        provider = SimulatedLLMProvider()
        async with summarization_handlers(transport=httpx.MockTransport(provider)) as handlers:
            assert set(handlers) == {TaskType.SUMMARIZATION}
            await handlers[TaskType.SUMMARIZATION](make_task(payload={"text": DOCUMENT}))
        assert len(provider.requests) == 1


class TestWorkerMetrics:
    @pytest.fixture
    def registry(self) -> CollectorRegistry:
        return CollectorRegistry()

    def test_observer_events_update_series(self, registry: CollectorRegistry) -> None:
        metrics = WorkerMetrics(registry)
        task = make_task()
        metrics.task_started(task)
        metrics.queue_depth("edaw:queue", 7)
        metrics.retry_scheduled(
            task, FailureClassification(ErrorClass.RATE_LIMITED, retryable=True), 1.5
        )
        metrics.task_finished(task, TaskStatus.COMPLETED, 0.3)
        metrics.dead_lettered(task, DeadLetterReason.NON_RETRYABLE, ErrorClass.CLIENT_ERROR)

        assert _sample(registry, "arq_queued_tasks", queue="edaw:queue") == 7
        assert (
            _sample(
                registry,
                "arq_retry_count_total",
                task_type="summarization",
                error_class="rate_limited",
            )
            == 1
        )
        assert (
            _sample(
                registry, "arq_task_latency_count", task_type="summarization", status="COMPLETED"
            )
            == 1
        )
        assert (
            _sample(
                registry,
                "arq_task_latency_bucket",
                task_type="summarization",
                status="COMPLETED",
                le="0.5",
            )
            == 1
        )
        assert (
            _sample(
                registry,
                "arq_dlq_total",
                task_type="summarization",
                reason="non_retryable",
                error_class="client_error",
            )
            == 1
        )

    def test_exposition_names_match_the_blueprint(self, registry: CollectorRegistry) -> None:
        metrics = WorkerMetrics(registry)
        metrics.queue_depth("q", 0)
        exposition = generate_latest(registry).decode()
        assert "# TYPE arq_queued_tasks gauge" in exposition
        assert "# TYPE arq_task_latency histogram" in exposition
        assert "# TYPE arq_retry_count_total counter" in exposition
        assert "# TYPE arq_dlq_total counter" in exposition

    def test_registering_twice_on_one_registry_fails(self, registry: CollectorRegistry) -> None:
        WorkerMetrics(registry)
        with pytest.raises(ValueError, match="Duplicated timeseries"):
            WorkerMetrics(registry)


async def _run_worker(
    pool: ArqRedis,
    settings: WorkerSettings,
    metrics: WorkerMetrics,
    llm: LLMSettings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> None:
    def handlers() -> AbstractAsyncContextManager[Mapping[TaskType, TaskHandler]]:
        return summarization_handlers(llm, transport=transport)

    options = create_worker_settings(handlers, worker=settings, observer=metrics)
    worker = Worker(**options, redis_pool=pool, burst=True, poll_delay=0.005, handle_signals=False)
    try:
        await worker.main()
    finally:
        await worker.close()


class TestEndToEnd:
    async def test_rate_limited_task_is_retried_and_metered(
        self, arq_pool: ArqRedis, fast_settings: WorkerSettings
    ) -> None:
        registry = CollectorRegistry()
        task = make_task(payload={"text": DOCUMENT, "max_length": 32})
        job = await enqueue_task(arq_pool, task, queue_name=fast_settings.queue_name)
        assert job is not None
        provider = SimulatedLLMProvider()
        calls = 0

        def first_call_rate_limited(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(429, headers={"retry-after": "0"}, json={"error": {}})
            return provider(request)

        await _run_worker(
            arq_pool,
            fast_settings,
            WorkerMetrics(registry),
            transport=httpx.MockTransport(first_call_rate_limited),
        )

        assert calls == 2
        result = TaskResult.model_validate_json(await job.result(timeout=1))
        assert result.status is TaskStatus.COMPLETED
        assert result.result is not None
        assert str(result.result["summary"]).startswith("Workers pull jobs")
        assert (
            _sample(
                registry,
                "arq_retry_count_total",
                task_type="summarization",
                error_class="rate_limited",
            )
            == 1
        )
        assert (
            _sample(
                registry, "arq_task_latency_count", task_type="summarization", status="COMPLETED"
            )
            == 1
        )
        depth = registry.get_sample_value("arq_queued_tasks", {"queue": fast_settings.queue_name})
        assert depth is not None

    async def test_persistent_rate_limit_exhausts_into_the_dlq(
        self, arq_pool: ArqRedis, fast_settings: WorkerSettings
    ) -> None:
        registry = CollectorRegistry()
        metrics = WorkerMetrics(registry)
        task = make_task(payload={"text": DOCUMENT})
        await enqueue_task(arq_pool, task, queue_name=fast_settings.queue_name)
        llm = LLMSettings(simulate_rate_limit_every=1, simulate_retry_after_s=0)

        await _run_worker(arq_pool, fast_settings, metrics, llm)

        assert (
            _sample(
                registry,
                "arq_retry_count_total",
                task_type="summarization",
                error_class="rate_limited",
            )
            == fast_settings.dlq_threshold
        )
        assert (
            _sample(
                registry,
                "arq_dlq_total",
                task_type="summarization",
                reason="retries_exhausted",
                error_class="rate_limited",
            )
            == 1
        )
        [envelope] = await RedisDeadLetterQueue(arq_pool, fast_settings.dlq_name).peek()
        assert envelope.original_task == task
        assert "upstream HTTP 429" in envelope.failure_reason

    async def test_invalid_payload_is_dead_lettered_without_calling_the_llm(
        self, arq_pool: ArqRedis, fast_settings: WorkerSettings
    ) -> None:
        registry = CollectorRegistry()
        await enqueue_task(
            arq_pool, make_task(payload={"text": ""}), queue_name=fast_settings.queue_name
        )
        await _run_worker(arq_pool, fast_settings, WorkerMetrics(registry), LLMSettings())
        assert (
            _sample(
                registry,
                "arq_dlq_total",
                task_type="summarization",
                reason="non_retryable",
                error_class="validation",
            )
            == 1
        )
        assert _sample(registry, "arq_retry_count_total", task_type="summarization") == 0
