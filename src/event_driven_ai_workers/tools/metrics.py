"""Prometheus metrics (Blueprint p.13), recorded through the engine's observer port.

Exposition names: ``arq_queued_tasks``, ``arq_task_latency_{bucket,sum,count}``,
``arq_retry_count_total`` and ``arq_dlq_total`` (prometheus_client appends ``_total``
to counters). Every label is drawn from a bounded enum, so cardinality stays fixed.
"""

from prometheus_client import REGISTRY, CollectorRegistry, Counter, Gauge, Histogram

from event_driven_ai_workers.core import DeadLetterReason, ErrorClass, FailureClassification
from event_driven_ai_workers.schemas import TaskPayload, TaskStatus

LATENCY_BUCKETS_S = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)
"""LLM workloads run for seconds to minutes; default buckets stop at 10s."""


class WorkerMetrics:
    """Prometheus-backed :class:`~event_driven_ai_workers.core.EngineObserver`.

    Create one per process: metrics register on ``registry`` (the global default
    unless given), and registering the same names twice raises.
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        registry = REGISTRY if registry is None else registry
        self.queued_tasks = Gauge(
            "arq_queued_tasks",
            "Jobs waiting in the arq queue, including deferred retries.",
            ["queue"],
            registry=registry,
        )
        self.task_latency = Histogram(
            "arq_task_latency",
            "Wall-clock seconds per task attempt that reached a terminal state.",
            ["task_type", "status"],
            buckets=LATENCY_BUCKETS_S,
            registry=registry,
        )
        self.retry_count = Counter(
            "arq_retry_count",
            "Task attempts that failed transiently and were rescheduled.",
            ["task_type", "error_class"],
            registry=registry,
        )
        self.dlq_total = Counter(
            "arq_dlq_total",
            "Tasks quarantined in the dead-letter queue.",
            ["task_type", "reason", "error_class"],
            registry=registry,
        )

    def task_started(self, task: TaskPayload) -> None:
        pass

    def task_finished(self, task: TaskPayload, status: TaskStatus, duration_s: float) -> None:
        self.task_latency.labels(task.task_type.value, status.value).observe(duration_s)

    def retry_scheduled(
        self,
        task: TaskPayload,
        failure: FailureClassification,
        delay_s: float,  # noqa: ARG002 — part of the EngineObserver protocol
    ) -> None:
        self.retry_count.labels(task.task_type.value, failure.error_class.value).inc()

    def dead_lettered(
        self, task: TaskPayload, reason: DeadLetterReason, error_class: ErrorClass
    ) -> None:
        self.dlq_total.labels(task.task_type.value, reason.value, error_class.value).inc()

    def queue_depth(self, queue_name: str, depth: int) -> None:
        self.queued_tasks.labels(queue_name).set(depth)
