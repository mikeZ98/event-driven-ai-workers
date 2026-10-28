"""Producer-side Prometheus metrics. Worker metrics live in each worker's own process."""

from enum import StrEnum

from prometheus_client import CollectorRegistry, Counter

from event_driven_ai_workers.schemas import TaskType


class SubmissionOutcome(StrEnum):
    """How the API answered a task submission."""

    ACCEPTED = "accepted"
    REPLAYED = "replayed"
    INVALID = "invalid"
    CONFLICT = "conflict"
    TOO_LARGE = "too_large"
    SATURATED = "saturated"
    UNAVAILABLE = "unavailable"


class ApiMetrics:
    """Counters registered on the app's own registry (labels drawn from bounded enums)."""

    def __init__(self, registry: CollectorRegistry) -> None:
        self.submissions = Counter(
            "edaw_api_submissions",
            "Task submissions by outcome.",
            ["task_type", "outcome"],
            registry=registry,
        )

    def submission(self, task_type: TaskType, outcome: SubmissionOutcome) -> None:
        self.submissions.labels(task_type.value, outcome.value).inc()
