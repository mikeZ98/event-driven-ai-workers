"""FastAPI producer: task ingestion, state polling, health and Prometheus metrics."""

from event_driven_ai_workers.api.app import PoolFactory, create_app
from event_driven_ai_workers.api.config import API_PREFIX, ApiSettings
from event_driven_ai_workers.api.metrics import ApiMetrics, SubmissionOutcome
from event_driven_ai_workers.api.models import ComponentHealth, HealthReport, TaskAccepted
from event_driven_ai_workers.api.problems import PROBLEM_MEDIA_TYPE, Problem, ProblemError
from event_driven_ai_workers.api.submission import (
    IdempotencyKeyInUseError,
    IdempotencyMismatchError,
    IdempotencyRecord,
    Submission,
    TaskSubmitter,
    fingerprint,
)

__all__ = [
    "API_PREFIX",
    "PROBLEM_MEDIA_TYPE",
    "ApiMetrics",
    "ApiSettings",
    "ComponentHealth",
    "HealthReport",
    "IdempotencyKeyInUseError",
    "IdempotencyMismatchError",
    "IdempotencyRecord",
    "PoolFactory",
    "Problem",
    "ProblemError",
    "Submission",
    "SubmissionOutcome",
    "TaskAccepted",
    "TaskSubmitter",
    "create_app",
    "fingerprint",
]
