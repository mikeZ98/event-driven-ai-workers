"""Core worker engine: configuration, retry policy, DLQ routing and arq wiring."""

from event_driven_ai_workers.core.config import RedisSettings, WorkerSettings
from event_driven_ai_workers.core.dlq import RedisDeadLetterQueue
from event_driven_ai_workers.core.engine import ExecutionOutcome, RetryScheduled, TaskEngine
from event_driven_ai_workers.core.errors import (
    DeadLetterPublishError,
    MalformedTaskError,
    PoisonPillError,
    UnsupportedTaskTypeError,
    UpstreamHTTPError,
)
from event_driven_ai_workers.core.ports import (
    DeadLetterReason,
    DeadLetterSink,
    EngineObserver,
    NullObserver,
    TaskHandler,
    TaskOutput,
    TaskStateStore,
)
from event_driven_ai_workers.core.retry import (
    BackoffPolicy,
    ErrorClass,
    FailureClassification,
    classify,
    classify_status,
    parse_retry_after,
)
from event_driven_ai_workers.core.tracking import RedisTaskStateStore
from event_driven_ai_workers.core.worker import (
    TASK_FUNCTION,
    HandlerFactory,
    WorkerRuntime,
    create_worker_settings,
    enqueue_task,
)

__all__ = [
    "TASK_FUNCTION",
    "BackoffPolicy",
    "DeadLetterPublishError",
    "DeadLetterReason",
    "DeadLetterSink",
    "EngineObserver",
    "ErrorClass",
    "ExecutionOutcome",
    "FailureClassification",
    "HandlerFactory",
    "MalformedTaskError",
    "NullObserver",
    "PoisonPillError",
    "RedisDeadLetterQueue",
    "RedisSettings",
    "RedisTaskStateStore",
    "RetryScheduled",
    "TaskEngine",
    "TaskHandler",
    "TaskOutput",
    "TaskStateStore",
    "UnsupportedTaskTypeError",
    "UpstreamHTTPError",
    "WorkerRuntime",
    "WorkerSettings",
    "classify",
    "classify_status",
    "create_worker_settings",
    "enqueue_task",
    "parse_retry_after",
]
