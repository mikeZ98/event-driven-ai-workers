"""Failure types shared by the engine and the workload executors."""

from uuid import UUID


class UpstreamHTTPError(Exception):
    """An LLM provider answered with a non-success HTTP status.

    Raised by workload clients so the retry classifier never needs to know
    which HTTP library produced the response.
    """

    def __init__(
        self, status_code: int, message: str = "", *, retry_after_s: float | None = None
    ) -> None:
        super().__init__(f"upstream HTTP {status_code}" + (f": {message}" if message else ""))
        self.status_code = status_code
        self.retry_after_s = retry_after_s


class PoisonPillError(Exception):
    """A failure that can never succeed on retry (bad input, broken contract)."""


class UnsupportedTaskTypeError(PoisonPillError):
    """No handler is registered for the task's type."""


class MalformedTaskError(PoisonPillError):
    """The queued message is not a valid ``TaskPayload`` and cannot be enveloped."""


class DeadLetterPublishError(Exception):
    """The dead-letter queue rejected a poison pill; the task is marked ``FAILED``."""

    def __init__(self, task_id: UUID) -> None:
        super().__init__(f"could not dead-letter task {task_id}")
        self.task_id = task_id
