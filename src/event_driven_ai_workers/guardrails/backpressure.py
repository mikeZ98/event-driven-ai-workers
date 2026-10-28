"""Queue backpressure: refuse new work while the arq queue is at capacity.

The depth is read straight from Redis (``ZCARD`` on the queue, O(1)): the
``arq_queued_tasks`` gauge lives in each worker's process, not the producer's. Like the
gauge, it counts deferred retries and in-flight jobs. The check is a soft limit:
concurrent submissions that pass it together can overshoot by their number.
"""

from arq.connections import ArqRedis

from event_driven_ai_workers.core.ports import resolve
from event_driven_ai_workers.guardrails.config import GuardrailSettings


class QueueSaturatedError(Exception):
    """The queue holds ``depth`` jobs, at or above ``limit``."""

    def __init__(self, depth: int, limit: int, retry_after_s: int) -> None:
        super().__init__(f"queue depth {depth} >= {limit}")
        self.depth = depth
        self.limit = limit
        self.retry_after_s = retry_after_s


class Backpressure:
    """Admission check against ``max_queue_depth`` for one queue."""

    def __init__(self, pool: ArqRedis, *, queue_name: str, settings: GuardrailSettings) -> None:
        self._pool = pool
        self._queue_name = queue_name
        self._settings = settings

    async def depth(self) -> int:
        depth: int = await resolve(self._pool.zcard(self._queue_name))
        return depth

    async def admit(self) -> None:
        """Raise :class:`QueueSaturatedError` when the queue is full."""
        depth = await self.depth()
        if depth >= self._settings.max_queue_depth:
            raise QueueSaturatedError(
                depth, self._settings.max_queue_depth, self._settings.backpressure_retry_after_s
            )
