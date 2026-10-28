"""Queue backpressure: refuse new work while the arq queue is at capacity.

Admission is one atomic Lua script: it counts the queue (``ZCARD``, like the workers'
``arq_queued_tasks`` gauge: deferred retries and in-flight jobs included) plus the
slots already reserved by submissions still on their way into the queue, and reserves
a slot only if the total is below ``max_queue_depth``. A check-then-enqueue would let
every request of a concurrent burst pass the check before any of them enqueued.

The reservation is released once the job is queued (or the submission fails). One left
behind by a crashed producer expires after :data:`RESERVATION_TTL_S`. A job counted
both as queued and as reserved for a moment only makes the limit stricter.
"""

import contextlib
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from arq.connections import ArqRedis
from redis.exceptions import RedisError

from event_driven_ai_workers.core.ports import resolve
from event_driven_ai_workers.guardrails.config import GuardrailSettings

RESERVATION_TTL_S = 30.0
"""Upper bound on claim + record + enqueue; stale reservations are dropped after it."""

_ADMIT = """
local queue, reservations = KEYS[1], KEYS[2]
local now_ms, ttl_ms, limit = tonumber(ARGV[1]), tonumber(ARGV[2]), tonumber(ARGV[3])
redis.call('ZREMRANGEBYSCORE', reservations, '-inf', now_ms - ttl_ms)
local depth = redis.call('ZCARD', queue) + redis.call('ZCARD', reservations)
if depth >= limit then
  return {0, depth}
end
redis.call('ZADD', reservations, now_ms, ARGV[4])
redis.call('PEXPIRE', reservations, ttl_ms)
return {1, depth}
"""


def reservations_key(queue_name: str) -> str:
    """Sorted set of admitted submissions not yet in ``queue_name``."""
    return queue_name + ":admissions"


class QueueSaturatedError(Exception):
    """The queue (with pending admissions) holds ``depth`` jobs, at or above ``limit``."""

    def __init__(self, depth: int, limit: int, retry_after_s: int) -> None:
        super().__init__(f"queue depth {depth} >= {limit}")
        self.depth = depth
        self.limit = limit
        self.retry_after_s = retry_after_s


class Backpressure:
    """Atomic admission against ``max_queue_depth`` for one queue."""

    def __init__(
        self,
        pool: ArqRedis,
        *,
        queue_name: str,
        settings: GuardrailSettings,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._pool = pool
        self._queue_name = queue_name
        self._settings = settings
        self._clock = clock
        self._admit = pool.register_script(_ADMIT)

    @asynccontextmanager
    async def admission(self, member: str) -> AsyncIterator[None]:
        """Hold a queue slot for ``member`` for the block, or raise
        :class:`QueueSaturatedError` when the queue is full."""
        reservations = reservations_key(self._queue_name)
        admitted, depth = await self._admit(
            keys=[self._queue_name, reservations],
            args=[
                int(self._clock() * 1_000),
                int(RESERVATION_TTL_S * 1_000),
                self._settings.max_queue_depth,
                member,
            ],
        )
        if not admitted:
            raise QueueSaturatedError(
                depth, self._settings.max_queue_depth, self._settings.backpressure_retry_after_s
            )
        try:
            yield
        finally:
            # Best effort: an unreleased slot expires with RESERVATION_TTL_S.
            with contextlib.suppress(RedisError, OSError):
                await resolve(self._pool.zrem(reservations, member))
