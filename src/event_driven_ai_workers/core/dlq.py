"""Redis-backed dead-letter queue (a list: ``LPUSH`` to quarantine, oldest at the tail)."""

from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from event_driven_ai_workers.core.ports import resolve
from event_driven_ai_workers.schemas import DeadLetterEnvelope


class RedisDeadLetterQueue:
    """Publishes :class:`DeadLetterEnvelope` JSON to a Redis list.

    The DLQ write is the last line of defence for a poison pill, so transient Redis
    errors are retried in-process (tenacity, jittered exponential) before giving up.
    """

    def __init__(
        self,
        redis: Redis,
        key: str,
        *,
        publish_attempts: int = 4,
        retry_base_s: float = 0.05,
        retry_max_s: float = 1.0,
    ) -> None:
        if publish_attempts < 1:
            raise ValueError("publish_attempts must be >= 1")
        self._redis = redis
        self._key = key
        self._retrying = AsyncRetrying(
            stop=stop_after_attempt(publish_attempts),
            wait=wait_exponential_jitter(
                multiplier=retry_base_s, max=retry_max_s, jitter=retry_base_s
            ),
            retry=retry_if_exception_type((RedisConnectionError, RedisTimeoutError)),
            reraise=True,
        )

    @property
    def key(self) -> str:
        return self._key

    async def publish(self, envelope: DeadLetterEnvelope) -> None:
        data = envelope.model_dump_json()
        async for attempt in self._retrying.copy():
            with attempt:
                await resolve(self._redis.lpush(self._key, data))

    async def size(self) -> int:
        return await resolve(self._redis.llen(self._key))

    async def peek(self, limit: int = 100) -> list[DeadLetterEnvelope]:
        """Oldest-first view of up to ``limit`` quarantined envelopes (non-destructive)."""
        if limit < 1:
            return []
        raw: list[bytes] = await resolve(self._redis.lrange(self._key, -limit, -1))
        return [DeadLetterEnvelope.model_validate_json(item) for item in reversed(raw)]
