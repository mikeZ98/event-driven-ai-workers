"""Redis-backed task state: one JSON ``TaskResult`` per task, expiring after a TTL."""

from uuid import UUID

from redis.asyncio import Redis

from event_driven_ai_workers.core.ports import resolve
from event_driven_ai_workers.schemas import TaskResult


class RedisTaskStateStore:
    """Stores the latest :class:`TaskResult` under ``<prefix><task_id>``."""

    def __init__(self, redis: Redis, *, prefix: str, ttl_s: int) -> None:
        self._redis = redis
        self._prefix = prefix
        self._ttl_s = ttl_s

    def key(self, task_id: UUID) -> str:
        return f"{self._prefix}{task_id}"

    async def save(self, result: TaskResult) -> None:
        await resolve(
            self._redis.set(self.key(result.task_id), result.model_dump_json(), ex=self._ttl_s)
        )

    async def get(self, task_id: UUID) -> TaskResult | None:
        raw: bytes | None = await resolve(self._redis.get(self.key(task_id)))
        return None if raw is None else TaskResult.model_validate_json(raw)
