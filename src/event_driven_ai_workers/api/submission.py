"""Idempotent task submission: claim the ``Idempotency-Key``, record ``PENDING``, enqueue.

The key is claimed with ``SET NX`` and stores the full :class:`TaskPayload`, so a repeated
submission returns the original task instead of creating a new one. A replay also
re-drives ``enqueue_task`` while the task is still ``PENDING``: that is a no-op while the
arq job exists (its id is the idempotency key) and heals a submission interrupted
between claiming the key and enqueueing.
"""

import contextlib
import hashlib
from dataclasses import dataclass
from uuid import UUID

from arq.connections import ArqRedis
from redis.exceptions import RedisError

from event_driven_ai_workers.core import RedisTaskStateStore, WorkerSettings, enqueue_task
from event_driven_ai_workers.core.ports import resolve
from event_driven_ai_workers.schemas import (
    TaskPayload,
    TaskResult,
    TaskStatus,
    TaskType,
    WorkerModel,
)


class IdempotencyMismatchError(Exception):
    """The key was already used for a different request."""


class IdempotencyKeyInUseError(Exception):
    """The key is held by an earlier job arq still keeps, or a concurrent claim vanished."""


class IdempotencyRecord(WorkerModel):
    """What an ``Idempotency-Key`` maps to: the request fingerprint and the task it created."""

    fingerprint: str
    task: TaskPayload


@dataclass(frozen=True, slots=True)
class Submission:
    task: TaskPayload
    status: TaskStatus
    replayed: bool


def fingerprint(task_type: TaskType, request: WorkerModel) -> str:
    """Digest of the validated (normalised) request, so equivalent bodies match."""
    canonical = f"{task_type.value}\n{request.model_dump_json()}"
    return hashlib.sha256(canonical.encode()).hexdigest()


class TaskSubmitter:
    """Turns validated requests into queued tasks, at most once per idempotency key."""

    def __init__(self, pool: ArqRedis, *, worker: WorkerSettings, key_prefix: str) -> None:
        self._pool = pool
        self._worker = worker
        self._key_prefix = key_prefix
        self.states = RedisTaskStateStore(
            pool, prefix=worker.state_key_prefix, ttl_s=worker.state_ttl_s
        )

    async def lookup(self, idempotency_key: str) -> IdempotencyRecord | None:
        raw: bytes | None = await resolve(self._pool.get(self._key(idempotency_key)))
        return None if raw is None else IdempotencyRecord.model_validate_json(raw)

    async def status(self, task_id: UUID) -> TaskResult | None:
        return await self.states.get(task_id)

    async def replay(self, record: IdempotencyRecord, request_fingerprint: str) -> Submission:
        """Return the task ``record`` created, provided the request is the same one."""
        if record.fingerprint != request_fingerprint:
            raise IdempotencyMismatchError(record.task.idempotency_key)
        task = record.task
        state = await self.states.get(task.task_id)
        if state is None or state.status is TaskStatus.PENDING:
            if state is None:
                state = TaskResult(task_id=task.task_id, status=TaskStatus.PENDING)
                await self.states.save(state)
            await enqueue_task(self._pool, task, queue_name=self._worker.queue_name)
        return Submission(task=task, status=state.status, replayed=True)

    async def submit(self, task: TaskPayload, request_fingerprint: str) -> Submission:
        """Claim ``task.idempotency_key`` and enqueue ``task``, or replay the earlier claim."""
        key = self._key(task.idempotency_key)
        record = IdempotencyRecord(fingerprint=request_fingerprint, task=task)
        claimed: bool | None = await resolve(
            self._pool.set(key, record.model_dump_json(), nx=True, ex=self._worker.state_ttl_s)
        )
        if not claimed:
            winner = await self.lookup(task.idempotency_key)
            if winner is None:
                raise IdempotencyKeyInUseError(task.idempotency_key)
            return await self.replay(winner, request_fingerprint)
        try:
            await self.states.save(TaskResult(task_id=task.task_id, status=TaskStatus.PENDING))
            job = await enqueue_task(self._pool, task, queue_name=self._worker.queue_name)
        except Exception:
            await self._release(key, task)
            raise
        if job is None:
            # The key's claim had expired, but arq still keeps the earlier job or result.
            await self._release(key, task)
            raise IdempotencyKeyInUseError(task.idempotency_key)
        return Submission(task=task, status=TaskStatus.PENDING, replayed=False)

    def _key(self, idempotency_key: str) -> str:
        return self._key_prefix + idempotency_key

    async def _release(self, key: str, task: TaskPayload) -> None:
        """Best effort: let the client retry with the same key after a failed enqueue."""
        with contextlib.suppress(RedisError, OSError):
            await resolve(self._pool.delete(key, self.states.key(task.task_id)))
