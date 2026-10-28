"""Runtime configuration, loaded from ``EDAW_REDIS_*`` / ``EDAW_WORKER_*`` environment variables.

Unlike the schemas, settings are not ``strict``: environment values always arrive as
strings, so lax coercion is required. Bounds are enforced explicitly.
"""

import dataclasses
from typing import Self

from arq.connections import RedisSettings as ArqRedisSettings
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

CANCELLATION_HEADROOM = 2
"""Extra arq tries beyond the DLQ threshold, absorbing runs cancelled by worker shutdown.

arq fails a job *without calling it* once ``job_try > max_tries``. Our engine must
always see the final attempt so it can dead-letter the task instead of losing it.
"""

JOB_TIMEOUT_GRACE_S = 10.0
"""arq's hard job timeout sits this far above the engine's own task timeout."""


class RedisSettings(BaseSettings):
    """Redis connection and pool settings shared by the producer and the workers."""

    model_config = SettingsConfigDict(
        env_prefix="EDAW_REDIS_", extra="forbid", frozen=True, validate_default=True
    )

    url: SecretStr = SecretStr("redis://localhost:6379/0")
    """``redis://``, ``rediss://`` or ``unix://`` DSN; secret because it may embed a password."""

    max_connections: int = Field(default=32, ge=1, le=10_000)
    conn_timeout_s: int = Field(default=2, ge=1, le=60)
    conn_retries: int = Field(default=5, ge=0, le=100)
    conn_retry_delay_s: int = Field(default=1, ge=0, le=60)
    retry_on_timeout: bool = True

    @field_validator("url")
    @classmethod
    def _check_dsn(cls, value: SecretStr) -> SecretStr:
        try:
            ArqRedisSettings.from_dsn(value.get_secret_value())
        except (RuntimeError, ValueError) as exc:
            raise ValueError("url must be a redis://, rediss:// or unix:// DSN") from exc
        return value

    def to_arq(self) -> ArqRedisSettings:
        """arq connection settings with this project's pool limits applied."""
        return dataclasses.replace(
            ArqRedisSettings.from_dsn(self.url.get_secret_value()),
            max_connections=self.max_connections,
            conn_timeout=self.conn_timeout_s,
            conn_retries=self.conn_retries,
            conn_retry_delay=self.conn_retry_delay_s,
            retry_on_timeout=self.retry_on_timeout,
        )


class WorkerSettings(BaseSettings):
    """Queue topology, concurrency, retry budget and backoff curve for the worker pool."""

    model_config = SettingsConfigDict(
        env_prefix="EDAW_WORKER_", extra="forbid", frozen=True, validate_default=True
    )

    queue_name: str = Field(default="edaw:queue", min_length=1, max_length=128)
    dlq_name: str = Field(default="edaw:dlq", min_length=1, max_length=128)
    state_key_prefix: str = Field(default="edaw:state:", min_length=1, max_length=128)
    state_ttl_s: int = Field(default=86_400, ge=60, le=30 * 86_400)

    dlq_threshold: int = Field(default=5, ge=0, le=20)
    """Retries allowed for transient failures before a task is dead-lettered."""

    max_jobs: int = Field(default=10, ge=1, le=1_000)
    """Concurrent jobs per worker process."""

    task_timeout_s: float = Field(default=120.0, gt=0, le=3_600)
    """Per-attempt deadline enforced by the engine; a timeout is a retryable failure."""

    backoff_base_s: float = Field(default=1.0, gt=0, le=600)
    backoff_max_s: float = Field(default=60.0, gt=0, le=3_600)
    backoff_multiplier: float = Field(default=2.0, ge=1, le=10)
    retry_after_cap_s: float = Field(default=300.0, gt=0, le=3_600)
    """Upper bound on an upstream ``Retry-After`` we are willing to honour."""

    queue_depth_sample_s: int = Field(default=15, ge=1, le=60)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.backoff_max_s < self.backoff_base_s:
            raise ValueError("backoff_max_s must be >= backoff_base_s")
        if self.dlq_name == self.queue_name:
            raise ValueError("dlq_name must differ from queue_name")
        if self.state_key_prefix in {self.queue_name, self.dlq_name}:
            raise ValueError("state_key_prefix must differ from the queue keys")
        return self

    @property
    def arq_max_tries(self) -> int:
        """arq's try limit: the first run, every retry, plus cancellation headroom."""
        return 1 + self.dlq_threshold + CANCELLATION_HEADROOM

    @property
    def arq_job_timeout_s(self) -> float:
        """arq's hard timeout, kept above :attr:`task_timeout_s` so ours fires first."""
        return self.task_timeout_s + JOB_TIMEOUT_GRACE_S
