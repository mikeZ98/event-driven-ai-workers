"""Producer API configuration, loaded from ``EDAW_API_*`` environment variables.

Like the worker settings, not ``strict``: environment values arrive as strings
(collections as JSON, e.g. ``EDAW_API_CORS_ORIGINS='["https://app.example"]'``).
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

API_PREFIX = "/v1"


class ApiSettings(BaseSettings):
    """HTTP listener, CORS, metrics exposition and idempotency bookkeeping."""

    model_config = SettingsConfigDict(
        env_prefix="EDAW_API_", extra="forbid", frozen=True, validate_default=True
    )

    host: str = Field(default="127.0.0.1", min_length=1)
    """Loopback by default; containers set ``0.0.0.0`` explicitly."""

    port: int = Field(default=8000, ge=1, le=65_535)

    cors_origins: tuple[str, ...] = ()
    """Browser origins allowed to call the API; empty disables CORS entirely."""

    metrics_enabled: bool = True
    """Serve ``GET /metrics`` for Prometheus."""

    idempotency_key_prefix: str = Field(default="edaw:idem:", min_length=1, max_length=128)
    """Redis prefix for ``Idempotency-Key`` records (kept for ``EDAW_WORKER_STATE_TTL_S``)."""

    health_timeout_s: float = Field(default=1.0, gt=0, le=30)
    """Deadline for the Redis round trips behind ``GET /health``."""

    @field_validator("cors_origins")
    @classmethod
    def _check_origins(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if bad := [o for o in value if o != "*" and not o.startswith(("http://", "https://"))]:
            raise ValueError(f"CORS origins must be '*' or http(s) URLs: {bad}")
        return value
