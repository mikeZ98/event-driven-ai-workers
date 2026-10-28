"""Ingestion guardrail limits, loaded from ``EDAW_GUARDRAILS_*`` environment variables.

Like the other settings, not ``strict``: environment values arrive as strings.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from event_driven_ai_workers.schemas import MAX_DOCUMENT_CHARS

MIB = 1_048_576


class GuardrailSettings(BaseSettings):
    """Limits enforced on every submission before it reaches Redis."""

    model_config = SettingsConfigDict(
        env_prefix="EDAW_GUARDRAILS_", extra="forbid", frozen=True, validate_default=True
    )

    max_payload_bytes: int = Field(default=MIB, ge=1_024, le=16 * MIB)
    """Request body cap, enforced while the body streams in (HTTP 413 beyond it)."""

    max_document_chars: int = Field(default=MAX_DOCUMENT_CHARS, ge=1, le=MAX_DOCUMENT_CHARS)
    """Document length cap in code points; may only lower the schema's ceiling."""

    max_queue_depth: int = Field(default=10_000, ge=1, le=10_000_000)
    """New submissions are refused (HTTP 503) once the queue holds this many jobs."""

    backpressure_retry_after_s: int = Field(default=5, ge=1, le=3_600)
    """``Retry-After`` sent with a backpressure refusal."""
