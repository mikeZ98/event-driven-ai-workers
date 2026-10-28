"""Shared base model and primitives used by every schema in the package."""

from datetime import UTC, datetime
from typing import Annotated

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict


class WorkerModel(BaseModel):
    """Immutable, strictly validated base for every domain contract.

    Strict mode means no type coercion: in Python mode values must already have
    the declared type; over the wire use ``model_dump_json``/``model_validate_json``,
    where JSON-native encodings (ISO-8601 datetimes, UUID strings, enum values) apply.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def utc_now() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


def _to_utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


UtcDatetime = Annotated[AwareDatetime, AfterValidator(_to_utc)]
"""Timezone-aware datetime normalised to UTC; naive datetimes are rejected."""
