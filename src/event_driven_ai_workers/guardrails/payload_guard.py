"""Pre-enqueue payload checks: a bounded body read, then size and content rules.

Everything here runs before the submission touches Redis. Oversized bodies are cut off
while streaming, so they are never buffered whole. Content rules reject what the strict
schema lets through but a worker cannot use: control characters (binary smuggled in as
text) and whitespace-only documents, which would spend an LLM call on nothing. Lone
surrogates and invalid UTF-8 never get this far; ``model_validate_json`` rejects them.

Errors carry field names and limits only, never the submitted values.
"""

import re
from collections.abc import AsyncIterable

from pydantic import ValidationError

from event_driven_ai_workers.guardrails.config import GuardrailSettings
from event_driven_ai_workers.schemas import SummarizationRequest

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
"""Unicode ``Cc`` except tab, LF and CR."""


class PayloadTooLargeError(Exception):
    """The request body exceeds ``max_payload_bytes``."""

    def __init__(self, limit: int) -> None:
        super().__init__(f"The request body exceeds {limit} bytes.")
        self.limit = limit


class DocumentTooLargeError(Exception):
    """The document exceeds ``max_document_chars``."""

    def __init__(self, limit: int) -> None:
        super().__init__(f"The document text exceeds {limit} characters.")
        self.limit = limit


class RejectedContentError(Exception):
    """A field holds content no worker should be handed."""

    def __init__(self, field: str, reason: str) -> None:
        super().__init__(f"{field}: {reason}")
        self.field = field
        self.reason = reason


class PayloadGuard:
    """Applies :class:`GuardrailSettings` to raw submissions."""

    def __init__(self, settings: GuardrailSettings) -> None:
        self.settings = settings

    async def read(self, chunks: AsyncIterable[bytes], *, content_length: str | None) -> bytes:
        """Collect the body, refusing it as soon as it is known to exceed the cap."""
        limit = self.settings.max_payload_bytes
        if content_length is not None and content_length.isdigit() and int(content_length) > limit:
            raise PayloadTooLargeError(limit)
        body = bytearray()
        async for chunk in chunks:
            body += chunk
            if len(body) > limit:
                raise PayloadTooLargeError(limit)
        return bytes(body)

    def summarization(self, raw: bytes) -> SummarizationRequest:
        """Parse and screen a summarization body.

        Raises :class:`DocumentTooLargeError` whichever limit the text breaks (the
        schema's ceiling or the configured one), :class:`RejectedContentError`, or the
        schema's ``ValidationError`` for any other defect.
        """
        limit = self.settings.max_document_chars
        try:
            request = SummarizationRequest.model_validate_json(raw)
        except ValidationError as exc:
            if any(e["loc"] == ("text",) and e["type"] == "string_too_long" for e in exc.errors()):
                raise DocumentTooLargeError(limit) from None
            raise
        if len(request.text) > limit:
            raise DocumentTooLargeError(limit)
        if request.text.isspace():
            raise RejectedContentError("text", "is blank")
        fields = [("text", request.text)]
        fields += [(f"focus_areas.{i}", area) for i, area in enumerate(request.focus_areas)]
        for name, value in fields:
            if _CONTROL.search(value):
                raise RejectedContentError(name, "contains control characters")
        return request
