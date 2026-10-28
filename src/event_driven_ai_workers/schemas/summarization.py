"""Document summarization request/result contracts (``TaskType.SUMMARIZATION``)."""

from typing import Annotated

from pydantic import Field, StringConstraints

from event_driven_ai_workers.schemas._base import WorkerModel

MAX_DOCUMENT_CHARS = 200_000
"""Contract ceiling on ``SummarizationRequest.text``; deployments may only lower it."""

FocusArea = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]


class SummarizationRequest(WorkerModel):
    """Text to summarise, the summary length budget in tokens and optional focus areas."""

    text: str = Field(min_length=1, max_length=MAX_DOCUMENT_CHARS)
    max_length: int = Field(default=256, ge=16, le=4096)
    focus_areas: tuple[FocusArea, ...] = Field(default=(), max_length=16)


class SummarizationResult(WorkerModel):
    """Summary produced by the LLM and the tokens billed for it."""

    summary: str = Field(min_length=1)
    tokens_used: int = Field(ge=0)
