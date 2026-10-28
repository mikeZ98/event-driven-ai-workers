"""Summarization task runner: the ``TaskType.SUMMARIZATION`` handler for the worker pool."""

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

import httpx
from pydantic_core import to_json

from event_driven_ai_workers.core import TaskHandler, TaskOutput
from event_driven_ai_workers.schemas import SummarizationRequest, TaskPayload, TaskType
from event_driven_ai_workers.tools.llm_client import LLMClient, LLMSettings, open_llm_client


class SummarizationRunner:
    """Validates the task payload as a :class:`SummarizationRequest` and calls the LLM."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    async def __call__(self, task: TaskPayload) -> TaskOutput:
        # The payload is JSON data: validate it in JSON mode, as it arrived on the wire.
        # A malformed payload raises ValidationError, a poison pill for the engine.
        request = SummarizationRequest.model_validate_json(to_json(task.payload))
        result = await self._client.summarize(request)
        return result.model_dump(mode="json")


@asynccontextmanager
async def summarization_handlers(
    settings: LLMSettings | None = None, *, transport: httpx.AsyncBaseTransport | None = None
) -> AsyncIterator[Mapping[TaskType, TaskHandler]]:
    """Handler factory for ``create_worker_settings``: one pooled LLM client per worker."""
    async with open_llm_client(settings, transport=transport) as client:
        yield {TaskType.SUMMARIZATION: SummarizationRunner(client)}
