"""LLM client: request shape, response parsing, 429/Retry-After handling, simulator."""

import json
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from event_driven_ai_workers.core import ErrorClass, UpstreamHTTPError, classify
from event_driven_ai_workers.schemas import SummarizationRequest, SummarizationResult
from event_driven_ai_workers.tools import (
    LLMClient,
    LLMRefusalError,
    LLMSettings,
    SimulatedLLMProvider,
    UpstreamContractError,
    open_llm_client,
)
from event_driven_ai_workers.tools.llm_client import (
    FALLBACK_BETA,
    MAX_ERROR_MESSAGE_CHARS,
    MESSAGES_PATH,
    SYSTEM_PROMPT,
    build_prompt,
)

Handler = Callable[[httpx.Request], httpx.Response]

REQUEST = SummarizationRequest(
    text="Queues decouple producers from workers. Retries absorb transient faults. Extra.",
    max_length=128,
    focus_areas=("reliability", "cost"),
)


def _message(
    content: list[dict[str, Any]] | None = None,
    stop_reason: str = "end_turn",
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content if content is not None else [{"type": "text", "text": "A summary."}],
        "stop_reason": stop_reason,
        "usage": usage if usage is not None else {"input_tokens": 100, "output_tokens": 20},
    }


def _error(status: int, error_type: str, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(
        status,
        headers=headers,
        json={"type": "error", "error": {"type": error_type, "message": "upstream says no"}},
    )


@pytest.fixture
def live_settings() -> LLMSettings:
    return LLMSettings.model_validate(
        {"mode": "live", "api_key": "sk-test-key", "base_url": "https://llm.test"}
    )


@pytest.fixture
def make_client(
    live_settings: LLMSettings,
) -> Callable[..., AbstractAsyncContextManager[LLMClient]]:
    def make(
        handler: Handler, settings: LLMSettings | None = None
    ) -> AbstractAsyncContextManager[LLMClient]:
        return open_llm_client(settings or live_settings, transport=httpx.MockTransport(handler))

    return make


async def _summarize(
    make_client: Callable[..., Any], handler: Handler, settings: LLMSettings | None = None
) -> SummarizationResult:
    async with make_client(handler, settings) as client:
        result: SummarizationResult = await client.summarize(REQUEST)
        return result


class TestRequestShape:
    async def test_messages_api_request(self, make_client: Callable[..., Any]) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=_message())

        await _summarize(make_client, handler)
        [request] = seen
        assert request.method == "POST"
        assert str(request.url) == f"https://llm.test{MESSAGES_PATH}"
        assert request.headers["x-api-key"] == "sk-test-key"
        assert request.headers["anthropic-version"] == "2026-10-21"
        assert request.headers["anthropic-beta"] == FALLBACK_BETA
        body = json.loads(request.content)
        assert body == {
            "model": "claude-opus-5-5",
            "max_tokens": 16_000,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": build_prompt(REQUEST)}],
            "output_config": {"effort": "medium"},
            "fallbacks": "default",
        }
        assert "thinking" not in body

    async def test_fallbacks_can_be_disabled(self, make_client: Callable[..., Any]) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=_message())

        settings = LLMSettings.model_validate(
            {"mode": "live", "api_key": "k", "server_side_fallbacks": False, "effort": "low"}
        )
        await _summarize(make_client, handler, settings)
        assert "anthropic-beta" not in seen[0].headers
        body = json.loads(seen[0].content)
        assert "fallbacks" not in body
        assert body["output_config"] == {"effort": "low"}

    def test_prompt_fences_untrusted_document_and_states_budget(self) -> None:
        prompt = build_prompt(REQUEST)
        assert "at most 128 tokens" in prompt
        assert "Focus on: reliability; cost." in prompt
        assert prompt.endswith(f"<document>\n{REQUEST.text}\n</document>")
        assert "Focus on" not in build_prompt(SummarizationRequest(text="doc"))


class TestResponseParsing:
    async def test_joins_text_blocks_and_ignores_others(
        self, make_client: Callable[..., Any]
    ) -> None:
        content: list[dict[str, Any]] = [
            {"type": "thinking", "thinking": ""},
            {"type": "text", "text": "Part one. "},
            {"type": "fallback", "from": {"model": "a"}, "to": {"model": "b"}},
            {"type": "text", "text": "Part two.  "},
        ]
        result = await _summarize(
            make_client, lambda _: httpx.Response(200, json=_message(content))
        )
        assert result.summary == "Part one. Part two."

    async def test_tokens_used_includes_cache_tokens(self, make_client: Callable[..., Any]) -> None:
        usage = {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_creation_input_tokens": 7,
            "cache_read_input_tokens": 3,
        }
        result = await _summarize(
            make_client, lambda _: httpx.Response(200, json=_message(usage=usage))
        )
        assert result.tokens_used == 25

    async def test_refusal_is_a_poison_pill(self, make_client: Callable[..., Any]) -> None:
        body = _message(content=[], stop_reason="refusal")
        with pytest.raises(LLMRefusalError) as excinfo:
            await _summarize(make_client, lambda _: httpx.Response(200, json=body))
        assert classify(excinfo.value).retryable is False

    @pytest.mark.parametrize(
        "body",
        [
            _message(stop_reason="max_tokens"),
            _message(content=[{"type": "thinking", "thinking": ""}]),
            _message(content=[{"type": "text", "text": "   "}]),
            {"unexpected": True},
        ],
    )
    async def test_unusable_success_is_a_contract_error(
        self, make_client: Callable[..., Any], body: dict[str, Any]
    ) -> None:
        with pytest.raises(UpstreamContractError) as excinfo:
            await _summarize(make_client, lambda _: httpx.Response(200, json=body))
        assert classify(excinfo.value).error_class is ErrorClass.VALIDATION

    async def test_non_json_success_is_a_contract_error(
        self, make_client: Callable[..., Any]
    ) -> None:
        with pytest.raises(UpstreamContractError):
            await _summarize(make_client, lambda _: httpx.Response(200, text="<html>"))


class TestRateLimiting:
    async def test_429_carries_retry_after_seconds(self, make_client: Callable[..., Any]) -> None:
        response = _error(429, "rate_limit_error", headers={"retry-after": "17"})
        with pytest.raises(UpstreamHTTPError) as excinfo:
            await _summarize(make_client, lambda _: response)
        error = excinfo.value
        assert (error.status_code, error.retry_after_s) == (429, 17.0)
        assert "rate_limit_error: upstream says no" in str(error)
        failure = classify(error)
        assert (failure.error_class, failure.retryable, failure.retry_after_s) == (
            ErrorClass.RATE_LIMITED,
            True,
            17.0,
        )

    async def test_429_with_http_date_retry_after(self, make_client: Callable[..., Any]) -> None:
        response = _error(
            429, "rate_limit_error", headers={"retry-after": "Thu, 01 Jan 1970 00:00:00 GMT"}
        )
        with pytest.raises(UpstreamHTTPError) as excinfo:
            await _summarize(make_client, lambda _: response)
        assert excinfo.value.retry_after_s == 0.0

    async def test_429_without_retry_after(self, make_client: Callable[..., Any]) -> None:
        with pytest.raises(UpstreamHTTPError) as excinfo:
            await _summarize(make_client, lambda _: _error(429, "rate_limit_error"))
        assert excinfo.value.retry_after_s is None

    @pytest.mark.parametrize(
        ("status", "error_type", "retryable"),
        [
            (529, "overloaded_error", True),
            (500, "api_error", True),
            (400, "invalid_request_error", False),
            (401, "authentication_error", False),
            (403, "permission_error", False),
            (413, "request_too_large", False),
        ],
    )
    async def test_status_errors_are_classified(
        self, make_client: Callable[..., Any], status: int, error_type: str, retryable: bool
    ) -> None:
        with pytest.raises(UpstreamHTTPError) as excinfo:
            await _summarize(make_client, lambda _: _error(status, error_type))
        assert excinfo.value.status_code == status
        assert classify(excinfo.value).retryable is retryable

    async def test_non_json_error_body_is_bounded(self, make_client: Callable[..., Any]) -> None:
        with pytest.raises(UpstreamHTTPError) as excinfo:
            await _summarize(make_client, lambda _: httpx.Response(502, text="x" * 5_000))
        assert len(str(excinfo.value)) <= len("upstream HTTP 502: ") + MAX_ERROR_MESSAGE_CHARS

    async def test_transport_timeouts_propagate_as_retryable(
        self, make_client: Callable[..., Any]
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow upstream", request=request)

        with pytest.raises(httpx.ReadTimeout) as excinfo:
            await _summarize(make_client, handler)
        assert classify(excinfo.value).error_class is ErrorClass.TIMEOUT


class TestSettings:
    def test_defaults_are_offline(self) -> None:
        settings = LLMSettings()
        assert settings.mode == "simulated"
        assert settings.model == "claude-opus-5-5"

    def test_live_mode_requires_api_key(self) -> None:
        with pytest.raises(ValidationError, match="api_key"):
            LLMSettings.model_validate({"mode": "live"})

    def test_env_configuration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EDAW_LLM_MODE", "live")
        monkeypatch.setenv("EDAW_LLM_API_KEY", "sk-env")
        settings = LLMSettings()
        assert settings.api_key is not None
        assert settings.api_key.get_secret_value() == "sk-env"
        assert "sk-env" not in repr(settings)

    @pytest.mark.parametrize(
        "overrides", [{"base_url": "ftp://x"}, {"effort": "extreme"}, {"max_output_tokens": 10}]
    )
    def test_rejects_invalid(self, overrides: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            LLMSettings.model_validate(overrides)


class TestSimulatedProvider:
    async def test_default_client_is_simulated_and_offline(self) -> None:
        async with open_llm_client() as client:
            result = await client.summarize(REQUEST)
        assert result.summary == (
            "Queues decouple producers from workers. Retries absorb transient faults."
        )
        assert result.tokens_used > 0

    async def test_rate_limits_every_nth_request(self) -> None:
        settings = LLMSettings(simulate_rate_limit_every=2, simulate_retry_after_s=3)
        async with open_llm_client(settings) as client:
            await client.summarize(REQUEST)
            with pytest.raises(UpstreamHTTPError) as excinfo:
                await client.summarize(REQUEST)
            await client.summarize(REQUEST)
        assert (excinfo.value.status_code, excinfo.value.retry_after_s) == (429, 3.0)

    async def test_long_documents_are_truncated(self) -> None:
        provider = SimulatedLLMProvider(max_summary_words=5)
        settings = LLMSettings()
        async with open_llm_client(settings, transport=httpx.MockTransport(provider)) as client:
            result = await client.summarize(
                SummarizationRequest(text="one two three four five six")
            )
        assert result.summary == "one two three four five"
        assert len(provider.requests) == 1

    @pytest.mark.parametrize(
        "content",
        [
            b"not json",
            b"{}",
            b'{"model": "m", "max_tokens": 5, "messages": []}',
            b'{"model": 1, "max_tokens": 5, "messages": [{"role": "user", "content": "x"}]}',
        ],
    )
    def test_malformed_requests_get_400(self, content: bytes) -> None:
        response = SimulatedLLMProvider()(
            httpx.Request("POST", f"https://sim{MESSAGES_PATH}", content=content)
        )
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "invalid_request_error"

    def test_unknown_endpoint_gets_404(self) -> None:
        response = SimulatedLLMProvider()(httpx.Request("GET", "https://sim/v1/models"))
        assert response.status_code == 404

    def test_prompt_without_document_tags_is_summarized_whole(self) -> None:
        body = {"model": "m", "max_tokens": 5, "messages": [{"role": "user", "content": "Hi."}]}
        response = SimulatedLLMProvider()(
            httpx.Request("POST", f"https://sim{MESSAGES_PATH}", json=body)
        )
        assert response.json()["content"][-1]["text"] == "Hi."

    def test_empty_document(self) -> None:
        body = {
            "model": "m",
            "max_tokens": 5,
            "messages": [{"role": "user", "content": "<document>\n \n</document>"}],
        }
        response = SimulatedLLMProvider()(
            httpx.Request("POST", f"https://sim{MESSAGES_PATH}", json=body)
        )
        assert response.json()["content"][-1]["text"] == "(empty document)"
