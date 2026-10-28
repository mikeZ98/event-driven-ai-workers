"""Schema contracts: strict, frozen, extra-forbidding models with validated lifecycles."""

import json
from datetime import UTC, datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ValidationError
from pydantic_core import to_json

from event_driven_ai_workers.schemas import (
    DeadLetterEnvelope,
    SummarizationRequest,
    SummarizationResult,
    TaskPayload,
    TaskResult,
    TaskStatus,
    TaskType,
    WorkerModel,
    utc_now,
)

CREATED = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)
TASK_ID = UUID("6f1c2a4e-8b7d-4c3a-9e2f-1a2b3c4d5e6f")


def _task(**overrides: Any) -> TaskPayload:
    fields: dict[str, Any] = {
        "task_id": TASK_ID,
        "idempotency_key": "doc-42:v1",
        "task_type": TaskType.SUMMARIZATION,
        "payload": {"text": "hello", "max_length": 64, "focus_areas": ["risk"]},
        "created_at": CREATED,
    } | overrides
    return TaskPayload(**fields)


def _completed(**overrides: Any) -> TaskResult:
    fields: dict[str, Any] = {
        "task_id": TASK_ID,
        "status": TaskStatus.COMPLETED,
        "result": {"summary": "short"},
        "duration_ms": 1500,
        "completed_at": CREATED,
    } | overrides
    return TaskResult(**fields)


def _envelope(**overrides: Any) -> DeadLetterEnvelope:
    fields: dict[str, Any] = {
        "original_task": _task(),
        "failure_reason": "LLM provider returned 503 after 5 attempts",
        "retry_attempts": 5,
        "error_trace": "Traceback (most recent call last): ...",
        "quarantined_at": CREATED,
    } | overrides
    return DeadLetterEnvelope(**fields)


SAMPLES: list[BaseModel] = [
    _task(),
    _completed(),
    TaskResult(task_id=TASK_ID, status=TaskStatus.PENDING),
    SummarizationRequest(text="Some long document.", max_length=128, focus_areas=("risk",)),
    SummarizationResult(summary="Short.", tokens_used=321),
    _envelope(),
]


class TestBaseContract:
    def test_every_domain_model_shares_the_strict_config(self) -> None:
        for model in SAMPLES:
            assert isinstance(model, WorkerModel)
            config = type(model).model_config
            assert config.get("frozen") is True
            assert config.get("extra") == "forbid"
            assert config.get("strict") is True

    @pytest.mark.parametrize("model", SAMPLES, ids=lambda m: type(m).__name__)
    def test_frozen(self, model: BaseModel) -> None:
        field = next(iter(type(model).model_fields))
        with pytest.raises(ValidationError, match="frozen"):
            setattr(model, field, getattr(model, field))

    @pytest.mark.parametrize("model", SAMPLES, ids=lambda m: type(m).__name__)
    def test_extra_fields_forbidden(self, model: BaseModel) -> None:
        data = model.model_dump() | {"unexpected": 1}
        with pytest.raises(ValidationError, match="extra_forbidden"):
            type(model).model_validate(data)

    @pytest.mark.parametrize("model", SAMPLES, ids=lambda m: type(m).__name__)
    def test_json_roundtrip(self, model: BaseModel) -> None:
        restored = type(model).model_validate_json(model.model_dump_json())
        assert restored == model

    @pytest.mark.parametrize("model", SAMPLES, ids=lambda m: type(m).__name__)
    def test_python_roundtrip(self, model: BaseModel) -> None:
        assert type(model).model_validate(model.model_dump()) == model

    def test_utc_now_is_aware_utc(self) -> None:
        assert utc_now().tzinfo is UTC


class TestTaskPayload:
    def test_defaults_generate_unique_id_and_utc_timestamp(self) -> None:
        before = utc_now()
        first = TaskPayload(idempotency_key="k1", task_type=TaskType.SUMMARIZATION, payload={})
        second = TaskPayload(idempotency_key="k1", task_type=TaskType.SUMMARIZATION, payload={})
        assert first.task_id != second.task_id
        assert first.task_id.version == 4
        assert first.created_at.tzinfo is UTC
        assert before <= first.created_at <= utc_now()

    def test_strict_mode_rejects_coercion_in_python_mode(self) -> None:
        with pytest.raises(ValidationError):
            _task(task_id=str(TASK_ID))
        with pytest.raises(ValidationError):
            _task(task_type="summarization")
        with pytest.raises(ValidationError):
            _task(created_at=CREATED.isoformat())

    def test_json_mode_accepts_wire_encodings(self) -> None:
        raw = (
            '{"task_id": "6f1c2a4e-8b7d-4c3a-9e2f-1a2b3c4d5e6f", "idempotency_key": "doc-42:v1",'
            ' "task_type": "summarization", "payload": {"text": "hello", "max_length": 64,'
            ' "focus_areas": ["risk"]}, "created_at": "2026-10-28T12:00:00Z"}'
        )
        assert TaskPayload.model_validate_json(raw) == _task()

    def test_rejects_naive_datetime(self) -> None:
        with pytest.raises(ValidationError, match="timezone_aware"):
            _task(created_at=datetime(2026, 10, 28, 12, 0))  # noqa: DTZ001

    def test_normalises_offsets_to_utc(self) -> None:
        warsaw = timezone(timedelta(hours=1))
        task = _task(created_at=datetime(2026, 10, 28, 13, 0, tzinfo=warsaw))
        assert task.created_at == CREATED
        assert task.created_at.tzinfo is UTC

    @pytest.mark.parametrize("key", ["", " lead", "has space", "-dash-first", "x" * 129, "ü"])
    def test_rejects_malformed_idempotency_keys(self, key: str) -> None:
        with pytest.raises(ValidationError):
            _task(idempotency_key=key)

    @pytest.mark.parametrize("key", ["a", "doc-42:v1", "tenant.7_job-9", "x" * 128])
    def test_accepts_idempotency_keys(self, key: str) -> None:
        assert _task(idempotency_key=key).idempotency_key == key

    def test_unknown_task_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TaskPayload.model_validate_json(
                '{"idempotency_key": "k", "task_type": "video", "payload": {}}'
            )

    def test_payload_must_be_json_object(self) -> None:
        with pytest.raises(ValidationError):
            _task(payload=["not", "an", "object"])
        with pytest.raises(ValidationError):
            _task(payload={"handle": object()})


class TestTaskStatus:
    @pytest.mark.parametrize(
        ("status", "terminal"),
        [
            (TaskStatus.PENDING, False),
            (TaskStatus.RUNNING, False),
            (TaskStatus.COMPLETED, True),
            (TaskStatus.FAILED, True),
            (TaskStatus.DEAD_LETTER, True),
        ],
    )
    def test_is_terminal(self, status: TaskStatus, terminal: bool) -> None:
        assert status.is_terminal is terminal

    def test_wire_values_are_upper_case(self) -> None:
        assert [s.value for s in TaskStatus] == [
            "PENDING",
            "RUNNING",
            "COMPLETED",
            "FAILED",
            "DEAD_LETTER",
        ]


class TestTaskResult:
    @pytest.mark.parametrize("status", [TaskStatus.PENDING, TaskStatus.RUNNING])
    def test_in_flight_states_carry_no_outcome(self, status: TaskStatus) -> None:
        result = TaskResult(task_id=TASK_ID, status=status)
        assert (result.result, result.duration_ms, result.completed_at) == (None, None, None)

    @pytest.mark.parametrize(
        "outcome",
        [{"result": {"summary": "x"}}, {"duration_ms": 5}, {"completed_at": CREATED}],
    )
    def test_in_flight_states_reject_outcome_fields(self, outcome: dict[str, Any]) -> None:
        with pytest.raises(ValidationError, match="must not carry outcome fields"):
            TaskResult(task_id=TASK_ID, status=TaskStatus.RUNNING, **outcome)

    @pytest.mark.parametrize("missing", ["completed_at", "duration_ms"])
    @pytest.mark.parametrize(
        "status", [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.DEAD_LETTER]
    )
    def test_terminal_states_require_timing(self, status: TaskStatus, missing: str) -> None:
        result_field = {"summary": "x"} if status is TaskStatus.COMPLETED else None
        with pytest.raises(ValidationError, match="requires completed_at and duration_ms"):
            _completed(status=status, result=result_field, **{missing: None})

    def test_completed_requires_result(self) -> None:
        with pytest.raises(ValidationError, match="result is required"):
            _completed(result=None)

    @pytest.mark.parametrize("status", [TaskStatus.FAILED, TaskStatus.DEAD_LETTER])
    def test_failed_states_forbid_result(self, status: TaskStatus) -> None:
        with pytest.raises(ValidationError, match="forbidden otherwise"):
            _completed(status=status)
        assert _completed(status=status, result=None).result is None

    def test_negative_duration_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _completed(duration_ms=-1)

    def test_strict_mode_rejects_float_duration(self) -> None:
        with pytest.raises(ValidationError):
            _completed(duration_ms=1.0)

    def test_status_from_json_value(self) -> None:
        raw = '{"task_id": "6f1c2a4e-8b7d-4c3a-9e2f-1a2b3c4d5e6f", "status": "DEAD_LETTER"}'
        with pytest.raises(ValidationError, match="requires completed_at"):
            TaskResult.model_validate_json(raw)
        pending = TaskResult.model_validate_json(raw.replace("DEAD_LETTER", "PENDING"))
        assert pending.status is TaskStatus.PENDING


class TestSummarization:
    def test_request_defaults(self) -> None:
        request = SummarizationRequest(text="doc")
        assert request.max_length == 256
        assert request.focus_areas == ()

    @pytest.mark.parametrize("max_length", [15, 4097, 0, -1])
    def test_request_rejects_out_of_range_max_length(self, max_length: int) -> None:
        with pytest.raises(ValidationError):
            SummarizationRequest(text="doc", max_length=max_length)

    @pytest.mark.parametrize("text", ["", "x" * 200_001])
    def test_request_rejects_empty_or_oversized_text(self, text: str) -> None:
        with pytest.raises(ValidationError):
            SummarizationRequest(text=text)

    def test_focus_areas_are_stripped_and_bounded(self) -> None:
        request = SummarizationRequest(text="doc", focus_areas=("  risk ", "cost"))
        assert request.focus_areas == ("risk", "cost")
        with pytest.raises(ValidationError):
            SummarizationRequest(text="doc", focus_areas=("   ",))
        with pytest.raises(ValidationError):
            SummarizationRequest(text="doc", focus_areas=tuple(f"a{i}" for i in range(17)))

    def test_focus_areas_strict_python_mode_requires_tuple(self) -> None:
        with pytest.raises(ValidationError):
            SummarizationRequest.model_validate({"text": "doc", "focus_areas": ["risk"]})
        parsed = SummarizationRequest.model_validate_json('{"text": "doc", "focus_areas": ["r"]}')
        assert parsed.focus_areas == ("r",)

    def test_request_validates_task_payload_on_the_wire(self) -> None:
        request = SummarizationRequest.model_validate_json(to_json(_task().payload))
        assert request == SummarizationRequest(text="hello", max_length=64, focus_areas=("risk",))

    def test_result_constraints(self) -> None:
        with pytest.raises(ValidationError):
            SummarizationResult(summary="", tokens_used=1)
        with pytest.raises(ValidationError):
            SummarizationResult(summary="ok", tokens_used=-1)
        with pytest.raises(ValidationError):
            SummarizationResult.model_validate({"summary": "ok", "tokens_used": "12"})


class TestDeadLetterEnvelope:
    def test_wraps_original_task_verbatim(self) -> None:
        envelope = _envelope()
        assert envelope.original_task == _task()
        assert envelope.original_task.task_id == TASK_ID

    def test_defaults(self) -> None:
        envelope = DeadLetterEnvelope(
            original_task=_task(), failure_reason="validation error", retry_attempts=0
        )
        assert envelope.error_trace is None
        assert envelope.quarantined_at.tzinfo is UTC

    def test_strictness_propagates_into_the_nested_task(self) -> None:
        assert _envelope(original_task=_task().model_dump()).original_task == _task()
        with pytest.raises(ValidationError, match=r"original_task\.task_id"):
            _envelope(original_task=_task().model_dump(mode="json"))

    @pytest.mark.parametrize(
        "overrides",
        [
            {"failure_reason": ""},
            {"failure_reason": "x" * 1025},
            {"retry_attempts": -1},
            {"retry_attempts": True},
            {"error_trace": "x" * 32_769},
            {"quarantined_at": datetime(2026, 1, 1)},  # noqa: DTZ001
        ],
    )
    def test_constraints(self, overrides: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            _envelope(**overrides)

    def test_nested_extra_fields_forbidden_on_the_wire(self) -> None:
        data = _envelope().model_dump(mode="json")
        data["original_task"]["sneaky"] = True
        with pytest.raises(ValidationError, match="extra_forbidden"):
            DeadLetterEnvelope.model_validate_json(json.dumps(data))

    def test_task_id_is_preserved_through_the_dlq(self) -> None:
        task = _task(task_id=uuid4())
        restored = DeadLetterEnvelope.model_validate_json(
            _envelope(original_task=task).model_dump_json()
        )
        assert restored.original_task.task_id == task.task_id
