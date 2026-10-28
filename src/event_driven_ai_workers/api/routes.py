"""Producer endpoints: task submission, state polling, health and metrics.

Request bodies are read raw and parsed with ``model_validate_json``: the schemas are
``strict`` and FastAPI's Python-mode body parsing would reject JSON-native encodings.
Responses are written with ``model_dump_json`` for the same reason.
"""

import asyncio
import contextlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, cast
from uuid import UUID

from arq.connections import ArqRedis
from fastapi import APIRouter, Depends, Header, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest
from pydantic import BaseModel, TypeAdapter, ValidationError
from redis.exceptions import RedisError

from event_driven_ai_workers import __version__
from event_driven_ai_workers.api.config import API_PREFIX, ApiSettings
from event_driven_ai_workers.api.metrics import ApiMetrics, SubmissionOutcome
from event_driven_ai_workers.api.models import ComponentHealth, HealthReport, TaskAccepted
from event_driven_ai_workers.api.problems import Problem, ProblemError
from event_driven_ai_workers.api.submission import (
    IdempotencyKeyInUseError,
    IdempotencyMismatchError,
    Submission,
    TaskSubmitter,
    fingerprint,
)
from event_driven_ai_workers.core import WorkerSettings, health_check_key
from event_driven_ai_workers.core.ports import resolve
from event_driven_ai_workers.schemas import (
    IdempotencyKey,
    SummarizationRequest,
    TaskPayload,
    TaskResult,
    TaskType,
    WorkerModel,
)

TASK_ROUTE = "get_task"
_MAX_VALIDATION_ERRORS = 5
_RETRY_AFTER_UNAVAILABLE_S = 5
_IDEMPOTENCY_KEY: TypeAdapter[str] = TypeAdapter(IdempotencyKey)


@dataclass(frozen=True, slots=True)
class ApiServices:
    """Per-app dependencies, created by the lifespan once the Redis pool is open."""

    settings: ApiSettings
    worker: WorkerSettings
    pool: ArqRedis
    submitter: TaskSubmitter
    metrics: ApiMetrics
    registry: CollectorRegistry


def get_services(request: Request) -> ApiServices:
    return cast("ApiServices", request.app.state.services)


Services = Annotated[ApiServices, Depends(get_services)]

router = APIRouter()
metrics_router = APIRouter()


def _problems(*statuses: int) -> dict[int | str, dict[str, Any]]:
    return {status: {"model": Problem} for status in statuses}


def _json(model: BaseModel, status: int, headers: dict[str, str] | None = None) -> Response:
    return Response(
        model.model_dump_json(),
        status_code=status,
        media_type="application/json",
        headers={"cache-control": "no-store", **(headers or {})},
    )


def validation_detail(errors: Sequence[Mapping[str, Any]]) -> str:
    """Locations and error types only: never echo the submitted values."""
    issues = [
        f"{'.'.join(str(part) for part in error['loc']) or 'body'}: {error['type']}"
        for error in errors[:_MAX_VALIDATION_ERRORS]
    ]
    return "Invalid request: " + "; ".join(issues)


def unavailable() -> ProblemError:
    return ProblemError(
        503,
        "queue-unavailable",
        "The task queue is temporarily unavailable; retry later.",
        headers={"retry-after": str(_RETRY_AFTER_UNAVAILABLE_S)},
    )


def require_idempotency_key(idempotency_key: Annotated[str | None, Header()] = None) -> str:
    """Dependency: the validated ``Idempotency-Key`` header (required on submissions)."""
    if idempotency_key is None:
        raise ProblemError(
            400, "idempotency-key-required", "The Idempotency-Key header is required."
        )
    try:
        return _IDEMPOTENCY_KEY.validate_python(idempotency_key)
    except ValidationError:
        raise ProblemError(
            400,
            "invalid-idempotency-key",
            "Idempotency-Key must be 1-128 characters from [A-Za-z0-9._:-], "
            "starting with a letter or digit.",
        ) from None


IdempotencyKeyHeader = Annotated[str, Depends(require_idempotency_key)]


async def _submit(
    services: ApiServices, task_type: TaskType, request: WorkerModel, idempotency_key: str
) -> Submission:
    submitter = services.submitter
    request_fingerprint = fingerprint(task_type, request)
    try:
        record = await submitter.lookup(idempotency_key)
        if record is not None:
            return await submitter.replay(record, request_fingerprint)
        task = TaskPayload(
            idempotency_key=idempotency_key,
            task_type=task_type,
            payload=request.model_dump(mode="json"),
        )
        return await submitter.submit(task, request_fingerprint)
    except IdempotencyMismatchError:
        services.metrics.submission(task_type, SubmissionOutcome.CONFLICT)
        raise ProblemError(
            422,
            "idempotency-key-mismatch",
            "This Idempotency-Key was already used for a different request.",
        ) from None
    except IdempotencyKeyInUseError:
        services.metrics.submission(task_type, SubmissionOutcome.CONFLICT)
        raise ProblemError(
            409,
            "idempotency-key-in-use",
            "An earlier task with this Idempotency-Key is still retained; retry later.",
            headers={"retry-after": "1"},
        ) from None
    except RedisError:
        services.metrics.submission(task_type, SubmissionOutcome.UNAVAILABLE)
        raise unavailable() from None


def _accepted(request: Request, services: ApiServices, submission: Submission) -> Response:
    task = submission.task
    outcome = SubmissionOutcome.REPLAYED if submission.replayed else SubmissionOutcome.ACCEPTED
    services.metrics.submission(task.task_type, outcome)
    location = str(request.app.url_path_for(TASK_ROUTE, task_id=str(task.task_id)))
    headers = {"location": location}
    if submission.replayed:
        headers["idempotent-replayed"] = "true"
    return _json(TaskAccepted.for_task(task, submission.status, location), 202, headers)


@router.post(
    f"{API_PREFIX}/tasks/summarize",
    status_code=202,
    summary="Queue a document summarization task",
    responses={202: {"model": TaskAccepted}, **_problems(400, 409, 422, 503)},
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {"application/json": {"schema": SummarizationRequest.model_json_schema()}},
        }
    },
)
async def submit_summarization(
    request: Request, services: Services, idempotency_key: IdempotencyKeyHeader
) -> Response:
    try:
        summarization = SummarizationRequest.model_validate_json(await request.body())
    except ValidationError as exc:
        services.metrics.submission(TaskType.SUMMARIZATION, SubmissionOutcome.INVALID)
        raise ProblemError(422, "invalid-request", validation_detail(exc.errors())) from None
    submission = await _submit(services, TaskType.SUMMARIZATION, summarization, idempotency_key)
    return _accepted(request, services, submission)


@router.get(
    f"{API_PREFIX}/tasks/{{task_id}}",
    name=TASK_ROUTE,
    summary="Current state of a task",
    responses={200: {"model": TaskResult}, **_problems(404, 422, 503)},
)
async def get_task(task_id: UUID, services: Services) -> Response:
    state = await services.submitter.status(task_id)
    if state is None:
        raise ProblemError(404, "task-not-found", "No task with this id (it may have expired).")
    return _json(state, 200)


@router.get(
    "/health",
    summary="Redis reachability and worker heartbeat",
    responses={200: {"model": HealthReport}, 503: {"model": HealthReport}},
)
@router.get("/healthz", include_in_schema=False)
async def health(services: Services) -> Response:
    redis_ok = workers_ok = False
    with contextlib.suppress(RedisError, OSError, TimeoutError):
        async with asyncio.timeout(services.settings.health_timeout_s):
            redis_ok = bool(await resolve(services.pool.ping()))
            heartbeat = health_check_key(services.worker.queue_name)
            workers_ok = bool(await resolve(services.pool.exists(heartbeat)))
    report = HealthReport(
        status="ok" if workers_ok else "degraded" if redis_ok else "unavailable",
        version=__version__,
        components=(
            ComponentHealth(
                name="redis", available=redis_ok, detail="ping ok" if redis_ok else "unreachable"
            ),
            ComponentHealth(
                name="workers",
                available=workers_ok,
                detail="heartbeat seen" if workers_ok else "no worker heartbeat",
            ),
        ),
    )
    return _json(report, 200 if redis_ok else 503)


@metrics_router.get("/metrics", include_in_schema=False)
async def metrics(services: Services) -> Response:
    return Response(generate_latest(services.registry), media_type=CONTENT_TYPE_LATEST)
