"""RFC 9457 problem details. Problems never echo the request body back to the client."""

from collections.abc import Mapping
from http import HTTPStatus

from pydantic import Field
from starlette.responses import Response

from event_driven_ai_workers.schemas import WorkerModel

PROBLEM_MEDIA_TYPE = "application/problem+json"
_TYPE_PREFIX = "urn:event-driven-ai-workers:problem:"


class Problem(WorkerModel):
    """``application/problem+json`` body."""

    type: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=128)
    status: int = Field(ge=400, le=599)
    detail: str = Field(min_length=1, max_length=512)


class ProblemError(Exception):
    """Raise from routes or dependencies to answer with a problem response."""

    def __init__(
        self, status: int, kind: str, detail: str, *, headers: Mapping[str, str] | None = None
    ) -> None:
        super().__init__(detail)
        self.problem = Problem(
            type=_TYPE_PREFIX + kind,
            title=HTTPStatus(status).phrase,
            status=status,
            detail=detail[:512],
        )
        self.headers = dict(headers or {})


def problem_response(error: ProblemError) -> Response:
    """Serialize ``error`` as a non-cacheable problem response."""
    return Response(
        error.problem.model_dump_json(),
        status_code=error.problem.status,
        media_type=PROBLEM_MEDIA_TYPE,
        headers={"cache-control": "no-store", **error.headers},
    )
