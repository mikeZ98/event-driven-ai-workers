# syntax=docker/dockerfile:1.7
#
# event-driven-ai-workers: one hardened image, two roles.
#   builder: uv resolves the frozen lockfile into /app/.venv (no dev deps, project as a
#            wheel, bytecode precompiled).
#   runtime: python:3.13-slim with the venv copied in. No pip/uv/compilers/curl,
#            non-root (10001:10001), read-only-rootfs compatible (only /tmp is written).
#
#   docker run IMAGE api      FastAPI producer on :8000 (the default)
#   docker run IMAGE worker   arq worker, Prometheus /metrics on :9100
#
# For reproducible releases, pin both FROM images by digest (@sha256:...).

ARG PYTHON_VERSION=3.13
ARG UV_VERSION=0.11.7

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ---------------------------------------------------------------------------
# Build stage
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

COPY --from=uv /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Dependencies first: this layer is cached until the lockfile changes.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Then the project itself, as a regular (non-editable) install: the runtime image
# needs no source tree.
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# Fail the build, not the first container start, if the entrypoint is broken.
RUN /app/.venv/bin/edaw --version

# ---------------------------------------------------------------------------
# Runtime stage
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ARG VERSION=0.1.0
LABEL org.opencontainers.image.title="event-driven-ai-workers" \
      org.opencontainers.image.description="FastAPI producer and arq workers for LLM tasks" \
      org.opencontainers.image.version="${VERSION}"

# Shrink the attack surface: no package installers in the final image.
RUN set -eux; \
    rm -rf /usr/local/bin/pip* \
           /usr/local/lib/python3*/site-packages/pip \
           /usr/local/lib/python3*/site-packages/pip-*.dist-info \
           /usr/local/lib/python3*/ensurepip; \
    groupadd --system --gid 10001 edaw; \
    useradd --system --uid 10001 --gid 10001 --no-create-home \
            --home-dir /nonexistent --shell /usr/sbin/nologin edaw

# Owned by root and read-only to edaw: the application cannot modify itself.
COPY --from=builder /app/.venv /app/.venv

# Listeners bind all interfaces inside the container (the code defaults to loopback).
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    HOME=/tmp \
    EDAW_API_HOST=0.0.0.0 \
    EDAW_API_PORT=8000 \
    EDAW_METRICS_HOST=0.0.0.0 \
    EDAW_METRICS_PORT=9100

WORKDIR /app
USER 10001:10001
EXPOSE 8000 9100

# No HEALTHCHECK here: the right probe depends on the role (api: GET /healthz on
# :8000, worker: GET /metrics on :9100), so the orchestrator defines it per service.
ENTRYPOINT ["edaw"]
CMD ["api"]
