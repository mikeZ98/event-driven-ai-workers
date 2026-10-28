# event-driven-ai-workers

Asynchronous background processing for heavy AI workloads (document summarization,
audio pipelines): a FastAPI producer, ARQ/Redis workers with exponential-backoff
retries on LLM 429/5xx responses, a dead-letter queue after 5 failed attempts and
Prometheus metrics.

## Development

```bash
direnv allow            # pins venv and caches under .cache/, enables .githooks
uv sync
uv run ruff check . && uv run ruff format --check .
uv run mypy --strict src tests
uv run pytest tests/ --cov=src/event_driven_ai_workers --cov-branch --cov-fail-under=90
```

Every commit must be authored and committed by
`mikeZ98 <145975972+mikeZ98@users.noreply.github.com>` with no AI attribution
trailers; `.githooks/` enforces this locally and `scripts/audit-attribution.sh`
audits history in CI.
