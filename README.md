# event-driven-ai-workers

Asynchronous, event-driven processing for heavy AI workloads. A FastAPI **producer**
accepts tasks and returns immediately; **arq workers** consume them from **Redis**, call
the LLM with retries tuned for 429/529 rate limits, quarantine poison pills in a
**dead-letter queue**, and record every state transition so clients can poll for the
result. Ingestion guardrails and atomic backpressure keep traffic spikes from reaching
Redis or the workers.

Python 3.13 · FastAPI · arq · Redis 7 · Pydantic v2 (strict) · Prometheus ·
`mypy --strict` · 99.9% branch coverage, hermetic test suite.

## Contents

- [Architecture](#architecture)
- [Core invariants](#core-invariants)
- [Bugs found and fixed](#bugs-found-and-fixed)
- [HTTP API](#http-api)
- [Retry and dead-letter policy](#retry-and-dead-letter-policy)
- [Prometheus metrics](#prometheus-metrics)
- [Local development and testing](#local-development-and-testing)
- [Docker](#docker)
- [Operational runbook](#operational-runbook)
- [Configuration reference](#configuration-reference)
- [CI pipeline](#ci-pipeline)
- [Known limitations](#known-limitations)
- [Contributing](#contributing)

## Architecture

```
                 POST /v1/tasks/summarize            GET /v1/tasks/{task_id}
  client ──────────────────────────────────┐   ┌───────────────────────────── client
   (Idempotency-Key header)                │   │        (poll until terminal)
                                           ▼   │
  ┌────────────────────────── producer (edaw api) ─────────────────────────────┐
  │ 1. payload guard: byte cap while streaming, document size, content rules   │
  │ 2. idempotency lookup: replay the original task for a known key            │
  │ 3. backpressure: atomic Lua admission, reserve one queue slot              │
  │ 4. SET NX claim of the key → state PENDING → enqueue arq job → 202         │
  └──────────────────────────────────────┬─────────────────────────────────────┘
                                         │ job id = Idempotency-Key
  ┌──────────────────────────────────── Redis ─────────────────────────────────┐
  │ edaw:queue (zset)          arq queue: queued, deferred and in-flight jobs  │
  │ edaw:queue:admissions      slots reserved by submissions on their way in   │
  │ edaw:idem:<key>            fingerprint + original TaskPayload (TTL)        │
  │ edaw:state:<task_id>       TaskResult: PENDING/RUNNING/terminal (TTL)      │
  │ edaw:dlq (list)            DeadLetterEnvelope JSON, oldest at the tail     │
  │ edaw:queue:health-check    worker heartbeat (TTL interval + 1s)            │
  └──────────────────────────────────────┬─────────────────────────────────────┘
                                         │
  ┌───────────────────── worker × N (edaw worker) ──────────────────────────┐
  │ TaskEngine: RUNNING → handler (LLM) under a per-attempt timeout         │
  │   success                → COMPLETED                                    │
  │   transient, budget left → PENDING, arq.Retry(defer=backoff)            │
  │   fatal or exhausted     → DeadLetterEnvelope → DLQ, DEAD_LETTER        │
  │ /metrics on :9100 (own registry)  · queue-depth sampler task            │
  └──────────────────────────────────────┬──────────────────────────────────┘
                                         │ POST /v1/messages (httpx, pooled)
                                         ▼
                      LLM provider (simulated by default, or live)
```

| Component | Module | Role |
|---|---|---|
| Producer | `api/` | Validates, de-duplicates and admits submissions; serves task state, health and metrics. |
| Guardrails | `guardrails/` | Body size and content screening; atomic queue admission. |
| Engine | `core/engine.py` | Runs one attempt; classifies failures; retries or dead-letters. |
| Worker runtime | `core/worker.py`, `worker_main.py` | arq wiring, lifecycle, heartbeat, `/metrics` listener, shutdown. |
| LLM client | `tools/llm_client.py` | Raw httpx Messages API client; deterministic simulated provider. |
| Contracts | `schemas/` | Frozen, `extra="forbid"`, `strict=True` Pydantic models shared by both sides. |

**Task lifecycle:** `PENDING → RUNNING → COMPLETED | DEAD_LETTER | FAILED`. A
rescheduled retry goes back to `PENDING`. `FAILED` is reserved for the case where even
the DLQ write fails. Terminal states carry `completed_at` and `duration_ms`; only
`COMPLETED` carries a `result`.

## Core invariants

These hold across the codebase and are pinned by tests; changes must preserve them.

**Atomic admission: backpressure that holds under bursts.** Admission is one Lua
script. It counts the queue (`ZCARD`) plus the slots already reserved by submissions
still on their way in, and reserves a slot only if the total is below
`EDAW_GUARDRAILS_MAX_QUEUE_DEPTH`. The submitter holds the slot from before the
idempotency claim until the job is queued, and releases it on failure too. A slot left
behind by a crashed producer expires after 30 s. A full queue answers `503
queue-saturated` with `Retry-After`. Because admission precedes the claim, a refused
client's key stays free for its retry, and replays of existing keys are still served
while the queue is saturated.

**Clean worker shutdown via `close_worker()`.** Shutdown never uses arq's bare
`Worker.close()` (see [Bugs found and fixed](#bugs-found-and-fixed)). `close_worker()`
cancels and reaps in-flight jobs first, so shutdown always deletes the heartbeat key,
runs `on_shutdown` (closing the LLM client and the queue-depth sampler) and exits 0 on
SIGTERM. Cancelled jobs stay queued and are retried by another worker; the arq try
budget keeps two tries of headroom for exactly this.

**Idempotency: one task per key.** Every submission carries an `Idempotency-Key`
(1–128 chars of `[A-Za-z0-9._:-]`). The key is claimed with `SET NX` and stores a
SHA-256 fingerprint of the *validated, normalised* request plus the full
`TaskPayload`. A repeat with the same body returns the original task with
`Idempotent-Replayed: true`. The same key with a different body is `422
idempotency-key-mismatch`. The arq job id *is* the key, so a second job cannot exist.
A replay of a still-`PENDING` task re-drives the enqueue, healing a submission that
crashed between claim and enqueue. Ten concurrent duplicates produce one task and one
LLM call.

**Dead-letter routing: no task is lost or retried forever.** Transient failures (429,
529, 5xx, timeouts, dropped connections) are retried up to `EDAW_WORKER_DLQ_THRESHOLD`
= 5 times, so a task gets **6 attempts** in total. Fatal failures (4xx, schema
violations, model refusals, unknown exceptions) go to the DLQ on the first attempt.
Each `DeadLetterEnvelope` carries the original task, the reason
(`retries_exhausted` or `non_retryable` plus the error class), the retry count, a
bounded traceback and `quarantined_at`. The DLQ write itself is retried with jittered
backoff before the engine gives up and marks the task `FAILED`.

**One retry loop.** Only the engine decides on retries; arq just executes
`Retry(defer=…)`. The LLM client never retries; it raises `UpstreamHTTPError` with the
parsed `Retry-After`. arq's `max_tries` is 1 + threshold + 2 and its job timeout is the
engine's timeout + 10 s, so the engine's decisions always win.

**Guardrails run before Redis.** The body is read under `EDAW_GUARDRAILS_MAX_PAYLOAD_BYTES`
while it streams. `Content-Length` is checked first; understated or chunked bodies are
cut off at the cap, never buffered whole. Then the size, schema and content checks run.
A rejected request, oversized or poisoned, never touches Redis.

**Strict contracts at every edge.** Domain models are `strict=True`, so JSON enters
only through `model_validate_json`, both on the API and when a worker reads a job.
Python-mode coercion never sees wire data.

## Bugs found and fixed

The end-to-end and burst tests surfaced three defects that unit tests had missed:

| Defect | Symptom | Fix |
|---|---|---|
| **Backpressure overshoot.** Check-then-enqueue admission. | 40 concurrent submissions against `max_queue_depth=5` were **all** accepted. Every request read the depth before any enqueued. | Atomic Lua admission with reservations ([invariant](#core-invariants)); a burst test pins exactly 5 admitted. |
| **Shutdown crash.** arq's `close()` gathers cancelled jobs without `return_exceptions`. | A job cancelled during arq's shielded `finish_job` re-raised `CancelledError` out of `close()`. `on_shutdown` never ran (LLM client and sampler leaked), `edaw worker` exited on the exception, and the heartbeat lingered until its TTL. The e2e suite was flaky about 2 runs in 3. | `close_worker()` cancels and reaps jobs first. A characterization test pins arq's behaviour, so the workaround can go once arq fixes `close()`. |
| **Self-counting queue gauge.** The depth sampler ran as an arq cron job. | An empty queue reported `arq_queued_tasks 1` (the sampler counted itself). With every `max_jobs` slot busy, the sampler could not run at all, so the gauge went stale exactly when the backlog grew. | The sampler is a plain asyncio task beside arq, started on worker startup and cancelled on shutdown. |

## HTTP API

| Method & path | Purpose | Success |
|---|---|---|
| `POST /v1/tasks/summarize` | Queue a summarization task (`Idempotency-Key` header required). | `202` + `Location`, `TaskAccepted` body |
| `GET /v1/tasks/{task_id}` | Current `TaskResult`. | `200` |
| `GET /health` (`/healthz`) | Redis ping and worker heartbeat: `ok`, `degraded` (no worker), `unavailable`. | `200`; `503` when Redis is down |
| `GET /metrics` | Producer Prometheus registry. | `200` |
| `GET /docs`, `/openapi.json` | OpenAPI documentation. | `200` |

```bash
curl -s -X POST localhost:8000/v1/tasks/summarize \
  -H 'content-type: application/json' -H 'idempotency-key: report-2026-10' \
  -d '{"text": "Queues decouple producers from consumers. They absorb bursts of load.",
       "max_length": 128, "focus_areas": ["queues"]}'
# 202 {"task_id": "3419ecd1-…", "status": "PENDING", "status_url": "/v1/tasks/3419ecd1-…", …}

curl -s localhost:8000/v1/tasks/3419ecd1-…
# {"status": "COMPLETED", "result": {"summary": "…", "tokens_used": 50}, "duration_ms": 1, …}
```

Request body (`SummarizationRequest`): `text` (1–200,000 characters), `max_length`
(16–4096 tokens, default 256), `focus_areas` (up to 16 strings of 1–64 characters).

Errors are [RFC 9457](https://www.rfc-editor.org/rfc/rfc9457)
`application/problem+json` with `type: urn:event-driven-ai-workers:problem:<kind>`.
They never echo submitted values.

| Status | `kind` | Meaning |
|---|---|---|
| 400 | `idempotency-key-required`, `invalid-idempotency-key` | Missing or malformed `Idempotency-Key`. |
| 404 | `task-not-found` | Unknown or expired task id. |
| 409 | `idempotency-key-in-use` | arq still retains an earlier job for this key (`Retry-After: 1`). |
| 413 | `payload-too-large`, `document-too-large` | Body over the byte cap; text over the character cap. |
| 422 | `invalid-request` | Schema violation (field locations and error types only). |
| 422 | `rejected-content` | Control characters, or a whitespace-only document. |
| 422 | `idempotency-key-mismatch` | Key reused for a different request body. |
| 503 | `queue-saturated` | Backpressure; retry after `Retry-After` seconds. |
| 503 | `queue-unavailable` | Redis unreachable (or refusing writes at `maxmemory`). |

## Retry and dead-letter policy

| Upstream outcome | Error class | Decision |
|---|---|---|
| HTTP 429 | `rate_limited` | Retry; honour `Retry-After` |
| HTTP 500/502/503/504/529 | `upstream_unavailable` | Retry; honour `Retry-After` |
| HTTP 408, timeouts | `timeout` | Retry |
| Connection resets, Redis connection errors | `connection` | Retry |
| Other HTTP 4xx | `client_error` | DLQ immediately |
| Schema violation, refusal, truncated or empty response | `validation` | DLQ immediately |
| Anything else | `unexpected` | DLQ immediately |

Retry `n` waits `uniform(c/2, c)`, where `c = min(EDAW_WORKER_BACKOFF_MAX_S, base ·
multiplier^(n−1))` (capped exponential backoff with equal jitter, decorrelating
concurrent retries). An upstream `Retry-After`, either delta-seconds or an HTTP date,
raises the wait to `min(Retry-After, EDAW_WORKER_RETRY_AFTER_CAP_S)`. With the
defaults (1 s base, ×2, 60 s cap) the five retries are spread over roughly 15–31 s.

Inspect the DLQ (oldest first) and re-submit a fixed task under a new idempotency key:

```bash
docker compose exec redis redis-cli LRANGE edaw:dlq -10 -1
docker compose exec redis redis-cli LLEN edaw:dlq
```

## Prometheus metrics

Workers and the producer run in separate processes, each with its own registry. Every
label comes from a bounded enum, so cardinality is fixed.

| Metric | Type | Labels | Served by | Meaning |
|---|---|---|---|---|
| `arq_queued_tasks` | gauge | `queue` | worker `:9100` | Jobs in the queue, including deferred retries and in-flight jobs; sampled every `EDAW_WORKER_QUEUE_DEPTH_SAMPLE_S`. |
| `arq_task_latency` | histogram (`_bucket`, `_sum`, `_count`) | `task_type`, `status` | worker | Seconds per attempt that reached a terminal state; buckets 0.1 s to 600 s. |
| `arq_retry_count_total` | counter | `task_type`, `error_class` | worker | Attempts that failed transiently and were rescheduled. |
| `arq_dlq_total` | counter | `task_type`, `reason`, `error_class` | worker | Tasks quarantined in the DLQ. |
| `edaw_api_submissions_total` | counter | `task_type`, `outcome` | producer `:8000` | `accepted`, `replayed`, `invalid`, `conflict`, `too_large`, `saturated`, `unavailable`. |

Process, platform and GC collectors are registered too. Starting points for alerts:

```promql
sum(rate(arq_dlq_total[5m])) > 0                                           # anything quarantined
sum(rate(arq_retry_count_total{error_class="rate_limited"}[5m])) > 1       # sustained 429s
max(arq_queued_tasks) > 0.8 * 10000                                        # nearing max_queue_depth
sum(rate(edaw_api_submissions_total{outcome="saturated"}[5m])) > 0         # shedding load
histogram_quantile(0.95, sum by (le) (rate(arq_task_latency_bucket{status="COMPLETED"}[10m])))
```

## Local development and testing

```bash
direnv allow     # venv and caches under .cache/, versioned git hooks, Colima paths
uv sync
uv run ruff check . && uv run ruff format --check .
uv run mypy --strict src tests
uv run bandit -c pyproject.toml -r src
uv run pytest tests/ --cov=src/event_driven_ai_workers --cov-branch --cov-fail-under=90
```

The suite is **hermetic**. `pytest-socket` disables inet sockets for every test
(`--disable-socket` in `pyproject.toml`, asserted by `tests/test_ci.py`). Redis is
fakeredis (with its Lua engine, so the admission script runs for real), and the LLM is
an `httpx.MockTransport`. No test needs Docker, a network or an API key.

| Suite | Covers |
|---|---|
| `test_schemas.py`, `test_config.py` | Strict contracts and settings validation. |
| `test_retry.py`, `test_engine.py` | Failure classification and backoff; engine retry/DLQ decisions and DLQ publishing. |
| `test_worker.py`, `test_llm_client.py`, `test_summarizer.py` | arq integration with a real `Worker`; LLM client and simulated provider; metrics. |
| `test_api.py`, `test_guardrails.py` | Producer endpoints, idempotency races, size/content guards, atomic admission. |
| `test_worker_main.py` | `/metrics` listener, run loop, signals, `close_worker()`, CLI. |
| `test_e2e_pipeline.py` | Client → API → Redis → worker → LLM → poll: happy path, 429/529 retries honouring `Retry-After`, DLQ exhaustion and fatal errors, burst shedding, concurrent duplicates. |
| `test_container.py`, `test_ci.py`, `test_docs.py` | Static checks on the Dockerfile, compose stack, CI workflow and this README. |
| `test_attribution.py`, `test_hermetic.py` | The git-hook and audit policy; the no-network guarantee. |

Run without Docker: `uv run edaw api` and `uv run edaw worker` against a local Redis
(`EDAW_REDIS_URL`); add `--burst` to make a worker exit once the queue is empty.

## Docker

```bash
docker compose up --build        # api on 127.0.0.1:8000, one worker, Redis 7.4
docker compose up --scale worker=3
docker compose port worker 9100  # a worker's ephemeral metrics port
scripts/smoke-compose.sh         # submit → poll → replay → metrics, end to end
docker compose down -v
```

One image serves both roles (`edaw api` is the default command; `edaw worker` runs a
worker). The build is multi-stage: uv installs the frozen lockfile without dev
dependencies and precompiles bytecode. The runtime is `python:3.13-slim` with no pip,
uv, compilers or curl, running as `edaw` (10001:10001) with the application read-only.

The compose stack is hardened throughout: every service is non-root, read-only
(writable `/tmp` tmpfs only), `cap_drop: ALL`, `no-new-privileges` and
resource-limited.
- **Redis** sits on an internal-only network. It runs with `noeviction` (a full Redis
  refuses writes, and the API answers 503, instead of silently dropping jobs), an AOF
  fsynced every second, and a 256 MB cap.
- **Health checks:** the API probes `/healthz` and the worker probes `/metrics`. Both
  wait for Redis to be healthy.
- **Shared queue settings:** the producer and worker read their queue settings from one
  shared block, so they cannot drift.
- **Secrets:** `.env` (see `.env.example`) reaches the worker only, because the
  producer never calls the LLM.

Building needs Docker BuildKit with the buildx plugin. With Colima, `.envrc` keeps the
VM under `.cache/`: run `colima start --cpu 2 --memory 4 --disk 20`.

## Operational runbook

| Signal | Likely cause | Action |
|---|---|---|
| `/health` `degraded` | No worker heartbeat within `EDAW_WORKER_HEALTH_CHECK_INTERVAL_S` + 1 s. Tasks are still accepted and wait in the queue. | Start or scale workers; check worker logs. |
| `/health` 503 `unavailable` | Redis unreachable. | Check the Redis container; submissions answer `503 queue-unavailable`. |
| `503 queue-saturated` rising | Backlog at `EDAW_GUARDRAILS_MAX_QUEUE_DEPTH`. | Scale workers or raise `EDAW_WORKER_MAX_JOBS`; raise the depth only if Redis memory allows. |
| `rate_limited` retries rising | Provider 429s. | Lower total concurrency (workers × `EDAW_WORKER_MAX_JOBS`); retries already honour `Retry-After`. |
| `arq_dlq_total` increasing | Poison pills or an outage longer than the retry budget. | Read envelopes (`LRANGE edaw:dlq`); fix and re-submit with a new key. |
| Redis near `maxmemory` | Large backlog or long state TTL. | Lower `EDAW_WORKER_STATE_TTL_S`, add capacity, raise `EDAW_REDIS_MAXMEMORY`. |

**Shutdown and restarts.** SIGTERM cancels a worker's in-flight jobs. arq re-queues
them for the next worker (at-least-once; see limitations), and the worker exits 0
after cleaning up its heartbeat. A crashed worker's jobs are picked up again once arq's
in-progress marker expires.

**Going live.** Set `EDAW_LLM_MODE=live` and `EDAW_LLM_API_KEY` in `.env`; without a
key, live mode refuses to start. Keep `EDAW_LLM_TIMEOUT_S` below
`EDAW_WORKER_TASK_TIMEOUT_S` so the HTTP timeout fires first.

## Configuration reference

Every setting is an environment variable, validated at startup; an invalid value stops
the process. A misspelt variable is simply ignored, so `tests/test_container.py`
checks that compose and the Dockerfile use only real names, and `tests/test_docs.py`
checks that this reference lists every one. Collections are JSON
(`EDAW_API_CORS_ORIGINS='["https://app.example"]'`).

### Producer (`EDAW_API_*`)

| Variable | Default | Description |
|---|---|---|
| `EDAW_API_HOST` | `127.0.0.1` | Bind address (`0.0.0.0` in the image). |
| `EDAW_API_PORT` | `8000` | Listen port. |
| `EDAW_API_CORS_ORIGINS` | `[]` | Allowed browser origins (`*` or http(s) URLs); empty disables CORS. |
| `EDAW_API_METRICS_ENABLED` | `true` | Serve `GET /metrics`. |
| `EDAW_API_IDEMPOTENCY_KEY_PREFIX` | `edaw:idem:` | Redis prefix for idempotency records (kept for the state TTL). |
| `EDAW_API_HEALTH_TIMEOUT_S` | `1.0` | Deadline for the Redis round trips behind `/health`. |

### Guardrails (`EDAW_GUARDRAILS_*`)

| Variable | Default | Description |
|---|---|---|
| `EDAW_GUARDRAILS_MAX_PAYLOAD_BYTES` | `1048576` | Request body cap, enforced while streaming (1 KiB–16 MiB). |
| `EDAW_GUARDRAILS_MAX_DOCUMENT_CHARS` | `200000` | Document cap in code points; may only lower the schema ceiling. |
| `EDAW_GUARDRAILS_MAX_QUEUE_DEPTH` | `10000` | Admission limit: queued + reserved jobs. |
| `EDAW_GUARDRAILS_BACKPRESSURE_RETRY_AFTER_S` | `5` | `Retry-After` on `503 queue-saturated`. |

### Worker and queue topology (`EDAW_WORKER_*`)

The first five settings below must be identical for the producer and every worker.

| Variable | Default | Description |
|---|---|---|
| `EDAW_WORKER_QUEUE_NAME` | `edaw:queue` | arq queue key. |
| `EDAW_WORKER_DLQ_NAME` | `edaw:dlq` | Dead-letter list key. |
| `EDAW_WORKER_STATE_KEY_PREFIX` | `edaw:state:` | Prefix of task-state keys. |
| `EDAW_WORKER_STATE_TTL_S` | `86400` | Retention of task state, idempotency records and arq results. |
| `EDAW_WORKER_DLQ_THRESHOLD` | `5` | Transient retries before dead-lettering (attempts = threshold + 1). |
| `EDAW_WORKER_MAX_JOBS` | `10` | Concurrent jobs per worker process. |
| `EDAW_WORKER_TASK_TIMEOUT_S` | `120` | Per-attempt deadline (a timeout is retryable). |
| `EDAW_WORKER_BACKOFF_BASE_S` | `1.0` | First retry's backoff ceiling. |
| `EDAW_WORKER_BACKOFF_MAX_S` | `60.0` | Backoff ceiling cap. |
| `EDAW_WORKER_BACKOFF_MULTIPLIER` | `2.0` | Growth factor per retry. |
| `EDAW_WORKER_RETRY_AFTER_CAP_S` | `300.0` | Longest upstream `Retry-After` honoured. |
| `EDAW_WORKER_QUEUE_DEPTH_SAMPLE_S` | `15` | `arq_queued_tasks` sampling interval. |
| `EDAW_WORKER_HEALTH_CHECK_INTERVAL_S` | `10` | Heartbeat refresh interval (the key lives interval + 1 s). |

### Redis (`EDAW_REDIS_*`)

| Variable | Default | Description |
|---|---|---|
| `EDAW_REDIS_URL` | `redis://localhost:6379/0` | `redis://`, `rediss://` or `unix://` DSN (treated as a secret). |
| `EDAW_REDIS_MAX_CONNECTIONS` | `32` | Pool size per process. |
| `EDAW_REDIS_CONN_TIMEOUT_S` | `2` | Connect timeout. |
| `EDAW_REDIS_CONN_RETRIES` | `5` | Connection attempts at startup. |
| `EDAW_REDIS_CONN_RETRY_DELAY_S` | `1` | Delay between connection attempts. |
| `EDAW_REDIS_RETRY_ON_TIMEOUT` | `true` | Retry commands that time out. |

### LLM client (`EDAW_LLM_*`, worker only)

| Variable | Default | Description |
|---|---|---|
| `EDAW_LLM_MODE` | `simulated` | `simulated` (deterministic, offline) or `live`. |
| `EDAW_LLM_API_KEY` | — | Required in live mode; set it in `.env`. |
| `EDAW_LLM_BASE_URL` | `https://api.anthropic.com` | Messages API base URL. |
| `EDAW_LLM_MODEL` | `claude-opus-5-5` | Model id. |
| `EDAW_LLM_API_VERSION` | `2026-10-21` | `anthropic-version` header. |
| `EDAW_LLM_EFFORT` | `medium` | `low`, `medium`, `high`, `xhigh` or `max`. |
| `EDAW_LLM_MAX_OUTPUT_TOKENS` | `16000` | Response ceiling, shared with thinking. |
| `EDAW_LLM_SERVER_SIDE_FALLBACKS` | `true` | Let the provider retry a refused request on its fallback model. |
| `EDAW_LLM_TIMEOUT_S` | `110.0` | HTTP timeout; keep below the task timeout. |
| `EDAW_LLM_CONNECT_TIMEOUT_S` | `5.0` | TCP/TLS connect timeout. |
| `EDAW_LLM_MAX_CONNECTIONS` | `20` | HTTP pool size per worker. |
| `EDAW_LLM_SIMULATE_RATE_LIMIT_EVERY` | `0` | Simulated mode: answer every Nth request with 429 (0 = never). |
| `EDAW_LLM_SIMULATE_RETRY_AFTER_S` | `1` | Simulated mode: `Retry-After` on those 429s. |

### Worker metrics listener (`EDAW_METRICS_*`)

| Variable | Default | Description |
|---|---|---|
| `EDAW_METRICS_ENABLED` | `true` | Serve the worker's `/metrics`. |
| `EDAW_METRICS_HOST` | `127.0.0.1` | Bind address (`0.0.0.0` in the image). |
| `EDAW_METRICS_PORT` | `9100` | Listen port. |
| `EDAW_METRICS_UDS` | — | Serve on this Unix socket instead of host:port. |

### Compose-only

| Variable | Default | Description |
|---|---|---|
| `EDAW_BIND` | `127.0.0.1` | Host interface for published ports. |
| `EDAW_REDIS_MAXMEMORY` | `256mb` | Redis `maxmemory`. |

## CI pipeline

`.github/workflows/ci.yml` runs on pushes and pull requests to `main`. Stale runs are
cancelled except on `main`. The token is read-only. Every action is pinned to a full
commit SHA from `actions`, `astral-sh` or `docker`.

```
audit ─┬─ quality ─┬─ docker
       └─ test ────┘
```

- **audit:** full-history attribution audit of the real PR head (`scripts/audit-attribution.sh`).
- **quality:** ruff lint and format, `mypy --strict src tests`, bandit.
- **test:** the hermetic suite with branch coverage ≥ 90%; coverage XML as an artifact.
- **docker:**
  - **Build context:** proves the context is only `README.md pyproject.toml src uv.lock`,
    even with planted decoys.
  - **Image:** builds the `runtime` target, then asserts the non-root user and the
    absence of installers and fetch tools.
  - **Live stack:** starts the compose stack, runs `scripts/smoke-compose.sh`, and
    requires workers to exit 0 on SIGTERM.

## Known limitations

- **At-least-once execution.** A job cancelled at shutdown, or orphaned by a crashed
  worker, runs again. If it had already finished its LLM call, that call is repeated.
  Results are idempotent per task id, so only the extra cost is visible.
- **Soft, conservative admission.** A job can be counted both as queued and as
  reserved for a moment, so the limit is never exceeded but may refuse slightly early.
  Reservations rely on loosely synchronised producer clocks for their 30 s expiry.
- **Single Redis.** The admission script touches two keys and arq does not support
  Redis Cluster; use a single primary (with a replica for failover).
- **One workload.** `TaskType.AUDIO_TRANSCRIPTION` exists in the contracts, but only
  summarization has an endpoint and a handler.
- **Unsigned images.** CI builds and verifies the image but does not publish, sign or
  attach provenance; that belongs to a release workflow.

## Contributing

Every commit must be authored *and* committed by
`mikeZ98 <145975972+mikeZ98@users.noreply.github.com>` with no AI attribution trailers.
`.githooks/` enforces this locally (`direnv allow` installs them), and
`scripts/audit-attribution.sh` audits the full history in CI. Commits follow
Conventional Commits, one concern per commit, and must pass every gate in
[Local development and testing](#local-development-and-testing).
