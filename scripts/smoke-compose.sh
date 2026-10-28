#!/usr/bin/env bash
# Smoke-test a running compose stack end to end over HTTP.
#
# Usage: scripts/smoke-compose.sh [<api base url>]   (default: http://127.0.0.1:8000)
# Expects `docker compose up --wait` to have finished. Used by CI and locally.
#
# Checks: /health is ok (Redis reachable, worker heartbeat seen); a submitted task
# completes with a summary; a resend replays the original task; the worker's and the
# producer's /metrics count what happened.
set -euo pipefail

BASE="${1:-http://127.0.0.1:8000}"
KEY="smoke-$(date +%s)-$$"
BODY='{"text": "Queues decouple producers from consumers. They absorb bursts of load. Workers drain them at their own pace."}'

fail() {
  echo "smoke: FAIL: $*" >&2
  exit 1
}

json_field() { # json_field <field> : read one top-level field from JSON on stdin
  python3 -c 'import json, sys; print(json.load(sys.stdin)[sys.argv[1]])' "$1"
}

submit() {
  curl -sS -X POST "$BASE/v1/tasks/summarize" \
    -H 'content-type: application/json' -H "idempotency-key: $KEY" \
    -d "$BODY" -D "$1"
}

health="$(curl -sS "$BASE/health")"
[[ "$(json_field status <<<"$health")" == "ok" ]] || fail "health is not ok: $health"

headers="$(mktemp)"
trap 'rm -f "$headers"' EXIT
accepted="$(submit "$headers")"
grep -q '^HTTP/[0-9.]* 202' "$headers" || fail "submission not accepted: $accepted"
task_id="$(json_field task_id <<<"$accepted")"
status_url="$(json_field status_url <<<"$accepted")"

status=PENDING
for _ in $(seq 1 100); do
  state="$(curl -sS "$BASE$status_url")"
  status="$(json_field status <<<"$state")"
  [[ "$status" == PENDING || "$status" == RUNNING ]] || break
  sleep 0.1
done
[[ "$status" == COMPLETED ]] || fail "task $task_id ended $status: $state"
python3 -c 'import json, sys; assert json.load(sys.stdin)["result"]["summary"]' <<<"$state" ||
  fail "completed task has no summary: $state"

replay="$(submit "$headers")"
grep -qi '^idempotent-replayed: true' "$headers" || fail "resend was not a replay"
[[ "$(json_field task_id <<<"$replay")" == "$task_id" ]] || fail "replay changed the task id"

worker_metrics="$(docker compose port worker 9100)"
curl -sS "http://$worker_metrics/metrics" |
  grep -q '^arq_task_latency_count{status="COMPLETED",task_type="summarization"} [1-9]' ||
  fail "worker metrics did not count the completed task"
curl -sS "$BASE/metrics" |
  grep -q '^edaw_api_submissions_total{outcome="replayed",task_type="summarization"} [1-9]' ||
  fail "producer metrics did not count the replay"

echo "smoke: OK (task $task_id completed and replayed)"
