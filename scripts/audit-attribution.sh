#!/usr/bin/env bash
# Audit commits for the sole-contributor + no-AI-attribution invariant.
#
# Usage: scripts/audit-attribution.sh [<git rev-list args>...]   (default: HEAD)
# Used by .githooks/pre-push locally and by CI.
#
# Fetch-depth safe: a shallow clone (actions/checkout default fetch-depth: 1) is
# audited down to its shallow boundary with a warning, an empty repository passes
# with zero commits, and a range whose endpoints were not fetched fails loudly
# instead of silently auditing nothing. Use fetch-depth: 0 for a full-history audit.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
# shellcheck source=../.githooks/_policy.sh
source "$ROOT/.githooks/_policy.sh"

if (($# == 0)); then
  if ! git rev-parse --verify --quiet HEAD >/dev/null; then
    policy_ok "Attribution audit passed (repository has no commits yet)."
    exit 0
  fi
  set -- HEAD
fi

if [[ "$(git rev-parse --is-shallow-repository)" == "true" ]]; then
  policy_warn "Shallow clone: auditing fetched commits only (use fetch-depth: 0 for full history)."
fi

# Capture first: a failing rev-list inside a process substitution would be ignored.
if ! shas="$(git rev-list "$@")"; then
  policy_err "git rev-list $* failed — is the range fetched? (shallow clone: raise fetch-depth)"
  exit 2
fi

fail=0
count=0
while IFS= read -r sha; do
  [[ -n "$sha" ]] || continue
  count=$((count + 1))
  IFS=$'\x1f' read -r an ae cn ce <<<"$(git log -1 --format='%an%x1f%ae%x1f%cn%x1f%ce' "$sha")"
  short="${sha:0:12}"
  policy_check_identity "[$short] Author" "$an" "$ae" || fail=1
  policy_check_identity "[$short] Committer" "$cn" "$ce" || fail=1
  policy_check_message "$(git log -1 --format=%B "$sha")" || { policy_err "  ↳ in commit $short"; fail=1; }
done <<<"$shas"

if ((fail)); then
  policy_err "Attribution audit FAILED ($count commit(s) scanned)."
  exit 1
fi
policy_ok "Attribution audit passed ($count commit(s) scanned)."
