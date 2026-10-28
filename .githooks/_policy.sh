# shellcheck shell=bash
# Shared attribution policy for git hooks and scripts/audit-attribution.sh.
# Sourced, never executed. Git ignores this file as a hook (not a hook name).

readonly POLICY_ALLOWED_NAME="mikeZ98"
readonly POLICY_ALLOWED_EMAIL="145975972+mikeZ98@users.noreply.github.com"

# Case-insensitive ERE matched against commit messages (after stripping
# comments and the literal file name CLAUDE.md, which docs commits may cite).
readonly POLICY_FORBIDDEN_MSG_RE='co-authored-by|generated-by|generated[ -]with|ai-generated|assisted-by|claude|anthropic|chatgpt|openai|gpt-[0-9]|copilot|gemini|cursor\.(sh|com)|🤖'

# Any "<Something>-by:" trailer (Signed-off-by, Reviewed-by, ...) names a person;
# only the sole contributor may appear in one.
readonly POLICY_TRAILER_RE='^[[:space:]]*[A-Za-z][A-Za-z-]*-by:'

readonly POLICY_CONVENTIONAL_RE='^(feat|fix|chore|docs|test|build|ci|refactor|perf|style|revert)(\([a-z0-9._/-]+\))?!?: [^ ].{0,98}$'

policy_err() { printf '\033[31m✖ %s\033[0m\n' "$*" >&2; }
policy_warn() { printf '\033[33m! %s\033[0m\n' "$*" >&2; }
policy_ok() { printf '\033[32m✔ %s\033[0m\n' "$*" >&2; }

# policy_check_identity <role> <name> <email>
policy_check_identity() {
  local role="$1" name="$2" email="$3"
  if [[ "$name" != "$POLICY_ALLOWED_NAME" || "$email" != "$POLICY_ALLOWED_EMAIL" ]]; then
    policy_err "$role identity '$name <$email>' violates the sole-contributor invariant."
    policy_err "Expected: '$POLICY_ALLOWED_NAME <$POLICY_ALLOWED_EMAIL>'."
    return 1
  fi
}

# policy_parse_ident <git-var-ident>  →  prints "name<TAB>email"
policy_parse_ident() {
  sed -E 's/^(.*) <([^>]*)> [0-9]+ [-+][0-9]{4}$/\1\t\2/' <<<"$1"
}

# policy_check_current_identity — validate the author/committer git would record now.
policy_check_current_identity() {
  local a_name a_email c_name c_email fail=0
  IFS=$'\t' read -r a_name a_email <<<"$(policy_parse_ident "$(git var GIT_AUTHOR_IDENT)")"
  IFS=$'\t' read -r c_name c_email <<<"$(policy_parse_ident "$(git var GIT_COMMITTER_IDENT)")"
  policy_check_identity "Author" "$a_name" "$a_email" || fail=1
  policy_check_identity "Committer" "$c_name" "$c_email" || fail=1
  return "$fail"
}

# policy_clean_message — strip comments and anything below the scissors line.
policy_clean_message() {
  sed -e '/^# ------------------------ >8 ------------------------$/,$d' -e '/^#/d'
}

# policy_check_message <message-text>
policy_check_message() {
  local msg="$1" hits trailers fail=0
  hits="$(sed -E 's/CLAUDE\.md//g' <<<"$msg" | grep -inE "$POLICY_FORBIDDEN_MSG_RE" || true)"
  if [[ -n "$hits" ]]; then
    policy_err "Commit message contains forbidden AI attribution / co-author metadata:"
    printf '    %s\n' "$hits" >&2
    fail=1
  fi
  trailers="$(grep -inE "$POLICY_TRAILER_RE" <<<"$msg" |
    grep -vF "$POLICY_ALLOWED_NAME <$POLICY_ALLOWED_EMAIL>" || true)"
  if [[ -n "$trailers" ]]; then
    policy_err "Commit message names a contributor other than '$POLICY_ALLOWED_NAME':"
    printf '    %s\n' "$trailers" >&2
    fail=1
  fi
  return "$fail"
}

# policy_check_subject <first-line>
policy_check_subject() {
  local subject="$1"
  case "$subject" in
    Merge\ *|fixup!\ *|squash!\ *|amend!\ *) return 0 ;;
  esac
  if ! grep -qE "$POLICY_CONVENTIONAL_RE" <<<"$subject"; then
    policy_err "Subject is not a Conventional Commit (max 100 chars): '$subject'"
    policy_err "Format: <type>(<scope>)?: <summary>   types: feat fix chore docs test build ci refactor perf style revert"
    return 1
  fi
}
