"""Git identity policy: sole contributor, no AI attribution, enforced by .githooks.

Each test drives the real hooks and audit script inside a throwaway repository with
an isolated git configuration, so the developer's own settings never leak in.
"""

import os
import re
import shutil
import subprocess
import tomllib
from collections.abc import Mapping
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_NAME = "mikeZ98"
ALLOWED_EMAIL = "145975972+mikeZ98@users.noreply.github.com"
HOOKS = ("pre-commit", "commit-msg", "pre-push")
AUDIT = "scripts/audit-attribution.sh"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _run(
    cwd: Path, *args: str, env: Mapping[str, str] | None = None, stdin: str | None = None
) -> subprocess.CompletedProcess[str]:
    base = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    base |= {
        "HOME": str(cwd.parent),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    }
    return subprocess.run(
        args,
        cwd=cwd,
        env=base | dict(env or {}),
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )


def _git(cwd: Path, *args: str, env: Mapping[str, str] | None = None) -> str:
    proc = _run(cwd, "git", *args, env=env)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def _commit(repo: Path, message: str, *extra: str, env: Mapping[str, str] | None = None) -> bool:
    proc = _run(repo, "git", "commit", "--allow-empty", "-q", "-m", message, *extra, env=env)
    return proc.returncode == 0


def _count(repo: Path) -> int:
    proc = _run(repo, "git", "rev-list", "--count", "HEAD")
    return int(proc.stdout) if proc.returncode == 0 else 0


def _install_policy(path: Path) -> None:
    """Copy this project's hooks and audit script into ``path`` (left untracked)."""
    shutil.copytree(ROOT / ".githooks", path / ".githooks")
    (path / "scripts").mkdir()
    shutil.copy2(ROOT / AUDIT, path / AUDIT)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A fresh repository with this project's hooks, audit script and allowed identity."""
    path = tmp_path / "repo"
    path.mkdir()
    _install_policy(path)
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "core.hooksPath", ".githooks")
    _git(path, "config", "user.name", ALLOWED_NAME)
    _git(path, "config", "user.email", ALLOWED_EMAIL)
    _git(path, "config", "user.useConfigOnly", "true")
    return path


class TestPolicySource:
    def test_policy_identity_matches_package_metadata(self) -> None:
        policy = (ROOT / ".githooks" / "_policy.sh").read_text()
        assert f'POLICY_ALLOWED_NAME="{ALLOWED_NAME}"' in policy
        assert f'POLICY_ALLOWED_EMAIL="{ALLOWED_EMAIL}"' in policy
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        assert project["authors"] == [{"name": ALLOWED_NAME, "email": ALLOWED_EMAIL}]

    @pytest.mark.parametrize("path", [*(f".githooks/{h}" for h in HOOKS), AUDIT])
    def test_hooks_and_audit_are_executable(self, path: str) -> None:
        assert os.access(ROOT / path, os.X_OK)

    def test_policy_library_is_not_a_hook_name(self) -> None:
        assert (ROOT / ".githooks" / "_policy.sh").is_file()
        assert not os.access(ROOT / ".githooks" / "_policy.sh", os.X_OK)

    def test_project_history_passes_audit(self) -> None:
        if _run(ROOT, "git", "rev-parse", "--verify", "HEAD").returncode != 0:
            pytest.skip("project has no commits yet")
        proc = _run(ROOT, "bash", AUDIT)
        assert proc.returncode == 0, proc.stderr


class TestCommitHooks:
    def test_clean_commit_by_sole_contributor_is_accepted(self, repo: Path) -> None:
        assert _commit(repo, "feat(schemas): define strict contracts")
        assert _git(repo, "log", "-1", "--format=%an <%ae>|%cn <%ce>") == (
            f"{ALLOWED_NAME} <{ALLOWED_EMAIL}>|{ALLOWED_NAME} <{ALLOWED_EMAIL}>"
        )

    @pytest.mark.parametrize(
        "message",
        [
            "feat: x\n\nCo-Authored-By: Someone <someone@example.com>",
            "feat: x\n\nco-authored-by: mikeZ98 <145975972+mikeZ98@users.noreply.github.com>",
            "feat: x\n\nGenerated-by: some-tool",
            "feat: x\n\nGenerated with an assistant",
            "feat: x\n\nAssisted-by: bot",
            "feat: use claude for summaries",
            "fix: Anthropic rate limit handling",
            "chore: drafted with ChatGPT",
            "chore: openai client",
            "docs: written by gpt-4",
            "feat: \N{ROBOT FACE} add worker",
            "feat: x\n\nSigned-off-by: Other Person <other@example.com>",
            "feat: x\n\nReviewed-by: Other Person <other@example.com>",
        ],
    )
    def test_ai_attribution_and_foreign_trailers_are_rejected(
        self, repo: Path, message: str
    ) -> None:
        assert not _commit(repo, message)
        assert _count(repo) == 0

    @pytest.mark.parametrize(
        "message",
        [
            f"feat: x\n\nSigned-off-by: {ALLOWED_NAME} <{ALLOWED_EMAIL}>",
            "docs(agents): reference CLAUDE.md conventions",
            "feat(api)!: drop v0 endpoints",
            "fixup! feat: x",
        ],
    )
    def test_allowed_messages(self, repo: Path, message: str) -> None:
        assert _commit(repo, message)

    @pytest.mark.parametrize(
        "subject", ["Add schemas", "feat:missing space", "feature: x", "feat(UP): x", "f" * 120]
    )
    def test_non_conventional_subject_is_rejected(self, repo: Path, subject: str) -> None:
        assert not _commit(repo, subject)

    def test_foreign_author_is_rejected(self, repo: Path) -> None:
        assert not _commit(repo, "feat: x", "--author=Someone <someone@example.com>")
        assert _count(repo) == 0

    @pytest.mark.parametrize(
        "env",
        [
            {"GIT_COMMITTER_NAME": "ci-bot"},
            {"GIT_COMMITTER_EMAIL": "mike@MacBook-Pro.local"},
            {"GIT_AUTHOR_EMAIL": "m.someone@gmail.com"},
        ],
    )
    def test_foreign_or_machine_identity_is_rejected(self, repo: Path, env: dict[str, str]) -> None:
        assert not _commit(repo, "feat: x", env=env)
        assert _count(repo) == 0

    def test_amend_cannot_smuggle_in_a_foreign_author(self, repo: Path) -> None:
        assert _commit(repo, "feat: x")
        assert not _commit(repo, "feat: x", "--amend", "--author=Bot <bot@example.com>")
        assert _git(repo, "log", "-1", "--format=%ae") == ALLOWED_EMAIL

    @pytest.mark.parametrize(
        "path", [".env", "config/.env.prod", ".cache/venv/x", "pkg/__pycache__/m.pyc", "id.pem"]
    )
    def test_pre_commit_blocks_cache_and_secret_files(self, repo: Path, path: str) -> None:
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("secret")
        _git(repo, "add", "-f", path)
        assert not _commit(repo, "chore: add file")

    def test_pre_commit_allows_env_example(self, repo: Path) -> None:
        (repo / ".env.example").write_text("REDIS_URL=redis://localhost:6379/0\n")
        _git(repo, "add", ".env.example")
        assert _commit(repo, "chore: add env template")


class TestAuditScript:
    def _bypass(self, repo: Path, message: str, *extra: str) -> None:
        """Simulate a commit that skipped local hooks (``--no-verify``)."""
        assert _commit(repo, message, "--no-verify", *extra)

    def test_empty_repository_passes(self, repo: Path) -> None:
        proc = _run(repo, "bash", AUDIT)
        assert proc.returncode == 0
        assert "no commits yet" in proc.stderr

    def test_clean_history_passes(self, repo: Path) -> None:
        for i in range(3):
            assert _commit(repo, f"feat: step {i}")
        proc = _run(repo, "bash", AUDIT)
        assert proc.returncode == 0
        assert "3 commit(s) scanned" in proc.stderr

    @pytest.mark.parametrize(
        ("message", "extra"),
        [
            ("feat: x\n\nCo-Authored-By: Claude <noreply@anthropic.com>", ()),
            ("feat: x", ("--author=Someone <someone@example.com>",)),
        ],
    )
    def test_bypassed_violation_is_caught(
        self, repo: Path, message: str, extra: tuple[str, ...]
    ) -> None:
        assert _commit(repo, "feat: clean base")
        self._bypass(repo, message, *extra)
        assert _commit(repo, "feat: clean tip")
        proc = _run(repo, "bash", AUDIT)
        assert proc.returncode == 1
        assert "Attribution audit FAILED (3 commit(s) scanned)" in proc.stderr

    def test_range_limits_the_audit(self, repo: Path) -> None:
        self._bypass(repo, "feat: x", "--author=Old <old@example.com>")
        base = _git(repo, "rev-parse", "HEAD")
        assert _commit(repo, "feat: new work")
        proc = _run(repo, "bash", AUDIT, f"{base}..HEAD")
        assert proc.returncode == 0
        assert "1 commit(s) scanned" in proc.stderr

    def test_unfetched_range_fails_loudly(self, repo: Path) -> None:
        assert _commit(repo, "feat: x")
        proc = _run(repo, "bash", AUDIT, "0123456789abcdef0123456789abcdef01234567..HEAD")
        assert proc.returncode == 2
        assert "is the range fetched" in proc.stderr

    def test_shallow_clone_is_audited_with_warning(self, repo: Path, tmp_path: Path) -> None:
        self._bypass(repo, "feat: x", "--author=Old <old@example.com>")
        assert _commit(repo, "feat: y")
        clone = tmp_path / "shallow"
        _git(tmp_path, "clone", "-q", "--depth", "1", f"file://{repo}", str(clone))
        _install_policy(clone)
        proc = _run(clone, "bash", AUDIT)
        assert proc.returncode == 0
        assert "Shallow clone" in proc.stderr
        assert "1 commit(s) scanned" in proc.stderr


class TestPrePush:
    @pytest.fixture
    def remote(self, tmp_path: Path, repo: Path) -> Path:
        path = tmp_path / "remote.git"
        _git(tmp_path, "init", "-q", "--bare", str(path))
        _git(repo, "remote", "add", "origin", str(path))
        return path

    def test_clean_push_is_accepted(self, repo: Path, remote: Path) -> None:
        assert _commit(repo, "feat: x")
        proc = _run(repo, "git", "push", "-q", "origin", "main")
        assert proc.returncode == 0, proc.stderr
        assert _git(remote, "rev-parse", "main") == _git(repo, "rev-parse", "HEAD")

    def test_push_with_bypassed_violation_is_rejected(self, repo: Path, remote: Path) -> None:
        assert _commit(repo, "feat: x")
        assert _run(repo, "git", "push", "-q", "origin", "main").returncode == 0
        assert _commit(repo, "feat: y\n\nGenerated-by: tool", "--no-verify")
        proc = _run(repo, "git", "push", "-q", "origin", "main")
        assert proc.returncode != 0
        assert re.search(r"forbidden AI attribution", proc.stderr)
        assert _git(remote, "rev-list", "--count", "main") == "1"

    @pytest.mark.usefixtures("remote")
    def test_branch_deletion_is_not_audited(self, repo: Path) -> None:
        assert _commit(repo, "feat: x")
        _git(repo, "push", "-q", "origin", "main", "main:topic")
        proc = _run(repo, "git", "push", "-q", "origin", ":topic")
        assert proc.returncode == 0, proc.stderr
