"""Static checks on the CI workflow, plus the hermetic guarantee its test job relies on.

The workflow is only as strong as its weakest step, so these pin the triggers, the job
graph, every gate, and the supply chain: each action must come from an allowed owner
and be pinned to a full commit SHA, and no ``run`` step may interpolate attacker-
controlled event data.
"""

import os
import re
import socket
import stat
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml
from pytest_socket import SocketBlockedError

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
ALLOWED_OWNERS = {"actions", "astral-sh", "docker"}
_PINNED = re.compile(r"^(?P<repo>[\w.-]+/[\w.-]+)@(?P<sha>[0-9a-f]{40})$")
_TAG_COMMENT = re.compile(r"uses:\s*(?P<ref>\S+)\s+#\s*v\d+\.\d+\.\d+\s*$")


@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    document: dict[Any, Any] = yaml.safe_load(WORKFLOW.read_text())
    # YAML 1.1 reads the bare key `on` as the boolean True.
    document["on"] = document.pop(True)
    return document


def _jobs(workflow: dict[str, Any]) -> dict[str, dict[str, Any]]:
    jobs: dict[str, dict[str, Any]] = workflow["jobs"]
    return jobs


def _steps(workflow: dict[str, Any], job: str | None = None) -> list[dict[str, Any]]:
    names = [job] if job else list(_jobs(workflow))
    return [step for name in names for step in _jobs(workflow)[name]["steps"]]


def _runs(workflow: dict[str, Any], job: str) -> str:
    return "\n".join(step.get("run", "") for step in _steps(workflow, job))


class TestTriggersAndPolicy:
    def test_runs_on_push_and_pull_requests_to_main(self, workflow: dict[str, Any]) -> None:
        assert workflow["on"] == {
            "push": {"branches": ["main"]},
            "pull_request": {"branches": ["main"]},
        }

    def test_stale_runs_are_cancelled_except_on_main(self, workflow: dict[str, Any]) -> None:
        concurrency = workflow["concurrency"]
        assert "github.event.pull_request.number" in concurrency["group"]
        assert concurrency["cancel-in-progress"] == "${{ github.ref != 'refs/heads/main' }}"

    def test_token_is_read_only(self, workflow: dict[str, Any]) -> None:
        assert workflow["permissions"] == {"contents": "read"}
        assert all("permissions" not in job for job in _jobs(workflow).values())

    def test_every_job_is_bounded(self, workflow: dict[str, Any]) -> None:
        for name, job in _jobs(workflow).items():
            assert 0 < job["timeout-minutes"] <= 30, name
            assert job["runs-on"] == "ubuntu-24.04", name


class TestJobGraph:
    def test_jobs_and_dependencies(self, workflow: dict[str, Any]) -> None:
        jobs = _jobs(workflow)
        assert set(jobs) == {"audit", "quality", "test", "docker"}
        assert "needs" not in jobs["audit"]
        assert jobs["quality"]["needs"] == "audit"
        assert jobs["test"]["needs"] == "audit"
        assert jobs["docker"]["needs"] == ["quality", "test"]


class TestSupplyChain:
    def test_actions_are_sha_pinned_from_allowed_owners(self, workflow: dict[str, Any]) -> None:
        uses = [step["uses"] for step in _steps(workflow) if "uses" in step]
        assert uses
        for ref in uses:
            pinned = _PINNED.match(ref)
            assert pinned, f"not pinned to a commit SHA: {ref}"
            assert pinned["repo"].split("/")[0] in ALLOWED_OWNERS, ref

    def test_every_pin_names_its_release_tag(self) -> None:
        lines = [line for line in WORKFLOW.read_text().splitlines() if "uses:" in line]
        assert lines
        for line in lines:
            assert _TAG_COMMENT.search(line), f"missing '# vX.Y.Z' comment: {line.strip()}"

    def test_one_version_per_action(self, workflow: dict[str, Any]) -> None:
        pins: dict[str, set[str]] = {}
        for step in _steps(workflow):
            if "uses" in step:
                repo, sha = step["uses"].split("@")
                pins.setdefault(repo, set()).add(sha)
        assert all(len(shas) == 1 for shas in pins.values()), pins

    def test_checkouts_do_not_persist_credentials(self, workflow: dict[str, Any]) -> None:
        checkouts = [
            s for s in _steps(workflow) if s.get("uses", "").startswith("actions/checkout@")
        ]
        assert len(checkouts) == len(_jobs(workflow))
        assert all(s["with"]["persist-credentials"] is False for s in checkouts)

    def test_run_steps_never_interpolate_event_data(self, workflow: dict[str, Any]) -> None:
        for step in _steps(workflow):
            assert "${{ github.event" not in step.get("run", ""), step.get("name")


class TestGates:
    def test_audit_checks_full_history_of_the_real_head(self, workflow: dict[str, Any]) -> None:
        checkout = _steps(workflow, "audit")[0]["with"]
        assert checkout["fetch-depth"] == 0
        assert "github.event.pull_request.head.sha" in checkout["ref"]
        assert "scripts/audit-attribution.sh HEAD" in _runs(workflow, "audit")

    def test_quality_runs_every_static_gate(self, workflow: dict[str, Any]) -> None:
        runs = _runs(workflow, "quality")
        for gate in (
            "uv sync --frozen",
            "uv run ruff check",
            "uv run ruff format --check .",
            "uv run mypy --strict src tests",
            "uv run bandit -c pyproject.toml -r src",
        ):
            assert gate in runs, gate

    def test_test_job_enforces_branch_coverage(self, workflow: dict[str, Any]) -> None:
        runs = _runs(workflow, "test")
        for flag in (
            "--cov=src/event_driven_ai_workers",
            "--cov-branch",
            "--cov-fail-under=90",
            "--cov-report=xml",
            "--cov-report=term",
        ):
            assert flag in runs, flag

    @pytest.mark.parametrize("job", ["quality", "test"])
    def test_uv_cache_is_keyed_on_the_lockfile(self, workflow: dict[str, Any], job: str) -> None:
        [setup] = [s for s in _steps(workflow, job) if s.get("uses", "").startswith("astral-sh/")]
        assert setup["with"]["enable-cache"] is True
        assert setup["with"]["cache-dependency-glob"] == "uv.lock"
        assert setup["with"]["python-version"] == "${{ env.PYTHON_VERSION }}"

    def test_docker_job_builds_and_exercises_the_runtime_image(
        self, workflow: dict[str, Any]
    ) -> None:
        steps = _steps(workflow, "docker")
        [build] = [s for s in steps if s.get("uses", "").startswith("docker/build-push-action@")]
        assert build["with"]["target"] == "runtime"
        assert build["with"]["push"] is False
        runs = _runs(workflow, "docker")
        for check in (
            "--output type=local",  # .dockerignore context dump
            'test "$actual" = "README.md pyproject.toml src uv.lock"',
            '\'{{.Config.User}}\' "$IMAGE")" = "10001:10001"',
            "docker compose up -d --no-build --wait",
            "scripts/smoke-compose.sh",
            "docker compose down -v",
        ):
            assert check in runs, check

    def test_docker_job_runs_the_image_compose_expects(self, workflow: dict[str, Any]) -> None:
        compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
        assert _jobs(workflow)["docker"]["env"]["IMAGE"] == compose["x-app"]["image"]


class TestDrift:
    def test_toolchain_versions_match_the_repo(self, workflow: dict[str, Any]) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text()
        uv_version = re.search(r"^ARG UV_VERSION=(\S+)$", dockerfile, re.MULTILINE)
        python_version = re.search(r"^ARG PYTHON_VERSION=(\S+)$", dockerfile, re.MULTILINE)
        assert uv_version
        assert python_version
        assert workflow["env"]["UV_VERSION"] == uv_version[1]
        assert workflow["env"]["PYTHON_VERSION"] == python_version[1]
        assert (ROOT / ".python-version").read_text().strip() == python_version[1]

    @pytest.mark.parametrize("script", ["audit-attribution.sh", "smoke-compose.sh"])
    def test_referenced_scripts_exist_and_are_executable(self, script: str) -> None:
        mode = (ROOT / "scripts" / script).stat().st_mode
        assert mode & stat.S_IXUSR

    def test_bandit_is_configured_for_src(self) -> None:
        bandit = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["bandit"]
        assert bandit["targets"] == ["src"]


class TestHermeticity:
    def test_pytest_disables_inet_sockets_by_default(self) -> None:
        pytest_ini = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["pytest"]
        assert "--disable-socket" in pytest_ini["ini_options"]["addopts"]

    @pytest.mark.parametrize("family", [socket.AF_INET, socket.AF_INET6])
    def test_inet_sockets_cannot_be_opened(self, family: socket.AddressFamily) -> None:
        with pytest.raises(SocketBlockedError):
            socket.socket(family, socket.SOCK_STREAM)

    def test_no_live_credentials_leak_into_tests(self) -> None:
        assert not os.environ.get("EDAW_LLM_API_KEY")
