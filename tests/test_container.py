"""Static checks on the container config: Dockerfile, compose stack and build context.

No daemon needed. These pin the hardening and catch the silent failures config drift
causes: an ``EDAW_*`` typo is just an ignored environment variable, and a producer
and worker that disagree on the queue topology lose tasks without an error.
"""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic_settings import BaseSettings

from event_driven_ai_workers.api import ApiSettings
from event_driven_ai_workers.core import RedisSettings, WorkerSettings
from event_driven_ai_workers.guardrails import GuardrailSettings
from event_driven_ai_workers.tools import LLMSettings
from event_driven_ai_workers.worker_main import MetricsServerSettings

ROOT = Path(__file__).resolve().parent.parent
SETTINGS: tuple[type[BaseSettings], ...] = (
    ApiSettings,
    WorkerSettings,
    RedisSettings,
    GuardrailSettings,
    LLMSettings,
    MetricsServerSettings,
)
APP_SERVICES = ("api", "worker")
_DEFAULTED = re.compile(r"^\$\{(?P<var>[A-Z0-9_]+):-(?P<default>.*)\}$")


def _env_names() -> set[str]:
    return {
        str(cls.model_config["env_prefix"]) + field.upper()
        for cls in SETTINGS
        for field in cls.model_fields
    }


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    document: dict[str, Any] = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    return document


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return (ROOT / "Dockerfile").read_text()


def _service(compose: dict[str, Any], name: str) -> dict[str, Any]:
    service: dict[str, Any] = compose["services"][name]
    return service


def _resolved(environment: dict[str, Any]) -> dict[str, str]:
    """``${VAR:-default}`` -> ``default``: the values an empty shell and .env produce."""
    resolved: dict[str, str] = {}
    for key, value in environment.items():
        match = _DEFAULTED.match(str(value))
        resolved[key] = match["default"] if match else str(value)
    return resolved


class TestDockerfile:
    def test_runtime_is_non_root_with_fixed_ids(self, dockerfile: str) -> None:
        runtime = dockerfile.split("AS runtime", 1)[1]
        assert "USER 10001:10001" in runtime
        assert "--uid 10001 --gid 10001" in runtime
        assert "--gid 10001 edaw" in runtime

    def test_runtime_ships_no_installers_or_fetch_tools(self, dockerfile: str) -> None:
        runtime = dockerfile.split("AS runtime", 1)[1]
        assert "rm -rf /usr/local/bin/pip*" in runtime
        assert "uv" not in re.findall(r"COPY --from=(\S+)", runtime)
        for tool in ("apt-get install", "curl", "wget", "gcc", "build-essential"):
            assert tool not in runtime

    def test_builder_installs_production_dependencies_only(self, dockerfile: str) -> None:
        syncs = re.findall(r"uv sync ([^\n]+)", dockerfile)
        assert len(syncs) == 2
        assert all("--frozen" in s and "--no-dev" in s for s in syncs)
        assert "UV_COMPILE_BYTECODE=1" in dockerfile

    def test_one_entrypoint_serves_both_roles(self, dockerfile: str) -> None:
        assert 'ENTRYPOINT ["edaw"]' in dockerfile
        assert 'CMD ["api"]' in dockerfile
        assert "EXPOSE 8000 9100" in dockerfile

    def test_environment_names_are_real_settings(self, dockerfile: str) -> None:
        names = set(re.findall(r"\b(EDAW_[A-Z0-9_]+)=", dockerfile))
        assert names
        assert names <= _env_names()


class TestCompose:
    @pytest.mark.parametrize("name", [*APP_SERVICES, "redis"])
    def test_services_are_hardened(self, compose: dict[str, Any], name: str) -> None:
        service = _service(compose, name)
        assert service["read_only"] is True
        assert service["cap_drop"] == ["ALL"]
        assert "no-new-privileges:true" in service["security_opt"]
        assert not str(service["user"]).startswith("0")
        assert service["deploy"]["resources"]["limits"]["memory"]
        assert service["healthcheck"]["test"]

    @pytest.mark.parametrize("name", APP_SERVICES)
    def test_app_services_run_the_image_user_and_wait_for_redis(
        self, compose: dict[str, Any], name: str
    ) -> None:
        service = _service(compose, name)
        assert service["user"] == "10001:10001"
        assert service["command"] == [name]
        assert service["depends_on"]["redis"]["condition"] == "service_healthy"
        assert any(mount.startswith("/tmp:") for mount in service["tmpfs"])  # noqa: S108 — a tmpfs mount spec

    @pytest.mark.parametrize("name", APP_SERVICES)
    def test_environment_names_are_real_settings(self, compose: dict[str, Any], name: str) -> None:
        assert set(_service(compose, name)["environment"]) <= _env_names()

    @pytest.mark.parametrize("name", APP_SERVICES)
    def test_defaults_validate(
        self, compose: dict[str, Any], name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for key, value in _resolved(_service(compose, name)["environment"]).items():
            monkeypatch.setenv(key, value)
        for cls in SETTINGS:
            cls()

    def test_producer_and_worker_share_the_queue_topology(self, compose: dict[str, Any]) -> None:
        shared = compose["x-queue-env"]
        assert {
            "EDAW_REDIS_URL",
            "EDAW_WORKER_QUEUE_NAME",
            "EDAW_WORKER_DLQ_NAME",
            "EDAW_WORKER_STATE_KEY_PREFIX",
            "EDAW_WORKER_STATE_TTL_S",
        } <= set(shared)
        api, worker = (_service(compose, n)["environment"] for n in APP_SERVICES)
        for key, value in shared.items():
            assert api[key] == worker[key] == value, key

    def test_llm_secret_reaches_the_worker_only(self, compose: dict[str, Any]) -> None:
        assert "env_file" not in _service(compose, "api")
        assert not any("LLM" in key for key in _service(compose, "api")["environment"])
        assert _service(compose, "worker")["env_file"] == [{"path": ".env", "required": False}]

    def test_redis_never_evicts_and_stays_internal(self, compose: dict[str, Any]) -> None:
        redis = _service(compose, "redis")
        command = redis["command"]
        assert command[command.index("--maxmemory-policy") + 1] == "noeviction"
        assert command[command.index("--appendonly") + 1] == "yes"
        assert "ports" not in redis
        assert redis["networks"] == ["backend"]
        assert compose["networks"]["backend"]["internal"] is True

    def test_published_ports_default_to_loopback(self, compose: dict[str, Any]) -> None:
        for name in APP_SERVICES:
            for port in _service(compose, name)["ports"]:
                assert port.startswith("${EDAW_BIND:-127.0.0.1}:")


class TestBuildContext:
    def test_dockerignore_is_default_deny(self) -> None:
        lines = [
            line.strip()
            for line in (ROOT / ".dockerignore").read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        assert lines[0] == "*"
        allowed = {line[1:] for line in lines if line.startswith("!")}
        assert allowed == {"pyproject.toml", "uv.lock", "README.md", "src"}

    def test_env_example_assigns_nothing(self) -> None:
        lines = (ROOT / ".env.example").read_text().splitlines()
        assert all(not line.strip() or line.startswith("#") for line in lines)
        assert any("EDAW_LLM_API_KEY=" in line for line in lines)
