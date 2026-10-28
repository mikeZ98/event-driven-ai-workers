"""README stays true to the code: settings and defaults, metrics, problems, tests, links."""

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from prometheus_client import CollectorRegistry
from pydantic import SecretStr
from pydantic_settings import BaseSettings

from event_driven_ai_workers.api import ApiMetrics, ApiSettings
from event_driven_ai_workers.core import RedisSettings, WorkerSettings
from event_driven_ai_workers.guardrails import GuardrailSettings
from event_driven_ai_workers.tools import LLMSettings, WorkerMetrics
from event_driven_ai_workers.worker_main import MetricsServerSettings

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text()
SETTINGS: tuple[type[BaseSettings], ...] = (
    ApiSettings,
    GuardrailSettings,
    WorkerSettings,
    RedisSettings,
    LLMSettings,
    MetricsServerSettings,
)
_ROW = re.compile(r"^\| `(?P<name>EDAW_[A-Z0-9_]+)` \| (?P<default>[^|]+?) \|", re.MULTILINE)
_PROBLEM = re.compile(r'ProblemError\(\s*\d{3},\s*"(?P<kind>[a-z]+(?:-[a-z]+)+)"')


def _documented() -> dict[str, str]:
    return {m["name"]: m["default"].strip().strip("`") for m in _ROW.finditer(README)}


def _settings_defaults() -> dict[str, Any]:
    return {
        str(cls.model_config["env_prefix"]) + name.upper(): field.default
        for cls in SETTINGS
        for name, field in cls.model_fields.items()
    }


def _same(documented: str, default: Any) -> bool:
    if default is None:
        return documented == "—"
    if isinstance(default, SecretStr):
        default = default.get_secret_value()
    if isinstance(default, bool):
        return documented == str(default).lower()
    if isinstance(default, tuple):
        return documented == json.dumps(list(default))
    if isinstance(default, int | float):
        return float(documented) == float(default)
    return documented == str(default)


def _slug(heading: str) -> str:
    """GitHub's anchor for a heading."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


class TestConfigurationReference:
    def test_every_setting_is_documented_with_its_default(self) -> None:
        documented = _documented()
        for name, default in _settings_defaults().items():
            assert name in documented, f"{name} missing from the configuration reference"
            assert _same(documented[name], default), (name, documented[name], default)

    def test_documents_no_unknown_variables(self) -> None:
        compose = (ROOT / "docker-compose.yml").read_text()
        compose_only = set(_documented()) - set(_settings_defaults())
        assert compose_only == {"EDAW_BIND", "EDAW_REDIS_MAXMEMORY"}
        assert all(f"${{{name}:-" in compose for name in compose_only)

    def test_compose_only_defaults_match_compose(self) -> None:
        compose = (ROOT / "docker-compose.yml").read_text()
        for name in ("EDAW_BIND", "EDAW_REDIS_MAXMEMORY"):
            assert f"${{{name}:-{_documented()[name]}}}" in compose, name

    def test_every_variable_mentioned_anywhere_exists(self) -> None:
        mentioned = set(re.findall(r"\bEDAW_[A-Z0-9_]*[A-Z0-9]\b", README))
        assert mentioned <= set(_documented()), mentioned - set(_documented())


class TestReferenceTables:
    def test_every_metric_family_has_a_table_row(self) -> None:
        registry = CollectorRegistry()
        WorkerMetrics(registry)
        ApiMetrics(registry)
        families = {m.name for m in registry.collect()}
        assert families
        rows = set(re.findall(r"^\| `([a-z_]+)` \|", README, re.MULTILINE))
        for family in families:  # counters are exposed with a _total suffix
            assert family in rows or f"{family}_total" in rows, family

    def test_every_problem_kind_is_documented(self) -> None:
        source = "".join(p.read_text() for p in (ROOT / "src").rglob("*.py"))
        kinds = {m["kind"] for m in _PROBLEM.finditer(source)} | {"queue-unavailable"}
        undocumented = {k for k in kinds if f"`{k}`" not in README} - {
            "http-error",
            "internal-error",
        }
        assert len(kinds) >= 10
        assert not undocumented, undocumented

    def test_every_test_module_is_listed(self) -> None:
        for module in (ROOT / "tests").glob("test_*.py"):
            assert f"`{module.name}`" in README, module.name


class TestStructure:
    @pytest.mark.parametrize(
        "section",
        [
            "Architecture",
            "Core invariants",
            "Prometheus metrics",
            "Local development and testing",
            "Docker",
            "Configuration reference",
        ],
    )
    def test_required_sections_exist(self, section: str) -> None:
        assert f"\n## {section}\n" in README

    def test_contents_links_resolve(self) -> None:
        anchors = {_slug(h) for h in re.findall(r"^#{2,3} (.+)$", README, re.MULTILINE)}
        links = re.findall(r"\]\(#([\w-]+)\)", README)
        assert links
        assert set(links) <= anchors, set(links) - anchors

    def test_documented_commands_match_the_gates(self) -> None:
        workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
        ci_runs = "\n".join(
            step.get("run", "") for job in workflow["jobs"].values() for step in job["steps"]
        )
        for gate in (
            "uv run mypy --strict src tests",
            "uv run bandit -c pyproject.toml -r src",
            "uv run ruff format --check .",
        ):
            assert gate in README
            assert gate in ci_runs

    def test_invariant_numbers_match_the_code(self) -> None:
        worker = WorkerSettings()
        attempts = worker.dlq_threshold + 1
        assert f"**{attempts} attempts**" in README
        assert f"`EDAW_WORKER_DLQ_THRESHOLD`\n= {worker.dlq_threshold}" in README
