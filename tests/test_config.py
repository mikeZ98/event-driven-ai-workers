"""Settings: defaults, environment overrides, bounds and arq translation."""

import pytest
from pydantic import ValidationError

from event_driven_ai_workers.core import RedisSettings, WorkerSettings
from event_driven_ai_workers.core.config import CANCELLATION_HEADROOM, JOB_TIMEOUT_GRACE_S


class TestWorkerSettings:
    def test_blueprint_defaults(self) -> None:
        settings = WorkerSettings()
        assert settings.dlq_threshold == 5
        assert settings.queue_name != settings.dlq_name
        assert settings.backoff_base_s <= settings.backoff_max_s

    def test_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EDAW_WORKER_DLQ_THRESHOLD", "3")
        monkeypatch.setenv("EDAW_WORKER_MAX_JOBS", "50")
        monkeypatch.setenv("EDAW_WORKER_QUEUE_NAME", "jobs")
        settings = WorkerSettings()
        assert (settings.dlq_threshold, settings.max_jobs, settings.queue_name) == (3, 50, "jobs")

    def test_arq_limits_leave_room_for_the_dead_letter_attempt(self) -> None:
        settings = WorkerSettings(dlq_threshold=5, task_timeout_s=30)
        assert settings.arq_max_tries == 1 + 5 + CANCELLATION_HEADROOM
        assert settings.arq_job_timeout_s == 30 + JOB_TIMEOUT_GRACE_S

    @pytest.mark.parametrize(
        "overrides",
        [
            {"backoff_base_s": 10, "backoff_max_s": 5},
            {"dlq_name": "edaw:queue"},
            {"state_key_prefix": "edaw:dlq"},
            {"dlq_threshold": -1},
            {"dlq_threshold": 21},
            {"max_jobs": 0},
            {"task_timeout_s": 0},
            {"backoff_multiplier": 0.5},
            {"unknown": 1},
        ],
    )
    def test_rejects_invalid(self, overrides: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            WorkerSettings.model_validate(overrides)

    def test_frozen(self) -> None:
        with pytest.raises(ValidationError):
            WorkerSettings().max_jobs = 1  # type: ignore[misc]


class TestRedisSettings:
    def test_dsn_and_pool_limits_reach_arq(self) -> None:
        settings = RedisSettings.model_validate(
            {"url": "rediss://:s3cret@cache.internal:6380/2", "max_connections": 64}
        )
        arq_settings = settings.to_arq()
        assert (arq_settings.host, arq_settings.port, arq_settings.database) == (
            "cache.internal",
            6380,
            2,
        )
        assert arq_settings.password == "s3cret"
        assert arq_settings.ssl is True
        assert arq_settings.max_connections == 64
        assert arq_settings.retry_on_timeout is True
        assert arq_settings.conn_timeout == settings.conn_timeout_s

    def test_password_is_not_leaked_by_repr(self) -> None:
        settings = RedisSettings.model_validate({"url": "redis://:s3cret@localhost:6379/0"})
        assert "s3cret" not in repr(settings)

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EDAW_REDIS_URL", "redis://redis:6379/1")
        assert RedisSettings().to_arq().host == "redis"

    @pytest.mark.parametrize("url", ["http://localhost:6379", "localhost:6379", "redis://h:x/0"])
    def test_rejects_non_redis_dsn(self, url: str) -> None:
        with pytest.raises(ValidationError):
            RedisSettings.model_validate({"url": url})
