from pathlib import Path

import pytest

from goagentx.config.settings import SettingsError, load_settings


def test_default_settings_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOAGENTX_DATABASE_PATH", raising=False)

    settings = load_settings()

    assert settings.database.path == Path("data/goagentx.db")
    assert settings.arena.quick_reject_rounds == 5
    assert settings.scoring.weights.quality == 0.70
    assert settings.scoring.normalization.max_latency_ms == 10000
    assert settings.promotion_gate.require_no_safety_violation is True


def test_missing_required_field_reports_field_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOAGENTX_DATABASE_PATH", raising=False)
    _write_config_set(tmp_path, database_yaml="database: {}\n")

    with pytest.raises(SettingsError) as exc_info:
        load_settings(tmp_path)

    message = str(exc_info.value)
    assert "database.path" in message
    assert "Field required" in message


def test_database_path_environment_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config_set(tmp_path)
    monkeypatch.setenv("GOAGENTX_DATABASE_PATH", "custom/goagentx.db")

    settings = load_settings(tmp_path)

    assert settings.database.path == Path("custom/goagentx.db")


def _write_config_set(tmp_path: Path, database_yaml: str | None = None) -> None:
    goagentx_yaml = database_yaml or "database:\n  path: data/test.db\n"
    (tmp_path / "goagentx.yaml").write_text(
        goagentx_yaml
        + """
reports:
  directory: reports
evolution:
  degradation_window: 50
  baseline_window: 300
  degradation_threshold: 0.15
arena:
  quick_reject_rounds: 5
  full_eval_rounds: 50
  min_win_rate: 0.55
  p_value_threshold: 0.05
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "scoring.yaml").write_text(
        """
weights:
  quality: 0.70
  cost: 0.10
  latency: 0.10
  safety: 0.10
normalization:
  max_cost: 1.0
  max_latency_ms: 10000
safety_penalty: 1.0
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "promotion_gate.yaml").write_text(
        """
min_win_rate: 0.55
max_p_value: 0.05
min_score_delta: 0.0
max_cost_delta: 0.20
max_latency_delta: 0.20
require_no_safety_violation: true
require_no_critical_bucket_regression: true
""".lstrip(),
        encoding="utf-8",
    )
