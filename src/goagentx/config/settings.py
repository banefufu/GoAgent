"""Typed YAML configuration loading for GoAgentX."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

DEFAULT_CONFIG_DIR = Path("configs")
DATABASE_PATH_ENV = "GOAGENTX_DATABASE_PATH"


class SettingsError(RuntimeError):
    """Raised when GoAgentX settings cannot be loaded or validated."""


class StrictModel(BaseModel):
    """Base model that rejects unknown configuration keys."""

    model_config = ConfigDict(extra="forbid")


class DatabaseSettings(StrictModel):
    """SQLite database configuration."""

    path: Path = Field(..., description="Path to the local SQLite database.")


class ReportsSettings(StrictModel):
    """Filesystem locations for generated audit reports."""

    directory: Path = Field(..., description="Directory for generated reports.")


class EvolutionSettings(StrictModel):
    """Settings for degradation detection and strategy evolution."""

    degradation_window: int = Field(..., gt=0)
    baseline_window: int = Field(..., gt=0)
    degradation_threshold: float = Field(..., ge=0.0, le=1.0)


class ArenaSettings(StrictModel):
    """Settings for quick reject and full evaluation."""

    quick_reject_rounds: int = Field(..., gt=0)
    full_eval_rounds: int = Field(..., gt=0)
    min_win_rate: float = Field(..., ge=0.0, le=1.0)
    p_value_threshold: float = Field(..., ge=0.0, le=1.0)


class ScoringWeights(StrictModel):
    """Relative weights used to combine scoring dimensions."""

    quality: float = Field(..., ge=0.0)
    cost: float = Field(..., ge=0.0)
    latency: float = Field(..., ge=0.0)
    safety: float = Field(..., ge=0.0)


class ScoringNormalization(StrictModel):
    """Normalization baselines for cost and latency scoring."""

    max_cost: float = Field(..., gt=0.0)
    max_latency_ms: int = Field(..., gt=0)


class ScoringSettings(StrictModel):
    """Scoring configuration."""

    weights: ScoringWeights
    normalization: ScoringNormalization
    safety_penalty: float = Field(..., ge=0.0, le=1.0)


class PromotionGateSettings(StrictModel):
    """Promotion gate thresholds for candidate strategies."""

    min_win_rate: float = Field(..., ge=0.0, le=1.0)
    max_p_value: float = Field(..., ge=0.0, le=1.0)
    min_score_delta: float
    max_cost_delta: float = Field(..., ge=0.0)
    max_latency_delta: float = Field(..., ge=0.0)
    require_no_safety_violation: bool
    require_no_critical_bucket_regression: bool


class Settings(StrictModel):
    """Complete GoAgentX configuration assembled from YAML files."""

    database: DatabaseSettings
    reports: ReportsSettings
    evolution: EvolutionSettings
    arena: ArenaSettings
    scoring: ScoringSettings
    promotion_gate: PromotionGateSettings


def load_settings(config_dir: str | Path = DEFAULT_CONFIG_DIR) -> Settings:
    """Load and validate GoAgentX settings.

    Args:
        config_dir: Directory containing GoAgentX YAML config files.

    Returns:
        A validated settings object.

    Raises:
        SettingsError: If a config file is missing, malformed, or invalid.
    """
    base_dir = Path(config_dir)
    main_config = _load_yaml_mapping(base_dir / "goagentx.yaml")
    scoring_config = _load_yaml_mapping(base_dir / "scoring.yaml")
    promotion_config = _load_yaml_mapping(base_dir / "promotion_gate.yaml")

    config = {
        **main_config,
        "scoring": scoring_config,
        "promotion_gate": promotion_config,
    }
    _apply_environment_overrides(config)

    try:
        return Settings.model_validate(config)
    except ValidationError as exc:
        raise SettingsError(f"Invalid GoAgentX configuration:\n{exc}") from exc


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Read a YAML file and require its root node to be a mapping."""
    if not path.exists():
        raise SettingsError(f"Config file not found: {path}")

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SettingsError(f"Invalid YAML in {path}: {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SettingsError(f"Config file must contain a YAML mapping: {path}")
    return data


def _apply_environment_overrides(config: dict[str, Any]) -> None:
    """Apply supported environment variable overrides in-place."""
    database_path = os.getenv(DATABASE_PATH_ENV)
    if database_path:
        database_config = config.setdefault("database", {})
        if not isinstance(database_config, dict):
            raise SettingsError("Invalid GoAgentX configuration: database must be a mapping")
        database_config["path"] = database_path
