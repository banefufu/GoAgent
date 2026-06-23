"""YAML import/export helpers for strategies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError
from yaml import YAMLError

from goagentx.core.strategy import Strategy, StrategyStatus
from goagentx.registry.strategy_registry import StrategyRegistry

IMPORTABLE_STATUSES = {StrategyStatus.DRAFT, StrategyStatus.CANDIDATE}


class StrategyIOError(RuntimeError):
    """Raised when a strategy YAML file cannot be read, written, or validated."""


def strategy_to_yaml_data(strategy: Strategy) -> dict[str, Any]:
    """Convert a Strategy model into YAML-friendly data."""
    return strategy.model_dump(mode="json")


def export_strategy_yaml(strategy: Strategy, path: str | Path) -> Path:
    """Write a strategy to a YAML file.

    Args:
        strategy: Strategy to serialize.
        path: Destination YAML path.

    Returns:
        The path written.

    Raises:
        StrategyIOError: If the target file cannot be written.
    """
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    data = strategy_to_yaml_data(strategy)

    try:
        target_path.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    except OSError as exc:
        raise StrategyIOError(f"Failed to write strategy YAML: {target_path}") from exc

    return target_path


def load_strategy_yaml(path: str | Path) -> Strategy:
    """Load and validate a Strategy from YAML.

    Args:
        path: YAML file path.

    Returns:
        A validated Strategy model.

    Raises:
        StrategyIOError: If the file is missing, malformed, or invalid.
    """
    source_path = Path(path)
    try:
        loaded = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StrategyIOError(f"Strategy YAML not found: {source_path}") from exc
    except OSError as exc:
        raise StrategyIOError(f"Failed to read strategy YAML: {source_path}") from exc
    except YAMLError as exc:
        raise StrategyIOError(f"Invalid strategy YAML syntax: {source_path}") from exc

    if not isinstance(loaded, dict):
        raise StrategyIOError(f"Strategy YAML must contain a mapping: {source_path}")

    try:
        return Strategy.model_validate(loaded)
    except ValidationError as exc:
        raise StrategyIOError(f"Invalid strategy YAML schema: {source_path}\n{exc}") from exc


def import_strategy_yaml(
    registry: StrategyRegistry,
    path: str | Path,
    *,
    status: StrategyStatus | str = StrategyStatus.CANDIDATE,
) -> Strategy:
    """Load a strategy YAML file and create it in a registry.

    Imported strategies are intentionally limited to draft/candidate states so
    file edits cannot bypass Arena and promotion controls.
    """
    import_status = StrategyStatus(status)
    if import_status not in IMPORTABLE_STATUSES:
        raise StrategyIOError("Imported strategies must use draft or candidate status")

    strategy = load_strategy_yaml(path)
    data = strategy.model_dump(mode="python")
    data["status"] = import_status
    importable_strategy = Strategy.model_validate(data)
    return registry.create(importable_strategy)
