import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from goagentx.core.task import Task, TaskModelError, TaskRun, TaskSet, load_task_set


FIXTURE_PATH = Path("tests/fixtures/task_sets/sample_task_set.json")


def test_task_can_store_generic_input_json() -> None:
    task = Task(
        id="task-001",
        task_type="doc_qa",
        bucket="baseline",
        input_json={
            "question": "What is GoAgentX?",
            "metadata": {"difficulty": "easy", "attempt": 1},
        },
        expected_json={"contains": ["GoAgentX"]},
        tags=["docs", "qa"],
    )

    serialized = task.model_dump(mode="json")

    assert serialized["task_type"] == "doc_qa"
    assert serialized["bucket"] == "baseline"
    json.dumps(serialized)


@pytest.mark.parametrize("missing_field", ["task_type", "bucket"])
def test_task_requires_task_type_and_bucket(missing_field: str) -> None:
    data = {
        "id": "task-001",
        "task_type": "doc_qa",
        "bucket": "baseline",
        "input_json": {"question": "What is GoAgentX?"},
    }
    data.pop(missing_field)

    with pytest.raises(ValidationError) as exc_info:
        Task(**data)

    assert missing_field in str(exc_info.value)


def test_task_rejects_non_json_serializable_input() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Task(
            id="task-001",
            task_type="doc_qa",
            bucket="baseline",
            input_json={"bad": {1, 2, 3}},
        )

    assert "JSON serializable" in str(exc_info.value)


def test_load_task_set_fixture() -> None:
    task_set = load_task_set(FIXTURE_PATH)

    assert isinstance(task_set, TaskSet)
    assert task_set.id == "sample-agent-tasks"
    assert len(task_set.tasks) == 2
    assert task_set.tasks[1].bucket == "critical"


def test_load_task_set_reports_invalid_fixture(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text('{"id": "empty", "tasks": []}', encoding="utf-8")

    with pytest.raises(TaskModelError) as exc_info:
        load_task_set(path)

    assert "Invalid task set schema" in str(exc_info.value)


def test_task_run_saves_score_breakdown_and_json_payloads() -> None:
    run = TaskRun(
        id="run-001",
        task_id="task-001",
        strategy_id="strategy-001",
        output_json={"answer": "GoAgentX evolves strategies."},
        score=0.82,
        score_breakdown={"quality": 0.9, "cost": 0.7},
        success=True,
        cost=0.02,
        latency_ms=1200,
        token_count=450,
        tool_calls=[{"name": "repo_search", "ok": True}],
    )

    serialized = run.model_dump(mode="json")

    assert serialized["score_breakdown"]["quality"] == 0.9
    assert serialized["tool_calls"][0]["name"] == "repo_search"
    json.dumps(serialized)


def test_task_run_rejects_negative_metrics() -> None:
    with pytest.raises(ValidationError) as exc_info:
        TaskRun(
            id="run-001",
            task_id="task-001",
            strategy_id="strategy-001",
            output_json={},
            score=0.0,
            success=False,
            cost=-0.01,
            latency_ms=0,
            token_count=0,
        )

    assert "cost" in str(exc_info.value)
