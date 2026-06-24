from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from goagentx.core.task import Task, TaskRun, load_task_set
from goagentx.registry.task_store import TaskStore, TaskStoreNotFoundError


FIXTURE_PATH = Path("tests/fixtures/task_sets/sample_task_set.json")


def test_save_task_set_and_query_by_task_set_id(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "goagentx.db")
    task_set = load_task_set(FIXTURE_PATH)

    saved_tasks = store.save_task_set(task_set)
    loaded_tasks = store.list_tasks(task_set_id=task_set.id)

    assert len(saved_tasks) == 2
    assert [task.id for task in loaded_tasks] == ["task-doc-001", "task-code-001"]
    assert {task.task_set_id for task in loaded_tasks} == {task_set.id}


def test_save_task_and_get_task_round_trips_model(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "goagentx.db")
    task = _task("task-001", task_type="doc_qa", bucket="baseline")

    saved = store.save_task(task, task_set_id="manual-set")
    loaded = store.get_task(task.id)

    assert loaded == saved
    assert loaded.task_set_id == "manual-set"
    assert loaded.input_json["question"] == "What is GoAgentX?"


def test_get_missing_task_raises(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "goagentx.db")

    with pytest.raises(TaskStoreNotFoundError):
        store.get_task("missing")


def test_list_tasks_filters_by_task_type_and_bucket(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "goagentx.db")
    store.save_task(_task("doc-001", task_type="doc_qa", bucket="baseline"))
    store.save_task(_task("doc-002", task_type="doc_qa", bucket="critical"))
    store.save_task(_task("code-001", task_type="code_review", bucket="critical"))

    tasks = store.list_tasks(task_type="doc_qa", bucket="critical")

    assert [task.id for task in tasks] == ["doc-002"]


def test_sample_tasks_filters_by_time_window(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "goagentx.db")
    old_time = datetime(2026, 1, 1, tzinfo=UTC)
    recent_time = datetime(2026, 6, 24, tzinfo=UTC)
    store.save_task(
        _task("old-001", task_type="doc_qa", bucket="baseline", created_at=old_time)
    )
    store.save_task(
        _task(
            "recent-001",
            task_type="doc_qa",
            bucket="baseline",
            created_at=recent_time,
        )
    )

    sampled = store.sample_tasks(
        task_type="doc_qa",
        bucket="baseline",
        created_after=recent_time - timedelta(days=1),
        limit=5,
    )

    assert [task.id for task in sampled] == ["recent-001"]


def test_save_run_and_list_recent_runs(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "goagentx.db")
    store.save_task(_task("task-001", task_type="doc_qa", bucket="baseline"))
    older_run = _task_run(
        "run-001",
        task_id="task-001",
        strategy_id="strategy-001",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    newer_run = _task_run(
        "run-002",
        task_id="task-001",
        strategy_id="strategy-001",
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    store.save_run(older_run)
    store.save_run(newer_run)
    recent_runs = store.list_recent_runs(limit=1, strategy_id="strategy-001")

    assert recent_runs == [newer_run]
    assert recent_runs[0].score_breakdown == {"quality": 0.8, "cost": 0.9}


def test_list_recent_runs_filters_by_experiment_id(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "goagentx.db")
    store.save_run(
        _task_run(
            "run-001",
            task_id="task-001",
            strategy_id="strategy-001",
            experiment_id="exp-001",
        )
    )
    store.save_run(
        _task_run(
            "run-002",
            task_id="task-002",
            strategy_id="strategy-001",
            experiment_id="exp-002",
        )
    )

    recent_runs = store.list_recent_runs(limit=10, experiment_id="exp-002")

    assert [task_run.id for task_run in recent_runs] == ["run-002"]


def test_list_recent_runs_filters_by_task_type(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "goagentx.db")
    store.save_task(_task("doc-001", task_type="doc_qa", bucket="baseline"))
    store.save_task(_task("code-001", task_type="code_review", bucket="critical"))
    store.save_run(
        _task_run("run-doc", task_id="doc-001", strategy_id="strategy-001")
    )
    store.save_run(
        _task_run("run-code", task_id="code-001", strategy_id="strategy-001")
    )

    recent_runs = store.list_recent_runs(
        limit=10,
        strategy_id="strategy-001",
        task_type="code_review",
    )

    assert [task_run.id for task_run in recent_runs] == ["run-code"]


def _task(
    task_id: str,
    *,
    task_type: str,
    bucket: str,
    created_at: datetime | None = None,
) -> Task:
    return Task(
        id=task_id,
        task_type=task_type,
        bucket=bucket,
        input_json={"question": "What is GoAgentX?"},
        expected_json={"contains": ["GoAgentX"]},
        tags=[task_type, bucket],
        created_at=created_at or datetime(2026, 1, 1, tzinfo=UTC),
    )


def _task_run(
    run_id: str,
    *,
    task_id: str,
    strategy_id: str,
    experiment_id: str | None = None,
    created_at: datetime | None = None,
) -> TaskRun:
    return TaskRun(
        id=run_id,
        task_id=task_id,
        strategy_id=strategy_id,
        experiment_id=experiment_id,
        output_json={"answer": "GoAgentX evolves strategies."},
        score=0.85,
        score_breakdown={"quality": 0.8, "cost": 0.9},
        success=True,
        cost=0.03,
        latency_ms=900,
        token_count=320,
        tool_calls=[{"name": "repo_search", "ok": True}],
        created_at=created_at or datetime(2026, 1, 1, tzinfo=UTC),
    )
