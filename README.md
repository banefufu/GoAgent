# GoAgentX

GoAgentX is a local, auditable agent strategy evolution toolkit. It treats prompts,
model parameters, tool policy, retry policy, and memory policy as versioned
`Strategy` objects, then evaluates candidate strategies through Arena, promotion
gates, and rollback-safe lifecycle states.

The current MVP is Python-based and uses deterministic fake runners by default.
Local tests and CLI demos do not require real model/API credentials.

## Quick Start

The commands below use PowerShell on Windows from the repository root.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\goagentx.exe --help
.\.venv\Scripts\goagentx.exe init --database-path data\demo.db
```

Seed the demo champion. Direct champion import is intentionally not exposed by
the CLI, so this local bootstrap uses the registry API for the first baseline.

```powershell
@'
from goagentx.registry.strategy_io import load_strategy_yaml
from goagentx.registry.strategy_registry import StrategyRegistry

registry = StrategyRegistry("data/demo.db")
champion = load_strategy_yaml("tests/fixtures/strategies/champion.yaml")
registry.create(champion)
print(f"seeded {champion.id} as {champion.status.value}")
'@ | .\.venv\Scripts\python.exe -
```

Import a candidate, run Arena Full Eval, and promote it to shadow with the manual
gate defaults.

```powershell
.\.venv\Scripts\goagentx.exe strategy import tests\fixtures\strategies\candidate_good.yaml --database-path data\demo.db

.\.venv\Scripts\goagentx.exe eval `
  --database-path data\demo.db `
  --champion champion `
  --candidate candidate_good `
  --task-set tests\fixtures\task_sets\sample_agent_tasks.json `
  --report-dir reports `
  --experiment-id demo-good-eval

.\.venv\Scripts\goagentx.exe promote `
  --database-path data\demo.db `
  --candidate candidate_good `
  --mode shadow `
  --champion champion `
  --reason readme_demo_shadow
```

Inspect what happened.

```powershell
.\.venv\Scripts\goagentx.exe strategy list --database-path data\demo.db
Get-ChildItem reports
```

## Configuration

GoAgentX reads YAML config from `configs/` by default.

| File | Purpose |
|---|---|
| `configs/goagentx.yaml` | Database path, report directory, evolution windows, Arena thresholds. |
| `configs/scoring.yaml` | Quality/cost/latency/safety weights and normalization. |
| `configs/promotion_gate.yaml` | Promotion gate thresholds and hard guardrails. |
| `configs/mutations.yaml` | Mutation ranges and tool allowlist for DreamCycle/Genome GA. |

Most commands accept `--config-dir` and `--database-path`. The database path can
also be overridden with `GOAGENTX_DATABASE_PATH`.

```powershell
$env:GOAGENTX_DATABASE_PATH = "data\local.db"
.\.venv\Scripts\goagentx.exe init
```

Runtime outputs are ignored by git:

- `data/` for SQLite databases
- `reports/` for Markdown reports and audit logs

## Strategy YAML

Strategies are structured, versioned genomes. A minimal strategy looks like this:

```yaml
id: candidate-docs
version: 1
name: Candidate docs strategy
task_type: doc_qa
status: candidate
genome:
  model:
    provider: openai_compatible
    name: gpt-4.1
    temperature: 0.4
    top_p: 0.9
  prompt_genome:
    role: senior_code_reviewer
    reasoning_style: evidence_first
    risk_policy: strict
    output_format: findings_first
  tools:
    enabled:
      - repo_search
      - shell_readonly
parent_ids: []
```

Useful commands:

```powershell
.\.venv\Scripts\goagentx.exe strategy import path\to\strategy.yaml --database-path data\demo.db
.\.venv\Scripts\goagentx.exe strategy list --database-path data\demo.db --status candidate
.\.venv\Scripts\goagentx.exe strategy show candidate-docs --database-path data\demo.db
.\.venv\Scripts\goagentx.exe strategy export candidate-docs --database-path data\demo.db --output exported.yaml
```

`strategy import` only imports as `draft` or `candidate`. This prevents a file
edit from bypassing Arena and promotion controls.

## Task Set Format

A task set is a JSON file with an `id` and a non-empty `tasks` list.

```json
{
  "id": "sample-agent-tasks",
  "description": "Small reusable evaluation set.",
  "tasks": [
    {
      "id": "task-doc-001",
      "task_type": "doc_qa",
      "bucket": "baseline",
      "input_json": {
        "question": "What does GoAgentX evolve?"
      },
      "expected_json": {
        "contains": ["structured agent strategies"]
      },
      "tags": ["docs", "qa"]
    }
  ]
}
```

Important fields:

- `task_type`: logical workload type, such as `doc_qa`, `code_review`, or `tool_use`
- `bucket`: evaluation slice, such as `baseline`, `edge`, or `critical`
- `input_json`: task input payload
- `expected_json`: fixture expectation used by the current fake runner
- `tags`: optional labels for filtering and analysis

The golden fixture is at `tests/fixtures/task_sets/sample_agent_tasks.json`.

## Arena Evaluation

Full Eval compares a champion and a candidate on the same task set and writes a
Markdown report plus persisted task runs.

```powershell
.\.venv\Scripts\goagentx.exe eval `
  --database-path data\demo.db `
  --champion champion `
  --candidate candidate_good `
  --task-set tests\fixtures\task_sets\sample_agent_tasks.json `
  --report-dir reports `
  --experiment-id demo-eval `
  --seed 0
```

The output includes verdict, selected task count, win rate, average score delta,
report path, and failed checks. With the default CLI fake runner, equal fixture
quality can still produce a `reject` verdict because significance and score
delta gates are intentionally strict.

## DreamCycle

DreamCycle checks score degradation, mutates the current champion, stores the new
candidates, and optionally runs Quick Reject.

```powershell
.\.venv\Scripts\goagentx.exe evolve dream `
  --database-path data\demo.db `
  --strategy champion `
  --task-set tests\fixtures\task_sets\sample_agent_tasks.json `
  --candidate-count 2 `
  --audit-log reports\dreamcycle-demo.jsonl `
  --seed 11
```

By default the CLI uses `--manual-trigger`, which generates candidates even if
there is not enough degradation history. Use `--require-degradation` when you
only want generation after the detector fires.

## Genome GA

Genome GA builds a candidate population from historically scored strategies. It
uses selection, crossover, and mutation, but does not run Arena or promote by
itself.

```powershell
.\.venv\Scripts\goagentx.exe evolve ga `
  --database-path data\demo.db `
  --task-type doc_qa `
  --population sample `
  --population-size 4 `
  --mutation-rate 0.34 `
  --seed 3
```

If the registry has too little scored history, GA exits with an explanatory
error rather than inventing parents.

## Promotion And Rollback

Promotion is an audited state machine:

```text
candidate -> shadow -> canary -> champion
```

Promote a candidate to shadow:

```powershell
.\.venv\Scripts\goagentx.exe promote `
  --database-path data\demo.db `
  --candidate candidate_good `
  --mode shadow `
  --champion champion `
  --reason manual_shadow_gate
```

Advance a shadow to canary, then champion:

```powershell
.\.venv\Scripts\goagentx.exe promote --database-path data\demo.db --candidate candidate_good --mode canary --champion champion
.\.venv\Scripts\goagentx.exe promote --database-path data\demo.db --candidate candidate_good --mode champion --champion champion
```

Rollback to a stable strategy:

```powershell
.\.venv\Scripts\goagentx.exe rollback `
  --database-path data\demo.db `
  --to champion `
  --failed candidate_good `
  --reason safety_regression
```

Promotion and rollback events are written to the `promotion_events` table.

## Tests

Run the full suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Focused suites:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\unit
.\.venv\Scripts\python.exe -m pytest -q tests\integration
.\.venv\Scripts\python.exe -m pytest -q tests\e2e
.\.venv\Scripts\python.exe -m pytest -q tests\e2e\test_evolution_flow.py
```

Current e2e coverage:

- golden Arena acceptance/rejection
- degradation -> DreamCycle -> Quick Reject -> Full Eval -> gate -> shadow/rejected

## FAQ

### Do tests need a real model?

No. The MVP uses deterministic fake runners in tests and CLI demos. Real model
integration should happen behind the `AgentRunner` adapter without changing the
Arena, promotion, or registry contracts.

### Why can I not import a champion directly?

Direct champion import would bypass evaluation and audit controls. Use registry
bootstrap only for a fresh local demo database, then move all future strategies
through `candidate -> shadow -> canary -> champion`.

### Why did Full Eval reject a candidate that looked fine?

Promotion requires more than successful task execution. The gate checks win
rate, score delta, p-value, cost, latency, safety, and critical bucket
regression. Ties and small samples can fail the significance checks.

### Where are reports and audit logs?

By default reports go to `reports/`. Full Eval writes Markdown reports, and
DreamCycle can write JSONL audit logs through `--audit-log`.

### What should I run before pushing changes?

Use the focused test for your change, then run the full suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
git diff --check
```
