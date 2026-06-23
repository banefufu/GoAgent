CREATE TABLE IF NOT EXISTS strategies (
  id TEXT PRIMARY KEY,
  version INTEGER NOT NULL,
  name TEXT NOT NULL,
  task_type TEXT,
  status TEXT NOT NULL,
  genome_json TEXT NOT NULL,
  parent_ids_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY,
  task_type TEXT NOT NULL,
  bucket TEXT NOT NULL,
  input_json TEXT NOT NULL,
  expected_json TEXT,
  tags_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS task_runs (
  id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  strategy_id TEXT NOT NULL,
  experiment_id TEXT,
  output_json TEXT NOT NULL,
  score REAL NOT NULL,
  success INTEGER NOT NULL,
  cost REAL NOT NULL,
  latency_ms INTEGER NOT NULL,
  token_count INTEGER NOT NULL,
  tool_calls_json TEXT NOT NULL,
  error_json TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_experiments (
  id TEXT PRIMARY KEY,
  champion_id TEXT NOT NULL,
  candidate_id TEXT NOT NULL,
  task_set_id TEXT NOT NULL,
  quick_reject_passed INTEGER NOT NULL,
  win_rate REAL NOT NULL,
  p_value REAL,
  avg_score_delta REAL NOT NULL,
  cost_delta REAL NOT NULL,
  latency_delta REAL NOT NULL,
  verdict TEXT NOT NULL,
  report_path TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS promotion_events (
  id TEXT PRIMARY KEY,
  strategy_id TEXT NOT NULL,
  from_status TEXT NOT NULL,
  to_status TEXT NOT NULL,
  reason TEXT NOT NULL,
  experiment_id TEXT,
  created_at TEXT NOT NULL
);
