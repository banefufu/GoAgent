# GoAgentX Developer Specification

> 版本：0.1 - 开发计划草案  
> 日期：2026-06-23  
> 参考：同目录 `DEV_SPEC.md` 的大型任务拆解写法  
> 定位：先把 GoAgentX 做成可验证、可回滚、可审计的 Agent 策略自适应运维系统

## 目录

- 项目概述
- 核心特点
- 技术选型
- 数据模型
- 测试方案
- 系统架构与模块设计
- 项目排期
- 可扩展性与未来展望

---

## 1. 项目概述

GoAgentX 是一个面向生产 Agent 的策略进化框架。它不试图直接训练模型，也不让 Agent 无约束地“自我修改”，而是把 prompt、模型参数、工具组合、工具策略、重试策略、输出格式等运行配置抽象成可版本化、可评估、可遗传的 Strategy Genome。

系统通过任务分数和运行轨迹检测策略退化，再通过 Arena 离线对照实验、Quick Reject、Full Eval、灰度上线和回滚机制，安全地选择更优策略。长期目标是让 Agent 在任务分布变化时具备自适应能力：先能试错，再能验证，最后才允许替换线上策略。

### 设计理念

> 核心定位：自适应 Agent 运维系统，而不是玄学式自我进化。

本项目的价值不在于发明新的训练算法，而在于把遗传算法、A/B 测试、混沌工程和 Agent 运行观测组合成一个工程闭环。

核心原则：

- 策略必须版本化：任何上线策略都能追踪来源、父本、实验记录和回滚点。
- 评估先于替换：候选策略必须在 Arena 里赢过当前 champion 才能进入灰度。
- 分数必须可解释：不只看总分，还要看任务类型、成本、延迟、安全和关键 bucket 退化。
- 线上必须可回滚：任何自动进化都不能绕过 promotion gate。
- 低成本优先：Genome GA 利用历史高分策略重组，DreamCycle 只在必要时消耗 LLM token。

### MVP 范围

第一版只做本地可运行的离线和半自动闭环：

- Strategy Registry：管理策略版本、基因字段和 lineage。
- Task Trace Store：记录任务输入、输出、工具调用、成本、延迟和分数。
- Scorer：计算基础组合分，并支持人工/LLM judge 扩展。
- Arena：对 champion 和 candidate 做同任务集对照评估。
- DreamCycle：退化触发后生成 1-3 个候选并送 Arena。
- Genome GA：从历史策略池做选择、交叉和变异，生成候选。
- Promotion Controller：支持 shadow、canary、promote、rollback 状态流转。
- CLI：提供最小可用命令，先不做复杂前端。

### 非目标

- 第一版不做模型训练、RLHF、知识蒸馏。
- 第一版不做完全自动全量替换，最多推进到 shadow/canary。
- 第一版不追求复杂 Dashboard，先保证 CLI 和结构化实验报告可用。
- 第一版不支持多租户权限模型，只保留未来扩展边界。

---

## 2. 核心特点

### 2.1 Strategy Genome 策略基因化

将 Agent 策略拆成结构化字段，而不是只保存一整段 prompt。

示例基因字段：

```yaml
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
    - browser
tool_policy:
  max_calls: 12
  prefer_read_before_edit: true
retry_policy:
  max_retries: 2
  retry_on_tool_error: true
memory_policy:
  read_project_memory: true
  write_long_term_memory: guarded
```

设计重点：

- Prompt 采用模块级拼接，避免默认做半句随机拼接。
- 每个策略保存 parent_ids，支持追踪遗传来源。
- 每个策略有 status：draft、candidate、shadow、canary、champion、retired、rejected。

### 2.2 Task Trace Store 任务轨迹库

每次任务运行都要沉淀成可复盘数据。

记录内容：

- task_id、task_type、input_hash、dataset_id
- strategy_id、strategy_version
- output_hash、success、score
- token_count、cost、latency_ms
- tool_calls、tool_error_count
- judge_result、human_rating
- created_at、run_env

这部分是后续进化的燃料。没有稳定的 trace，就没有可信的进化。

### 2.3 Scorer 评分器

第一版采用组合评分：

```text
score =
  0.40 * task_success
+ 0.25 * judge_score
+ 0.15 * tool_correctness
+ 0.10 * cost_score
+ 0.10 * latency_score
- safety_penalty
```

注意：

- 总分只用于排序，不直接决定上线。
- promotion gate 必须额外检查安全、成本、延迟和关键任务 bucket。
- 后续可加入人工评分、LLM-as-judge、单元测试结果、业务成功率等指标。

### 2.4 Arena 策略验证场

Arena 是 GoAgentX 的核心安全网。它负责把 champion 和 candidate 放到同一批任务上做 paired evaluation。

评估流程：

```text
champion_strategy + candidate_strategy + task_set
        |
        v
Quick Reject: 5 个代表任务
        |
        v
Full Eval: 50 个历史任务
        |
        v
paired win/loss/tie + bootstrap/permutation test
        |
        v
verdict: reject / shadow / canary / promote_ready
```

默认上线门槛：

```text
win_rate >= 0.55
p_value < 0.05
avg_score_delta > 0
cost_delta <= 20%
latency_delta <= 20%
safety_violation_count == 0
critical_bucket_regression == false
```

### 2.5 DreamCycle 在线进化

DreamCycle 用于探索新策略，允许调用 LLM 或模板生成候选。

触发条件：

- 最近 50 个任务平均分比过去 300 个任务均值下降 15%。
- 某个 task_type 连续低于阈值。
- 工具错误率、失败率或人工差评明显升高。
- 手动触发。

候选类型：

- 参数变异：temperature、top_p、max_tool_calls、retry_policy。
- Prompt 模块变异：role、reasoning_style、risk_policy、output_format。
- 工具组合变异：启用/禁用工具、调整工具优先级、增加只读约束。

### 2.6 Genome GA 零 Token 进化

Genome GA 不调用 LLM，只从已有策略池中重组高分基因。

默认参数：

```text
population_size = 20
selection_pool = top 60%
elite_ratio = 20%
mutation_rate = 20%
crossover = uniform + module-level prompt crossover
```

适用场景：

- 空闲时段后台生成候选策略。
- 某类任务长期稳定但想低成本微调。
- 从历史高分策略中寻找更优组合。

### 2.7 Promotion Controller 上线控制

策略状态流：

```text
draft -> candidate -> shadow -> canary -> champion
                    \-> rejected
champion -> retired
canary -> rolled_back
```

第一版建议：

- shadow：旁路运行，不影响真实输出。
- canary：只接 5%-10% 流量，观察真实表现。
- champion：成为默认策略。
- rollback：任何 safety/cost/score 异常触发回滚。

---

## 3. 技术选型

### 3.1 默认实现路线

本 DEV_SPEC 默认按 Python MVP 编排，原因是 LLM 调用、统计检验、任务回放、数据分析和测试生态更适合快速验证。若后续决定改为 Go 实现，模块边界、数据库表结构、CLI 命令和验收标准保持不变，只替换 runtime 与目录结构。

推荐技术栈：

| 模块 | 选型 | 说明 |
|---|---|---|
| Runtime | Python 3.11+ | 快速验证 Agent Ops 闭环 |
| CLI | Typer | 命令清晰，适合本地 MVP |
| 配置 | YAML + Pydantic | 策略和系统配置结构化校验 |
| 数据库 | SQLite | 单机 MVP 足够，易迁移 |
| 实验报告 | JSONL + Markdown | 便于审计、diff、人工复盘 |
| 统计检验 | scipy / numpy | paired bootstrap、permutation test |
| Prompt 模板 | Jinja2 | 模块级 prompt 拼装 |
| 测试 | pytest | 单元、集成、E2E |
| 调度 | APScheduler | 后台 GA 和周期检测 |
| 可选 API | FastAPI | 第二阶段再暴露服务接口 |

### 3.2 CLI 命令设计

最小可用命令：

```bash
goagentx init
goagentx strategy list
goagentx strategy show <strategy_id>
goagentx run --strategy <strategy_id> --task <task_file>
goagentx eval --champion <id> --candidate <id> --task-set <dataset_id>
goagentx evolve dream --strategy <id>
goagentx evolve ga --population <dataset_id>
goagentx promote --candidate <id> --mode shadow
goagentx rollback --to <strategy_id>
```

### 3.3 配置文件

建议使用：

```text
configs/
  goagentx.yaml
  scoring.yaml
  promotion_gate.yaml
  mutations.yaml
  task_buckets.yaml
```

示例：

```yaml
evolution:
  degradation_window: 50
  baseline_window: 300
  degradation_threshold: 0.15
arena:
  quick_reject_rounds: 5
  full_eval_rounds: 50
  min_win_rate: 0.55
  p_value_threshold: 0.05
promotion:
  max_cost_delta: 0.20
  max_latency_delta: 0.20
  require_no_safety_violation: true
```

---

## 4. 数据模型

### 4.1 SQLite 表设计

#### strategies

```sql
CREATE TABLE strategies (
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
```

#### tasks

```sql
CREATE TABLE tasks (
  id TEXT PRIMARY KEY,
  task_type TEXT NOT NULL,
  bucket TEXT NOT NULL,
  input_json TEXT NOT NULL,
  expected_json TEXT,
  tags_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
```

#### task_runs

```sql
CREATE TABLE task_runs (
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
```

#### eval_experiments

```sql
CREATE TABLE eval_experiments (
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
```

#### promotion_events

```sql
CREATE TABLE promotion_events (
  id TEXT PRIMARY KEY,
  strategy_id TEXT NOT NULL,
  from_status TEXT NOT NULL,
  to_status TEXT NOT NULL,
  reason TEXT NOT NULL,
  experiment_id TEXT,
  created_at TEXT NOT NULL
);
```

### 4.2 文件型产物

```text
data/
  goagentx.db
  task_sets/
  reports/
    eval_<experiment_id>.md
  traces/
    task_runs.jsonl
strategies/
  champion.yaml
  candidates/
  archived/
```

---

## 5. 测试方案

### 5.1 测试理念

GoAgentX 的测试重点不是“函数能不能跑”，而是“坏策略能不能被拦住，好策略能不能被可解释地推进”。

测试层级：

- 单元测试：基因变异、交叉、评分、统计检验、状态流转。
- 集成测试：策略运行、trace 记录、Arena 实验、Promotion gate。
- E2E 测试：从退化检测到候选生成、评估、shadow promotion 全链路。
- 回归测试：固定 golden task set，防止评分器和 gate 误改。

### 5.2 必测场景

- 分数下降 15% 时能触发 DreamCycle。
- Quick Reject 能淘汰明显退化候选。
- Full Eval 能输出 win_rate、p_value、avg_score_delta。
- cost_delta 超过阈值时，即使 win_rate 达标也不能上线。
- safety_violation > 0 时必须 reject。
- critical bucket 退化时必须 reject。
- rollback 能把 champion 恢复到上一版本。
- Genome GA 生成的子代必须保留 parent_ids。

### 5.3 测试命令

```bash
pytest -q
pytest -q tests/unit
pytest -q tests/integration
pytest -q tests/e2e/test_evolution_flow.py
```

---

## 6. 系统架构与模块设计

### 6.1 整体架构图

```text
                    +----------------------+
                    | Evolution Scheduler  |
                    +----------+-----------+
                               |
          +--------------------+--------------------+
          |                                         |
          v                                         v
   +-------------+                          +----------------+
   | DreamCycle  |                          |   Genome GA    |
   +------+------+                          +-------+--------+
          |                                         |
          +--------------------+--------------------+
                               |
                               v
                      +----------------+
                      | Candidate Pool |
                      +-------+--------+
                              |
                              v
+-------------+       +-------+--------+       +----------------------+
| Task Store  +------>|     Arena      +------>| Promotion Controller |
+-------------+       +-------+--------+       +----------+-----------+
                              |                           |
                              v                           v
                      +---------------+           +-------------------+
                      | Eval Reports  |           | Strategy Registry |
                      +---------------+           +-------------------+
```

### 6.2 推荐目录结构

```text
goagentx/
  pyproject.toml
  README.md
  configs/
    goagentx.yaml
    scoring.yaml
    promotion_gate.yaml
    mutations.yaml
  src/
    goagentx/
      cli.py
      config/
        settings.py
      core/
        strategy.py
        task.py
        run.py
        scoring.py
      registry/
        strategy_registry.py
        task_store.py
        experiment_store.py
      arena/
        runner.py
        paired_eval.py
        report.py
      evolution/
        scheduler.py
        dreamcycle.py
        genome_ga.py
        mutation.py
        crossover.py
        selection.py
      promotion/
        gate.py
        controller.py
        rollback.py
      adapters/
        agent_runner.py
        llm_client.py
        tool_runtime.py
      observability/
        trace.py
        metrics.py
  tests/
    unit/
    integration/
    e2e/
    fixtures/
      task_sets/
      strategies/
```

### 6.3 模块说明

#### Strategy Registry

职责：

- 读写策略。
- 维护版本、状态、父本关系。
- 提供 champion 查询和候选列表。

核心接口：

```python
class StrategyRegistry:
    def create(self, strategy: Strategy) -> Strategy: ...
    def get(self, strategy_id: str) -> Strategy: ...
    def list_by_status(self, status: str) -> list[Strategy]: ...
    def get_champion(self, task_type: str | None = None) -> Strategy: ...
    def update_status(self, strategy_id: str, status: str) -> None: ...
```

#### Task Trace Store

职责：

- 保存 task 和 task_run。
- 支持按 task_type、bucket、时间窗口采样。
- 支持为 Arena 构造固定 task set。

#### Scorer

职责：

- 将任务运行结果转成可比较分数。
- 支持多维指标和加权配置。
- 输出 score breakdown，方便解释。

#### Arena

职责：

- 对两个策略跑同一批任务。
- 输出 paired win/loss/tie。
- 生成 Markdown 评估报告。
- 根据 gate 输出 verdict。

#### DreamCycle

职责：

- 根据退化信号读取 champion。
- 生成 1-3 个候选策略。
- 将候选写入 registry。
- 触发 Arena 或进入候选池。

#### Genome GA

职责：

- 从历史策略池选择父本。
- 做 crossover 和 mutation。
- 生成子代候选。
- 不调用 LLM，不产生 token 成本。

#### Promotion Controller

职责：

- 根据 Arena verdict 推进状态。
- 支持 shadow、canary、champion。
- 记录 promotion event。
- 支持 rollback。

---

## 7. 项目排期

### 阶段总览

| 阶段 | 目标 | 交付物 |
|---|---|---|
| A | 工程骨架与配置基座 | 可运行 CLI、配置加载、测试框架 |
| B | 策略注册表 | Strategy Genome、Registry、SQLite 持久化 |
| C | 任务轨迹与评分 | Task Store、Task Run、Scorer |
| D | Arena 对照实验 | Quick Reject、Full Eval、统计检验、报告 |
| E | DreamCycle | 退化检测、候选变异、Arena 接入 |
| F | Genome GA | 选择、交叉、变异、候选池 |
| G | Promotion | shadow/canary/promote/rollback |
| H | CLI 与审计报告 | 常用命令、报告输出、操作日志 |
| I | E2E 验收与文档 | golden task set、全链路测试、README |

### 进度跟踪表

| 阶段 | 状态 | 备注 |
|---|---|---|
| A | 已完成 | A1-A3 已完成，本地工程可安装、可运行、可测试 |
| B | 已完成 | B1-B3 已完成，策略对象化和 YAML 文件流转可用 |
| C | 待开始 | 评分可信度决定进化质量 |
| D | 待开始 | Arena 是安全网，优先级最高 |
| E | 待开始 | DreamCycle 不直接上线 |
| F | 待开始 | GA 只产生候选，不绕过 Arena |
| G | 待开始 | 上线和回滚必须可审计 |
| H | 待开始 | CLI 先于 Dashboard |
| I | 待开始 | 用 E2E 防止闭环断裂 |

---

## 阶段 A：工程骨架与配置基座

目标：先让项目可安装、可运行、可测试。

### A1：初始化工程结构（已完成）

- 目标：建立 Python 包结构和最小 CLI。
- 修改文件：
  - `pyproject.toml`
  - `src/goagentx/__init__.py`
  - `src/goagentx/cli.py`
  - `tests/unit/test_cli.py`
- 实现要点：
  - 使用 Typer 定义 `goagentx --help`。
  - 配置 pytest。
  - 保留 `src/` layout。
- 验收标准：
  - `goagentx --help` 可显示命令。
  - `pytest -q` 可执行。
- 测试方法：
  - `pytest -q tests/unit/test_cli.py`

### A2：配置加载与校验（已完成）

- 目标：支持读取 YAML 配置并做类型校验。
- 前置依赖：A1
- 修改文件：
  - `configs/goagentx.yaml`
  - `configs/scoring.yaml`
  - `configs/promotion_gate.yaml`
  - `src/goagentx/config/settings.py`
  - `tests/unit/test_settings.py`
- 实现要点：
  - 使用 Pydantic 定义 Settings。
  - 缺少必要字段时给出明确错误。
  - 支持环境变量覆盖数据库路径。
- 验收标准：
  - 默认配置可加载。
  - 错误配置能失败并提示字段。
- 测试方法：
  - `pytest -q tests/unit/test_settings.py`

### A3：SQLite 初始化（已完成）

- 目标：建立本地数据库和 migration 初始化逻辑。
- 前置依赖：A2
- 修改文件：
  - `src/goagentx/registry/db.py`
  - `src/goagentx/registry/schema.sql`
  - `tests/unit/test_db_init.py`
- 实现要点：
  - `goagentx init` 创建 `data/goagentx.db`。
  - schema 创建具备幂等性。
- 验收标准：
  - 多次运行 init 不报错。
  - strategies/tasks/task_runs/eval_experiments/promotion_events 表存在。
- 测试方法：
  - `pytest -q tests/unit/test_db_init.py`

---

## 阶段 B：Strategy Registry

目标：把策略配置变成可版本化、可继承、可查询的对象。

### B1：定义 Strategy 数据结构（已完成）

- 目标：实现 Strategy、Genome、PromptGenome、ToolPolicy 等类型。
- 前置依赖：A2
- 修改文件：
  - `src/goagentx/core/strategy.py`
  - `tests/unit/test_strategy_model.py`
- 实现要点：
  - genome 必须可 JSON 序列化。
  - parent_ids 默认空数组。
  - status 只能取枚举值。
- 验收标准：
  - 合法 strategy 可创建。
  - 非法 status、非法参数范围会报错。
- 测试方法：
  - `pytest -q tests/unit/test_strategy_model.py`

### B2：实现 StrategyRegistry（已完成）

- 目标：支持策略增删查改和状态流转。
- 前置依赖：A3、B1
- 修改文件：
  - `src/goagentx/registry/strategy_registry.py`
  - `tests/unit/test_strategy_registry.py`
- 实现要点：
  - `create/get/list_by_status/update_status/get_champion`。
  - 同一 task_type 最多一个 champion。
  - 更新状态时写 updated_at。
- 验收标准：
  - 可创建 champion 和 candidate。
  - 设置新 champion 时旧 champion 自动 retired 或显式要求先 retire。
- 测试方法：
  - `pytest -q tests/unit/test_strategy_registry.py`

### B3：策略 YAML 导入导出（已完成）

- 目标：让策略可以用文件编辑和版本管理。
- 前置依赖：B1、B2
- 修改文件：
  - `src/goagentx/registry/strategy_io.py`
  - `strategies/champion.yaml`
  - `tests/unit/test_strategy_io.py`
- 实现要点：
  - `strategy export` 导出 YAML。
  - `strategy import` 导入为 draft/candidate。
  - 导入时校验 schema。
- 验收标准：
  - YAML round-trip 后内容一致。
- 测试方法：
  - `pytest -q tests/unit/test_strategy_io.py`

---

## 阶段 C：任务轨迹与评分

目标：让每次 Agent 运行都变成可计算、可回放的数据。

### C1：定义 Task 和 TaskRun

- 目标：实现任务、任务集、运行结果的数据结构。
- 前置依赖：A2
- 修改文件：
  - `src/goagentx/core/task.py`
  - `tests/unit/test_task_model.py`
- 实现要点：
  - task_type、bucket 必填。
  - input_json 保持通用，避免绑定某个业务。
  - TaskRun 保存 score breakdown。
- 验收标准：
  - 能加载 fixture task set。
- 测试方法：
  - `pytest -q tests/unit/test_task_model.py`

### C2：实现 TaskStore

- 目标：持久化任务和运行记录。
- 前置依赖：A3、C1
- 修改文件：
  - `src/goagentx/registry/task_store.py`
  - `tests/unit/test_task_store.py`
- 实现要点：
  - 支持按 bucket、task_type、时间窗口采样。
  - 支持固定 task_set_id。
- 验收标准：
  - 能保存任务、保存运行、查询最近 N 条。
- 测试方法：
  - `pytest -q tests/unit/test_task_store.py`

### C3：实现 Scorer

- 目标：计算组合分和分项说明。
- 前置依赖：A2、C1
- 修改文件：
  - `src/goagentx/core/scoring.py`
  - `tests/unit/test_scorer.py`
- 实现要点：
  - 权重来自 `configs/scoring.yaml`。
  - cost_score、latency_score 做归一化。
  - safety_penalty 可直接压低总分。
- 验收标准：
  - 同一输入得到稳定分数。
  - safety violation 场景分数明显降低。
- 测试方法：
  - `pytest -q tests/unit/test_scorer.py`

### C4：AgentRunner 适配层

- 目标：定义统一的策略运行接口，先用 fake runner 跑通闭环。
- 前置依赖：B1、C1、C3
- 修改文件：
  - `src/goagentx/adapters/agent_runner.py`
  - `src/goagentx/core/run.py`
  - `tests/unit/test_agent_runner.py`
- 实现要点：
  - 第一版实现 `FakeAgentRunner`，根据 fixture 返回稳定结果。
  - 后续真实 Agent 接入只替换 adapter。
- 验收标准：
  - 给定 strategy + task 能返回 TaskRun。
- 测试方法：
  - `pytest -q tests/unit/test_agent_runner.py`

---

## 阶段 D：Arena 对照实验

目标：实现 GoAgentX 的策略验证网关。

### D1：实现 paired evaluation

- 目标：比较 champion 和 candidate 在同一批任务上的表现。
- 前置依赖：B2、C2、C4
- 修改文件：
  - `src/goagentx/arena/paired_eval.py`
  - `tests/unit/test_paired_eval.py`
- 实现要点：
  - 对每个 task 输出 win/lose/tie。
  - 支持 tie_threshold，避免微小噪声。
  - 输出 win_rate、avg_score_delta。
- 验收标准：
  - 已知 fixture 能得到预期 win_rate。
- 测试方法：
  - `pytest -q tests/unit/test_paired_eval.py`

### D2：实现统计显著性检验

- 目标：使用 paired bootstrap 或 permutation test 计算 p_value。
- 前置依赖：D1
- 修改文件：
  - `src/goagentx/arena/stats.py`
  - `tests/unit/test_stats.py`
- 实现要点：
  - 默认 permutation test。
  - 样本太少时标记 `insufficient_sample`。
  - 随机种子可配置，保证测试稳定。
- 验收标准：
  - 明显优势样本 p_value 较低。
  - 小样本不会误判 promote。
- 测试方法：
  - `pytest -q tests/unit/test_stats.py`

### D3：Quick Reject

- 目标：先用 5 个代表任务快速淘汰垃圾候选。
- 前置依赖：D1、D2
- 修改文件：
  - `src/goagentx/arena/runner.py`
  - `tests/integration/test_quick_reject.py`
- 实现要点：
  - 从 task set 中按 bucket 分层抽样。
  - 明显低于 champion 时直接 reject。
- 验收标准：
  - fixture 中弱候选不会进入 Full Eval。
- 测试方法：
  - `pytest -q tests/integration/test_quick_reject.py`

### D4：Full Eval 与报告生成

- 目标：完成 50 轮评估并输出 Markdown 报告。
- 前置依赖：D1-D3
- 修改文件：
  - `src/goagentx/arena/report.py`
  - `tests/integration/test_full_eval.py`
- 实现要点：
  - 报告包含总览、bucket 表、失败样例、成本延迟、安全项。
  - 报告路径写入 eval_experiments。
- 验收标准：
  - `goagentx eval ...` 生成 report。
  - report 中可看出 verdict 原因。
- 测试方法：
  - `pytest -q tests/integration/test_full_eval.py`

---

## 阶段 E：DreamCycle

目标：在检测到退化时生成少量高质量候选，但不直接上线。

### E1：退化检测

- 目标：实现 score drop 检测。
- 前置依赖：C2
- 修改文件：
  - `src/goagentx/evolution/scheduler.py`
  - `tests/unit/test_degradation_detector.py`
- 实现要点：
  - 比较最近窗口和基线窗口。
  - 支持按 task_type 检测。
  - 输出 trigger reason。
- 验收标准：
  - 下降 15% 触发。
  - 正常波动不触发。
- 测试方法：
  - `pytest -q tests/unit/test_degradation_detector.py`

### E2：变异器 Mutation

- 目标：从当前策略生成参数、prompt、工具三类候选。
- 前置依赖：B1
- 修改文件：
  - `src/goagentx/evolution/mutation.py`
  - `configs/mutations.yaml`
  - `tests/unit/test_mutation.py`
- 实现要点：
  - 参数变异范围受配置限制。
  - Prompt 模块级变异，不默认半句切分。
  - 工具变异必须遵守安全 allowlist。
- 验收标准：
  - 每次 mutation 产生合法 strategy。
  - parent_ids 正确记录。
- 测试方法：
  - `pytest -q tests/unit/test_mutation.py`

### E3：DreamCycle 编排

- 目标：退化触发后生成 1-3 个候选并送入 Arena。
- 前置依赖：D4、E1、E2
- 修改文件：
  - `src/goagentx/evolution/dreamcycle.py`
  - `tests/integration/test_dreamcycle.py`
- 实现要点：
  - 候选先写入 registry，status=candidate。
  - 可配置是否自动运行 Arena。
  - 所有行为写 audit log。
- 验收标准：
  - `goagentx evolve dream --strategy <id>` 能生成候选并跑 Quick Reject。
- 测试方法：
  - `pytest -q tests/integration/test_dreamcycle.py`

---

## 阶段 F：Genome GA

目标：零 Token 从历史高分策略池中生成候选。

### F1：选择 Selection

- 目标：按历史表现选择父本池。
- 前置依赖：B2、C2
- 修改文件：
  - `src/goagentx/evolution/selection.py`
  - `tests/unit/test_selection.py`
- 实现要点：
  - 默认选择 top 60%。
  - 精英策略单独保留。
  - 支持按 task_type 选择。
- 验收标准：
  - 低分策略不会进入默认父本池。
- 测试方法：
  - `pytest -q tests/unit/test_selection.py`

### F2：交叉 Crossover

- 目标：实现 uniform crossover 和 prompt module crossover。
- 前置依赖：F1
- 修改文件：
  - `src/goagentx/evolution/crossover.py`
  - `tests/unit/test_crossover.py`
- 实现要点：
  - 数值参数逐字段继承。
  - prompt_genome 按模块继承。
  - tools 做集合交叉并校验 allowlist。
- 验收标准：
  - 子代合法，且能追踪两个父本。
- 测试方法：
  - `pytest -q tests/unit/test_crossover.py`

### F3：GA 编排

- 目标：生成下一代 population。
- 前置依赖：F1、F2、E2
- 修改文件：
  - `src/goagentx/evolution/genome_ga.py`
  - `tests/integration/test_genome_ga.py`
- 实现要点：
  - population_size 默认 20。
  - elite_ratio 默认 20%。
  - mutation_rate 默认 20%。
  - 子代写入 candidate_pool，不直接评估或上线。
- 验收标准：
  - `goagentx evolve ga` 能生成候选策略。
- 测试方法：
  - `pytest -q tests/integration/test_genome_ga.py`

---

## 阶段 G：Promotion 与回滚

目标：把策略上线做成受控状态机。

### G1：Promotion Gate

- 目标：根据实验结果判断候选是否可进入 shadow/canary。
- 前置依赖：D4
- 修改文件：
  - `src/goagentx/promotion/gate.py`
  - `tests/unit/test_promotion_gate.py`
- 实现要点：
  - 检查 win_rate、p_value、score_delta。
  - 检查 cost_delta、latency_delta。
  - 检查 safety 和 critical bucket。
- 验收标准：
  - 任一硬性门槛失败即 reject。
- 测试方法：
  - `pytest -q tests/unit/test_promotion_gate.py`

### G2：Promotion Controller

- 目标：实现状态推进和事件记录。
- 前置依赖：B2、G1
- 修改文件：
  - `src/goagentx/promotion/controller.py`
  - `tests/integration/test_promotion_controller.py`
- 实现要点：
  - 支持 candidate -> shadow -> canary -> champion。
  - 每次状态变化写 promotion_events。
  - 禁止未评估策略直接 champion。
- 验收标准：
  - 合法路径可推进。
  - 非法路径被拒绝。
- 测试方法：
  - `pytest -q tests/integration/test_promotion_controller.py`

### G3：Rollback

- 目标：支持回滚到最近稳定 champion。
- 前置依赖：G2
- 修改文件：
  - `src/goagentx/promotion/rollback.py`
  - `tests/integration/test_rollback.py`
- 实现要点：
  - 保存 previous champion。
  - 回滚事件可审计。
  - 回滚后失败策略标记 rolled_back 或 retired。
- 验收标准：
  - `goagentx rollback --to <strategy_id>` 能恢复指定策略。
- 测试方法：
  - `pytest -q tests/integration/test_rollback.py`

---

## 阶段 H：CLI 与审计报告

目标：让开发者能通过命令完成完整闭环。

### H1：Strategy CLI

- 目标：实现策略查看、导入、导出、状态查看。
- 前置依赖：B3
- 修改文件：
  - `src/goagentx/cli.py`
  - `tests/integration/test_strategy_cli.py`
- 验收标准：
  - `goagentx strategy list/show/import/export` 可用。
- 测试方法：
  - `pytest -q tests/integration/test_strategy_cli.py`

### H2：Eval CLI

- 目标：实现 Arena 命令入口。
- 前置依赖：D4
- 修改文件：
  - `src/goagentx/cli.py`
  - `tests/integration/test_eval_cli.py`
- 验收标准：
  - `goagentx eval --champion s1 --candidate s2 --task-set sample` 输出 verdict 和 report 路径。
- 测试方法：
  - `pytest -q tests/integration/test_eval_cli.py`

### H3：Evolve CLI

- 目标：实现 DreamCycle 和 Genome GA 命令入口。
- 前置依赖：E3、F3
- 修改文件：
  - `src/goagentx/cli.py`
  - `tests/integration/test_evolve_cli.py`
- 验收标准：
  - `goagentx evolve dream` 和 `goagentx evolve ga` 都能生成 candidate。
- 测试方法：
  - `pytest -q tests/integration/test_evolve_cli.py`

### H4：Promotion CLI

- 目标：实现 promote 和 rollback 命令入口。
- 前置依赖：G2、G3
- 修改文件：
  - `src/goagentx/cli.py`
  - `tests/integration/test_promotion_cli.py`
- 验收标准：
  - `goagentx promote --candidate <id> --mode shadow` 可推进状态。
  - `goagentx rollback --to <id>` 可回滚。
- 测试方法：
  - `pytest -q tests/integration/test_promotion_cli.py`

---

## 阶段 I：端到端验收与文档收口

目标：证明 GoAgentX 的最小闭环真实可用。

### I1：Golden Task Set

- 目标：建立固定样例任务集。
- 修改文件：
  - `tests/fixtures/task_sets/sample_agent_tasks.json`
  - `tests/fixtures/strategies/champion.yaml`
  - `tests/fixtures/strategies/candidate_good.yaml`
  - `tests/fixtures/strategies/candidate_bad.yaml`
- 实现要点：
  - 覆盖文档问答、代码审查、工具调用三类任务。
  - 包含一个明显好候选和一个明显坏候选。
- 验收标准：
  - Arena 能稳定接受 good candidate、拒绝 bad candidate。
- 测试方法：
  - `pytest -q tests/e2e/test_arena_golden_set.py`

### I2：全链路 E2E

- 目标：跑通从退化检测到候选生成、评估、shadow promotion 的全链路。
- 前置依赖：A-H
- 修改文件：
  - `tests/e2e/test_evolution_flow.py`
- 验收标准：
  - 退化触发 DreamCycle。
  - 候选进入 Arena。
  - 通过 gate 的候选进入 shadow。
  - 未通过 gate 的候选 rejected。
- 测试方法：
  - `pytest -q tests/e2e/test_evolution_flow.py`

### I3：README 与运行手册

- 目标：让新开发者 10 分钟内跑通本地 MVP。
- 修改文件：
  - `README.md`
- README 必须包含：
  - 快速开始
  - 配置说明
  - 策略 YAML 示例
  - 任务集格式
  - Arena 评估命令
  - DreamCycle 命令
  - Genome GA 命令
  - Promotion 和 rollback 命令
  - 常见问题
- 验收标准：
  - 按 README 可完成 init -> import strategy -> eval -> promote shadow。

---

## 8. 交付里程碑

### M1：本地工程可运行

范围：阶段 A  
验收：

```bash
goagentx init
goagentx --help
pytest -q
```

### M2：策略和任务可沉淀

范围：阶段 B + C  
验收：

```bash
goagentx strategy list
goagentx run --strategy champion --task tests/fixtures/task_sets/sample_agent_tasks.json
```

### M3：Arena 可拦截坏策略

范围：阶段 D  
验收：

```bash
goagentx eval --champion champion --candidate candidate_bad --task-set sample
```

输出 verdict 应为 reject。

### M4：DreamCycle 可生成候选

范围：阶段 E  
验收：

```bash
goagentx evolve dream --strategy champion
```

输出 1-3 个 candidate，并生成 Quick Reject 结果。

### M5：Genome GA 可零 Token 生成候选

范围：阶段 F  
验收：

```bash
goagentx evolve ga --population sample
```

生成候选策略且不调用 LLM。

### M6：Promotion 闭环可回滚

范围：阶段 G + H + I  
验收：

```bash
goagentx promote --candidate candidate_good --mode shadow
goagentx rollback --to champion
pytest -q tests/e2e/test_evolution_flow.py
```

---

## 9. 可扩展性与未来展望

### 9.1 从本地 CLI 到服务化 Agent Ops

MVP 使用本地 SQLite 和 CLI。后续可以扩展：

- FastAPI 服务化。
- PostgreSQL 替换 SQLite。
- Dashboard 展示策略 lineage、Arena 报告和分数趋势。
- 多 Agent、多任务类型、多租户隔离。

### 9.2 更可信的评分体系

后续可增加：

- 人工评分校准。
- LLM judge 多评委投票。
- 业务指标回流。
- 单元测试或执行验证作为 hard signal。
- 分 bucket 的置信区间。

### 9.3 更稳的统计与实验治理

后续可增加：

- Sequential testing。
- 多候选比较修正。
- Shadow mode 真实流量旁路对照。
- Canary 自动扩缩比例。
- 长期回归任务集。

### 9.4 更强的 Genome 表达能力

后续可增加：

- Prompt AST 或 DSL。
- 工具策略图。
- Memory policy 基因。
- Safety policy 基因。
- 针对不同 task_type 的局部 champion。

### 9.5 和混沌工程结合

Arena 可继续承担压力测试网关：

- 工具超时。
- 网络失败。
- 错误上下文注入。
- 模型输出格式漂移。
- 长上下文干扰。

候选策略不仅要在正常任务上赢，还要在混沌任务下不崩。
