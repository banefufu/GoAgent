# GoAgentX Implementation Plan

> 来源：`GOAGENTX_DEV_SPEC.md` v0.1  
> 用途：把总 DEV_SPEC 拆成可执行的实施导航，配合 `skills/*/SKILL.md` 使用。  
> 规则：总规格仍以 `GOAGENTX_DEV_SPEC.md` 为准；拆分文件由 `skills/spec-sync/sync_spec.py` 生成。

## 1. 使用方式

### 开始一个新任务

1. 同步规格：
   ```powershell
   python skills/spec-sync/sync_spec.py
   ```
2. 读取导航：
   ```text
   skills/spec-sync/SPEC_INDEX.md
   ```
3. 定位任务：
   ```text
   skills/spec-sync/specs/07-schedule.md
   ```
4. 根据任务类型补读：
   - 技术实现：`skills/spec-sync/specs/03-tech-stack.md`
   - 数据模型：`skills/spec-sync/specs/04-data-model.md`
   - 测试要求：`skills/spec-sync/specs/05-testing.md`
   - 架构边界：`skills/spec-sync/specs/06-architecture.md`

### 每次只做一个子任务

开发节奏固定为：

```text
spec-sync -> progress-tracker -> implement -> testing-stage -> checkpoint
```

每次只推进一个任务，例如 A1、A2、A3，不一次性吞掉整个阶段 A。

## 2. 拆分后的规格地图

| 目的 | 文件 |
|---|---|
| 项目定位、MVP 范围、非目标 | `skills/spec-sync/specs/01-overview.md` |
| GoAgentX 核心能力解释 | `skills/spec-sync/specs/02-features.md` |
| Python MVP、CLI、配置选型 | `skills/spec-sync/specs/03-tech-stack.md` |
| SQLite 表和文件产物 | `skills/spec-sync/specs/04-data-model.md` |
| 测试分层和必测场景 | `skills/spec-sync/specs/05-testing.md` |
| 模块边界、目录结构、核心接口 | `skills/spec-sync/specs/06-architecture.md` |
| A-I 阶段任务细分 | `skills/spec-sync/specs/07-schedule.md` |
| M1-M6 里程碑验收 | `skills/spec-sync/specs/08-milestones.md` |
| 后续扩展方向 | `skills/spec-sync/specs/09-future.md` |

## 3. 实施顺序

| 阶段 | 子任务 | 目标 |
|---|---|---|
| A | A1 -> A2 -> A3 | 工程骨架、配置、SQLite 初始化 |
| B | B1 -> B2 -> B3 | Strategy Genome、Registry、YAML 导入导出 |
| C | C1 -> C2 -> C3 -> C4 | Task/Run、TaskStore、Scorer、Fake AgentRunner |
| D | D1 -> D2 -> D3 -> D4 | paired eval、统计检验、Quick Reject、Full Eval 报告 |
| E | E1 -> E2 -> E3 | 退化检测、Mutation、DreamCycle 编排 |
| F | F1 -> F2 -> F3 | Selection、Crossover、Genome GA |
| G | G1 -> G2 -> G3 | Promotion Gate、状态推进、Rollback |
| H | H1 -> H2 -> H3 -> H4 | Strategy/Eval/Evolve/Promotion CLI |
| I | I1 -> I2 -> I3 | Golden Task Set、E2E、README 收口 |

## 4. 阶段验收节奏

| 里程碑 | 覆盖范围 | 验收重点 |
|---|---|---|
| M1 | A | `goagentx init`、`goagentx --help`、`pytest -q` |
| M2 | B + C | 策略和任务运行记录能落库 |
| M3 | D | Arena 能拒绝坏候选 |
| M4 | E | DreamCycle 能生成候选并进入 Quick Reject |
| M5 | F | Genome GA 零 Token 生成候选 |
| M6 | G + H + I | shadow promotion、rollback、E2E 全链路通过 |

## 5. 当前推荐起点

当前项目还没有实现代码时，从 A1 开始：

```text
阶段 A：工程骨架与配置基座
任务 A1：初始化工程结构
主读：skills/spec-sync/specs/07-schedule.md
补读：skills/spec-sync/specs/03-tech-stack.md、skills/spec-sync/specs/06-architecture.md
测试：pytest -q tests/unit/test_cli.py
```

