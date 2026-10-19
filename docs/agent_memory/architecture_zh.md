<a id="agent-memory-evaluation-架构"></a>
# Agent Memory Evaluation（智能体记忆评测）架构

[English](./architecture.md) · [文档索引](../README_zh.md) ·
[运行指南](./README_zh.md) · [评测结果](./results_zh.md)

本文档说明当前 Agent Memory Evaluation 的架构和公开扩展契约。环境准备和评测命令
仍以 [Agent Memory Evaluation 指南](./README_zh.md)为准。

## 架构概览

Agent Memory Evaluation 评估以下组合：

```text
智能体运行时 + 可选记忆插件 + 任务域 + 验证器
```

AgentBench 入口负责协议和记忆生命周期的执行顺序，调度器负责任务、试次（trial）、
重试和反馈的执行。运行时相关行为由 `AgentAdapter` 实现；任务加载、提示词
构造、环境准备和验证由 `DomainAdapter` 实现。记忆插件生命周期操作通常配置在
`configs/agentbench/memory_plugins/` 下。

## 核心组件

### `AgentAdapter`

[`AgentAdapter`](../../scripts/agentbench/agents/base.py) 定义调度器如何驱动一个
智能体运行时。它负责构造会话元数据、调用运行时、收集会话产物和
token 用量，以及判断运行时结果是否可重试。

当前公开智能体注册表包含
[`OpenClawAgentAdapter`](../../scripts/agentbench/agents/openclaw.py)。

### `DomainAdapter`

[`DomainAdapter`](../../scripts/agentbench/domains/base.py) 定义任务域如何加载任务、
准备单任务资源、构造提示词、验证智能体输出并清理任务状态。

当前任务域注册表覆盖 `reasoning`、`information_retrieval`、`knowledge_work`、
`code_implementation` 和 `software_engineering`。

<a id="runner-与记忆生命周期"></a>
### 调度器与记忆生命周期

[`run_agent_eval.py`](../../scripts/agentbench/run_agent_eval.py) 选择评测协议，并负责
评测阶段和记忆生命周期的执行顺序。

[`runner.py`](../../scripts/agentbench/runner.py) 执行任务和试次、写入单次试次
产物、应用重试策略并生成阶段汇总。

[`memory_lifecycle.py`](../../scripts/agentbench/memory_lifecycle.py) 执行配置中的插件
生命周期命令。`memory_train_backup_test` 协议负责验证、模式切换、清理、
训练、等待沉淀、备份、恢复和测试的执行顺序。可选的结构化反馈集成
仍在调度器和插件配置中显式声明。

## 评测执行流程

每个任务试次按以下顺序执行：

```text
domain.setup
agent.prepare_task
domain.build_prompt
agent.call
domain.verify
optional train feedback
domain.record_agent_outcome
domain.cleanup
agent.collect_session
agent.cleanup_task
write result.json
```

任务域级初始化（initialize）和收尾（finalize）包围完整评测阶段。
全部选定任务和试次完成后生成阶段汇总。

<a id="session-标识契约"></a>
## 会话标识契约

[`SessionSpec`](../../scripts/agentbench/session.py) 将不同用途的标识分开：

- `cli_session_id`：经过清理、可被智能体 CLI 接受的标识。
- `semantic_session_id`：保留 benchmark 阶段、任务域、数据划分、任务和试次
  语义的标识。
- `source_ref`：稳定的 benchmark 来源引用。
- `metadata`：benchmark、阶段、任务域、数据划分、任务、试次和智能体的结构化
  元数据。
- `agent_session_ref`：可选的运行时专用会话引用。
- `openclaw_session_key`：使用 OpenClaw 时对应的会话 key。
- `openclaw_gateway_session_id`：对应的 OpenClaw 网关标识。

OpenClaw 适配器将 `cli_session_id` 传给 `openclaw agent --session-id`，并默认通过
`OMNIMEMEVAL_AGENT_CONTEXT` 暴露序列化后的完整 `SessionSpec`。这样可以让
CLI 安全标识与插件和结果元数据保持对应。

## 结果与指标契约

每个试次写入一个 `result.json`，包含任务、阶段、数据划分、试次、会话、
智能体结果、验证器结果、反馈结果、token 用量、时间戳和异常信息。辅助产物
可以包括 `response.txt`、`session.jsonl` 和验证器输出。

每个阶段写入 `summary.json` 和 `report.md`。汇总包括任务数和试次数、
`pass@1`、配置的 `pass@n`、平均通过率、奖励分数、耗时、轮次、token 用量、
失败类别、排除基础设施错误后的统计，以及任务域专用指标。

## 扩展点

- 新增智能体运行时：实现 `AgentAdapter`，并注册到
  `scripts/agentbench/agents/__init__.py`。
- 新增任务域：实现 `DomainAdapter`，并注册到
  `scripts/agentbench/domains/__init__.py`。
- 新增或调整记忆插件生命周期：通过
  `configs/agentbench/memory_plugins/` 配置；当通用命令生命周期不足时，使用显式
  集成代码。

Agent Memory Evaluation 与 User Memory 的 `scripts/client_factory/` 适配器层保持
分离，因为两条评测线评估的是不同系统边界。
