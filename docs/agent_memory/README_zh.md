<a id="agent-memory-evaluation"></a>
# Agent Memory Evaluation（智能体记忆评测）

[English](./README.md) · [文档索引](../README_zh.md) · [评测结果](./results_zh.md)

## 概览

本评测线使用 OmniMemEval 的 AgentBench 模块，评估智能体运行时（agent runtime）
在有无记忆插件（memory plugin）时的任务表现。
当前公开的运行时适配器覆盖 OpenClaw 和五个任务域。

评测设置分为两类：

- 基线：不启用待测记忆插件，使用 `test_only` 或 `train_then_test`。
- 记忆插件：使用 `memory_train_backup_test`，包含清理、训练、等待沉淀、
  备份、恢复和测试。

首次运行建议按本文顺序阅读。已有环境可以直接查看
[进阶运行](#进阶运行)、[输出与结果](#输出与结果)或[扩展与贡献](#扩展与贡献)。
所有命令均在仓库根目录执行。

## 环境与配置

<a id="运行环境安装"></a>
### 安装依赖

为 Agent Memory 创建独立环境，不与 User Memory 混用：

```bash
conda create -n agentmem python=3.12 -y
conda activate agentmem
python -m pip install -U pip
python -m pip install -r requirements_agentbench.txt
npm install -g openclaw
```

为 OpenClaw 配置可用的模型/provider，再确认 CLI 可调用：

```bash
openclaw --version
openclaw agent --help
```

<a id="环境与模型配置"></a>
### 配置凭证

复制 Agent Memory 模板，再填写所选任务域需要的配置：

```bash
cp env_examples/.env.agent .env.agent
```

AgentBench 默认加载项目根目录的 `.env.agent`；`--env FILE` 可额外加载环境文件。
这些凭证与 User Memory 的 `.env.memos` 等后端文件分开维护。

| 用途 | 配置 |
|---|---|
| 智能体模型/provider | `configs/agentbench/agents/openclaw.yaml`，使用 `LLM_BASE_URL` 和 `LLM_API_KEY` |
| 评分模型 | `JUDGE_MODEL`、`JUDGE_API_BASE`、`JUDGE_API_KEY` |
| 信息检索 embedding | `IR_EMBEDDING_ENDPOINT`、`IR_EMBEDDING_MODEL`、`IR_EMBEDDING_API_KEY` |
| 外部评测服务 | `EVALUATION_API_BASE`、`EVALUATION_API_KEY`、`EVALUATION_MODEL_OWNER`、`EVALUATION_MODEL_NAME`、`EVALUATION_TIMEOUT`、`EVALUATION_MAX_RETRIES` |

字段说明以[环境模板](../../env_examples/.env.agent)和所选
`configs/agentbench/domains/*.yaml` 为准。
未通过智能体 YAML/环境变量配置的 provider 字段，可以回退到
`~/.openclaw/openclaw.json` 中可解析的配置。不要提交填写了凭证的文件。

### 按需准备记忆插件

基线运行不需要待测记忆插件。使用 `memory_train_backup_test` 时，
先安装并配置该插件。例如，MemOS local plugin 提供以下安装脚本：

```bash
curl -fsSL https://raw.githubusercontent.com/MemTensor/MemOS/main/apps/memos-local-plugin/install.sh | bash
```

执行前应审阅上游安装脚本。补齐生成的插件配置、凭证、本地路径和服务地址，
并确认插件在 OpenClaw 中已启用。

记忆插件运行强制使用 `runtime.home_mode: isolated_copy`，但 `home_links`
仍可能把真实插件数据链接到临时 OpenClaw home。
生命周期中的清理/恢复命令可能删除或覆盖这些数据，并重启服务。
请检查所选插件 YAML，使用独立评测数据，不要使用个人记忆。

<a id="数据来源声明"></a>
## 数据准备

AgentBench 使用 `EverMind-AI/EvoAgentBench` 数据集。
数据归属、划分和基础 benchmark 定义遵循上游项目；
OmniMemEval 提供运行适配、配置、生命周期管理与结果组织。

- [Hugging Face 数据集](https://huggingface.co/datasets/EverMind-AI/EvoAgentBench)
- [上游项目](https://github.com/EverMind-AI/EvoAgentBench)

| 任务域 | Benchmark | Train | Test |
|---|---|---:|---:|
| `information_retrieval` | BrowseCompPlus | 154 | 65 |
| `reasoning` | OmniMath | 478 | 100 |
| `software_engineering` | SWE-Bench | 101 | 26 |
| `code_implementation` | LiveCodeBench | 97 | 39 |
| `knowledge_work` | GDPVal | 87 | 58 |

### 下载数据集

```bash
mkdir -p data/agentbench
hf download EverMind-AI/EvoAgentBench \
  --repo-type dataset \
  --local-dir ./data/agentbench
```

预期目录：

```text
data/agentbench/
  BrowseComp-Plus/
  Reasoning & Problem Decomposition/
  gdpval/
  livecode/
  swebench/
```

`hf` CLI 随 `huggingface_hub` 提供，详见
[上游 CLI 指南](https://huggingface.co/docs/huggingface_hub/en/guides/cli)。
若使用自定义数据位置，请更新 `configs/agentbench/domains/*.yaml` 中的相应路径。

### 准备任务域特有依赖

只需额外准备所选任务域需要的工具。LiveCodeBench 使用官方验证器包：

```bash
git clone https://github.com/LiveCodeBench/LiveCodeBench.git ./LiveCodeBench
python -m pip install --no-deps -e ./LiveCodeBench
```

BrowseComp-Plus 的 dense 检索需要预处理、索引和 embedding 服务配置：

```bash
python scripts/agentbench/utils/browsecomp-plus-tools/setup_data.py \
  --output-dir ./data/agentbench/BrowseComp-Plus \
  --skip-index
python scripts/agentbench/utils/browsecomp-plus-tools/build_dense_index.py \
  --config configs/agentbench/domains/information_retrieval.yaml
```

SWE-Bench 需要 Docker，评测容器需要访问 PyPI/GitHub；
GDPVal 需要 PDF/Office 处理工具。Debian/Ubuntu 可使用管理员权限安装所需系统包：

```bash
apt-get update
apt-get install -y docker.io poppler-utils libreoffice openjdk-21-jdk
systemctl start docker
```

Java 与 Pyserini/BM25 检索有关；仅使用预构建 dense index 时可能不需要。
这些是任务域特有要求，不是下文 reasoning 快速开始的额外前置条件。

## 快速开始

完成环境、凭证和数据准备后，运行一道 reasoning 任务：

```bash
./scripts/run_agent_eval.sh \
  --agent openclaw \
  --domain reasoning \
  --protocol test_only \
  --version smoke_agentbench \
  --test-split 1 \
  --trials 1 \
  --parallel 1
```

该命令调用真实模型/评分服务，可能产生费用。对于 reasoning，
`--test-split 1` 选择第一道测试题；省略它则选择完整测试集。
`--parallel` 只控制并发，不限制题数。小规模检查不等于完整 benchmark 结果。

完整协议见[进阶运行](#进阶运行)。Runner 的 `--help` 提供 CLI 选项说明。

## 进阶运行

<a id="测评协议"></a>
### 评测协议

以下命令使用配置中的完整数据划分。

#### `test_only`

运行测试集：

```bash
./scripts/run_agent_eval.sh \
  --agent openclaw --domain reasoning --protocol test_only \
  --version baseline_reasoning_test --trials 1 --parallel 1
```

#### `train_then_test`

先训练再测试，不执行记忆插件生命周期命令：

```bash
./scripts/run_agent_eval.sh \
  --agent openclaw --domain reasoning --protocol train_then_test \
  --version train_then_test_reasoning --trials 1 --parallel 1
```

#### `memory_train_backup_test`

记忆插件与基线分开评测：

1. 设置插件为训练模式并清理记忆。
2. 执行训练。
3. 在同一 OpenClaw 会话的下一轮提交验证器反馈。
4. 如已启用，按相同会话/episode 语义提交插件侧结构化反馈。
5. 等待记忆沉淀/进化。
6. 备份当前任务域记忆。
7. 每次测试前恢复该域备份，即使已禁止测试写入也仍恢复。
8. 执行测试，将结果写入 `results/agentbench/`。

单任务域：

```bash
./scripts/run_agent_eval.sh \
  --agent openclaw --domain reasoning --protocol memory_train_backup_test \
  --memory-plugin memos --version memos_reasoning_eval \
  --test-runs 1 --trials 1 --parallel 1
```

全部五个任务域：

```bash
./scripts/run_agentbench_memory_train_backup_test.sh \
  --memory-plugin memos --version memos_5domain_eval \
  --test-runs 1 --trials 1 --parallel 1
```

任务域子集：

```bash
./scripts/run_agentbench_memory_train_backup_test.sh \
  --memory-plugin memos --domains reasoning,code_implementation \
  --version memos_reasoning_code_eval
```

### 记忆插件配置

生命周期配置位于 `configs/agentbench/memory_plugins/`：
`memos.yaml`、`everos.yaml`、`openviking.yaml`、`supermemory.yaml`
和 `hindsight.yaml`。

每个配置负责声明：

- `plugin`：结果目录中的插件标签。
- `backup_dir` 和 `backup_file_template`：备份位置。
- `settle_seconds` 或 `commands.wait_settle`：训练后的等待沉淀逻辑。
- `home_links`：链接到隔离 OpenClaw home 的插件路径。
- `modes.train/test` 或 `commands.set_mode_*`：读写模式。
- `commands.clear/backup/restore`：生命周期操作。
- `feedback`：训练反馈和可选的插件侧结构化反馈。

例如，`memos.yaml` 启用结构化反馈：

```yaml
feedback:
  enabled: true
  timeout: 300
  memos_structured_submit: true
  memos_submit_timeout: 900
```

其他插件配置默认使用普通训练反馈，不执行上述结构化提交。
请检查所选 YAML，不要假定所有插件行为相同。

<a id="目录结构"></a>
<a id="输出与结果读取"></a>
<a id="验证结果"></a>
## 输出与结果

记忆插件协议的输出结构如下：

```text
results/agentbench/<profile>-<version>-<domain>/
  experiment_config.json
  memory_lifecycle.log
  memory_lifecycle.json
  train/
    summary.json
    report.md
    <task>__trial_1/
      result.json
      response.txt
      session.jsonl
      verifier/
  test_run_1/
    summary.json
    report.md
    <task>__trial_1/
      result.json
      response.txt
      session.jsonl
      verifier/
```

关键字段：

- `summary.json`：`pass@1`、平均 reward 和平均耗时。
- `result.json.agent_result`：完成状态、耗时和会话恢复信息。
- `result.json.feedback_result`：同会话训练反馈状态。
- `result.json.memos_feedback_result`：启用时的插件侧结构化反馈状态。
- `memory_lifecycle.json`：清理、备份与恢复事件。

公开评测设置与表格见[评测结果](./results_zh.md)。
快照也包含 Hermes 结果，但当前公开运行时适配器覆盖 OpenClaw，
并未提供 Hermes 的复现路径。模型、评分与 embedding 服务可用性均可能影响单条奖励分数。

## 扩展与贡献

[架构与扩展契约](./architecture_zh.md)说明调度、会话标识、结果字段，
以及智能体/任务域扩展点。Agent Memory 适配器与 User Memory 的
`scripts/client_factory/` 层分离。

PR 流程和证据要求见 [CONTRIBUTING.md（英文）](../../CONTRIBUTING.md)。
代码、配置与数据准备脚本的位置见根目录的[仓库结构](../../README_zh.md#目录结构)。

## 验证与许可证

在独立的 Python 3.11 或 3.12 开发环境运行离线检查：

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/check_doc_links.py
```

[测试指南（英文）](../../scripts/tests/README.md)区分离线测试、仓库检查和真实后端集成测试。
离线 CI 不验证 benchmark 分数，也不调用真实服务。

代码许可证见 [LICENSE](../../LICENSE)。外部数据集保留上游许可证，详见
[THIRD_PARTY_NOTICES.md（英文）](../../THIRD_PARTY_NOTICES.md)。
