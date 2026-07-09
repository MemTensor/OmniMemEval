# OmniMemEval AgentBench

[English](./README.md)

AgentBench 是 OmniMemEval 中面向 Agent Runtime 的评测模块，用于评估 OpenClaw 在五个任务域上的任务完成能力，并支持在相同任务集合上独立评测记忆插件。当前提供以下评测协议：

- plain AgentBench：不启用待测记忆插件，执行 `test_only` 或 `train_then_test`。
- memory plugin AgentBench：按 `memory_train_backup_test` 协议清理、训练、等待沉淀、备份、恢复并测试记忆。

评测结果见 [eval_res_zh.md](./eval_res_zh.md)。

## 数据来源声明

AgentBench 使用 EverMind（Hugging Face/GitHub namespace: `EverMind-AI`）发布的 `EvoAgentBench` 数据集。数据集、任务划分和基础 benchmark 归属以 EverMind-AI/EvoAgentBench 及其数据集说明为准；本目录仅提供 OmniMemEval 侧的运行适配、配置和结果组织方式，不重新发布或声明拥有上述数据。

参考来源：

- HuggingFace 数据集：`EverMind-AI/EvoAgentBench`
- GitHub 项目：`EverMind-AI/EvoAgentBench`
- EvoAgentBench 覆盖的五个基础 benchmark：BrowseCompPlus、OmniMath、SWE-Bench、LiveCodeBench、GDPVal

数据规模依据 EvoAgentBench 的公开说明整理如下：

| 域 | 基础 benchmark | Train | Test |
|---|---|---:|---:|
| `information_retrieval` | BrowseCompPlus | 154 | 65 |
| `reasoning` | OmniMath | 478 | 100 |
| `software_engineering` | SWE-Bench | 101 | 26 |
| `code_implementation` | LiveCodeBench | 97 | 39 |
| `knowledge_work` | GDPVal | 87 | 58 |

## 数据准备

以下命令均在仓库根目录执行。首先下载公开数据集：

```bash
mkdir -p data/agentbench
huggingface-cli download EverMind-AI/EvoAgentBench \
  --repo-type dataset \
  --local-dir ./data/agentbench
```

下载完成后，数据目录应包含以下任务数据：

```text
data/agentbench/
  BrowseComp-Plus/
  Reasoning & Problem Decomposition/
  gdpval/
  livecode/
  swebench/
```

各域数据路径由 `configs/agentbench/domains/*.yaml` 管理。若采用自定义数据目录，请同步更新相应 domain yaml 中的 `data_path`、`train_file`、`test_file`、`repo_root` 或同类字段。

LiveCodeBench 域依赖官方 verifier 包：

```bash
git clone https://github.com/LiveCodeBench/LiveCodeBench.git ./LiveCodeBench
pip install --no-deps -e ./LiveCodeBench
```

BrowseComp-Plus dense 检索依赖数据预处理、索引文件以及 embedding 服务配置：

```bash
python scripts/agentbench/utils/browsecomp-plus-tools/setup_data.py \
  --output-dir ./data/agentbench/BrowseComp-Plus \
  --skip-index
python scripts/agentbench/utils/browsecomp-plus-tools/build_dense_index.py \
  --config configs/agentbench/domains/information_retrieval.yaml
```

SWE-Bench 域依赖 Docker daemon，且容器环境需能够访问 PyPI/GitHub。GDPVal 的 PDF/表格任务依赖文档处理工具。Debian/Ubuntu 环境可参考以下命令安装系统依赖：

```bash
apt-get update
apt-get install -y docker.io poppler-utils libreoffice openjdk-21-jdk
systemctl start docker
```

其中 `openjdk-21-jdk` 主要用于兼容依赖 Pyserini/BM25 的检索流程；若仅使用已构建的 dense index，可根据实际配置决定是否安装。

## 运行环境安装

推荐使用独立 conda 环境 `agentmem`：

```bash
conda create -n agentmem python=3.12 -y
conda activate agentmem
python -m pip install -U pip
pip install -r requirements_agentbench.txt
```

不要复用 User Memory 的环境运行 AgentBench。User Memory 与 AgentBench 依赖分别维护在独立的 requirements 文件中。

安装 OpenClaw CLI，并确保其位于当前 shell 的 `PATH` 中：

```bash
npm install -g openclaw
```

如需运行记忆插件生命周期评测，需要先安装待测记忆插件。以下以 MemOS local plugin 为例：

```bash
curl -fsSL https://raw.githubusercontent.com/MemTensor/MemOS/main/apps/memos-local-plugin/install.sh | bash
```

插件安装完成后，还需要完成 OpenClaw 和记忆插件配置，再进行验证。具体配置项取决于待测插件。以 MemOS local plugin 为例，安装脚本完成后，需要按插件要求补齐 OpenClaw 配置目录下生成的插件配置，并确认该插件在 OpenClaw 配置中处于启用状态。

运行 AgentBench 前至少确认：

- OpenClaw 已配置可用的模型/provider。
- 记忆插件已安装，并在 OpenClaw 配置中启用。
- 插件所需的凭证、本地路径或服务 endpoint 已配置。
- 下文所述 `.env.agent` 已准备完成。

安装和配置都完成后，验证 OpenClaw CLI 是否可以正常调用：

```bash
openclaw --version
openclaw agent --help
```

运行 LiveCodeBench 域前，应完成上文的 `LiveCodeBench` 安装。SWE-Bench 域依赖 Docker；信息检索域依赖 embedding endpoint；GDPVal 域依赖 PDF/Office 文件处理工具。

域相关依赖：

- BrowseComp-Plus dense 检索依赖 embedding 配置和索引文件；相关工具位于 `scripts/agentbench/utils/browsecomp-plus-tools/`。
- GDPVal 依赖 PDF/表格处理工具，PDF 场景使用 `poppler-utils`，PPTX 场景可使用 `libreoffice`。
- SWE-Bench 依赖 Docker，容器内评测脚本需访问 PyPI/GitHub。
- LiveCodeBench 使用 `LiveCodeBench/` verifier。

## 环境与模型配置

AgentBench 默认读取项目根目录 `.env.agent`，也支持通过 `--env FILE` 指定额外环境变量文件。使用独立的 `.env.agent` 可以避免与 User Memory Evaluation 使用的 `.env.memos`、`.env.mem0` 等 backend-specific 配置混淆。

从模板创建 Agent Memory Evaluation 环境文件：

```bash
cp env_examples/.env.agent .env.agent
```

OpenClaw 模型配置文件为：

```text
configs/agentbench/agents/openclaw.yaml
```

常用 `.env.agent` 字段如下：

```bash
LLM_BASE_URL=...
LLM_API_KEY=...

JUDGE_MODEL=...
JUDGE_API_BASE=...
JUDGE_API_KEY=...

IR_EMBEDDING_ENDPOINT=...
IR_EMBEDDING_MODEL=...
IR_EMBEDDING_API_KEY=...

EVALUATION_API_BASE=...
EVALUATION_API_KEY=...
EVALUATION_MODEL_OWNER=...
EVALUATION_MODEL_NAME=...
EVALUATION_TIMEOUT=240
EVALUATION_MAX_RETRIES=3
```

配置说明：

- OpenClaw agent 的模型来自 `configs/agentbench/agents/openclaw.yaml`，其中 provider credential 由 `.env.agent` 注入。
- 若某些 provider 字段未在 `.env.agent` 中配置，adapter 会回退到 `~/.openclaw/openclaw.json` 中可解析的配置。
- memory plugin 协议下始终使用 `runtime.home_mode: isolated_copy`，并通过 memory plugin 配置中的 `home_links` 把必要插件目录链接到临时 `OPENCLAW_HOME`。

## 目录结构

```text
scripts/agentbench/
  run_agent_eval.py                 # 单域评测入口
  runner.py                         # phase/trial/retry 调度
  memory_lifecycle.py               # 记忆插件清理、备份、恢复命令执行
  feedback.py                       # train 后 verifier feedback prompt
  plugin_feedback.py                # 插件侧 structured feedback 通用入口
  session_capture.py                # 从 OpenClaw session 构造 task/feedback trace
  memos_feedback.py                 # 插件侧 structured feedback 适配
  agents/openclaw.py                # OpenClaw adapter
  domains/                          # 五个 domain adapter

scripts/
  run_agentbench_memos_5domain_1each_smoke.sh  # MemOS 五域/单域 1-each smoke

configs/agentbench/
  agents/openclaw.yaml
  domains/*.yaml
  memory_plugins/*.yaml

results/agentbench/
  <profile>-<version>-<domain>/
    train/
    test_run_1/
    memory_lifecycle.log
    memory_lifecycle.json
```

## 测评协议

### 最小 smoke 验证

baseline smoke 可直接运行单域 `test_only`：

```bash
./scripts/run_agent_eval.sh \
  --agent openclaw \
  --domain reasoning \
  --protocol test_only \
  --version baseline_reasoning_smoke \
  --trials 1 \
  --parallel 1
```

MemOS lifecycle smoke 建议先跑单域 1 train / 1 test：

```bash
./scripts/run_agentbench_memos_5domain_1each_smoke.sh \
  --domains reasoning \
  --version memos_reasoning_1each_smoke \
  --tasks-per-split 1 \
  --trials 1 \
  --test-runs 1 \
  --parallel 1 \
  --settle-seconds 0
```

该脚本会执行完整 `memory_train_backup_test` 生命周期：清理记忆、训练、发送
verifier feedback、提交 MemOS structured feedback、备份、恢复、测试和 finalize。
建议先用它验证本地 OpenClaw、MemOS、模型、judge 和 embedding 配置，再启动五域完整评测。

### `test_only`

执行 test split，适用于 baseline smoke test 或单域功能验证。

```bash
./scripts/run_agent_eval.sh \
  --agent openclaw \
  --domain reasoning \
  --protocol test_only \
  --version baseline_reasoning_test \
  --trials 1 \
  --parallel 1
```

### `train_then_test`

依次执行 train split 和 test split；该协议不执行记忆插件生命周期。

```bash
./scripts/run_agent_eval.sh \
  --agent openclaw \
  --domain reasoning \
  --protocol train_then_test \
  --version train_then_test_reasoning \
  --trials 1 \
  --parallel 1
```

### `memory_train_backup_test`

该协议用于独立评测记忆插件，结果目录与 baseline 评测结果分离。流程如下：

1. 设置插件为 train 模式。
2. 清理记忆。
3. 执行 train split。
4. train verifier 完成后，将 feedback 作为同一 OpenClaw session 的下一轮消息提交。
5. 若插件配置启用 structured feedback，则提交插件侧显式反馈，并保证 task turn 和 feedback turn 保持同一 session/episode 语义。
6. 等待插件沉淀/进化，默认由 `settle_seconds` 或 `wait_settle` 配置控制。
7. 备份当前域的记忆。
8. 每次 test 前都恢复该域备份，即使插件支持关闭写入也仍恢复。
9. 执行 test split，并写入独立结果目录。

单域运行：

```bash
./scripts/run_agent_eval.sh \
  --agent openclaw \
  --domain reasoning \
  --protocol memory_train_backup_test \
  --memory-plugin memos \
  --version memos_reasoning_eval \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

调试生命周期时，可以显式指定 train/test 任务：

```bash
./scripts/run_agent_eval.sh \
  --agent openclaw \
  --domain reasoning \
  --protocol memory_train_backup_test \
  --memory-plugin memos \
  --version memos_reasoning_debug \
  --train-task omni_35 \
  --test-task omni_2080 \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

五域顺序运行：

```bash
./scripts/run_agentbench_memory_train_backup_test.sh \
  --memory-plugin memos \
  --version memos_5domain_eval \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

指定域子集：

```bash
./scripts/run_agentbench_memory_train_backup_test.sh \
  --memory-plugin memos \
  --domains reasoning,code_implementation \
  --version memos_reasoning_code_eval
```

## 记忆插件配置

插件生命周期配置位于：

```text
configs/agentbench/memory_plugins/
  memos.yaml
  everos.yaml
  openviking.yaml
  supermemory.yaml
  hindsight.yaml
```

每个配置负责声明：

- `plugin`：插件标签，会进入结果目录名。
- `backup_dir` 和 `backup_file_template`：备份位置。
- `settle_seconds` 或 `commands.wait_settle`：训练后等待沉淀/进化的逻辑。
- `home_links`：在 isolated OpenClaw home 中需链接的插件目录。
- `modes.train/test` 或 `commands.set_mode_*`：训练/测试时插件读写模式。
- `commands.clear/backup/restore`：清理、备份、恢复记忆。
- `execution`：可选 agent 执行策略。MemOS 使用 `transport: gateway` 和
  `capture_mode: manual_after_feedback`，让任务执行可并发，同时在 verifier feedback
  后显式提交记忆写入。
- `max_parallel`：可选并发上限，用于不能安全并发启动多个本地 runtime 的插件。
- `feedback`：是否启用 train feedback、timeout，以及插件侧 structured feedback。

以 `memos.yaml` 为例，structured feedback 配置如下：

```yaml
feedback:
  enabled: true
  timeout: 300
  structured_submit: true
  backend: memos
  submit_timeout: 900
  memos_structured_submit: true
  memos_submit_timeout: 900
```

其他插件默认仅执行普通 train feedback，不执行 structured submit；需要显式提交反馈的插件可在对应 yaml 中开启相关开关。

对于 MemOS，`structured_submit: true` 会在 verifier feedback turn 后调用 MemOS
bridge 提交显式反馈。该步骤依赖 MemOS embedding/LLM 配置，可能比 agent 调用耗时更长；服务较慢时可调整 `submit_timeout`。

## 输出与结果读取

每个 domain 的输出目录格式如下：

```text
results/agentbench/openclaw-memos-memos_5domain_eval-reasoning/
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

- `summary.json`：`pass@1`、平均 reward、平均耗时。
- `result.json.agent_result`：OpenClaw 完成状态、耗时、是否从 session/trajectory 恢复。
- `result.json.feedback_result`：train 后同 session feedback turn 状态。
- `result.json.plugin_feedback_result`：插件侧 structured feedback 状态。
- `result.json.memos_feedback_result`：MemOS structured feedback 的兼容字段。
- `memory_lifecycle.json`：插件 clear/backup/restore 等命令事件。

## 排查建议

- 如果 smoke run 在 agent 已写出回答后长时间不返回，优先检查 domain verifier 是否正在等待外部 judge 服务。`reasoning` 域默认使用 `verify_mode: llm`。
- 若只想验证生命周期机制，可临时复制 `configs/agentbench/domains/reasoning.yaml`，把 `verify_mode` 改为 `exact`，再通过 `--domain-config` 传入。该方式仅用于流程验证，不用于正式评测结果。
- 如果卡在 MemOS structured feedback 阶段，检查 `result.json.plugin_feedback_result`、`memory_lifecycle.log` 和 `~/.openclaw/memos-plugin/data/memos.db`。成功提交后，MemOS DB 中应能看到 episode、task/feedback traces，以及一条 explicit feedback。
- 中断运行后，可确认 MemOS viewer 端口没有残留监听：

```bash
lsof -nP -iTCP:18799 -sTCP:LISTEN
```

## 验证结果

5 域评测结果见 [eval_res_zh.md](./eval_res_zh.md)。模型服务、外部 judge 服务和 embedding 服务状态均可能影响单条样本 reward。
