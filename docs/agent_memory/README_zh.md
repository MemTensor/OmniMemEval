# OmniMemEval AgentBench

[English](./README.md)

AgentBench 是 OmniMemEval 中面向 Agent Runtime 的评测模块，用于评估 OpenClaw 和 Hermes 在五个任务域上的任务完成能力，并支持在相同任务集合上独立评测记忆插件。当前提供以下评测协议：

- plain AgentBench：不启用待测记忆插件，执行 `test_only` 或 `train_then_test`。
- memory plugin AgentBench：按 `memory_train_backup_test` 协议清理、训练、等待沉淀、备份、恢复并测试记忆。

评测结果见 [eval_res_zh.md](./eval_res_zh.md)。

产品评测配置与运行说明：

- [Mem0](./products/mem0_zh.md)
- [EverOS](./products/everos_zh.md)
- [Hindsight](./products/hindsight_zh.md)
- [OpenViking](./products/openviking_zh.md)
- [Supermemory](./products/supermemory_zh.md)

## 数据来源声明

AgentBench 使用 EverMind（Hugging Face/GitHub namespace: `EverMind-AI`）发布的 `EvoAgentBench` 数据集。数据集、任务划分和基础 benchmark 归属以 EverMind-AI/EvoAgentBench 及其数据集说明为准；本目录仅提供 OmniMemEval 侧的运行适配、配置和结果组织方式，不重新发布或声明拥有上述数据。

参考来源：

- HuggingFace 数据集：`EverMind-AI/EvoAgentBench`
- GitHub 项目：`EverMind-AI/EvoAgentBench`
- EvoAgentBench 覆盖的五个基础 benchmark：BrowseCompPlus、OmniMath、SWE-Bench、LiveCodeBench、GDPVal

具体任务数量取决于下载时指定的数据 revision。当前数据准备命令使用的公开 `main`
revision 包含：

| 域 | 基础 benchmark | Train | Test |
|---|---|---:|---:|
| `information_retrieval` | BrowseCompPlus | 154 | 65 |
| `reasoning` | OmniMath | 478 | 100 |
| `software_engineering` | SWE-Bench | 87 | 56 |
| `code_implementation` | LiveCodeBench | 182 | 86 |
| `knowledge_work` | GDPVal | 105 | 60 |

## 数据准备

`EverMind-AI/EvoAgentBench` 提供任务 split、内含的 OmniMath 题目和 GDPVal
meta-prompts，但不包含全部上游 benchmark payload。只执行
`huggingface-cli download EverMind-AI/EvoAgentBench`，不会得到 SWE-Bench
parquet、LiveCodeBench 测试数据、GDPVal 元数据/参考文件和解密后的
BrowseComp-Plus 题目。

请在仓库根目录使用统一数据准备命令：

```bash
python scripts/agentbench/prepare_data.py download --domains all
```

该命令会下载并整理以下上游数据：

| 运行目录 | 上游来源 | 准备内容 |
|---|---|---|
| `Reasoning & Problem Decomposition/` | `EverMind-AI/EvoAgentBench` | OmniMath train/test JSONL |
| `BrowseComp-Plus/` | `Tevatron/browsecomp-plus` | 解密题目和 EvoAgentBench split |
| `swebench/` | `princeton-nlp/SWE-bench_Verified` | Verified parquet 和 EvoAgentBench split |
| `livecode/` | `livecodebench/code_generation_lite` | release-v6 原始 JSONL 和 split 任务缓存 |
| `gdpval/` | `openai/gdpval` | `dataset.json`、参考文件、task-ID 映射、meta-prompts 和 split |

当前公开 `main` 数据准备完成后约占 9 GB；首次冷启动还应为 Hugging Face cache
额外预留约 6 GB，以上均不包含 SWE-Bench Docker 镜像。如果 Xet 下载不稳定，
可增加 `--disable-xet`。只准备部分域时可传入逗号分隔列表，例如：

```bash
python scripts/agentbench/prepare_data.py download \
  --domains reasoning,software_engineering
```

准备完成后的运行目录结构为：

```text
data/agentbench/
  BrowseComp-Plus/browsecomp_plus_decrypted.jsonl
  Reasoning & Problem Decomposition/test_set_100/{train,test}.jsonl
  gdpval/{dataset.json,clusters.json,meta_prompts/,reference_files/}
  livecode/{task_split.json,release_v6.json,source/}
  swebench/{task_split.json,test-00000-of-00001.parquet}
  prepare_manifest.json
```

命令支持断点续传，并保留已经存在的真实数据目录。如果旧部署使用了指向其他
EvoAgentBench checkout 的软链接，可以继续使用 `migrate`，也可以通过 `--force`
只移除这些链接并准备自包含的本地数据：

```bash
python scripts/agentbench/prepare_data.py migrate \
  --source /path/to/EvoAgentBench

# 或准备自包含数据：
python scripts/agentbench/prepare_data.py download --domains all --force
```

使用以下命令检查五域文件和 split ID 覆盖：

```bash
python scripts/agentbench/prepare_data.py verify --domains all --deep
```

GDPVal reference files 默认会完整下载。只有在允许评测首次运行时按需下载的情况下，
才使用 `--skip-gdpval-references`。

各域数据路径由 `configs/agentbench/domains/*.yaml` 管理。使用自定义目录时，请传入
`--data-dir`，并同步修改对应 domain YAML。

LiveCodeBench 域依赖官方 verifier 包：

```bash
git clone https://github.com/LiveCodeBench/LiveCodeBench.git ./LiveCodeBench
pip install --no-deps -e ./LiveCodeBench
```

BrowseComp-Plus dense 检索依赖数据预处理、索引文件以及 embedding 服务配置：

```bash
python scripts/agentbench/prepare_data.py download \
  --domains information_retrieval \
  --build-ir-index

python scripts/agentbench/prepare_data.py verify \
  --domains information_retrieval \
  --require-ir-index \
  --deep
```

使用 `--build-ir-index` 前，先在 `.env.agent` 中配置
`IR_EMBEDDING_ENDPOINT`、`IR_EMBEDDING_MODEL` 和 `IR_EMBEDDING_API_KEY`。
openai-compatible 索引必须与评测时使用的 embedding 模型一致，不能直接混用其他模型
生成的预构建索引。

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

如需评测 Hermes，请先按 Hermes 的部署方式安装 CLI，并确保当前 shell 能直接调用：

```bash
hermes --version
hermes chat --help
```

如需运行记忆插件生命周期评测，需要先安装待测记忆插件。以下以 MemOS local plugin 为例：

```bash
# 可选值为 openclaw、hermes 或 all
TARGET_AGENT=hermes
curl -fsSL https://raw.githubusercontent.com/MemTensor/MemOS/main/apps/memos-local-plugin/install.sh \
  | bash -s -- --agent "$TARGET_AGENT"
```

插件安装完成后，还需要完成目标 Runtime 和记忆插件配置，再进行验证。具体配置项取决于待测插件。以 MemOS local plugin 为例，OpenClaw 使用 `~/.openclaw/memos-plugin/`，Hermes 使用 `~/.hermes/memos-plugin/`；需要按插件要求补齐对应 `config.yaml` 中的 LLM、embedding 和凭证配置。Hermes 安装还应生成 `~/.hermes/plugins/memory/memtensor` provider。OmniMemEval 的 Hermes MemOS profile 会在隔离的评测配置中选择评测专用 provider，不会改写用户的全局 Hermes 配置。

运行 AgentBench 前至少确认：

- 待测的 OpenClaw 或 Hermes 已配置可用的模型/provider。
- 记忆插件已安装，并在目标 Runtime 中可被发现。
- 插件所需的凭证、本地路径或服务 endpoint 已配置。
- 下文所述 `.env.agent` 已准备完成。

安装和配置都完成后，根据目标 Runtime 验证 CLI 是否可以正常调用：

```bash
openclaw --version
openclaw agent --help

hermes --version
hermes chat --help
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
- 仓库默认不固定 Hermes 的模型或 provider；Hermes 从 `~/.hermes/config.yaml`
  继承相关配置。若某次评测必须锁定模型，可复制
  `configs/agentbench/agents/hermes.yaml`，加入覆盖项后通过 `--agent-config` 指定。
- memory plugin 协议通过对应的 runtime profile 设置 `runtime.home_mode: isolated_copy` 和 `home_links`，把必要插件目录链接到临时 agent home。

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

上述默认 OpenClaw smoke 会执行完整 `memory_train_backup_test` 生命周期：清理记忆、
训练、发送 verifier feedback、提交 MemOS structured feedback、备份、恢复、测试和
finalize。建议先用它验证本地 OpenClaw、MemOS、模型、judge 和 embedding 配置，
再启动五域完整评测。Hermes + MemOS 默认不执行额外的 structured submit，具体流程见
下一节。

### Hermes + MemOS 插件测评

下面以 Hermes + MemOS 为例说明完整流程。Hindsight、OpenViking、Supermemory
等其他 Hermes memory provider 的安装和配置差异见本目录下对应的产品文档；运行入口
和 `memory_train_backup_test` 协议保持一致。

#### 1. 前置检查

默认 Hermes home 为 `~/.hermes`。如果使用其他目录，应在启动评测前设置
`HERMES_HOME`。Hermes + MemOS lifecycle 会检查 Hermes 配置、MemOS bridge 以及
所需系统命令：

```bash
command -v hermes
command -v node
command -v sqlite3
command -v timeout
command -v curl

test -f "${HERMES_HOME:-$HOME/.hermes}/config.yaml"
test -f "${HERMES_HOME:-$HOME/.hermes}/memos-plugin/config.yaml"
test -f "${HERMES_HOME:-$HOME/.hermes}/memos-plugin/dist/bridge.cjs"
test -e "${HERMES_HOME:-$HOME/.hermes}/plugins/memory/memtensor"
```

其中 `~/.hermes/memos-plugin/config.yaml` 是 MemOS 自身的配置；隔离评测 home 使用的
Hermes 模型和 provider 则继承自 `~/.hermes/config.yaml`。两者用途不同，都需要保证
所依赖的服务和凭证可用。

#### 2. 先运行单域 smoke

先用 reasoning 域的 1 条 train 和 1 条 test 验证完整链路：

```bash
./scripts/run_agentbench_memos_5domain_1each_smoke.sh \
  --agent hermes \
  --domains reasoning \
  --version hermes_memos_reasoning_1each_smoke \
  --tasks-per-split 1 \
  --trials 1 \
  --test-runs 1 \
  --parallel 5 \
  --settle-seconds 0
```

如需定位具体样本，可以直接指定 train/test task：

```bash
./scripts/run_agent_eval.sh \
  --agent hermes \
  --domain reasoning \
  --protocol memory_train_backup_test \
  --memory-plugin memos \
  --version hermes_memos_reasoning_debug \
  --train-task omni_35 \
  --test-task omni_2080 \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

#### 3. 启动正式评测

单域完整 train/test：

```bash
./scripts/run_agent_eval.sh \
  --agent hermes \
  --domain reasoning \
  --protocol memory_train_backup_test \
  --memory-plugin memos \
  --version hermes_memos_reasoning_eval \
  --test-runs 1 \
  --trials 1 \
  --parallel 5
```

五域顺序运行：

```bash
./scripts/run_agentbench_memory_train_backup_test.sh \
  --agent hermes \
  --memory-plugin memos \
  --version hermes_memos_5domain_eval \
  --test-runs 1 \
  --trials 1 \
  --parallel 5
```

也可以通过 `--domains reasoning,code_implementation` 只运行指定域。正式运行前
应先完成上面的单域 smoke；不同域的 verifier、Docker、检索索引和文档工具依赖仍按
“数据准备”一节配置。

#### 4. Hermes + MemOS 实际执行语义

该流程不是直接在用户的全局 MemOS 数据库上训练和测试：

1. lifecycle 先备份用户原有的 Hermes MemOS SQLite，并在结果目录下建立隔离的
   run-scoped `HERMES_HOME` 和 `MEMOS_HOME`；不会清空或覆盖原始
   `~/.hermes/memos-plugin/data/memos.db`。
2. `profiles/hermes/memos.yaml` 为每个任务创建临时 Hermes home，并加载
   `omnimemeval_memos` 评测 provider。每个任务的 stdio proxy 连接同一个 run-scoped
   shared runtime，该 runtime 是本次运行 SQLite 的唯一 MemoryCore writer。
3. train task 完成验证后，verifier feedback 通过 `hermes chat --resume` 发送到同一个
   真实 Hermes session。Hermes MemOS 默认依靠这个正常对话 turn 捕获 feedback，
   因此 lifecycle 中 `structured_submit` 默认为 `false`，不要为该流程额外开启
   `--memos-structured-feedback`。
4. 每个成功的 train 和 test 调用都必须在 MemOS DB 中形成已关闭且非空的 episode；
   只返回 Hermes CLI 成功但没有持久化记忆，会被视为技术失败。
5. 每个 phase 结束后，lifecycle 都会审计 benchmark trial 与 Hermes session 的一一
   对应关系；train 结束后还会排空 embedding/evolution 队列，SQLite 完整性检查通过
   后才生成当前域的训练备份。
6. 每轮 test 前恢复同一份训练备份，再运行 test split。test 完成后执行 finalize、
   checkpoint 和 cleanup，避免不同域或重复测试之间共享意外状态。

结果目录示例：

```text
results/agentbench/hermes-memos-hermes_memos_reasoning_eval-reasoning/
  experiment_config.json
  memory_lifecycle.log
  memory_lifecycle.json
  memory_backups/
    user-global-hermes-*.sqlite3
    memos-hermes-reasoning-*.sqlite3
  train/
  test_run_1/
```

排查 Hermes + MemOS 时，优先检查：

- `memory_lifecycle.log`：validate、clear、settle、backup、restore 和 cleanup 是否成功。
- `memory_lifecycle.json`：各生命周期阶段的命令、时间和退出状态。
- `train/<task>__trial_1/result.json` 和
  `test_run_1/<task>__trial_1/result.json` 中的 `agent_result.memos_capture`：是否记录了
  持久化的 Hermes session/episode。
- 运行期间的 `<run_dir>/runtime/hermes/memos-plugin/logs/`：bridge、embedding 或
  evolution 的详细错误。cleanup 后该隔离目录会被删除，应在失败现场先保留日志。
- Hermes + MemOS 评测使用 headless shared runtime，并设置
  `MEMOS_START_DAEMON=0`，不会启动旧 viewer daemon；排障时应检查 run-scoped 日志和
  SQLite 备份。

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
4. train verifier 完成后，将 feedback 作为同一个 Agent session（OpenClaw 或
   Hermes）的下一轮消息提交。
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

## Agent、Profile 与记忆生命周期配置

配置按三层组织：

```text
configs/agentbench/
  agents/{openclaw,hermes}.yaml
  profiles/openclaw/{plain,memos,everos,hindsight,openviking,supermemory}.yaml
  profiles/hermes/{plain,memos,hindsight,openviking,supermemory}.yaml
  memory_plugins/memos/lifecycle/{openclaw,hermes}.yaml
  memory_plugins/{everos,hindsight,openviking,supermemory}/lifecycle/openclaw.yaml
  memory_plugins/{hindsight,openviking,supermemory}/lifecycle/hermes.yaml
```

`agents` 只声明 Runtime 和模型；`profiles/<agent>` 声明 plain baseline 或
插件如何接入该 Runtime；`memory_plugins/<plugin>/lifecycle/<agent>` 只负责
安全快照、clear、settle、backup、restore 和 cleanup。memory 协议未传
`--profile` 时自动使用与
`--memory-plugin` 同名的 profile，agent/plugin 不匹配会在运行任务前失败。

Hermes profile 不覆盖用户全局的 `platform_toolsets.cli`。adapter 会继承该列表，
仅在非 `knowledge_work` 域删除 `web`，并只在 `information_retrieval` 域将工具面
收缩为本地 search MCP。
除 MemOS 外，产品 profile 也不会替用户选择或启用 memory provider、OpenClaw
plugin slot。测评用户必须先配置目标产品；lifecycle 会在任何破坏性阶段前校验
该选择是否正确。

每个配置负责声明：

- `plugin`：插件标签，会进入结果目录名。
- `backup_dir` 和 `backup_file_template`：备份位置。
- `settle_seconds` 或 `commands.wait_settle`：训练后等待沉淀/进化的逻辑。
- `modes.train/test` 或 `commands.set_mode_*`：训练/测试时插件读写模式。
- `commands.clear/backup/restore`：清理、备份、恢复记忆。
- `execution`：可选生命周期执行策略。OpenClaw MemOS 使用
  `capture_mode: manual_after_feedback`，让任务执行可并发，同时在 verifier feedback
  后显式提交记忆写入；Hermes MemOS 通过 memory provider 自动捕获任务和 feedback。
- `execution.max_parallel`：插件/runtime 组合的可选并发上限。Hermes + MemOS 使用
  shared runtime，目前支持最多 5 个并发 trial；更大的 `--parallel` 会被限制为 5。
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
