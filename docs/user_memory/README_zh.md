<a id="user-memory-evaluation"></a>
# User Memory Evaluation（用户记忆评测）

[English](./README.md) · [文档索引](../README_zh.md) · [评测结果](./results_zh.md)

## 概览

本评测线通过共享评测流程和适配器（adapter）层评估记忆后端（memory backend）
API。通过 `--lib` 选择后端，可以在同一套 benchmark 流程下评估记忆产品、
自托管框架和自定义适配器。

目前有 15 个适配器入口，覆盖 14 种记忆方案（Zep 和 Graphiti 使用不同入口），
支持 LoCoMo、LongMemEval、BEAM、PersonaMem v2 和 HaluMem 五个 benchmark。

首次运行建议按本文顺序阅读。已有环境可以直接查看
[进阶运行](#进阶运行)、[输出与结果](#输出与结果)或[扩展与贡献](#扩展与贡献)。
所有命令均在仓库根目录执行。

## 环境与配置

<a id="1-创建环境"></a>
### 安装依赖

为 User Memory 创建独立环境：

```bash
conda create -n omnimemeval python=3.12 -y
conda activate omnimemeval
python -m pip install -r requirements_user_memory.txt
```

Agent Memory 的依赖应安装在另一个环境中，详见
[Agent Memory 指南](../agent_memory/README_zh.md)。

<a id="2-配置凭证"></a>
### 配置凭证

复制待评测后端的模板，再填写配置。例如：

```bash
cp env_examples/.env.memos .env.memos
```

配置答案模型、适用时的评分模型，以及后端凭证：

- `ANSWER_MODEL`、`ANSWER_API_KEY`、`ANSWER_BASE_URL`
- 使用 LLM 评分时的 `EVAL_MODEL`、`EVAL_API_KEY`、`EVAL_BASE_URL`
- 后端凭证，例如 `MEMOS_API_KEY` 或 `MEM0_API_KEY`

后端特有配置、默认值和各 benchmark 的适用范围见
[模板索引（英文）](../../env_examples/README.md)和
[参数说明（英文）](../../env_examples/PARAMETERS.md)。
PersonaMem v2 使用选项匹配，不使用 LLM 评分。不要提交填写了凭证的文件。

<a id="3-准备数据"></a>
## 数据准备

只下载计划运行的 benchmark：

```bash
# LoCoMo
python data/locomo/prepare_locomo.py

# LongMemEval-S
python data/longmemeval/prepare_longmemeval.py

# BEAM 100K
python data/beam/prepare_beam.py

# PersonaMem v2
python data/personamem_v2/prepare_personamem.py

# HaluMem-Medium
python data/halumem/prepare_halumem.py
```

数据按需下载，不提交到仓库。数据准备与许可证信息见
[benchmark 说明](#benchmarks)、各数据目录 README 和
[第三方声明（英文）](../../THIRD_PARTY_NOTICES.md)。

## 快速开始

配置后端并准备 LongMemEval 数据后，先检查一个对话单元的写入与检索：

```bash
./scripts/run_lme_eval.sh --lib memos --env .env.memos --version smoke_lme \
  --streaming 1 --start-idx 0 --end-idx 0 --to-step 2
```

该命令调用真实后端，可能产生费用。索引范围按对话单元选择，此处只处理第一个单元。
流式模式（streaming）会写入、检索、保存结果，再删除该单元的后端记忆。
`--to-step 2` 在答案生成和评分前停止，因此这不是完整 benchmark 分数。
请使用独立的评测账号。

完整评分运行见[进阶运行](#进阶运行)。各 runner 的 `--help` 也提供选项说明。

## Benchmarks

### LoCoMo

长对话问答，覆盖多跳召回、时序推理和开放域记忆使用。
复现结果排除对抗性问题。
详见[数据与许可证说明（英文）](../../data/locomo/README.md)。

### LongMemEval

跨会话长期记忆，覆盖知识更新、时序推理和偏好。
共享加载器使用 `longmemeval_s_cleaned.json`，对写入和检索一致地清除已知异常特殊字符。
详见[数据与许可证说明（英文）](../../data/longmemeval/README.md)。

### BEAM

大规模长期记忆问答。Runner 支持 100K（约 128K token）、500K、1M 和 10M
规模，使用 LLM 对原子事实点（nugget）评分。
详见[数据与许可证说明（英文）](../../data/beam/README.md)。

<a id="personamem-v2"></a>
### PersonaMem v2

个性化记忆与偏好感知多选问答。将所选选项与标准答案匹配，不使用 LLM 评分。
详见[数据与许可证说明（英文）](../../data/personamem_v2/README.md)。

### HaluMem

记忆鲁棒性评测，覆盖幻觉、记忆边界、冲突、多跳推理和动态更新。
详见[数据与许可证说明（英文）](../../data/halumem/README.md)。

## 进阶运行

<a id="4-运行评测"></a>
### 完整 benchmark 运行

以下每条命令运行对应 benchmark 配置的数据集，不是小规模检查。
请先准备相应数据和凭证。

```bash
./scripts/run_locomo_eval.sh --lib memos --env .env.memos
./scripts/run_lme_eval.sh --lib memos --env .env.memos
./scripts/run_beam_eval.sh --lib memos --env .env.memos
./scripts/run_pmv2_eval.sh --lib memos --env .env.memos
./scripts/run_halumem_eval.sh --lib memos --env .env.memos
```

替换 `memos` 和环境文件即可切换后端。通过 `--version <name>` 隔离运行产物。
共享默认值与 benchmark 特有选项统一维护在
[参数说明（英文）](../../env_examples/PARAMETERS.md)。

<a id="评测流程"></a>
### 流程与评分

标准检索模式的流程如下：

```text
数据集 → 写入（add）→ 检索 → 答案生成 → 评估 → 指标计算 → 报告
             ↕          ↕
                记忆后端
```

- 写入和检索调用所选适配器的 `add()` 和 `search()`。
- 答案生成通常使用配置的 OpenAI-compatible 答案模型。
  后端特有的直接回答模式可能替代或跳过该阶段；提交结果时必须披露所选模式。
- LoCoMo、LongMemEval 和 HaluMem 使用 LLM 评分，并计算已实现的辅助指标；
  BEAM 使用基于 LLM 的 nugget 评分；PersonaMem v2 使用选项匹配。
- Runner 支持阶段检查点和重放（replay）。部分运行不等于完整 benchmark 结果。

### 流式与部分运行

LongMemEval、BEAM、PersonaMem v2 和 HaluMem 支持 `--streaming 1`，
对每个数据单元依次写入、检索、保存、删除，再处理下一个单元。
支持 `--start-idx`、`--end-idx`、`--restart-unit`、`--no-resume` 和
`--skip-failed-streaming`；具体语义以所选 runner 的 `--help` 为准。

`--to-step 2` 可用于写入与检索检查，但它只限制阶段，不限制数据量。
如果没有显式使用受支持的子集选择参数，仍可能处理整个数据集。
[快速开始](#快速开始)中的 LongMemEval 示例同时限制了阶段和数据范围。

### 重放

在已有运行中继续后续阶段时，使用相同后端和版本，并保留所需的前序产物。
例如，从答案生成阶段开始重新执行：

```bash
./scripts/run_lme_eval.sh --lib memos --env .env.memos \
  --version existing_run --from-step 3
```

将 `existing_run` 替换为已有运行版本。显式指定 `--from-step`
会强制重新执行该阶段及后续阶段。

相比之下，`--replay` 加载已有实验配置，并创建带时间戳的新运行版本，
不会把前序产物复制到新目录。完整重放示例：

```bash
./scripts/run_lme_eval.sh --env .env.memos \
  --replay results/lme/{LIB}-{VERSION}/
```

其他 runner 也接受 `--replay`，传入其对应结果目录即可。
请将 `{LIB}-{VERSION}` 替换为已有后端标识和运行版本，使用上面的
`--env FILE --replay DIR` 顺序；后端标识和运行设置会从已保存配置恢复，
无需额外添加 `--lib`。只使用可信配置：重放会将 `experiment_config.sh` 作为 Shell 代码执行。
提交分数时应记录配置变化，并披露检查点复用或重新评估的情况。

<a id="清理"></a>
### 清理

先预览评测用户 ID，再删除某次运行写入的后端记忆：

```bash
./scripts/run_memory_clear.sh --lib memos --env .env.memos --version '<run-version>' --datasets locomo,lme,beam,pmv2,hm --dry-run
./scripts/run_memory_clear.sh --lib memos --env .env.memos --version '<run-version>' --datasets locomo,lme,beam,pmv2,hm --yes
```

请将 `<run-version>` 替换为目标运行版本。`--dry-run` 不删除数据；实际删除必须传入 `--yes`。
清理非默认 BEAM 或 HaluMem 数据时，根据需要使用可重复的 `--beam-scale`
或 `--halumem-variant`。请核对账号和运行版本。

<a id="benchmark-结果"></a>
## 输出与结果

产物按后端标识和运行版本写入以下目录：

| Benchmark | 结果目录 |
|---|---|
| LoCoMo | `results/locomo/{LIB}-{VERSION}/` |
| LongMemEval | `results/lme/{LIB}-{VERSION}/` |
| BEAM | `results/beam/{LIB}-{VERSION}/` |
| PersonaMem v2 | `results/pmv2/{LIB}-{VERSION}/` |
| HaluMem | `results/halumem/{LIB}-{VERSION}/` |

[评测结果](./results_zh.md)提供公开快照、评测配置、汇总规则和上下文 token 定义的入口。
统一流程复现结果（Reproduced Results）与公开参考结果（Published Reference Results）
的证据要求和可比性不同，不应混用。

<a id="支持的-memory-backend"></a>
## 扩展与贡献

适配器层提供通用写入、检索操作和用户级清理。支持以下后端标识：

| `--lib` | 后端 |
|---|---|
| `memos` | MemOS |
| `mem0` | Mem0 |
| `zep` | Zep |
| `supermemory` | Supermemory |
| `everos` | EverOS |
| `letta` | Letta |
| `hindsight` | Hindsight |
| `graphiti` | Zep Graphiti 本地/自托管 |
| `cognee` | Cognee |
| `viking` | Viking Memory |
| `memori` | Memori |
| `memmachine` | MemMachine |
| `memorylake` | MemoryLake |
| `backboard` | Backboard.io |
| `mem9` | mem9 |

接入、配置、隔离与测试要求见
[适配器清单（英文）](../../CONTRIBUTING.md#adding-a-user-memory-adapter)，
分数提交要求见
[结果提交规范（英文）](../../CONTRIBUTING.md#submitting-benchmark-results)。
<a id="项目结构"></a>
代码、配置与数据准备脚本的位置见根目录的[仓库结构](../../README_zh.md#目录结构)。

<a id="验证"></a>
## 验证与许可证

在独立的 Python 3.11 或 3.12 开发环境运行离线检查：

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/check_doc_links.py
```

[测试指南（英文）](../../scripts/tests/README.md)区分离线测试、仓库检查和真实后端集成测试。
离线 CI 不验证 benchmark 分数，也不调用真实服务。

<a id="license"></a>
代码许可证见 [LICENSE](../../LICENSE)。外部数据集保留上游许可证，详见
[THIRD_PARTY_NOTICES.md（英文）](../../THIRD_PARTY_NOTICES.md)。
