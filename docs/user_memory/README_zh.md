# User Memory Evaluation

[English](./README.md)

[文档索引](../README_zh.md)

本文档说明 OmniMemEval 的 User Memory Evaluation 评测线。该评测线通过统一 benchmark pipeline 和 adapter 层评估 memory backend API。用户可以通过 `--lib` 切换不同 memory backend，在同一套 benchmark 流程下对比主流 memory 产品、自托管 memory framework 和自定义 adapter。

adapter 层通过 15 个入口覆盖 14 种主流 memory 方案，包括 MemOS、Mem0、Zep/Graphiti、Supermemory、EverOS、Letta、Hindsight、Cognee、Viking Memory、Memori、MemMachine、MemoryLake、Backboard.io 和 mem9。该评测线支持 LoCoMo、LongMemEval、BEAM、PersonaMem v2 和 HaluMem 5 个评测任务，覆盖长期记忆系统中几类关键能力：对话记忆、跨 session 更新、超长上下文检索、个性化偏好，以及在
幻觉、冲突和动态更新场景下的鲁棒性。

Benchmark 覆盖：

- [LoCoMo](#locomo)：长对话 QA，覆盖多跳召回、时序推理和开放域记忆使用。
- [LongMemEval](#longmemeval)：跨 session 长期记忆，覆盖知识更新、时序推理和偏好问题。
- [BEAM](#beam)：128K 到 10M token 上下文规模下的大规模记忆检索评测。
- [PersonaMem v2](#personamem-v2)：面向用户偏好、敏感信息和个性化行为的记忆评测。
- [HaluMem](#halumem)：面向记忆幻觉、边界识别、冲突处理、多跳推理和动态更新的鲁棒性评测。

## 评测流程

OmniMemEval 的 benchmark 链路共享同一套阶段：

```text
┌──────────────────┐
│ Benchmark 数据   │
│ 数据集特定格式   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐      add()       ┌──────────────────┐
│ 1. 写入          ├─────────────────▶│ Memory Backend   │
│ 对话内容         │                  │ 由 --lib 选择    │
└────────┬─────────┘                  └────────┬─────────┘
         │                                     │
         ▼                                     │ search()
┌──────────────────┐                           │
│ 2. 检索          │◀──────────────────────────┘
│ 召回上下文       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐      ANSWER LLM
│ 3. 生成答案      ├─────────────────▶ 生成答案
│ ANSWER 模型      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐      EVAL LLM / NLP
│ 4. 评估          ├─────────────────▶ 评估记录
│ LLM-as-Judge     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 5. 指标计算      │
│ 准确率/延迟      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 6. 报告          │
│ Markdown/结果    │
└──────────────────┘
```

- 写入阶段调用 memory client 的 `add()`。
- 检索阶段调用 memory client 的 `search()`。
- 答案生成阶段使用 OpenAI-compatible ANSWER 模型生成答案。
- 评估阶段使用 OpenAI-compatible EVAL 模型做 LLM-as-Judge，并计算 NLP 指标。
- 指标和报告写入 `results/<benchmark>/<LIB>-<VERSION>/`。

Shell runner 和 Python 阶段都支持 checkpoint/resume。

## 快速开始

### 1. 创建环境

```bash
conda create -n omnimemeval python=3.12 -y
conda activate omnimemeval
pip install -r requirements_user_memory.txt
```

AgentBench 依赖放在 `requirements_agentbench.txt`，应安装在独立的 Agent Memory 环境中。

### 2. 配置凭证

从产品模板开始：

```bash
cp env_examples/.env.memos .env.memos
```

填写 memory 产品凭证和 ANSWER/EVAL LLM 配置：

- `ANSWER_MODEL`, `ANSWER_API_KEY`, `ANSWER_BASE_URL`
- `EVAL_MODEL`, `EVAL_API_KEY`, `EVAL_BASE_URL`
- 产品侧凭证，例如 `MEMOS_API_KEY` 或 `MEM0_API_KEY`

完整参数见 [env_examples/README.md](../../env_examples/README.md) 和
[env_examples/PARAMETERS.md](../../env_examples/PARAMETERS.md)。

### 3. 准备数据

```bash
# LoCoMo
python data/locomo/prepare_locomo.py

# LongMemEval S
python data/longmemeval/prepare_longmemeval.py

# BEAM 100K
python data/beam/prepare_beam.py

# PersonaMem v2
python data/personamem_v2/prepare_personamem.py

# HaluMem Medium
python data/halumem/prepare_halumem.py
```

Benchmark 数据按需下载，不提交到仓库。上游数据许可证和再分发说明见
[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) 和各数据目录 README。

### 4. 运行评测

```bash
./scripts/run_locomo_eval.sh --lib memos --env .env.memos
./scripts/run_lme_eval.sh --lib memos --env .env.memos
./scripts/run_beam_eval.sh --lib memos --env .env.memos
./scripts/run_pmv2_eval.sh --lib memos --env .env.memos
./scripts/run_halumem_eval.sh --lib memos --env .env.memos
```

常用参数：

| 参数 | 用途 |
|------|------|
| `--version <name>` | 结果目录后缀，默认 `omnimemeval_<date>`。 |
| `--from-step N` / `--to-step N` | 只运行部分阶段。 |
| `--replay <result_dir>` | 从已有结果目录重算后续阶段。 |
| `--top-k N` | 检索数量，覆盖 env 中的 `TOPK`。 |
| `--llm-workers N` | Answer/Eval LLM 并发数。 |
| `--allow-empty-search 1` | 允许 raw memory 为空的 search 结果通过。 |
| `--skip-failed-search 1` | search 失败时标记 skipped，而不是失败退出。 |
| `--skip-failed-answer 1` | answer 失败时标记 skipped，而不是失败退出。 |
| `--skip-failed-judge 1` | judge 失败时标记 skipped，而不是失败退出。 |

LongMemEval、BEAM、PersonaMem v2 和 HaluMem 支持 streaming 模式。在
streaming 模式下，OmniMemEval 会对每个 benchmark unit 依次执行 add、search、
保存结果和 delete，再进入下一个 unit。对应 runner 均通过 `--streaming 1` 启用：

```bash
./scripts/run_lme_eval.sh --lib memos --env .env.memos --streaming 1
./scripts/run_beam_eval.sh --lib memos --env .env.memos --streaming 1
./scripts/run_pmv2_eval.sh --lib memos --env .env.memos --streaming 1
./scripts/run_halumem_eval.sh --lib memos --env .env.memos --streaming 1
```

Streaming 模式可配合 `--start-idx`、`--end-idx`、`--restart-unit`、
`--no-resume` 和 `--skip-failed-streaming` 使用。

最小 smoke 命令：

```bash
# LoCoMo：只跑 ingestion 和 search
./scripts/run_locomo_eval.sh --lib memos --env .env.memos --version smoke_locomo --to-step 2

# LongMemEval：只跑一个 streaming conversation 到 search
./scripts/run_lme_eval.sh --lib memos --env .env.memos --version smoke_lme \
  --streaming 1 --start-idx 0 --end-idx 0 --to-step 2

# BEAM：默认 100K scale，只跑 ingestion 和 search
./scripts/run_beam_eval.sh --lib memos --env .env.memos --version smoke_beam --to-step 2

# PersonaMem v2：只跑 ingestion 和 search
./scripts/run_pmv2_eval.sh --lib memos --env .env.memos --version smoke_pmv2 --to-step 2

# HaluMem：只跑 ingestion 和 search
./scripts/run_halumem_eval.sh --lib memos --env .env.memos --version smoke_hm --to-step 2
```

从已有结果目录 replay 后续阶段：

```bash
./scripts/run_locomo_eval.sh --lib memos --env .env.memos --replay results/locomo/{LIB}-{VERSION}/
./scripts/run_lme_eval.sh --lib memos --env .env.memos --replay results/lme/{LIB}-{VERSION}/
./scripts/run_beam_eval.sh --lib memos --env .env.memos --replay results/beam/{LIB}-{VERSION}/
./scripts/run_pmv2_eval.sh --lib memos --env .env.memos --replay results/pmv2/{LIB}-{VERSION}/
./scripts/run_halumem_eval.sh --lib memos --env .env.memos --replay results/halumem/{LIB}-{VERSION}/
```

## Benchmark 结果

公开评测结果见 [benchmark results（英文）](./results.md)。该文档整理了
当前公开 benchmark 链路在 OmniMemEval 统一评测配置下复现的分数、context token
指标、部署说明、公开参考分数和复现命令。

## 支持的 Memory Backend

公开 adapter 层统一暴露 `add()` / `search()` / `delete()` 接口，覆盖主流
memory 产品和自托管 memory framework：

通过 `--lib` 可以在不改 benchmark 阶段、prompt 流程和指标计算的前提下，对不同
memory 方案运行同一套评测。

| `--lib` | Adapter |
|---------|---------|
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

## Benchmarks

<a id="locomo"></a>
### LoCoMo

LoCoMo 评估长对话记忆、多跳推理和时序记忆。数据说明见
[data/locomo/README.md](../../data/locomo/README.md)。

```bash
./scripts/run_locomo_eval.sh --lib memos --env .env.memos
```

结果目录：`results/locomo/{LIB}-{VERSION}/`

<a id="longmemeval"></a>
### LongMemEval

LongMemEval 评估跨 session 长期记忆。OmniMemEval 通过共享 loader 读取
`longmemeval_s_cleaned.json`，并对 ingestion 和 search 使用同一份清洗后的数据。
数据说明见 [data/longmemeval/README.md](../../data/longmemeval/README.md)。

```bash
./scripts/run_lme_eval.sh --lib memos --env .env.memos
```

结果目录：`results/lme/{LIB}-{VERSION}/`

<a id="beam"></a>
### BEAM

BEAM 评估 128K、500K、1M 和 10M token 规模下的长期记忆能力，采用
per-nugget LLM-as-Judge 评分。数据说明见
[data/beam/README.md](../../data/beam/README.md)。

```bash
./scripts/run_beam_eval.sh --lib memos --env .env.memos
```

结果目录：`results/beam/{LIB}-{VERSION}/`

<a id="personamem-v2"></a>
### PersonaMem v2

PersonaMem v2 评估个性化记忆和偏好感知多选问答。数据说明见
[data/personamem_v2/README.md](../../data/personamem_v2/README.md)。

```bash
./scripts/run_pmv2_eval.sh --lib memos --env .env.memos
```

结果目录：`results/pmv2/{LIB}-{VERSION}/`

<a id="halumem"></a>
### HaluMem

HaluMem 评估记忆幻觉、冲突处理、动态更新和记忆边界鲁棒性。数据说明见
[data/halumem/README.md](../../data/halumem/README.md)。

```bash
./scripts/run_halumem_eval.sh --lib memos --env .env.memos
```

结果目录：`results/halumem/{LIB}-{VERSION}/`

## 清理

删除某次评测写入的后端 memory：

```bash
./scripts/run_memory_clear.sh --lib memos --env .env.memos --version <name> --datasets locomo,lme,beam,pmv2,hm --dry-run
./scripts/run_memory_clear.sh --lib memos --env .env.memos --version <name> --datasets locomo,lme,beam,pmv2,hm --yes
```

`--dry-run` 只打印目标 id；实际删除必须显式传入 `--yes`。清理非默认 BEAM 或 HaluMem 数据时，可配合可重复的 `--beam-scale` 和 `--halumem-variant` 使用。

## 项目结构

```text
OmniMemEval/
├── data/
│   ├── beam/
│   ├── halumem/
│   ├── locomo/
│   ├── longmemeval/
│   └── personamem_v2/
├── docs/
│   └── user_memory/
│       ├── README.md
│       ├── README_zh.md
│       └── results.md
├── env_examples/
├── scripts/
│   ├── client_factory/
│   ├── beam/
│   ├── halumem/
│   ├── locomo/
│   ├── longmemeval/
│   ├── personamem_v2/
│   ├── tests/
│   ├── utils/
│   ├── run_beam_eval.sh
│   ├── run_halumem_eval.sh
│   ├── run_locomo_eval.sh
│   ├── run_lme_eval.sh
│   ├── run_pmv2_eval.sh
│   └── run_memory_clear.sh
├── README.md
├── README_zh.md
├── THIRD_PARTY_NOTICES.md
├── requirements_user_memory.txt  # User Memory 依赖
└── requirements_agentbench.txt   # AgentBench 依赖
```

## 验证

```bash
git ls-files -z '*.sh' | xargs -0 -n 1 bash -n
conda run -n omnimemeval python -m compileall -q scripts data
conda run -n omnimemeval python -m pip install -r requirements-dev.txt
conda run -n omnimemeval python -m pytest -q
```

离线测试、CI 检查和后端集成 smoke test 的区别见
[测试指南](../../scripts/tests/README.md)。

## License

见 [LICENSE](../../LICENSE)。第三方 benchmark 数据保留其上游许可证；OmniMemEval
代码许可证不重新授权外部数据集。详见
[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)。
