# MemEval

[English](./README.md)

MemEval 是 memory system API 的标准化评测框架。本次开源版本只包含
LoCoMo 和 LongMemEval 两条 benchmark 评测链路，同时保留公开的 memory
adapter 层。

MemEval 不是 memory 服务，也不是 MemOS 部署包。MemOS 是被评测的 memory
system；MemEval 是评测框架，用统一 adapter 对 MemOS 和其他 memory backend
运行可复现评测。

支持的 benchmark：

- [LoCoMo](#locomo)：长对话 QA，多跳和时序记忆。
- [LongMemEval](#longmemeval)：跨 session 长期记忆，500 个问题。

## 评测流程

两条链路共享同一套阶段：

```text
Ingest -> Search -> Answer Response -> Eval -> Metric -> Report
```

- Ingest 调用 memory client 的 `add()`。
- Search 调用 memory client 的 `search()`。
- Answer 使用 OpenAI-compatible ANSWER 模型生成答案。
- Eval 使用 OpenAI-compatible EVAL 模型做 LLM-as-Judge，并计算 NLP 指标。
- Metric 和 Report 写入 `results/<benchmark>/<LIB>-<VERSION>/`。

Shell runner 和 Python 阶段都支持 checkpoint/resume。

## 快速开始

### 1. 创建环境

```bash
conda create -n memeval python=3.12 -y
conda activate memeval
pip install -r requirements.txt
```

### 2. 配置凭证

从产品模板开始：

```bash
cp env_examples/.env.memos .env.memos
```

填写 memory 产品凭证和 ANSWER/EVAL LLM 配置：

- `ANSWER_MODEL`, `ANSWER_API_KEY`, `ANSWER_BASE_URL`
- `EVAL_MODEL`, `EVAL_API_KEY`, `EVAL_BASE_URL`
- 产品侧凭证，例如 `MEMOS_API_KEY` 或 `MEM0_API_KEY`

完整参数见 [env_examples/README.md](./env_examples/README.md) 和
[env_examples/PARAMETERS.md](./env_examples/PARAMETERS.md)。

### 3. 准备数据

```bash
# LoCoMo
python data/locomo/prepare_locomo.py

# LongMemEval S
python data/longmemeval/prepare_longmemeval.py
```

Benchmark 数据按需下载，不提交到仓库。LoCoMo 使用 CC BY-NC 4.0 许可证，
LongMemEval 使用 MIT 许可证；详见
[THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md) 和各数据目录 README。

### 4. 运行评测

```bash
./scripts/run_locomo_eval.sh --lib memos --env .env.memos
./scripts/run_lme_eval.sh --lib memos --env .env.memos
```

常用参数：

| 参数 | 用途 |
|------|------|
| `--version <name>` | 结果目录后缀，默认 `memeval_<date>`。 |
| `--from-step N` / `--to-step N` | 只运行部分阶段。 |
| `--replay <result_dir>` | 从已有结果目录重算后续阶段。 |
| `--top-k N` | 检索数量，覆盖 env 中的 `TOPK`。 |
| `--llm-workers N` | Answer/Eval LLM 并发数。 |
| `--allow-empty-search 1` | 允许 raw memory 为空的 search 结果通过。 |
| `--skip-failed-search 1` | search 失败时标记 skipped，而不是失败退出。 |
| `--skip-failed-answer 1` | answer 失败时标记 skipped，而不是失败退出。 |
| `--skip-failed-judge 1` | judge 失败时标记 skipped，而不是失败退出。 |

LongMemEval 支持按 conversation streaming：

```bash
./scripts/run_lme_eval.sh --lib memos --env .env.memos --streaming 1
```

Streaming 会对每个 conversation 执行 add、search、保存结果、delete，再进入下一个
conversation。可配合 `--start-idx`、`--end-idx`、`--restart-unit`、
`--no-resume` 和 `--skip-failed-streaming` 使用。

最小 smoke 命令：

```bash
# LoCoMo：只跑 ingestion 和 search
./scripts/run_locomo_eval.sh --lib memos --env .env.memos --version smoke_locomo --to-step 2

# LongMemEval：只跑一个 streaming conversation 到 search
./scripts/run_lme_eval.sh --lib memos --env .env.memos --version smoke_lme \
  --streaming 1 --start-idx 0 --end-idx 0 --to-step 2
```

从已有结果目录 replay 后续阶段：

```bash
./scripts/run_locomo_eval.sh --lib memos --env .env.memos --replay results/locomo/{LIB}-{VERSION}/
./scripts/run_lme_eval.sh --lib memos --env .env.memos --replay results/lme/{LIB}-{VERSION}/
```

## 支持的 Memory Adapter

公开 adapter 层统一暴露 `add()` / `search()` / `delete()` 接口，支持以下
`--lib`：

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
[data/locomo/README.md](./data/locomo/README.md)。

```bash
./scripts/run_locomo_eval.sh --lib memos --env .env.memos
```

结果目录：`results/locomo/{LIB}-{VERSION}/`

<a id="longmemeval"></a>
### LongMemEval

LongMemEval 评估跨 session 长期记忆。MemEval 通过共享 loader 读取
`longmemeval_s_cleaned.json`，并对 ingestion 和 search 使用同一份清洗后的数据。

```bash
./scripts/run_lme_eval.sh --lib memos --env .env.memos
```

结果目录：`results/lme/{LIB}-{VERSION}/`

## 清理

删除某次评测写入的后端 memory：

```bash
./scripts/run_memory_clear.sh --lib memos --env .env.memos --version <name> --datasets locomo,lme --dry-run
./scripts/run_memory_clear.sh --lib memos --env .env.memos --version <name> --datasets locomo,lme --yes
```

`--dry-run` 只打印目标 id；实际删除必须显式传入 `--yes`。

## 项目结构

```text
MemEval/
├── data/
│   ├── locomo/
│   └── longmemeval/
├── env_examples/
├── scripts/
│   ├── client_factory/
│   ├── locomo/
│   ├── longmemeval/
│   ├── tests/
│   ├── utils/
│   ├── run_locomo_eval.sh
│   ├── run_lme_eval.sh
│   └── run_memory_clear.sh
├── README.md
├── README_zh.md
├── THIRD_PARTY_NOTICES.md
└── requirements.txt
```

## 验证

```bash
bash -n scripts/_experiment_utils.sh scripts/run_locomo_eval.sh scripts/run_lme_eval.sh scripts/run_memory_clear.sh
conda run -n memeval python -m compileall -q scripts
conda run -n memeval python -m unittest discover -s scripts/tests -p 'test_*.py'
```

## License

见 [LICENSE](./LICENSE)。第三方 benchmark 数据保留其上游许可证；MemEval
代码许可证不重新授权外部数据集。详见
[THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md)。
