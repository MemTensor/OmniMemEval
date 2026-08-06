# OmniMemEval

[English](./README.md)

OmniMemEval 是面向记忆系统和记忆增强 Agent 的评测框架，当前包含两条互补的评测线：

| 评测线 | 评测对象 | Benchmark / 任务域 | 文档 |
|---|---|---|---|
| User Memory Evaluation | 通过 `add()` 和 `search()` 暴露能力的 memory backend API | LoCoMo、LongMemEval、BEAM、PersonaMem v2、HaluMem | [docs/user_memory/README_zh.md](./docs/user_memory/README_zh.md) |
| Agent Memory Evaluation | 安装记忆插件后的 Agent Runtime | AgentBench domains：reasoning、information retrieval、knowledge work、code implementation、software engineering | [docs/agent_memory/README_zh.md](./docs/agent_memory/README_zh.md) |

## 评测线

### User Memory Evaluation

User Memory Evaluation 通过统一 adapter 层评估 memory backend 系统能力。使用 `--lib` 可以在同一套 benchmark pipeline 下切换不同 memory 产品、自托管 memory framework 或自定义 adapter。

该流程覆盖写入、检索、答案生成、LLM-as-Judge 评估、指标聚合和报告生成。结果写入 `results/<benchmark>/<LIB>-<VERSION>/`。

入口文档：[docs/user_memory/README_zh.md](./docs/user_memory/README_zh.md)

### Agent Memory Evaluation

Agent Memory Evaluation 评估安装记忆插件后的 Agent Runtime 在任务域中的实际表现。当前实现基于 AgentBench，评估 OpenClaw 在五个任务域上的能力。记忆插件协议包含记忆清理、训练、沉淀、备份、恢复和测试执行。

结果写入 `results/agentbench/`。

入口文档：[docs/agent_memory/README_zh.md](./docs/agent_memory/README_zh.md)

## 安装

User Memory Evaluation 和 Agent Memory Evaluation 使用两套独立环境。User Memory Evaluation 可在仓库根目录创建基础 Python 环境：

```bash
conda create -n omnimemeval python=3.12 -y
conda activate omnimemeval
pip install -r requirements_user_memory.txt
```

Agent Memory Evaluation 推荐使用独立的 `agentmem` 环境，因为 AgentBench 任务域依赖 OpenClaw 和额外领域依赖：

```bash
conda create -n agentmem python=3.12 -y
conda activate agentmem
python -m pip install -U pip
pip install -r requirements_agentbench.txt
```

OpenClaw、系统包和各 domain 的额外安装步骤详见 [docs/agent_memory/README_zh.md](./docs/agent_memory/README_zh.md)。

不同评测线存在额外依赖：

- User Memory Evaluation：memory backend 凭证、ANSWER/EVAL LLM 凭证，以及各 benchmark 的数据准备。详见 [docs/user_memory/README_zh.md](./docs/user_memory/README_zh.md)。
- Agent Memory Evaluation：OpenClaw CLI、AgentBench 数据，以及 Docker、LiveCodeBench、BrowseComp-Plus index 等任务域依赖。详见 [docs/agent_memory/README_zh.md](./docs/agent_memory/README_zh.md)。

## 快速开始

### User Memory Evaluation

```bash
cp env_examples/.env.memos .env.memos
python data/locomo/prepare_locomo.py
./scripts/run_locomo_eval.sh --lib memos --env .env.memos
```

### Agent Memory Evaluation

```bash
cp env_examples/.env.agent .env.agent
mkdir -p data/agentbench
huggingface-cli download EverMind-AI/EvoAgentBench \
  --repo-type dataset \
  --local-dir ./data/agentbench
./scripts/run_agent_eval.sh \
  --agent openclaw \
  --domain reasoning \
  --protocol test_only \
  --version smoke_agentbench \
  --trials 1 \
  --parallel 1
```

## 结果

- User Memory 公开结果快照：[docs/benchmark-results.md](./docs/benchmark-results.md)
- AgentBench 评测结果：[docs/agent_memory/eval_res_zh.md](./docs/agent_memory/eval_res_zh.md)

## 目录结构

```text
configs/
  agentbench/                 # Agent Memory Evaluation 配置
data/
  locomo/ longmemeval/ beam/  # User Memory benchmark 数据准备
  personamem_v2/ halumem/
docs/
  user_memory/                # User Memory Evaluation 中英文文档
  agent_memory/               # Agent Memory Evaluation 中英文文档和结果
  benchmark-results.md        # User Memory 公开结果快照
env_examples/                 # 环境变量模板和参数说明
scripts/
  agentbench/                 # AgentBench runner 实现
  client_factory/             # User Memory backend 适配器
  locomo/ longmemeval/ beam/  # User Memory benchmark 流水线
  personamem_v2/ halumem/
  tests/                      # 测试套件
  utils/                      # 通用工具
requirements_user_memory.txt  # User Memory Evaluation 依赖
requirements_agentbench.txt   # Agent Memory Evaluation 依赖
```

## License

见 [LICENSE](./LICENSE)。第三方 benchmark 数据保留其上游许可证；OmniMemEval 代码许可证不重新授权外部数据集。详见 [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md)。
