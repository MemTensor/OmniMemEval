# Mem0 评测配置与运行

[English](./mem0.md)

本文说明如何使用 OmniMemEval AgentBench 评测 OpenClaw/Hermes + Mem0。
产品及插件安装请按 Mem0 和对应 Agent 的官方文档完成；本文重点说明评测框架所需
配置、隔离方式和运行命令。

## 支持范围

| Agent | 当前支持的 Mem0 后端 | 生命周期配置 |
|---|---|---|
| OpenClaw | `open-source`/OSS + Qdrant server | `configs/agentbench/memory_plugins/mem0/lifecycle/openclaw.yaml` |
| Hermes | OSS + 本地 embedded Qdrant path | `configs/agentbench/memory_plugins/mem0/lifecycle/hermes.yaml` |

当前生命周期不支持 Mem0 Cloud、PGVector 或 OpenClaw/Hermes 以外的集成。若部署
使用其他后端，应复制 lifecycle YAML 和控制脚本后通过
`--memory-plugin-config` 选择自定义实现。

## 评测前配置

### 公共环境

按 [AgentBench 中文文档](../README_zh.md)准备 `.env.agent`、模型服务和
EvoAgentBench 数据。Mem0 使用的 LLM、embedding 和向量库凭证应保存在 Agent
配置或受保护的环境文件中，不要写入仓库。

### OpenClaw

框架不会安装、启用或选择插件。用户应提前完成下列配置：

- `plugins.slots.memory` 为 `openclaw-mem0`。
- `plugins.entries.openclaw-mem0` 存在且已启用。
- `plugins.allow` 存在时包含 `openclaw-mem0`。
- 插件 `mode` 为 `open-source` 或 `oss`。
- `oss.vectorStore.provider` 为 `qdrant`，且配置了可访问的 URL 和 collection。
- `~/.openclaw/npm/node_modules` 中可以解析插件使用的 `mem0ai` OSS SDK。

检查示例：

```bash
openclaw config get plugins.slots.memory
openclaw mem0 status --json
curl -fsS http://127.0.0.1:6333/collections >/dev/null
```

生命周期复制用户配置到运行级 home，并把隔离副本改为：

```text
userId = omnimemeval:<run_id>
train: autoRecall=false, autoCapture=false, skills.triage.enabled=false
test:  autoRecall=true,  autoCapture=false, skills.triage.enabled=false
```

训练结束后，框架通过插件使用的官方 Mem0 OSS SDK，以 `infer=false` 显式提交
每条任务的 expected answer、agent answer、verifier feedback 和 agent
reflection，并等待命令完成。
这样可以避免 skills 模式不自动捕获，以及一次性 OpenClaw CLI 退出时丢失
fire-and-forget 写入。

### Hermes

用户应手动选择 Mem0：

```bash
hermes memory status
hermes config set memory.provider mem0
```

并确认：

- `~/.hermes/config.yaml` 中 `memory.provider: mem0`。
- `~/.hermes/mem0.json` 存在，`mode` 为 `oss`。
- `oss.vector_store.provider` 为 `qdrant`，并使用本地 `path`。
- Hermes 自身 Python 环境可导入 `mem0` 和 `qdrant_client`。

生命周期复制 `mem0.json`，然后为当前 run 重写 `user_id`、`agent_id`、Qdrant
path 和 `MEM0_DIR`。训练后使用 Hermes 环境中的 Mem0 SDK，以 `infer=False`
显式写入已验证反馈；这避免 Hermes 后台同步线程在 CLI 退出前未落库。

embedded Qdrant 对本地目录使用独占锁，因此 Hermes + Mem0 生命周期自动把
`--parallel` 限制为 `1`。

## 两条数据流程验证

OpenClaw：

```bash
./scripts/run_agent_eval.sh \
  --agent openclaw \
  --domain reasoning \
  --protocol memory_train_backup_test \
  --memory-plugin mem0 \
  --version openclaw_mem0_reasoning2_smoke \
  --train-task omni_35,omni_159 \
  --test-task omni_2080,omni_2185 \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

Hermes：

```bash
./scripts/run_agent_eval.sh \
  --agent hermes \
  --domain reasoning \
  --protocol memory_train_backup_test \
  --memory-plugin mem0 \
  --version hermes_mem0_reasoning2_smoke \
  --train-task omni_35,omni_159 \
  --test-task omni_2080,omni_2185 \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

流程依次执行：

1. 校验用户已经选择 Mem0 和当前支持的 Qdrant 模式。
2. 创建运行级 Agent home、用户标识和数据路径。
3. 只清空当前 `omnimemeval:<run_id>` 的记忆。
4. 执行训练任务和 verifier feedback。
5. 显式写入 verified feedback，并拒绝空记忆备份。
6. 等待记忆数量稳定，创建训练备份。
7. 测试前恢复完全相同的训练快照。
8. 测试阶段只召回、不自动写入。
9. cleanup 时删除运行级数据和隔离 home。

OpenClaw 备份保存当前用户在主 collection 和 `_entities` collection 中的完整
points、payload 和 vectors。Hermes 备份保存运行级 embedded Qdrant 目录以及
`MEM0_DIR` 下的 history 数据。

## 正式五域评测

```bash
./scripts/run_agentbench_memory_train_backup_test.sh \
  --agent openclaw \
  --memory-plugin mem0 \
  --version openclaw_mem0_5domain_eval \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

将 `--agent` 改为 `hermes` 即可运行 Hermes；Hermes 的并行度会被限制为 1。

## 结果检查

结果目录：

```text
results/agentbench/<agent>-mem0-<version>-<domain>/
```

重点检查：

- `memory_lifecycle.log` 中 explicit ingest 数量大于 0。
- backup、restore 和 test 阶段均成功。
- OpenClaw restore 后主 collection 的 run-scoped point 数与备份一致。
- Hermes restore 后 `memory_count` 与 manifest 一致。
- cleanup 后当前 run 的数据已清除，用户原有 Mem0 namespace 未被修改。

## 常见问题

- **训练成功但备份为空**：检查 explicit ingest 日志。只打开 skills triage 或
  `autoCapture` 不足以保证一次性 CLI 已完成写入。
- **OpenClaw recall timeout**：检查 embedding endpoint 延迟、模型配置和 Qdrant
  连接；插件的召回超时会导致本轮不注入记忆。
- **Hermes Qdrant storage lock**：不要对同一生命周期目录并行运行多个 Hermes
  trial；框架会把单次运行限制为 `--parallel 1`。
- **维度或 collection 不匹配**：OpenClaw embedding dimension 必须与 Qdrant
  collection 一致，且 lifecycle 配置应指向插件实际使用的 collection。
