# Supermemory 评测配置与运行

[English](./supermemory.md)

本文说明如何使用 OmniMemEval AgentBench 评测 OpenClaw/Hermes + Supermemory。
产品安装请参考 [Supermemory 插件主页](https://supermemory.ai/plugins/)、
[OpenClaw 插件项目](https://github.com/supermemoryai/openclaw-supermemory)和
[Hermes Memory Providers 文档](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory-providers/)。

## 支持范围

| Agent | 支持状态 | 生命周期配置 |
|---|---|---|
| OpenClaw | 已支持并完成实机验证 | `configs/agentbench/memory_plugins/supermemory/lifecycle/openclaw.yaml` |
| Hermes | 已支持并完成实机验证 | `configs/agentbench/memory_plugins/supermemory/lifecycle/hermes.yaml` |

Supermemory 可以使用云服务或兼容的自托管服务。无论采用哪种方式，运行评测的机器都
必须能够访问配置中的 API endpoint。

## 评测前配置

### 1. 准备公共环境

按 [AgentBench 中文文档](../README_zh.md)准备 `.env.agent`、模型服务和
EvoAgentBench 数据。不要把 Supermemory API key 写入仓库或评测结果目录。

### 2. OpenClaw 配置

框架不会替用户安装、启用或选择插件。用户配置必须满足：

- `plugins.enabled` 不能为 `false`。
- `plugins.allow` 存在时包含 `openclaw-supermemory`。
- `plugins.slots.memory` 为 `openclaw-supermemory`。
- `plugins.entries.openclaw-supermemory` 存在且已启用。
- 插件配置或环境变量中包含可用的 API key 和 base URL。
- `~/.openclaw/npm/node_modules` 中可以解析 `supermemory` Node SDK。

检查示例：

```bash
openclaw supermemory status
openclaw config get plugins.slots.memory
test -d ~/.openclaw/npm/node_modules
```

评测时框架会强制把隔离配置改为：

```text
containerTag = omnimemeval_<run_id>
train: autoRecall=true, autoCapture=true
test:  autoRecall=true, autoCapture=false
```

用户原配置中的普通 container 不会被清空。生命周期拒绝管理不以
`omnimemeval_` 开头的 container。

### 3. Hermes 配置

用户必须提前选择 Supermemory：

```bash
hermes memory status
hermes config set memory.provider supermemory
```

应满足：

- `~/.hermes/config.yaml` 中 `memory.provider: supermemory`。
- `~/.hermes/supermemory.json` 存在并配置了 endpoint/container 等信息。
- API key 位于 Hermes `.env`、运行环境或部署约定的安全凭证文件中。

检查示例：

```bash
hermes memory status
test -f ~/.hermes/supermemory.json
```

Hermes 生命周期会复制 `supermemory.json`，并将隔离副本的 `container_tag` 改为
当前 run 的 `omnimemeval_<run_id>`。

如果自托管 endpoint 或凭证获取方式与默认实现不同，应复制对应 lifecycle YAML，
必要时调整 `hermes_supermemory_ctl.py` 的配置来源，再通过
`--memory-plugin-config` 使用自定义配置。

## 两条数据流程验证

OpenClaw：

```bash
./scripts/run_agent_eval.sh \
  --agent openclaw \
  --domain reasoning \
  --protocol memory_train_backup_test \
  --memory-plugin supermemory \
  --version openclaw_supermemory_reasoning2_smoke \
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
  --memory-plugin supermemory \
  --version hermes_supermemory_reasoning2_smoke \
  --train-task omni_35,omni_159 \
  --test-task omni_2080,omni_2185 \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

生命周期会：

1. 校验用户已选择 Supermemory。
2. 创建运行级 Agent home 和 container tag。
3. 清空本次 run 的 container。
4. 训练阶段开启自动召回和自动写入。
5. 等待文档完成 extraction、chunking、embedding 和 indexing。
6. 导出运行级记忆并生成训练备份。
7. 测试前导入备份并再次等待索引完成。
8. 测试阶段关闭自动写入，只保留召回。
9. cleanup 时删除运行级 container 和隔离 home。

OpenClaw 备份以 document JSONL 为主；Hermes 备份同时保存 documents 和生成后的
memories，恢复时优先恢复 memories。

## 正式评测

OpenClaw 五域：

```bash
./scripts/run_agentbench_memory_train_backup_test.sh \
  --agent openclaw \
  --memory-plugin supermemory \
  --version openclaw_supermemory_5domain_eval \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

Hermes 五域：

```bash
./scripts/run_agentbench_memory_train_backup_test.sh \
  --agent hermes \
  --memory-plugin supermemory \
  --version hermes_supermemory_5domain_eval \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

## 结果检查

结果目录格式：

```text
results/agentbench/<agent>-supermemory-<version>-<domain>/
```

重点检查：

- lifecycle 所有阶段返回码为 0。
- `wait_settle` 没有 `failed/error/cancelled` 文档。
- OpenClaw 备份 `manifest.json` 中的 `document_count` 与预期一致。
- Hermes 备份中记录了 documents/memories 数量。
- restore 完成并等待索引后才开始 test。
- cleanup 后本次 `omnimemeval_<run_id>` container 为空或已删除。

## 常见问题

- **HTTP 401/403**：检查插件使用的 API key 是否与实际 Supermemory 服务一致，
  尤其不要只更新 Agent 配置而遗漏自托管服务自身的环境文件。
- **训练结束但 container 为空**：确认插件和 lifecycle 使用的是同一个运行级
  container tag；框架会强制覆盖隔离配置中的 tag。
- **文档状态为 failed**：检查自托管服务的 memory extraction LLM、embedding
  模型和队列 worker；框架会失败关闭，不会把失败文档当作已沉淀。
- **测试期间新增记忆**：确认 `set_mode_test` 已执行且
  `autoCapture=false`；可在 `memory_lifecycle.log` 中查看模式切换记录。
