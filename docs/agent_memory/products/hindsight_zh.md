# Hindsight 评测配置与运行

[English](./hindsight.md)

本文说明如何使用 OmniMemEval AgentBench 评测 OpenClaw/Hermes + Hindsight。
产品安装请参考 [Hindsight OpenClaw 集成文档](https://hindsight.vectorize.io/sdks/integrations/openclaw)
和 [Hermes Memory Providers 文档](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory-providers/)；
本文重点说明评测框架配置和运行方式。

## 支持范围

| Agent | 支持状态 | 生命周期配置 |
|---|---|---|
| OpenClaw | 已支持并完成实机验证 | `configs/agentbench/memory_plugins/hindsight/lifecycle/openclaw.yaml` |
| Hermes | 已支持并完成实机验证 | `configs/agentbench/memory_plugins/hindsight/lifecycle/hermes.yaml` |

## 评测前配置

### 1. 准备公共环境

按 [AgentBench 中文文档](../README_zh.md)准备 `.env.agent`、模型服务和
EvoAgentBench 数据。评测框架不会安装或启动用户未配置好的 Hindsight 产品。

### 2. OpenClaw 配置

用户应手动启用 Hindsight；框架只校验，不替用户选择插件。当前生命周期要求：

- `plugins.enabled` 不能为 `false`。
- `plugins.allow` 存在时必须包含 `hindsight-openclaw`。
- `plugins.slots.memory` 必须为 `hindsight-openclaw`。
- `plugins.entries.hindsight-openclaw` 存在且已启用。
- Hindsight API 默认位于 `http://127.0.0.1:9077`。
- 本地嵌入式部署的实例描述文件默认位于
  `/var/lib/hindsight-openclaw/.pg0/instances/hindsight/instance.json`。

检查示例：

```bash
openclaw config get plugins.slots.memory
curl -fsS http://127.0.0.1:9077/health
command -v hindsight-admin
```

评测时框架会复制 OpenClaw 配置，并在隔离副本中使用固定的运行级 bank：

```text
omnimemeval-<run_id>
```

不会修改或清空用户已有 bank。

### 3. Hermes 配置

用户必须提前把 Hermes provider 配为 Hindsight：

```bash
hermes memory status
hermes config set memory.provider hindsight
```

应满足：

- `~/.hermes/config.yaml` 中 `memory.provider: hindsight`。
- `~/.hermes/hindsight/config.json` 存在且凭证、模式可用。
- Hindsight API 默认位于 `http://127.0.0.1:9177`。
- `hindsight-admin` 对运行评测的用户可用。

检查示例：

```bash
hermes memory status
test -f ~/.hermes/hindsight/config.json
curl -fsS http://127.0.0.1:9177/health
command -v hindsight-admin
```

如 API 地址或配置位置不同，应复制对应 lifecycle YAML，修改 `env` 后使用
`--memory-plugin-config`。

## 两条数据流程验证

OpenClaw：

```bash
./scripts/run_agent_eval.sh \
  --agent openclaw \
  --domain reasoning \
  --protocol memory_train_backup_test \
  --memory-plugin hindsight \
  --version openclaw_hindsight_reasoning2_smoke \
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
  --memory-plugin hindsight \
  --version hermes_hindsight_reasoning2_smoke \
  --train-task omni_35,omni_159 \
  --test-task omni_2080,omni_2185 \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

生命周期会：

1. 校验用户确实选择了 Hindsight。
2. 创建隔离 Agent home，并把 provider 指向运行级 bank。
3. 删除并重建该运行级 bank。
4. 训练后等待 Hindsight operations 连续空闲。
5. 使用 `hindsight-admin export-bank` 备份 bank。
6. 测试前使用 `import-bank` 恢复相同 bank。
7. 测试结束后删除运行级 bank 和隔离 home。

任何 operation 进入 `failed`、`error` 或 `cancelled` 状态都会使评测失败。

## 正式评测

以下命令中的 `AGENT` 替换为 `openclaw` 或 `hermes`：

```bash
AGENT=openclaw
./scripts/run_agentbench_memory_train_backup_test.sh \
  --agent "$AGENT" \
  --memory-plugin hindsight \
  --version "${AGENT}_hindsight_5domain_eval" \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

指定域子集：

```bash
./scripts/run_agentbench_memory_train_backup_test.sh \
  --agent hermes \
  --memory-plugin hindsight \
  --domains reasoning,code_implementation \
  --version hermes_hindsight_reasoning_code_eval
```

## 结果检查

结果目录格式：

```text
results/agentbench/<agent>-hindsight-<version>-<domain>/
```

重点检查：

- `memory_lifecycle.json` 中所有 lifecycle command 的 `returncode` 为 0。
- `memory_lifecycle.log` 包含 bank idle、backup 和 restore 成功记录。
- 备份归档中的 `manifest.json` 显示正确的 `has_bank`。
- train/test 的 `summary.json` 均生成，基础设施失败未被计入任务质量分数。
- cleanup 后不存在本次 `omnimemeval-<run_id>` bank。

## 常见问题

- **Hermes 报 provider 不匹配**：在用户配置中手动设置
  `memory.provider: hindsight`，框架不会替用户修改。
- **找不到 `hindsight-admin`**：确保它位于 Hindsight daemon 的环境或系统
  `PATH`；控制器也会尝试从 daemon 可执行文件附近发现它。
- **导出时数据库连接失败**：检查本地 pg0 instance 文件、daemon 运行用户和端口。
- **operation 长时间不结束**：检查 Hindsight LLM/embedding 凭证和服务日志；
  不要通过缩短 timeout 绕过未完成的记忆写入。
