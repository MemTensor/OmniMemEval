# OpenViking 评测配置与运行

[English](./openviking.md)

本文说明如何使用 OmniMemEval AgentBench 评测 OpenClaw/Hermes + OpenViking。
产品安装请参考 [OpenViking 官方项目](https://github.com/volcengine/OpenViking)、
[OpenClaw 插件说明](https://github.com/volcengine/OpenViking/blob/main/examples/openclaw-plugin/INSTALL.md)
和 [Hermes 集成说明](https://docs.openviking.ai/en/agent-integrations/05-hermes)。

## 支持范围

| Agent | 支持状态 | 生命周期配置 |
|---|---|---|
| OpenClaw | 已支持并完成实机验证 | `configs/agentbench/memory_plugins/openviking/lifecycle/openclaw.yaml` |
| Hermes | 已支持并完成实机验证 | `configs/agentbench/memory_plugins/openviking/lifecycle/hermes.yaml` |

## 评测前配置

### 1. 准备公共环境

按 [AgentBench 中文文档](../README_zh.md)准备 `.env.agent`、模型服务和评测数据。
OpenViking 服务及 Agent 集成必须先由评测用户安装并验证。

### 2. OpenClaw 配置

OpenViking 在当前 OpenClaw 集成中占用的是 `contextEngine` slot，不是 memory slot。
框架要求用户配置满足：

- `plugins.enabled` 不能为 `false`。
- `plugins.allow` 存在时包含 `openviking`。
- `plugins.slots.contextEngine` 为 `openviking`。
- `plugins.entries.openviking` 存在且已启用。
- 插件目录默认位于 `~/.openclaw/extensions/openviking`。

检查示例：

```bash
openclaw config get plugins.slots.contextEngine
openclaw config get plugins.entries.openviking.config
test -d ~/.openclaw/extensions/openviking
curl -fsS http://127.0.0.1:1933/health
```

默认 OpenClaw 生命周期还假设：

| 项目 | 默认值 |
|---|---|
| OpenViking server | `/opt/openviking/venv/bin/openviking-server` |
| 用户配置 | `~/.openviking/ov.conf` |
| 用户数据 | `/opt/openviking/data` |
| API | `http://127.0.0.1:1933` |

评测框架会复制 `ov.conf`，并把隔离副本的 `storage.workspace` 改为当前 run 的数据目录。

### 3. Hermes 配置

用户应手动把 provider 配置为 OpenViking：

```bash
hermes memory status
hermes config set memory.provider openviking
```

默认 Hermes 生命周期使用以下部署约定：

| 项目 | 默认值 |
|---|---|
| 用户 OpenViking 配置目录 | `~/.openviking` |
| 用户服务容器名 | `openviking` |
| 评测容器名 | `omnimemeval-openviking-<run_id>` |
| 镜像 | `ghcr.io/volcengine/openviking:latest` |
| API | `http://127.0.0.1:1933` |

运行前检查：

```bash
hermes memory status
docker inspect openviking >/dev/null
test -f ~/.openviking/ov.conf
curl -fsS http://127.0.0.1:1933/health
```

评测期间用户原容器会暂停，框架使用隔离配置和数据启动评测容器，cleanup 后再恢复
原容器。因此同一台机器不应并发启动多个占用 `1933` 端口的 OpenViking 评测。

如果部署目录、容器名、镜像或端口不同，应复制对应 lifecycle YAML 修改 `env`，
并通过 `--memory-plugin-config` 使用自定义配置。

## 两条数据流程验证

OpenClaw：

```bash
./scripts/run_agent_eval.sh \
  --agent openclaw \
  --domain reasoning \
  --protocol memory_train_backup_test \
  --memory-plugin openviking \
  --version openclaw_openviking_reasoning2_smoke \
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
  --memory-plugin openviking \
  --version hermes_openviking_reasoning2_smoke \
  --train-task omni_35,omni_159 \
  --test-task omni_2080,omni_2185 \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

生命周期会：

1. 校验用户选择的 OpenViking provider/slot。
2. 保存用户配置和数据快照。
3. 创建运行级 Agent home、OpenViking 配置和数据目录。
4. 清空评测数据并启动隔离服务。
5. 训练后查询 `/api/v1/tasks`，要求任务连续 30 秒无活动。
6. 停止隔离服务后备份数据。
7. 恢复备份、重新启动服务并执行测试。
8. 删除隔离状态并恢复用户原服务。

任务出现 `failed`、`error` 或 `cancelled` 时，生命周期直接失败。

## 正式评测

OpenClaw 五域：

```bash
./scripts/run_agentbench_memory_train_backup_test.sh \
  --agent openclaw \
  --memory-plugin openviking \
  --version openclaw_openviking_5domain_eval \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

Hermes 五域：

```bash
./scripts/run_agentbench_memory_train_backup_test.sh \
  --agent hermes \
  --memory-plugin openviking \
  --version hermes_openviking_5domain_eval \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

## 结果检查

结果目录格式：

```text
results/agentbench/<agent>-openviking-<version>-<domain>/
```

重点检查：

- lifecycle 所有阶段返回码为 0。
- `wait_settle` 日志显示 task queue 已稳定空闲。
- 备份归档包含运行级 OpenViking 数据，而不是只有配置文件。
- restore 后 health check 成功，随后才开始 test。
- cleanup 后评测容器已删除、用户原服务恢复健康。

## 常见问题

- **OpenClaw 校验 memory slot 失败**：OpenViking 应选择
  `plugins.slots.contextEngine`，不是 `plugins.slots.memory`。
- **Hermes 返回 401/403**：检查运行级 `.env` 中 API key，以及多租户部署所需的
  account/user 配置。框架会传递 Authorization 和租户 header。
- **备份时数据库损坏或锁文件报错**：必须先停止 server/container 再归档；
  不要绕过 lifecycle 的 stop 步骤。
- **任务队列不收敛**：检查 OpenViking 的 embedding、向量库和 LLM 配置及容器日志。
