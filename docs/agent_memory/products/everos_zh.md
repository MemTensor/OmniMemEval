# EverOS 评测配置与运行

[English](./everos.md)

本文说明如何使用 OmniMemEval AgentBench 评测 OpenClaw + EverOS。产品安装请参考
[EverOS 官方项目](https://github.com/EverMind-AI/EverOS)和所使用的 OpenClaw
连接插件说明；本文只列出评测框架需要的配置和运行方式。

## 支持范围

| Agent | 支持状态 | 生命周期配置 |
|---|---|---|
| OpenClaw | 已支持并完成实机验证 | `configs/agentbench/memory_plugins/everos/lifecycle/openclaw.yaml` |
| Hermes | 当前未提供 | — |

## 评测前配置

### 1. 准备公共环境

在仓库根目录准备 AgentBench 环境、`.env.agent` 和 EvoAgentBench 数据。完整说明见
[AgentBench 中文文档](../README_zh.md)。

如果数据已经位于其他项目中，可以把它链接到本项目期望的位置：

```bash
mkdir -p data
ln -s /path/to/EvoAgentBench/data data/agentbench
```

链接后应能找到 `data/agentbench/Reasoning & Problem Decomposition/` 等数据目录。

### 2. 手动选择 EverOS

评测框架不会替用户修改 OpenClaw 的插件选择。运行前必须在用户自己的
`~/.openclaw/openclaw.json` 中启用已安装的 EverOS/EverMemOS OpenClaw 连接插件，
并确认普通 OpenClaw 会话能够写入和召回 EverOS。

当前 profile 会把以下插件目录链接到隔离的评测 home：

```text
~/.openclaw/plugins/evermemos-openclaw-plugin
~/.openclaw/npm
```

如果实际插件安装目录不同，应复制并修改
`configs/agentbench/profiles/openclaw/everos.yaml` 中的 `home_links`，然后通过
`--profile-config` 使用自定义 profile。

### 3. 核对生命周期依赖

默认生命周期按以下部署形态运行：

| 项目 | 默认值 |
|---|---|
| 用户 EverOS 根目录 | `~/.everos` |
| EverOS API | `http://127.0.0.1:8000` |
| EverMemOS 兼容 API | `http://127.0.0.1:1995` |
| 兼容服务目录 | `/opt/everos-compat` |
| 兼容服务入口 | `uvicorn evermemos_v0_compat:app` |

运行前检查：

```bash
command -v everos
command -v uvicorn
test -f ~/.everos/everos.toml
test -d ~/.openclaw/plugins/evermemos-openclaw-plugin
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:1995/health
```

如果端口、目录或服务入口不同，应复制
`configs/agentbench/memory_plugins/everos/lifecycle/openclaw.yaml`，修改 `env`
和相关命令，再通过 `--memory-plugin-config` 指定该文件。

## 两条数据流程验证

建议先用 reasoning 域两条训练数据和两条测试数据验证完整生命周期：

```bash
./scripts/run_agent_eval.sh \
  --agent openclaw \
  --domain reasoning \
  --protocol memory_train_backup_test \
  --memory-plugin everos \
  --version everos_reasoning2_smoke \
  --train-task omni_35,omni_159 \
  --test-task omni_2080,omni_2185 \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

框架会依次执行：

1. 保存用户 OpenClaw/EverOS 状态快照。
2. 创建运行级 OpenClaw home 和 EverOS 根目录。
3. 清空运行级记忆并启动隔离服务。
4. 执行训练数据及 verifier feedback。
5. 调用 EverOS flush 和 cascade sync，等待记忆落盘。
6. 备份当前域的 Markdown 记忆和索引。
7. 恢复训练备份后执行测试数据。
8. 清理运行级状态并重新启动用户原来的 EverOS 服务。

## 正式评测

单域完整评测：

```bash
./scripts/run_agent_eval.sh \
  --agent openclaw \
  --domain reasoning \
  --protocol memory_train_backup_test \
  --memory-plugin everos \
  --version everos_reasoning_eval \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

五域顺序评测：

```bash
./scripts/run_agentbench_memory_train_backup_test.sh \
  --agent openclaw \
  --memory-plugin everos \
  --version everos_5domain_eval \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

## 结果检查

结果默认位于：

```text
results/agentbench/openclaw-everos-<version>-<domain>/
```

至少检查：

- `memory_lifecycle.json` 中 validate、clear、wait、backup、restore、cleanup 均成功。
- `memory_lifecycle.log` 中 flush、cascade sync 和两个 health check 没有报错。
- `memory_backups/` 下生成的 EverOS 训练备份不是空归档。
- `train/summary.json` 和 `test_run_1/summary.json` 均存在。
- 测试结束后 `8000` 和 `1995` 两个用户服务恢复健康。

## 常见问题

- **插件目录不存在**：修正 profile 的 `home_links`，不要把运行级 home 链接回可变的用户数据目录。
- **卡在等待记忆沉淀**：检查 EverOS server 日志、兼容服务日志以及 flush/cascade 返回值。
- **备份没有训练记忆**：确认插件使用的 session/app/project 与生命周期配置一致，并确认训练会话确实调用了兼容 API。
- **服务被意外终止**：不要在同一台机器上并发运行多个占用 `8000/1995` 的 EverOS 评测。
