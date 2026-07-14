# AgentBench Migration Design

本文档总结将 OpenClaw agent 评测经验迁移到
`OmniMemEval` 的设计方案。目标不是把外部实验工程全量合并进来，
也不是为每个记忆插件产品写专用适配器，而是在 `OmniMemEval` 中增加一套
可复用的 agent 能力评测协议。

## 目标

`OmniMemEval` 当前主要评测 memory backend API，例如 LoCoMo 和
LongMemEval 流水线中的 `add/search/delete` 能力。新增 AgentBench 后，
评测对象扩展为：

```text
Agent Runtime + Agent 配置中的记忆插件 + Task Domain + Verifier
```

其中 OpenClaw、Nanobot 等是 agent runtime；记忆插件属于 agent 自身配置的一部分。
如果某个记忆插件已经适配 OpenClaw 或 Nanobot，`OmniMemEval` 不再逐个适配该
记忆产品，而是通过统一 agent adapter 调用对应 agent。

## 非目标

- 不迁移外部实验工程的全部脚本、历史 job、备份恢复脚本。
- 不内置 memos、Hindsight 或其他具体记忆插件产品的专用适配逻辑。
- 不把 agent 评测塞进现有 `client_factory` memory backend 层。
- 第一阶段不强制迁移 SWE-bench、LiveCodeBench 等重依赖 domain。

## 推荐目录结构

```text
OmniMemEval/
  scripts/
    agentbench/
      __init__.py
      runner.py
      summary.py
      session.py
      agents/
        __init__.py
        base.py
        openclaw.py
        nanobot.py
      domains/
        __init__.py
        base.py
        reasoning/
        information_retrieval/
        knowledge_work/
      protocols/
        __init__.py
        test_only.py
        train_then_test.py
  scripts/run_agent_eval.sh
  configs/agentbench/
    agents/
      openclaw.yaml
      nanobot_plain.yaml
```

结果目录沿用 `OmniMemEval` 风格：

```text
results/agentbench/<agent-profile>-<domain>-<version>/
```

## 核心抽象

### AgentAdapter

`AgentAdapter` 只负责如何驱动某个 agent 完成任务，不关心具体记忆插件是什么。

```python
class AgentAdapter:
    name: str

    def prepare_run(self, config: dict, run_dir: Path) -> None:
        ...

    def build_session_spec(self, *, phase, domain, split, task, trial) -> SessionSpec:
        ...

    def prepare_task(self, task: dict, env_info: dict, session: SessionSpec) -> None:
        ...

    def call(self, prompt: str, session: SessionSpec, timeout: int) -> dict:
        ...

    def collect_session(self, session: SessionSpec, trial_dir: Path) -> dict:
        ...

    def cleanup_task(self) -> None:
        ...

    def finalize_run(self) -> None:
        ...
```

OpenClaw adapter 负责：

- 调用 `openclaw agent`。
- 准备临时或指定的 `OPENCLAW_HOME`。
- 写入或合并 `openclaw.json`。
- 按 agent profile 暴露 `home_links`、env、MCP server。
- 收集 OpenClaw session JSONL。
- 解析最终 assistant answer、token、tool calls。

Nanobot adapter 负责：

- 调用 `nanobot agent`。
- 准备临时 workspace/config。
- 收集 Nanobot session JSONL。
- 暴露与 OpenClaw 一致的 `SessionSpec` 和结果格式。

### DomainAdapter

`DomainAdapter` 继续使用 EvoAgentBench 的最小任务协议：

```python
class DomainAdapter:
    name: str

    def load_tasks(self, args) -> list[dict]:
        ...

    def setup(self, task: dict, agent_name: str, trial: int) -> dict:
        ...

    def build_prompt(self, task: dict, env_info: dict) -> str:
        ...

    def verify(self, task: dict, env_info: dict, trial_dir: Path, agent_result: dict) -> dict:
        ...

    def cleanup(self, task: dict, env_info: dict) -> None:
        ...
```

第一阶段建议迁移：

- `reasoning`
- `information_retrieval`
- `knowledge_work`

第二阶段再考虑：

- `code_implementation`
- `software_engineering`

## 实验协议

AgentBench 提供两个协议，而不是把 baseline 固定写死在一次实验里。

### test_only

只跑测试集，用于评测当前 agent profile 的原始能力或某个已配置插件状态下的能力。

```text
test_only
  test split
```

输出：

```text
results/agentbench/openclaw-plain-reasoning-v1/
  experiment_config.json
  test/
    summary.json
    <task>__trial_1/
      result.json
      session.jsonl
      verifier/
```

### train_then_test

先跑训练集，让 agent 配置中的记忆插件沉淀经验；再跑测试集，观察训练后效果。

```text
train_then_test
  train split
  test split
```

输出：

```text
results/agentbench/openclaw-plugin-reasoning-v1/
  experiment_config.json
  train/
    summary.json
  test/
    summary.json
  report.md
```

baseline 可以用 `test_only` 跑一个无记忆插件的 agent profile 得到；如果要比较
不同 agent profile，先分别运行单域评测，再在外部分析对应 `summary.json`。

示例：

```bash
./scripts/run_agent_eval.sh \
  --agent openclaw \
  --domain reasoning \
  --protocol test_only \
  --version plain_v1

./scripts/run_agent_eval.sh \
  --agent openclaw \
  --domain reasoning \
  --protocol train_then_test \
  --version plugin_v1

./scripts/run_agent_eval.sh \
  --agent openclaw \
  --domain reasoning \
  --protocol test_only \
  --version plain_3run \
  --runs 3 \
  --pass-at 3
```

## Agent Profile 配置

记忆插件不作为 `OmniMemEval` 的 agent adapter，而是拆分为 runtime-specific
agent profile 和独立 lifecycle。当前配置位于：

```text
configs/agentbench/agents/<agent>.yaml
configs/agentbench/profiles/<agent>/<profile>.yaml
configs/agentbench/memory_plugins/<plugin>/lifecycle/<agent>.yaml
```

OpenClaw 示例：

```yaml
agent:
  name: openclaw
  command: openclaw

  runtime:
    home_mode: isolated_copy
    home_links:
      - extensions/my-memory-plugin
      - my-memory-data

  env:
    MY_MEMORY_PLUGIN_KEY: ${MY_MEMORY_PLUGIN_KEY}

  prompt_prefix:
    train: "new task"
    test: "new task"

  session:
    mode: gateway_compatible
    expose_context_env: OMNIMEMEVAL_AGENT_CONTEXT
```

Nanobot 示例：

```yaml
agent:
  name: nanobot
  command: nanobot

  runtime:
    workspace_mode: isolated_copy

  env:
    MEMORY_API_KEY: ${MEMORY_API_KEY}

  session:
    mode: default
    expose_context_env: OMNIMEMEVAL_AGENT_CONTEXT
```

## Session ID 兼容设计

实际评测中会遇到一个问题：部分 agent CLI 对 `session_id` 字符集有限制，
但部分记忆插件希望看到带来源语义的 session id。

例如 OpenClaw CLI 的 `--session-id` 不能包含冒号，而 OpenClaw gateway/memos
事件中使用的 session id 格式是：

```text
openclaw::main::agent:main:explicit:<safe-cli-session-id>
```

因此 AgentBench 必须拆分 session id：

```text
base_session_id
  评测框架生成的安全 ID。

cli_session_id
  传给 agent CLI 的 ID。必须满足对应 agent 的格式限制。

semantic_session_id
  带 benchmark/domain/split/task/phase 语义的 ID，可以包含冒号。

agent_session_ref
  agent 内部或 gateway 侧使用的 session 引用。
```

### OpenClaw gateway-compatible 策略

OpenClaw 在实际 gateway feedback 路径中的格式为：

```python
def openclaw_session_key(cli_session_id: str) -> str:
    return f"agent:main:explicit:{cli_session_id}"

def openclaw_gateway_session_id(cli_session_id: str) -> str:
    return f"openclaw::main::{openclaw_session_key(cli_session_id)}"
```

也就是：

```text
cli_session_id:
  reasoning-omni_2533-3e779d

openclaw_session_key:
  agent:main:explicit:reasoning-omni_2533-3e779d

openclaw_gateway_session_id:
  openclaw::main::agent:main:explicit:reasoning-omni_2533-3e779d
```

OpenClaw CLI 调用仍然使用安全 ID：

```bash
openclaw agent --session-id reasoning-omni_2533-3e779d ...
```

但通过环境变量向 agent/plugin 暴露完整上下文：

```json
{
  "benchmark": "omnimemeval-agentbench",
  "phase": "train",
  "domain": "reasoning",
  "split": "train",
  "task": "omni_2533",
  "trial": 1,
  "cli_session_id": "reasoning-omni_2533-3e779d",
  "semantic_session_id": "omnimemeval:train_then_test:train:reasoning:omni_2533:trial:1",
  "openclaw_session_key": "agent:main:explicit:reasoning-omni_2533-3e779d",
  "openclaw_gateway_session_id": "openclaw::main::agent:main:explicit:reasoning-omni_2533-3e779d"
}
```

这样可以同时满足：

- CLI 不因非法字符失败。
- 插件或外部存储可以读取 gateway-compatible session id。
- 结果文件可以把 CLI session、语义 session、agent 内部 session 对齐。

如果某个插件必须让 agent 内部自动捕获链路直接使用完整 gateway session id，
而 OpenClaw CLI 又不支持该格式，则需要以下之一：

- OpenClaw 支持额外 `--metadata`、`--source-ref` 或类似参数。
- 改走 OpenClaw gateway API。
- 用户通过 agent profile 的 overlay/patch hook 自行兼容具体插件。

主框架不内置具体插件 patch，但可以提供通用 overlay 机制：

```yaml
agent:
  runtime:
    overlays:
      - source: ./patches/my-plugin
        target: node_modules/my-plugin
        prepend_plugin_path: true
```

## Runner 行为

AgentBench runner 复用 EvoAgentBench 的核心执行模型：

```text
domain.setup
agent.prepare_task
domain.build_prompt
agent.call
domain.verify
agent.collect_session
domain.cleanup
agent.cleanup_task
```

每个 trial 写入：

```text
<task>__trial_<n>/
  result.json
  session.jsonl
  verifier/
```

`result.json` 必须包含：

```json
{
  "task_name": "omni_2533",
  "agent": "openclaw",
  "phase": "train",
  "split": "train",
  "trial": 1,
  "session": {
    "cli_session_id": "...",
    "semantic_session_id": "...",
    "agent_session_ref": "...",
    "openclaw_gateway_session_id": "..."
  },
  "agent_result": {},
  "verifier_result": {},
  "token_usage": {}
}
```

## Metrics

单 phase `summary.json` 继续包含：

- task count
- trial count
- pass@1
- average reward
- average elapsed seconds
- average tokens
- failure classes
- domain-specific metrics

每个 phase 的 `summary.json` 和 `report.md` 负责输出单域测评结果：

- 每个 task 的每次 trial 结果
- 每次 trial 的整体 pass rate
- 平均 reward、平均耗时、平均 token
- pass@1
- pass@n，其中 n 由 `--pass-at` 控制，默认等于 `--trials`

对于支持解析 tool calls 的 agent，可额外统计：

- memory tool calls
- memory read calls
- memory write calls
- memory errors
- empty retrieval count

这些指标只作为可选解析结果，不要求每个 agent 或插件都支持。

## 迁移步骤

1. 建立 `scripts/agentbench/` 基础目录、`AgentAdapter`、`DomainAdapter`、
   `SessionSpec`。
2. 迁移通用 runner 和 summary，调整结果目录到
   `results/agentbench/`。
3. 迁移 OpenClaw adapter，保留 CLI 调用、临时 home、session 收集、MCP 注入，
   去掉 memos 专用硬编码。
4. 实现 OpenClaw `gateway_compatible` session 策略。
5. 迁移 Nanobot adapter，复用同一 runner/protocol/session/result 格式。
6. 先迁移 `reasoning` domain 并跑通 `test_only` 和 `train_then_test`。
7. 迁移 `information_retrieval`，补齐 MCP server 生命周期。
8. 迁移 `knowledge_work`。
9. 增加 phase 级 markdown report。
10. 后续按需迁移 `code_implementation`、`software_engineering`。

## 设计原则

- `OmniMemEval` 适配 agent runtime，不适配每个记忆插件产品。
- 记忆插件通过 agent profile 配置进入评测。
- runner、domain、protocol 不依赖 OpenClaw。
- OpenClaw adapter 不知道具体插件产品，只知道 OpenClaw 的运行方式。
- session id 必须区分 CLI 安全 ID 和插件/存储语义 ID。
- 具体产品 patch 不进入主框架，最多通过 overlay hook 由用户显式配置。
