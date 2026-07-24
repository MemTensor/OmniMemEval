# MemOS 插件源码改动与影响说明

本文只记录为解决 OpenClaw/Hermes 并发调用和后台记忆处理问题而对 **MemOS 本地插件源码**所做的修改，以及这些修改带来的行为变化、收益、成本和兼容性影响。

修改基于本机安装的 MemOS 2.0.10 源码：

`/root/gyh/MemOS/apps/memos-local-plugin`

当前源码属于在 2.0.10 基础上的本地修订版。只有重新构建、打包并安装该修订版插件后，本文所述行为才会生效；官方未修改的 2.0.10 不包含这些能力。

## 1. 修改目标

本次修改主要解决以下问题：

1. OpenClaw gateway 已经加载 MemOS 时，并发 CLI 再次加载插件会创建第二个 MemoryCore，触发 `DuplicateOpenClawRuntimeError`。
2. 如果简单绕过 runtime lock，多个进程会同时访问同一数据库，并重复执行 reflection、reward、L2/L3 和 Skill 进化。
3. CLI 或 session 很快退出时，原来的 fire-and-forget 记忆任务可能尚未持久化，或者 `session.close` 因同步等待整条模型链路而超时。
4. 同一个 episode/policy 的多个事件路径可能并发触发记忆进化，造成重复 policy、重复 Skill 和额外模型调用。
5. 后台任务缺少持久状态，进程异常退出后无法可靠判断任务是否完成，也难以安全恢复。
6. OpenClaw 并发 hook 可能出现 `turn.start` 尚未完成、`agent_end` 已经到达的竞态，导致 episode 绑定错误或额外创建 episode。
7. Hermes 使用了 JavaScript 风格的正则调用 `pgrep`，而 `pgrep` 使用 POSIX ERE，可能直接返回正则错误。

## 2. 修改后的整体结构

```text
OpenClaw gateway / CLI / feedback client
                    │
                    │ Unix socket JSON-RPC
                    ▼
     一个 MemOS Home 对应一个 runtime daemon
                    │
                    ▼
          唯一 MemoryCore + SQLite
                    │
          ┌─────────┴─────────┐
          │                   │
   同步持久化 L1        evolution_jobs
                              │
                              ▼
                     单消费者进化 worker
                              │
                  reflection / reward / L2 / L3
                              │
                              ▼
                       Skill 单通道调度
```

并发 CLI 只负责向同一个 runtime 提交检索、turn、feedback 等请求。原始对话和任务状态先写入 SQLite，耗时较长的记忆进化由后台唯一 worker 串行完成。

## 3. 具体源码修改

### 3.1 OpenClaw 改为共享 MemoryCore runtime

原来的 OpenClaw 插件在每个加载插件的进程内直接初始化 MemoryCore、SQLite 和 viewer，并用 runtime lock 阻止第二个实例。这可以避免数据库被多个实例同时写入，但也意味着 gateway 存活时，任何需要加载插件的 CLI 都会因重复实例而失败。

修改后增加独立的 `runtime-daemon`：

- 一个 MemOS Home 只允许一个 daemon 成为 MemoryCore owner。
- gateway、并发 CLI 和其他调用方只创建轻量 socket client，不再各自初始化 MemoryCore。
- 多个进程同时尝试启动 daemon 时，原有 runtime lock 只用于 owner election；胜出的进程继续启动，其他进程正常退出并连接胜出的 owner。
- OpenClaw 插件改为懒连接。只有工具或 hook 真正需要 MemOS 时才连接或启动 daemon。
- `cli-metadata` 等纯插件发现流程不再启动数据库、daemon 或 viewer，避免只查询 CLI 元数据也产生运行时副作用。
- viewer 由 daemon 统一持有。viewer 端口被占用时，MemoryCore 可以继续以 headless 方式提供服务，不再因为 viewer 冲突关闭记忆能力。

影响：

- gateway 存活时，并发 CLI 可以正常使用同一个 MemOS 数据库。
- `DuplicateOpenClawRuntimeError` 不再是正常并发调用的结果。
- 一个 Home 内不会因为每个 CLI 启动一个 MemoryCore 而重复执行记忆进化。
- 插件运行时从“嵌入每个 OpenClaw 进程”变成了“独立 daemon + 多 client”，部署和排障时需要同时关注 daemon 日志。

### 3.2 新增本地 Unix socket JSON-RPC 通道

新增多客户端 Unix socket server/client，用于把原有 MemoryCore 接口转发给共享 daemon：

- socket 文件名由 MemOS Home 的 SHA-256 摘要和 uid 生成，避免 Home 路径过长，也避免不同用户或不同 Home 发生冲突。
- socket 权限设置为 `0600`。
- client 连接后通过 `core.health` 校验 agent 类型和 SQLite 绝对路径，防止误连另一个 runtime。
- transport 断开时允许重连。
- JSON-RPC 应用错误和请求超时不会自动重试，因为请求可能已经在 owner 中执行；自动重放写请求会造成重复 capture 或 feedback。
- 新增 `runtime-stdio-proxy` 兼容需要 stdio bridge 的调用方。
- OpenClaw 调用被强制路由到 proxy；如果配置试图直接启动普通 `bridge.mjs`，会明确失败，不会静默创建第二个 MemoryCore。
- `tool_outcome.record` 被补充为正式 RPC 方法，远程 core 可以完整转发工具执行结果。

影响：

- 多进程共享 MemoryCore 有了明确的进程间通信边界。
- runtime identity 校验降低了跨 Home 写错数据库的风险。
- timeout 不重试优先保证写入幂等性，但调用方仍可能看到“请求超时、后台实际已执行”的不确定状态。
- 当前 transport 依赖 Unix domain socket，主要适用于 Linux/macOS；Windows 需要 named pipe 或 loopback TCP 适配。

### 3.3 新增持久化 evolution job queue

新增 SQLite migration `017-evolution-jobs.sql` 和 `evolution_jobs` repository。耗时的语义进化不再只存在于 Node.js 内存任务中，而是先作为 job 写入数据库。

当前 job 类型包括：

- `turn_enrichment`：补充 trace summary 和 embedding。
- `episode_evolution`：执行 episode reflection、reward、L2/L3 和后续 Skill 处理。

job 状态包括：

- `queued`
- `leased`
- `failed`
- `succeeded`
- `dead_letter`

队列提供以下机制：

- 使用 dedupe key 合并相同 episode 的重复调度。
- job 正在执行时再次收到相同请求，会设置 `rerun_requested`，当前执行完成后再安全重跑一次。
- SQLite `BEGIN IMMEDIATE` 和有效 lease 检查保证同一个数据库任意时刻只有一个进化 job 被消费。
- 默认 lease 为 5 分钟，并在长模型调用期间定期续租。
- runtime 异常退出后，过期 lease 可以由下一次启动恢复。
- 失败任务使用指数退避重试；默认最大尝试次数为 3，之后进入 `dead_letter`。
- 已成功 job 定期清理，默认保留 7 天用于诊断。

影响：

- CLI 退出后，已入队的记忆处理不会因为调用进程消失而丢失。
- runtime 崩溃后可以恢复未完成任务。
- 数据库能够直接反映后台处理的真实状态。
- 新数据库 schema 与原版 2.0.10 不再完全一致；降级使用不了解该队列的旧插件时，不保证具有相同处理语义。
- 进程被强制杀死后，任务通常要等 lease 过期才能恢复，最坏会增加约 5 分钟延迟。

### 3.4 将 L1 持久化与模型进化解耦

原来的 `agent_end` 路径可能同步执行 summary、embedding、reflection、reward 和分层记忆进化，容易超过 OpenClaw hook 的 30 秒预算。简单改成 fire-and-forget 又会在一次性 CLI 很快退出时丢失尚未发出的 capture 请求。

修改后将 capture 拆成两段：

1. `runPersistOnly` 同步提取和规范化 turn，持久化原始 L1 trace，并写入 evolution job。
2. 后台 worker 执行 `runEnrich`，补充 summary、vector，并继续执行 episode evolution。

待补充的 trace 会带有 `capture_pending_enrichment` 标记。enrichment 完成后更新原记录并移除标记，而不是重复插入一份 trace。

`agent_end` 现在会等待“L1 trace 和 evolution job 已经 durable”这一较短路径完成后再返回，但不会等待全部模型进化完成。

影响：

- 避免纯 fire-and-forget 导致的原始记忆丢失。
- `agent_end` 更容易在 OpenClaw 的 hook 超时内完成。
- 对话结束时通常只能保证原始 trace 已持久化，不能保证 L2/L3/Skill 已经生成。
- viewer 中可能短暂出现已 capture、尚未 enrichment 的记录，这是正常中间状态。

### 3.5 修改 `session.close` 语义和 daemon 退出行为

`MemoryCore.closeSession()` 现在只负责关闭 session/episode，不再同步调用整条 pipeline 的 `flush()`。

OpenClaw client 全部断开后，daemon 不会立即退出，而是继续检查：

- evolution queue 是否仍有 active job。
- embedding retry queue 是否仍有 pending/in-progress job。
- 队列清空后是否持续一个 quiet interval，避免 evolution 完成后刚产生的 embedding job 被漏掉。

确认后台任务稳定结束后，daemon 才关闭 viewer、MemoryCore 和 runtime lock。也提供 drain-only 入口，用于只恢复和排空已有后台任务。

影响：

- session 和 CLI 可以及时退出，后台 reflection、reward、L2/L3、Skill 和 embedding 仍会继续执行。
- `session.close` 成功不再等价于所有记忆处理完成。
- client 退出后一段时间仍可能存在模型调用、CPU/内存占用和推理费用。
- 如果直接 SIGKILL daemon，未完成任务依赖下次启动和 lease recovery 恢复。

### 3.6 所有模型进化进入同一个串行通道

新增 evolution worker 的 `runExclusive()`，将模型相关的记忆操作串行化。除自动 episode evolution 外，以下显式路径也进入同一个通道：

- feedback reward
- feedback repair
- feedback experience
- L2/L3 drain
- 手工 reward fallback
- Skill 触发与 flush

这避免自动事件路径和显式 feedback API 绕过后台队列，同时对同一个数据库并发调用模型进化。

影响：

- 并发 CLI 仍可以并发提交检索、turn 和 feedback，原始数据会分别持久化。
- 后续的高层记忆进化由一个 worker 逐项处理，不会因为 CLI 并发而同时跑多份。
- SQLite 写竞争、重复 reward、重复 L2/L3 和重复 Skill 的概率显著降低。
- 高并发下高级记忆处理吞吐会降低，队列等待时间会增加；这是用吞吐换取一致性和避免重复模型成本。

### 3.7 Skill 增加 single-flight 和同键合并

Skill subscriber 原来使用一个简单的 microtask/debounce 状态。事件触发和手工 `runOnce()` 可能走不同路径，仍可能同时对相同 policy 结晶 Skill。

修改后：

- 所有 Skill 生成进入同一个 process-local 串行 scheduler。
- 以 `policyId`、`skillId` 或 global 作为 single-flight key。
- 同一个 key 已在等待或执行时，新请求复用已有 Promise，不再启动第二次 Skill 生成。
- `flush()` 会持续等待 scheduler 尾部，直到期间新增的任务也处理完成。

影响：

- 相同 policy 被多个事件同时触发时只进行一次 Skill 结晶。
- 解决少量对话因重复进化产生过多 Skill 的主要并发触发路径。
- 被合并的触发不会各自生成独立 Skill；这是预期的去重语义。

### 3.8 修复 OpenClaw 并发 turn/episode 绑定竞态

并发 hook 下，`turn.start` 的检索和 episode 路由可能尚未返回，`agent_end` 已经到达。旧逻辑只能查找最近绑定，可能绑定到错误 turn，或者因找不到 episode 而创建新 episode。

修改后：

- `turn.start` 开始时先按 session、run 和 user text 注册 pending binding Promise。
- `agent_end` 优先查找精确 binding；尚未完成时等待对应 pending binding。
- 只有精确和 pending binding 都不存在时，才回退到旧的最近绑定逻辑。
- session 结束时统一在 `finally` 中清理 cursor、episode binding、tool call 和 user text 状态。
- 一次性 session 的 L1 capture 已成功后，即使 `closeSession` 的 transport 调用失败，也只记录 session-close warning，不会把它误报为 `onTurnEnd` 失败。

影响：

- 并发请求下 turn 与 episode 的对应关系更稳定。
- 减少额外 episode、trace 归错 session 和主任务没有正确持久化的问题。
- 如果 `turn.start` 本身长时间阻塞，`agent_end` 会等待同一个 pending Promise；最终仍由上层 RPC/hook timeout 控制，不会无限等待。

### 3.9 扩展 runtime health 和后台状态可观测性

`core.health` 增加两组统计：

- evolution：active、queued、leased、retrying、succeeded、deadLetter
- embedding retry：pending、inProgress、failed、succeeded

daemon 的 background drain 使用这些状态判断真实后台工作是否完成，而不是只判断日志是否静默。

影响：

- viewer、诊断工具和调用方可以区分“对话已结束”和“记忆处理已结束”。
- failed/dead-letter 不再被隐藏为正常 idle。
- `succeeded` 计数会保留到定期清理，因此判断活跃任务时应使用 active/queued/leased/retrying，而不是要求总 job 数为零。

### 3.10 修复 Hermes 进程检测正则

Hermes bridge 使用 `pgrep -f` 检测 `hermes ... chat` 进程。原模式包含 JavaScript 的非捕获分组 `(?:...)`，但 `pgrep` 使用 POSIX ERE，不支持该语法，可能以退出码 2 返回正则错误。

修改后：

- 传给 `pgrep` 的模式改为 POSIX ERE，使用 `[[:space:]]` 和普通捕获分组。
- JavaScript 单元测试使用语义等价的独立 JS RegExp，不再错误地把 JS 正则可编译当作 `pgrep` 兼容证明。
- 模式仍允许 `hermes` 和 `chat` 之间存在全局参数，并避免匹配 `chatter`、`chat-server` 等无关命令。

影响：Hermes 进程检测不再因正则方言不兼容而失败，且保留了对不同 CLI 参数排列的兼容性。

## 4. 主要影响汇总

| 维度 | 正面影响 | 代价或风险 |
| --- | --- | --- |
| OpenClaw 并发 | gateway 和多个 CLI 共享一个 MemoryCore | 增加独立 daemon 和 socket 运维边界 |
| 数据持久性 | CLI 退出前保证 L1 和 job 已写入；后台任务可恢复 | session 结束不代表高级记忆立即可用 |
| 进化一致性 | 一个数据库只串行执行一条模型进化链 | 高并发时后台队列等待变长 |
| 去重 | episode job、Skill policy 触发均有去重/合并 | 重复触发不会各自生成独立结果 |
| 故障恢复 | lease、重试和 dead-letter 提供可追踪状态 | SIGKILL 后可能等待约 5 分钟 lease 过期 |
| 模型成本 | 避免同一记忆被多个进程重复进化 | client 退出后仍可能继续产生模型调用成本 |
| Viewer | daemon 统一提供 viewer；端口冲突可 headless 运行 | viewer 未启动不代表 MemoryCore 未运行 |
| 数据库兼容 | 后台工作状态持久化、可诊断 | 新 migration 与原版 2.0.10 行为不完全相同 |
| 平台兼容 | Linux/macOS 本地 IPC 简单且权限明确 | Windows 暂无对应 Unix socket transport |

## 5. 未修改的能力边界

本次 MemOS 源码修改没有改变以下内容：

- 没有修改记忆检索算法、召回排序或检索模型。
- 没有给 MemOS 检索增加内部软超时。
- 没有改变模型服务本身的超时、限流或稳定性。
- 没有把每个会话或每个任务拆成独立数据库；共享粒度仍由调用时选择的 MemOS Home 决定。
- 没有让多个不同 Home 共用一个 daemon。每个 Home 仍会有自己的 owner、数据库和后台 worker。
- 没有保证进程遭到 SIGKILL 后任务立即恢复；恢复仍受 lease 到期时间约束。
- 没有把插件改成 read-only。正常检索、capture、feedback 和记忆进化行为仍然保留。

## 6. 主要代码位置

共享 OpenClaw runtime：

- `adapters/openclaw/runtime-daemon.ts`
- `adapters/openclaw/runtime-client.ts`
- `adapters/openclaw/remote-core.ts`
- `adapters/openclaw/runtime-drain.ts`
- `adapters/openclaw/runtime-paths.ts`
- `adapters/openclaw/runtime-stdio-proxy.ts`
- `adapters/openclaw/index.ts`
- `bridge/socket.ts`

持久化进化和 capture：

- `core/storage/migrations/017-evolution-jobs.sql`
- `core/storage/repos/evolution_jobs.ts`
- `core/evolution/worker.ts`
- `core/capture/capture.ts`
- `core/capture/subscriber.ts`
- `core/pipeline/orchestrator.ts`
- `core/pipeline/memory-core.ts`
- `core/skill/subscriber.ts`

接口和适配修复：

- `agent-contract/memory-core.ts`
- `agent-contract/jsonrpc.ts`
- `bridge/methods.ts`
- `adapters/openclaw/bridge.ts`
- `adapters/hermes/memos_provider/bridge_client.py`
- `bridge/hermes-process.ts`

## 7. 构建与测试状态

当前修订版已经完成以下验证：

- MemOS 测试：1195 passed，2 skipped。
- TypeScript lint 和 build 通过。
- npm pack 通过，并已使用本地 tarball 重新安装。
- 并发 OpenClaw 调用可以共享同一 runtime，未再触发正常并发场景下的 `DuplicateOpenClawRuntimeError`。
- 两条并发对话的原始记忆可以持久化，后台进化队列最终归零，未再出现由并发重复进化造成的异常 Skill 数量。

## 8. 发布注意事项

该插件虽然仍以 MemOS 2.0.10 为基础，但运行时结构和数据库 schema 已发生实质变化。发布或开源时建议：

1. 使用可区分的版本号或附加 commit SHA，不要让用户误认为它与官方原版 2.0.10 完全一致。
2. 保留升级前数据库备份，并明确说明 migration 017 和后台 job 语义。
3. 在 CI 中覆盖“gateway 存活 + 并发 CLI”“client 全部退出后继续 drain”“daemon 崩溃后 lease recovery”和“同 policy Skill single-flight”。
4. 文档中明确说明：CLI/session 退出只代表请求结束，不代表后台高级记忆已经全部生成；需要通过 health 中的 queue 状态判断完成情况。
5. 如果需要支持 Windows，应先实现并验证 named pipe 或本机 TCP transport。
