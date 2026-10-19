# OmniMemEval 文档

[English](./README.md)

OmniMemEval 包含两条互补的评测线。请根据待评测对象选择对应入口：

| 评测线 | 评测对象 | 指南 | 结果 |
|---|---|---|---|
| User Memory Evaluation | 通过 `add()` 和 `search()` 暴露能力的 Memory Backend API | [指南](./user_memory/README_zh.md) | [结果（英文）](./user_memory/results.md) |
| Agent Memory Evaluation | 安装记忆插件后的 Agent Runtime | [指南](./agent_memory/README_zh.md) | [结果](./agent_memory/results_zh.md) |

参与 Agent Memory 开发时，还可以阅读当前的
[架构与扩展契约](./agent_memory/architecture_zh.md)。

与具体文件直接相关的文档继续就近维护：

- 数据准备与许可证：`data/<benchmark>/README.md`
- 环境模板与参数：
  [env_examples/README.md](../env_examples/README.md) 和
  [env_examples/PARAMETERS.md](../env_examples/PARAMETERS.md)
- 测试命令：[scripts/tests/README.md](../scripts/tests/README.md)
- 开发与结果提交规范：[CONTRIBUTING.md（英文）](../CONTRIBUTING.md)
- 第三方归属：[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)
