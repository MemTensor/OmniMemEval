# OmniMemEval 文档

[English](./README.md)

OmniMemEval 包含两条互补的评测线。请根据待评测对象选择对应入口：

| 评测线 | 评测对象 | 指南 | 结果 |
|---|---|---|---|
| User Memory Evaluation（用户记忆评测） | 通过 `add()` 和 `search()` 暴露能力的记忆后端 API | [指南](./user_memory/README_zh.md) | [结果](./user_memory/results_zh.md) |
| Agent Memory Evaluation（智能体记忆评测） | 有无记忆插件时的智能体运行时 | [指南](./agent_memory/README_zh.md) | [结果](./agent_memory/results_zh.md) |

参与 Agent Memory 开发时，还可以阅读当前的
[架构与扩展契约](./agent_memory/architecture_zh.md)。

## 阅读顺序

两份运行指南采用相同顺序：概览、环境与配置、数据准备、快速开始、进阶运行、
输出与结果、扩展与贡献。User Memory 另外介绍五个 benchmark。
建议先运行有明确范围限制的快速检查，再运行完整评测；两者都会调用真实服务。

中英文指南保持章节与可执行命令对应。结果数值只在英文结果页维护，
中文结果页提供指标说明和表格入口。产品名与代码标识在两种语言中保持原样。

## 相关参考

与具体文件直接相关的文档继续就近维护：

- 数据准备与许可证：`data/<benchmark>/README.md`
- 环境模板与参数：
  [env_examples/README.md（英文）](../env_examples/README.md) 和
  [env_examples/PARAMETERS.md（英文）](../env_examples/PARAMETERS.md)
- 测试命令：[scripts/tests/README.md（英文）](../scripts/tests/README.md)
- 开发与结果提交规范：[CONTRIBUTING.md（英文）](../CONTRIBUTING.md)
- 第三方归属：[THIRD_PARTY_NOTICES.md（英文）](../THIRD_PARTY_NOTICES.md)
