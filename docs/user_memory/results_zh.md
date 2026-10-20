# User Memory Evaluation（用户记忆评测）结果

[English / 完整结果表](./results.md) · [文档索引](../README_zh.md) ·
[运行指南](./README_zh.md)

本文提供中文阅读说明与结果表入口。数值统一维护在
[英文结果页](./results.md)，避免中英文表格更新不同步。

## 评测设置与可比性

结果分为两类：

- **统一流程复现结果（Reproduced Results）**：同一 benchmark 内使用共享数据、
  提示词、答案/评分模型与指标逻辑。后端部署和配置差异仍需披露。
- **公开参考结果（Published Reference Results）**：来自外部公开报告或贡献者自测，
  可能采用不同模型、数据、检索配置或评分方式，不能直接与统一流程复现结果比较。

模型标识、后端部署说明和指标定义见
[评测设置](./results.md#evaluation-setup)，各后端的整体分数见
[结果汇总](./results.md#result-summary)。横线 `-` 表示当前快照未收录可比的完整复现结果，
不表示零分。

## 指标与结果入口

| Benchmark | 主要指标 | 结果与范围说明 |
|---|---|---|
| LoCoMo | LLM 评分准确率 | [非对抗性问题子集与分类结果](./results.md#locomo) |
| LongMemEval | LLM 评分准确率 | [LongMemEval-S 与按题数加权的总体分数](./results.md#longmemeval) |
| BEAM | Nugget 分数 | [100K / 10M 的结果、范围与评分定义](./results.md#beam) |
| PersonaMem v2 | 选项匹配准确率 | [偏好类型与多次运行汇总](./results.md#personamem-v2) |
| HaluMem | LLM 评分准确率 | [问题类型与分类结果](./results.md#halumem) |

`Context Tokens` 是每道问题发送给答案模型的平均输入 token 数，
包含答案提示词和检索上下文，不是单纯的后端返回文本长度。
在准确率可比时，较低数值表示较高的 token 效率。
具体定义和各 benchmark 的汇总方式以英文结果页为准。

## 复现与提交结果

运行准备和命令见[运行指南](./README_zh.md)，
快照复现说明见[Reproduction Notes](./results.md#reproduction-notes)。

提交结果时，请附上运行作者、版本、数据范围、配置、指标汇总方式、
失败/跳过情况，以及可重算分数的公开逐题产物。
适配器接入并不自动证明其分数可收录。
完整要求见[结果提交规范（英文）](../../CONTRIBUTING.md#submitting-benchmark-results)。
