<a id="agentbench-评测结果"></a>
# Agent Memory Evaluation（智能体记忆评测）结果

[English / 完整结果表](./results.md) · [文档索引](../README_zh.md) ·
[运行指南](./README_zh.md) · [架构](./architecture_zh.md)

本文提供中文指标说明与结果表入口。数值统一维护在
[英文结果页](./results.md)，避免中英文表格更新不同步。

该快照记录 OpenClaw 和 Hermes 两套评测设置。
当前公开运行指南和已注册运行时适配器覆盖 OpenClaw，并未提供 Hermes 的复现路径。

## 评测指标

| 指标 | 含义 |
|---|---|
| Acc（%） | 同一任务 3 次独立运行的平均单次通过率 |
| Avg turns | 每个任务平均触发的模型回复轮次 |
| Avg chars | 每个任务回复文本的平均字符数 |

Acc 衡量单次运行的完成表现，不是 pass-at-3。
表格数值不使用排名或统计显著性标记；当前快照没有提供显著性检验说明。
服务可用性可能影响单条奖励分数。

## 数据与评测设置

数据来自 [EverMind-AI/EvoAgentBench](https://huggingface.co/datasets/EverMind-AI/EvoAgentBench)。
基线是对应智能体不启用记忆插件的运行结果；未标明云服务的产品使用本地服务和插件或集成。

快照记录的配置如下：

- OpenClaw：2026.5.7。
- Hermes：Hermes Agent v0.18.0（2026.7.1）。
- MemOS：Memos-Local-Plugin 2.0.8。
- 回答模型：qwen3.6-flash，no_thinking 模式。
- 评分模型：qwen3.6-flash，thinking 模式。

这些是快照配置，不代表当前安装命令会自动得到相同版本。
完整设置见[英文说明](./results.md#data-and-evaluation-setup)。

## 结果

### OpenClaw

查看 [OpenClaw 完整结果表](./results.md#openclaw)，包含五个任务域的准确率和交互成本指标。

### Hermes

查看 [Hermes 完整结果表](./results.md#hermes)。
此表保留历史评测记录，不表示当前公开 runner 支持该运行时的完整复现。

## 提交结果

请提供运行时/插件版本、协议、任务范围、运行与 trial 次数、配置、失败处理、
汇总方式，以及足以重算指标的稳定公开产物。
详见[结果证据要求（英文）](../../CONTRIBUTING.md#submitting-benchmark-results)。
