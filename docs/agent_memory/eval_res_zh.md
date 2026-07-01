# AgentBench 评测结果

[English](./eval_res.md)

## 评测指标

> Acc：统计同一任务 3 次独立运行中的平均一次通过率，用于衡量单次运行的稳定完成情况。

> Avg turns：表示每个任务中 agent 平均触发的模型回复轮次，用于衡量完成任务过程中的交互/推理迭代深度。

> Avg chars：Agent 回答文本的平均字符长度，用于衡量输出答案的文本长度。

## 数据与评测设置

数据集来源及切分逻辑：

数据参考：[https://huggingface.co/datasets/EverMind-AI/EvoAgentBench](https://huggingface.co/datasets/EverMind-AI/EvoAgentBench)

测评 Agent：OpenClaw + 对应产品的插件。

baseline 是指 OpenClaw 不加任何插件运行的结果。

评测使用的 OpenClaw 版本为 2026.5.7。

未标明云服务的产品，均使用本地部署的服务和对应的 OpenClaw 插件。

MemOS 使用 Memos-Local-Plugin 2.0.8 版本。

OpenClaw 回答模型配置：qwen3.6-flash no_thinking 模式。

评测判别模型：qwen3.6-flash thinking 模式。

## 结果

| **Method** | BrowseComp-Plus Acc | BrowseComp-Plus Avg turns | OmniMath <br>Acc | OmniMath Cost <br>Avg chars（单位：k tokens） | SWE-Bench <br>Acc | SWE-Bench <br>Avg turns | LiveCodeBench <br>Acc | LiveCodeBench Avg turns | GDPVal <br>Acc | GDPVal <br>Avg turns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Baseline** | 18.46 | 35.1 | 52 | 5658.6 | 26.92 | 58.8 | 51.28 | 23.9 | 34.48 | 17.2 |
| **Mem0** | 13.33 | 36.4 | 53 | 6090.53 | 26.92 | 55.78 | 60.68 | 6.27 | 41.38 | 17.28 |
| **EverOS** | 18.98 | 30.8 | 54.67 | 6085.87 | 32.05 | 54.77 | 59.83 | 7.2 | 38.51 | 16.97 |
| **Supermemory** | 14.36 | 32.03 | 58 | 5632.1 | 30.7 | 51.37 | 59.83 | 6.83 | 43.10 | 16.70 |
| **Hindsight** | 16.92 | 35.6 | 55 | 6177 | 38.46 | 56.7 | 71.79 | 14.2 | 50.00 | 17.5 |
| **mem9**<br>**（云服务）** | 12.31 | 31.6 | 54.33 | 5880.9 | 30.77 | 59.1 | 64.1 | 8.23 | 47.7 | 15.8 |
| **Memori**<br>**（云服务）** | 16.92 | 36.3 | 52 | 6189.2 | 34.62 | 43.1 | 51.28 | 9.3 | 37.93 | 15.8 |
| **MemOS** | **23.85*** | 43.3 | **61*** | 5164.2 | **38.46*** | 49.55 | <u>64.96</u> | 9.43 | **62.07*** | 23.93 |
