# OmniMemEval Documentation

[中文版](./README_zh.md)

OmniMemEval has two complementary evaluation tracks. Choose the track that
matches the system you want to evaluate:

| Track | Evaluation target | Guide | Results |
|---|---|---|---|
| User Memory Evaluation | Memory backend APIs exposed through `add()` and `search()` | [Guide](./user_memory/README.md) | [Results](./user_memory/results.md) |
| Agent Memory Evaluation | Agent runtimes equipped with memory plugins | [Guide](./agent_memory/README.md) | [Results](./agent_memory/results.md) |

Agent Memory contributors can also read the current
[architecture and extension contracts](./agent_memory/architecture.md).

Related documentation remains next to the files it describes:

- Dataset preparation and licenses: `data/<benchmark>/README.md`
- Environment templates and parameters:
  [env_examples/README.md](../env_examples/README.md) and
  [env_examples/PARAMETERS.md](../env_examples/PARAMETERS.md)
- Test commands: [scripts/tests/README.md](../scripts/tests/README.md)
- Contribution and result submission rules: [CONTRIBUTING.md](../CONTRIBUTING.md)
- Third-party attribution: [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)
