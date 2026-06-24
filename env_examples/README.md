# Environment Configuration Examples

This directory contains product-specific environment templates for the public
MemEval adapter layer. The open-source benchmark pipelines are LoCoMo and
LongMemEval.

## Quick Start

```bash
cp env_examples/.env.memos .env.memos
```

Fill in the required fields:

- Memory product credentials, for example `MEMOS_API_KEY` or `MEM0_API_KEY`.
- `ANSWER_MODEL`, `ANSWER_API_KEY`, `ANSWER_BASE_URL`.
- `EVAL_MODEL`, `EVAL_API_KEY`, `EVAL_BASE_URL`.

Then run:

```bash
./scripts/run_locomo_eval.sh --lib memos --env .env.memos
./scripts/run_lme_eval.sh --lib memos --env .env.memos
```

See [PARAMETERS.md](./PARAMETERS.md) for shared runner and LLM settings.

## Available Templates

| File | Product |
|------|---------|
| `.env.memos` | MemOS |
| `.env.mem0` | Mem0 |
| `.env.zep` | Zep |
| `.env.supermemory` | Supermemory |
| `.env.everos` | EverOS |
| `.env.letta` | Letta |
| `.env.hindsight` | Hindsight |
| `.env.graphiti` | Graphiti local/self-hosted |
| `.env.cognee` | Cognee |
| `.env.viking` | Viking Memory |
| `.env.memori` | Memori |
| `.env.memmachine` | MemMachine |
| `.env.memorylake` | MemoryLake |
| `.env.backboard` | Backboard.io |
| `.env.mem9` | mem9 |

Templates include only placeholders and public defaults. Do not commit real
`.env.*` files at the repository root.
