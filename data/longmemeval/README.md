# LongMemEval Dataset

LongMemEval is a comprehensive benchmark for evaluating **long-term interactive
memory** in chat assistants, published by UCSB et al. It contains 500 carefully
designed questions across six categories, covering information extraction,
cross-session reasoning, temporal reasoning, knowledge update, preferences, and
abstention-style behavior.

- Paper: [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813) (ICLR 2025)
- Repository: https://github.com/xiaowu0162/LongMemEval
- Data source: Hugging Face [`xiaowu0162/longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
- License: MIT

## Files

| File | Size | Description |
|------|------|-------------|
| `longmemeval_oracle.json` | ~14.7 MB | Oracle version with evidence sessions only |
| `longmemeval_s_cleaned.json` | ~264.5 MB | LongMemEval-S: ~48 sessions / ~115K tokens per question |
| `longmemeval_m_cleaned.json` | ~2.6 GB | LongMemEval-M: ~500 sessions / ~1.5M tokens per question |

Only `README.md` and `prepare_longmemeval.py` are intended to be
version-controlled by MemEval. Downloaded JSON files are generated artifacts
and should remain local.

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total questions | 500 |
| S-version context | ~115K tokens / question (~48 sessions) |
| M-version context | ~1.5M tokens / question (~500 sessions) |
| Question categories | 6 |

## Question Categories

| Category | Count | Description |
|----------|-------|-------------|
| `temporal-reasoning` | 133 | Requires understanding temporal order and time-related information |
| `multi-session` | 133 | Requires integrating information across multiple sessions |
| `knowledge-update` | 78 | Information changes over time; must answer with the latest version |
| `single-session-user` | 70 | Extract user-related facts from a single session |
| `single-session-assistant` | 56 | Extract assistant-related facts from a single session |
| `single-session-preference` | 30 | Extract user preferences from a single session |

## Data Structure

Top-level JSON array of 500 question objects. Each question contains:

| Field | Type | Description |
|-------|------|-------------|
| `question_id` | string | Unique identifier, e.g., `"gpt4_2655b836"` |
| `question_type` | string | One of the 6 categories above |
| `question` | string | Question text |
| `answer` | string | Gold answer |
| `question_date` | string | Timestamp when the question is asked, e.g., `"2023/04/10 (Mon) 23:07"` |
| `haystack_dates` | list | Dates of haystack sessions |
| `haystack_session_ids` | list | IDs of haystack sessions |
| `haystack_sessions` | list | Full session dialogue content |
| `answer_session_ids` | list | Session IDs containing evidence for the answer |

The S and M versions share the same structure; M includes significantly more filler sessions.

## Data Preparation

The JSON files are downloaded directly from Hugging Face
[`xiaowu0162/longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned).
The default download is the S variant, matching MemEval's runner default.

```bash
python data/longmemeval/prepare_longmemeval.py                       # download S variant (~265 MB, used by default)
python data/longmemeval/prepare_longmemeval.py --variant oracle s m  # download all variants (M is ~2.6 GB)
python data/longmemeval/prepare_longmemeval.py --force               # overwrite existing files
```

The script validates each downloaded JSON file before replacing the local copy.

## License And Redistribution

`xiaowu0162/longmemeval-cleaned` is published on Hugging Face with the MIT
license. The MemEval code license does not override the upstream LongMemEval
dataset license.

## Evaluation

- **Metric**: LLM-as-a-Judge Accuracy (default judge: gpt-4o-mini)
- **Pipeline**: Ingest sessions → Search relevant context → Generate answer → LLM judge scores correctness
- **6 question categories**: Single-session user facts, assistant facts, preferences, multi-session reasoning, temporal reasoning, and knowledge update

## References

- Paper: [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813)
- Repository: https://github.com/xiaowu0162/LongMemEval
- Hugging Face: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
