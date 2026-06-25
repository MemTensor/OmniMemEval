# LoCoMo Dataset

LoCoMo (Long Conversation Memory) is a benchmark dataset for evaluating
long-conversation memory capabilities. OmniMemEval uses 10 conversation samples
from the [original LoCoMo dataset](https://github.com/snap-research/locomo) to
assess memory systems' retrieval and reasoning abilities across extended
multi-session dialogues.

## Files

| File | Size | Description |
|------|------|-------------|
| `locomo10.json` | ~2.8 MB | Downloaded artifact with original structure: conversations, QA pairs, event summaries, observations |

`locomo10.json` is not version-controlled by OmniMemEval. Download it with
`prepare_locomo.py`; the upstream CC BY-NC 4.0 dataset license remains in force.

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Conversations | 10 |
| QA pairs | 1,986 |
| Sessions | 272 |
| Dialogue turns | 5,882 |
| Turns with images | 910 |
| Time span | Dec 2022 – Jan 2024 |

## Data Structure (`locomo10.json`)

Top-level JSON array of 10 sample objects. Each sample contains:

- **`sample_id`** — Unique identifier (e.g., `conv-26`, `conv-30`, ..., `conv-50`)
- **`conversation`** — Multi-session dialogue with `speaker_a`, `speaker_b`, timestamped sessions, and per-turn text/image data
- **`qa`** — Question-answer pairs with `question`, `answer` (or `adversarial_answer`), `evidence`, and `category`
- **`event_summary`** — Per-session event summaries for each speaker
- **`observation`** — Memory observation entries with dialogue references
- **`session_summary`** — Free-text summaries of each session

## QA Categories

| Category | Count | Description |
|----------|-------|-------------|
| 1 | 282 | **Multi-hop** — Requires integrating information across multiple sessions |
| 2 | 321 | **Temporal reasoning** — Requires understanding time-related information |
| 3 | 96 | **Open-domain** — Requires external or commonsense knowledge |
| 4 | 841 | **Single-hop** — Factual extraction from a single session |
| 5 | 446 | **Adversarial** — Contains misleading premises; excluded from evaluation |

Categories 1–4 (1,540 questions) are used for evaluation; category 5 is excluded.

## Conversation Participants

| sample_id | Speaker A | Speaker B |
|-----------|-----------|-----------|
| conv-26 | Caroline | Melanie |
| conv-30 | Jon | Gina |
| conv-41 | John | Maria |
| conv-42 | Joanna | Nate |
| conv-43 | Tim | John |
| conv-44 | Audrey | Andrew |
| conv-47 | James | John |
| conv-48 | Deborah | Jolene |
| conv-49 | Evan | Sam |
| conv-50 | Calvin | Dave |

## Data Preparation

The data file is sourced from the [LoCoMo GitHub repository](https://github.com/snap-research/locomo). To regenerate:

```bash
python data/locomo/prepare_locomo.py          # download locomo10.json
python data/locomo/prepare_locomo.py --force   # overwrite existing files
```

The script validates that the downloaded file is JSON with 10 top-level
conversation records before replacing the local copy.

## License

`locomo10.json` is downloaded from [`snap-research/locomo`](https://github.com/snap-research/locomo)
and follows the upstream dataset license,
[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://github.com/snap-research/locomo/blob/main/LICENSE.txt).
The OmniMemEval repository license applies to OmniMemEval code; it does not override the
upstream LoCoMo dataset license. Use of this data must remain non-commercial and
include attribution to the LoCoMo authors.

## Evaluation

- **Metric**: LLM-as-a-Judge Accuracy (default judge: gpt-4o-mini, `num_runs=1`)
- **Pipeline**: `locomo_ingestion.py` → `locomo_search.py` → `locomo_responses.py` → `locomo_eval.py` → `locomo_metric.py`
- **Entry script**: `scripts/run_locomo_eval.sh`
- **Dual-speaker design**: Most products create separate memory spaces per speaker; conversation-level products (Letta, Cognee, Hindsight, Backboard) share one agent/dataset/bank per conversation
- **Evaluated questions**: 1,540 (category 5 adversarial questions excluded)

## References

- Paper: [LoCoMo: Long-Context Conversation Understanding with Memory Operations](https://arxiv.org/abs/2402.17753)
- Repository: https://github.com/snap-research/locomo
