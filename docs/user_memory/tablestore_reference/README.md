# Alibaba Cloud Tablestore Memory Storage reference results

A benchmark contributor evaluated the managed [Tablestore Memory Storage service](https://www.alibabacloud.com/help/en/tablestore/memory-storage-service-sub-product-test/memory-storage-overview) in the Beijing region on November 26, 2026. The run used the [structured memory API](https://www.alibabacloud.com/help/en/tablestore/memory-storage-service-sub-product-test/structured-memory) and OmniMemEval's prompts and grading functions at [`7c79d46e`](https://github.com/MemTensor/OmniMemEval/tree/7c79d46e31dd85f3da2509b430bba9406656dd28). The service did not expose its deployed build or internal memory-extraction model in the captured store metadata.

| Benchmark | Correct / evaluated | Incorrect | Unscored | Accuracy | Reader prompt tokens / calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoCoMo, categories 1–4 | 1,359 / 1,540 | 181 | 0 | 88.25% | 4,510,033 / 1,540 |
| LongMemEval-S, with history omissions detailed below | 464 / 500 | 36 | 0 | 92.80% | 1,687,782 / 500 |

### Configuration and scoring

Both the Reader and Judge used Alibaba Cloud Model Studio's `qwen3.8-max` at temperature 0. The contributor used the provider-reported model ID. Backend-specific transport submitted messages to `AddMemories`, tracked asynchronous tasks with `GetMemoryTask`, and queried `SearchMemories` with the original question, `topK=20`, `enableRerank=true`, and `includeEvidence=false`. All 2,040 queries returned 20 structured memory texts. The Reader received those texts in their returned order. Original messages were read back separately to check ingestion. The Judge received the question, the original reference answer and the Reader output.

The retained run environments were inspected for this submission. LoCoMo used `tablestore-agent-storage` 1.0.10 for the gateway and `openai` 3.16.2 for model calls. LongMemEval-S used `tablestore-agent-storage` 1.0.11 for ingestion/readback and 1.0.10 for retrieval, with `openai` 3.57.0 for model calls.

The contributor performed one scored run per benchmark, with one accepted answer and one binary judgment per question. Work resumed from per-question checkpoints within the same run. A separate single-session LoCoMo smoke test contributed no results or tokens. All benchmark questions within the stated splits reached a final judgment. Accuracy is the count of correct judgments divided by the number of questions. Context Tokens is the mean of the Qwen provider's `usage.prompt_tokens` across accepted **Reader** calls, including the answer prompt. The displayed means are rounded to the nearest token. These results measure end-to-end retrieval, answering and grading with this model pair.

### LoCoMo

The source was `locomo10.json`, SHA-256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`. Categories 1–4 contain 1,540 questions across 10 conversations. The 446 category-5 adversarial questions were outside the split. No answer-key changes were made.

All 5,882 source messages across 272 sessions were submitted in 399 batches and read back with matching speaker, role, content and timestamp. Both human participants had role `user`, the original participant name in `speaker`, and the original utterance in `content`. Each question searched the sessions of its own conversation. The harness used OmniMemEval's shared-conversation context formatting, `locomo_response` and `locomo_grader`.

### LongMemEval-S

The source was `longmemeval_s_cleaned.json`, SHA-256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`. All 500 original questions, reference answers and question dates were retained. The source contains 23,867 sessions and 246,750 turns. The adapter kept each native `user` or `assistant` role and used its value as the message-level `speaker`. Empty turns (12) were skipped and four oversized turns were split losslessly to fit the API. The contributor additionally omitted 13 source turns across seven cases before indexing for content safety.

Of the 493 cases without these additional omissions, 457 were correct (92.70%). All seven adjusted cases were correct. The reported 92.80% describes the submitted histories. The 30 abstention questions are included in the denominator, with 27 correct. Each question searched only its own historical sessions. Dates were interpreted as UTC. The Reader received the question date through the official `lme_response` prompt. The original question alone was sent to retrieval. Grading used `lme_grader`.

All 246,730 submitted messages were read back. Of 24,134 write batches, 24,133 completed and one six-message batch failed during memory persistence because the service reported a duplicate primary key. An exact session readback found zero structured memories for that session. The affected question was scored against the available memories and judged correct.

### Verify the submission

The [LoCoMo records](./locomo.csv) use the original sample ID and zero-based QA index, so duplicate question strings remain separate. The [LongMemEval-S records](./longmemeval.csv) use the original question IDs. Both exports contain a category, final binary judgment and per-question Reader prompt-token count. The [summary](./summary.json) includes per-category correct counts, denominators, token totals and SHA-256 checksums of the score files.

Run these commands from the repository root using Python 3.11 or later:

```bash
python3 docs/user_memory/tablestore_reference/verify.py

# Optional: also check the IDs and categories against your copies of the official files.
python3 docs/user_memory/tablestore_reference/verify.py \
  --locomo /path/to/locomo10.json \
  --longmemeval /path/to/longmemeval_s_cleaned.json
```

The verifier checks the score exports offline. With dataset arguments it also checks the official source hashes and full ID/category coverage. The history-omission and ingestion counts above come from the contributor's private ingestion ledger. The public records support arithmetic and coverage checks. Independent replay of retrieval or model judgments requires the original service state and model inputs.

For run provenance, these were the operator's stage invocations, shown relative to the driver directory with interpreter paths and credentials omitted. The transport drivers were external to OmniMemEval; the public verifier above is the executable review path in this repository.

```bash
# LoCoMo
python bench.py start
python bench.py search
python score.py --watch
python score.py --all --groups 7 8 9 --workers 24
python bench.py readback

# LongMemEval-S
python longmem/runner.py plan
python longmem/runner.py setup
python longmem/runner.py canary
python longmem/runner.py prepare
python longmem/runner.py ingest --workers 96
python longmem/score.py --watch --workers 20
python longmem/runner.py readback --workers 8
```

LoCoMo used locks and input fingerprints to share checkpoints across the two scoring workers. LongMemEval-S used a per-case ingestion barrier before retrieval. Original dialogue, generated answer text, model requests, private endpoints and service identifiers are omitted from the public export.
