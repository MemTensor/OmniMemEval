# Contributing to OmniMemEval

Contributions to adapters, evaluation pipelines, documentation, and public
benchmark results are welcome. Use an issue to report a reproducible problem
or discuss a substantial change, and a pull request to propose a focused change.

## Development workflow

Create a branch from the latest `main` (use a fork if you do not have write
access). Names such as `fix/search-status`, `chore/test-ci`, or
`docs/quickstart` describe the purpose; no specific prefix is required.
Keep one PR focused on one goal. All changes to `main` go through a PR.

For offline development, use Python 3.11 or 3.12 in a virtual or conda environment:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/check_doc_links.py
git diff --check
```

See the [test guide](scripts/tests/README.md) for targeted suites and repository
checks. Install [User Memory](docs/user_memory/README.md) or
[Agent Memory](docs/agent_memory/README.md) runtime dependencies separately when
running real evaluations. Record the dependency versions used for a run;
the requirements files are dependency lists, not locked reproduction environments.

Match the surrounding code style, avoid unrelated formatting changes, and add
a regression test for a behavior fix. Documentation-only changes should pass
the link checks; include a minimal before/after example when explaining a bug.

## Maintaining documentation

- Keep track guides in `docs/user_memory/` and `docs/agent_memory/`, with
  `README.md` for English and `README_zh.md` for Chinese. Use the existing
  `name.md` / `name_zh.md` convention for other paired pages; avoid renaming URLs
  just for style.
- Update both languages when changing guide sections, commands, options, or
  required setup. Translate explanations, not product names, paths, or flags.
- Maintain numeric result tables only in each track's English `results.md`.
  Chinese result pages provide translated context and links to the canonical
  tables rather than copying scores. Keep methodology explanations aligned.
- Keep parameter defaults, dataset/license details, and test commands in their
  existing reference files; link to them instead of duplicating full references.
  Distinguish bounded smoke checks, full evaluations, and destructive operations.
- Check file links and preserve existing heading anchors with aliases when
  reorganizing sections.
  The automated link checker validates relative file targets, not anchors or
  external URLs.

## Adding a User Memory adapter

The public entry points use `add(messages, user_id, ...)` and
`search(query, user_id, top_k, ...)`, plus a user-scoped cleanup method.
Inspect an existing adapter with similar behavior and its tests; product-specific
results are normalized by the search dispatch helpers.

1. Add the client under `scripts/client_factory/` and register its class in
   [registry.py](scripts/client_factory/registry.py). Keep optional SDK imports
   lazy so other backends remain usable without that SDK.
2. Register its search behavior in
   [search_helpers.py](scripts/utils/search_helpers.py) and the LoCoMo-specific
   dispatch in [locomo_search.py](scripts/locomo/locomo_search.py).
3. Check timestamp and session handling in
   [ingest_helpers.py](scripts/utils/ingest_helpers.py), along with any
   benchmark-specific ingestion behavior. Registering the client class alone
   does not define these behaviors.
4. Expose cleanup through a method supported by
   [delete_user_data](scripts/utils/streaming.py), such as `delete(user_id)`.
   Verify that deleting one evaluation user leaves all other users intact.
5. Add a credential-free template under `env_examples/`, dependency/setup
   instructions, and entries in the template index and both backend guide tables.
   Document the backend version, deployment mode, required settings, and
   non-default settings used by any submitted results.
6. Add offline tests for configuration, registration, add/search/cleanup,
   user isolation, and failures. Document a small real-backend smoke run and
   its version/configuration when integration has been tested.

For Agent Memory extensions, follow the separate
[architecture and extension contracts](docs/agent_memory/architecture.md).

## Submitting benchmark results

The result tables distinguish **Published Reference Results** from
**Reproduced Results**. Adding an adapter does not automatically add its scores.
Only submit benchmarks whose supporting materials are ready; other rows can
follow in another PR. Missing or unrun benchmarks use `-`.

For every proposed result, provide a stable public report/artifact link and:

- Run author and date; backend/deployment version; OmniMemEval or other harness
  commit; dataset revision or checksum, split, answer-key version, and any corrections.
- Evaluated question counts and coverage; failed, skipped, timed-out, and excluded
  items; the denominator used for the reported metric.
- Answer/judge model identifiers, prompts or prompt versions, retrieval settings,
  exact commands, and credential-free configuration.
- Scoring and aggregation rules, repetition counts, and how multiple runs are
  combined. Explain checkpoint reuse and any re-evaluation; do not combine retained
  high scores with score-selected reruns into an unexplained aggregate.
- Per-question records sufficient to check coverage and recompute the aggregate,
  along with summary files. Provide answer prompt-token totals and call counts
  when submitting `Context Tokens`; otherwise use `-`.

Link large artifacts in a public repository or release rather than adding them
to this repository's ignored `results/` directory. Share the records needed for
verification with credentials and private endpoints removed. Respect the
upstream dataset licenses listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

### Published Reference Results

Add rows to the relevant sections of
[docs/user_memory/results.md](docs/user_memory/results.md), with a direct source
link. These are externally reported results and may use different data,
models, prompts, or budgets. State such differences next to the result or in
the linked methodology. For the LoCoMo four-category table, identify the
non-adversarial subset; report any full-set score separately.

An adapter contribution and a new shared-baseline run are not prerequisites
for submitting published reference results.

### Reproduced Results

Use an available/reviewable OmniMemEval adapter and the shared pipeline,
dataset scope, prompts, metrics, and baseline answer/judge settings documented
in [Evaluation Setup](docs/user_memory/results.md#evaluation-setup).
PersonaMem v2 uses option matching rather than an LLM judge. Backend-specific
settings must be disclosed, including modes that generate answers directly.

Contributor-run results are eligible for review when these conditions and the
artifact requirements are met. Identify who ran them. The review should record
whether verification covered artifact arithmetic, replay of answer/judge stages,
or a complete rerun; these establish different levels of evidence.

## Review and merge

Describe the problem, change, and validation in the PR. If a command was not
run, say so. Separate blocking issues from optional suggestions and reply to
review comments when updating the PR. New commits may require another review.

CI checks offline tests, Python/Shell syntax, and relative Markdown file links.
It does not establish benchmark validity or run live integrations. Maintainers
review evidence and test results before merging; squash merging is the normal
choice for a focused PR. Follow-up fixes should use another branch and PR.
