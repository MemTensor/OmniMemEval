"""Export score-only audit records from privately retained result directories.

This deliberately excludes questions, answers, conversation content, user IDs,
retrieval context, endpoints, logs and environment snapshots.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

RUNS = {
    'locomo': 'locomo/hojo-hojo_locomo_full_max10k_20261012a',
    'beam100k': 'beam100k/hojo_beam100k_grounded_full_p6_20261015a',
    'personamem_v2': 'personamem_v2/hojo_personamem_v2_full_p3_20261002_retry1',
    'halumem': 'halumem/hojo_hm_dev_gpt_full_20261022c',
    'longmemeval_s': 'longmemeval/hojo_lme_full_gpt_20261022a_completed',
}
JUDGED = {
    'locomo': 'hojo_locomo_judged.json', 'beam100k': 'hojo_beam_judged.json',
    'personamem_v2': 'hojo_personamem_v2_judged.json',
    'halumem': 'hojo_hm_judged.json', 'longmemeval_s': 'hojo_lme_judged.json',
}
COUNTS = {'locomo': 1540, 'beam100k': 400, 'personamem_v2': 5000,
          'halumem': 3467, 'longmemeval_s': 500}
LOCOMO_CATEGORIES = {1: 'multi-hop', 2: 'temporal', 3: 'open-domain', 4: 'single-hop'}
HARNESS_SOURCE_OVERRIDES = {
    # Preserve the v3 prompt/retrieval behavior recorded for the 69.58% run.
    'run_omnimemeval_beam100k.py': 'e983688562daf22cece0fc050bea76d1a4aee4a8',
}
SAFE_CONFIG = {
    'version', 'benchmark', 'dataset_sha256', 'dataset_revision', 'top_k',
    'candidate_limit', 'min_score', 'max_tokens', 'answer_model', 'eval_model',
    'workers', 'ingest_mode', 'answer_profile', 'time_profile', 'strict_fair',
    'method', 'action_query_expansion', 'multi_pass_retrieval',
    'assistant_response_query_expansion', 'ordering_summary_candidate_limit',
    'ordering_summary_min_score', 'broad_max_tokens', 'prompt_profile',
    'retrieval_profile', 'context_profile', 'evidence_profile',
    'assistant_recall_profile', 'index_assistant_messages', 'scope',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--service-source', type=Path, help='Optional MemoryEngine source checkout for a credential-free harness snapshot')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for benchmark, relative in RUNS.items():
        root = args.source_root / relative
        source = root / JUDGED[benchmark]
        payload = json.loads(source.read_text())
        records = []
        for group_index, (key, value) in enumerate(payload.items()):
            values = value if isinstance(value, list) else [value]
            for question_index, item in enumerate(values):
                row = {'status': item.get('status', 'success')}
                category = item.get('category', item.get('dimension', 'unknown'))
                if benchmark == 'locomo':
                    category = LOCOMO_CATEGORIES.get(int(category), str(category))
                    row.update(conversation_index=group_index, question_index=question_index)
                    row['id'] = f'{group_index}:{question_index}'
                elif benchmark == 'halumem':
                    # The exported key is converted to numeric coordinates only.
                    parts = str(item.get('key', key)).split(':')
                    numeric = [int(part) for part in parts if part.isdigit()]
                    row['id'] = ':'.join(map(str, numeric)) if numeric else str(group_index)
                    row.update(session_index=item.get('session_idx'), question_index=item.get('question_idx'))
                elif benchmark == 'longmemeval_s':
                    row['index'] = item['index']
                    row['id'] = str(item['index'])
                elif benchmark == 'personamem_v2':
                    row['index'] = item['row_index']
                    row['id'] = str(item['row_index'])
                else:
                    # Dataset conversation/question coordinates, without service user IDs.
                    parts = str(item.get('key', key)).split(':')
                    row['id'] = str(group_index)
                    row['question_index'] = int(parts[-1]) if parts[-1].isdigit() else group_index
                    row['rubric_scores'] = [float(value['score']) for value in item.get('rubric_item_scores', [])]
                row['category'] = category
                question = item.get('question')
                if isinstance(question, str):
                    row['question_sha256'] = hashlib.sha256(question.encode()).hexdigest()
                if benchmark == 'beam100k':
                    row['score'] = float(item['nugget_score'])
                    row['scoring_method'] = item['scoring_method']
                    if row['scoring_method'] == 'per_rubric_item':
                        assert row['rubric_scores']
                        assert abs(sum(row['rubric_scores']) / len(row['rubric_scores']) - row['score']) < 1e-10
                    else:
                        assert row['scoring_method'] == 'kendall_tau_b'
                else:
                    judgment = item.get('llm_judgments', {}).get('judgment_1')
                    correct = (item['is_correct'] if benchmark == 'personamem_v2' else
                               item['correct'] if benchmark == 'longmemeval_s' else
                               judgment == 'CORRECT' if isinstance(judgment, str) else judgment)
                    assert isinstance(correct, bool), (benchmark, row['id'])
                    row['correct'] = correct
                    row['score'] = float(correct)
                records.append(row)
        assert len(records) == COUNTS[benchmark]
        assert len({row['id'] for row in records}) == len(records)
        assert all(row['status'] == 'success' for row in records)
        if benchmark in ('longmemeval_s', 'personamem_v2'):
            assert {row['index'] for row in records} == set(range(COUNTS[benchmark]))
        output = args.output_dir / f'{benchmark}.jsonl'
        output.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in records))
        categories = collections.defaultdict(list)
        for row in records:
            categories[row['category']].append(row['score'])
        config = {}
        report = {}
        for filename in ['run-state.json', 'exp_report.json', 'effective_retrieval_config.json']:
            path = root / filename
            if path.exists():
                data = json.loads(path.read_text())
                for mapping in [data, data.get('config', {}), data.get('configuration', {})]:
                    config.update({key: value for key, value in mapping.items() if key in SAFE_CONFIG})
                if filename == 'exp_report.json':
                    report = data
        if benchmark == 'longmemeval_s':
            shard = root / 'shards/worker-01/run-state.json'
            data = json.loads(shard.read_text())['config']
            config.update({key: value for key, value in data.items() if key in SAFE_CONFIG and key != 'workers'})
        if benchmark == 'locomo':
            config.update(version=root.name.removeprefix('hojo-'),
                          answer_model='gpt-4.1-mini-2025-04-14', eval_model='gpt-4o-mini-2024-07-18')
        summary = {
            'version': config.get('version', root.name), 'questions': len(records),
            'valid_questions': len(records), 'execution_failures': 0, 'excluded_questions': 0,
            'score': sum(row['score'] for row in records) / len(records),
            'aggregation': 'arithmetic mean of unrounded per-question scores',
            'categories': {key: {'questions': len(values), 'score': sum(values) / len(values)}
                           for key, values in sorted(categories.items())},
            'configuration': config, 'source_judged_sha256': sha(source),
            'public_records_sha256': sha(output),
        }
        if 'elapsed_seconds' in report:
            summary['elapsed_seconds'] = report['elapsed_seconds']
        if benchmark == 'locomo':
            report_path = root / 'exp_report.md'
            text = report_path.read_text()
            context_tokens = re.search(r'\| Context Tokens \(avg\) \| ([\d,.]+) \|', text)
            answer_calls = re.search(r'\| Answer API Calls \| ([\d,]+) \|', text)
            assert context_tokens and answer_calls
            summary['context_tokens_mean'] = float(context_tokens.group(1).replace(',', ''))
            summary['answer_api_calls'] = int(answer_calls.group(1).replace(',', ''))
            summary['source_report_sha256'] = sha(report_path)
        summaries[benchmark] = summary
    (args.output_dir / 'summary.json').write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + '\n')
    if args.service_source:
        names = ['run_omnimemeval_lme.py', 'run_omnimemeval_lme_parallel.py',
                 'run_omnimemeval_halumem.py', 'run_omnimemeval_beam100k.py',
                 'run_omnimemeval_personamem_v2.py', 'omnimemeval_badcase_ids.py']
        scripts = args.output_dir / 'harness/scripts'
        scripts.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for name in names:
            source = args.service_source / 'backend/scripts' / name
            destination = scripts / name
            if name in HARNESS_SOURCE_OVERRIDES:
                content = subprocess.check_output([
                    'git', '-C', str(args.service_source), 'show',
                    f'{HARNESS_SOURCE_OVERRIDES[name]}:backend/scripts/{name}',
                ])
                destination.write_bytes(content)
            else:
                shutil.copyfile(source, destination)
            hashes[name] = sha(destination)
        commit = subprocess.check_output(['git', '-C', str(args.service_source), 'rev-parse', 'HEAD'], text=True).strip()
        provenance = {'snapshot_commit': commit,
                      'file_source_commits': HARNESS_SOURCE_OVERRIDES,
                      'files_sha256': hashes,
                      'note': 'Reference harness source snapshot; this is not proof of the deployed service commit for historical runs.'}
        (args.output_dir / 'harness/source_manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    for benchmark, summary in summaries.items():
        print(f'{benchmark}: {summary["questions"]} questions, {summary["score"]:.2%}')


if __name__ == '__main__':
    main()
