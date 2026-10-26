"""Recompute the published aggregates without model or memory-service calls."""
import collections
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    summary = json.loads((root / 'summary.json').read_text())
    for benchmark, expected in summary.items():
        source = root / f'{benchmark}.jsonl'
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected['public_records_sha256']
        records = [json.loads(line) for line in source.read_text().splitlines()]
        assert len(records) == expected['questions'] == expected['valid_questions']
        assert len({row['id'] for row in records}) == len(records)
        assert all(row['status'] == 'success' for row in records)
        if benchmark in ('longmemeval_s', 'personamem_v2'):
            assert {row['index'] for row in records} == set(range(len(records)))
        groups = collections.defaultdict(list)
        for row in records:
            assert 0 <= row['score'] <= 1
            if benchmark == 'beam100k':
                if row['scoring_method'] == 'per_rubric_item':
                    assert row['rubric_scores']
                    assert abs(sum(row['rubric_scores']) / len(row['rubric_scores']) - row['score']) < 1e-10
                else:
                    assert row['scoring_method'] == 'kendall_tau_b'
            else:
                assert isinstance(row['correct'], bool)
                assert row['score'] == float(row['correct'])
            groups[row['category']].append(row['score'])
        overall = sum(row['score'] for row in records) / len(records)
        assert abs(overall - expected['score']) < 1e-10
        assert set(groups) == set(expected['categories'])
        for category, values in groups.items():
            target = expected['categories'][category]
            assert len(values) == target['questions']
            assert abs(sum(values) / len(values) - target['score']) < 1e-10
        print(f'{benchmark}: {len(records)} questions, {overall:.2%}; coverage and hashes verified')


if __name__ == '__main__':
    main()
