"""Recompute the Tablestore reference scores and optionally check official dataset IDs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FILES = {
    "locomo": ("sample_id", "qa_index", "category", "correct", "reader_prompt_tokens"),
    "longmemeval": ("question_id", "question_type", "is_abstention", "correct", "reader_prompt_tokens"),
}
COUNTS = {"locomo": 1540, "longmemeval": 500}
DATA_SHA256 = {
    "locomo": "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
    "longmemeval": "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def key_and_category(dataset, row):
    if dataset == "locomo":
        return (row["sample_id"], int(row["qa_index"])), row["category"]
    return row["question_id"], row["question_type"]


def summarize(dataset):
    path = ROOT / (dataset + ".csv")
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        require(tuple(reader.fieldnames or ()) == FILES[dataset], f"{dataset}: columns")
        rows = list(reader)
    categories = {}
    tokens = 0
    correct = 0
    abstentions = {"correct": 0, "questions": 0}
    keys = {}
    for row in rows:
        key, category = key_and_category(dataset, row)
        require(key not in keys and all(row.values()), f"{dataset}: duplicate key or missing field")
        keys[key] = category
        require(row["correct"] in {"0", "1"}, f"{dataset}: invalid judgment")
        hit = int(row["correct"])
        count = int(row["reader_prompt_tokens"])
        require(count > 0, f"{dataset}: invalid Reader usage")
        tokens += count
        correct += hit
        tally = categories.setdefault(category, {"correct": 0, "questions": 0})
        tally["correct"] += hit
        tally["questions"] += 1
        if dataset == "longmemeval":
            require(row["is_abstention"] == str(int("_abs" in key)), "abstention flag mismatch")
            if row["is_abstention"] == "1":
                abstentions["correct"] += hit
                abstentions["questions"] += 1
    require(len(rows) == COUNTS[dataset], f"{dataset}: incomplete coverage")
    stats = {
        "questions": len(rows), "correct": correct, "incorrect": len(rows) - correct,
        "unscored": 0, "by_category": categories, "reader_calls": len(rows),
        "reader_prompt_tokens": tokens, "records_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if dataset == "longmemeval":
        stats["abstention"] = abstentions
    return stats, keys


def check_dataset(dataset, path, keys):
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == DATA_SHA256[dataset], f"{dataset}: unexpected source file hash")
    source = json.loads(raw)
    if dataset == "locomo":
        expected = {(item["sample_id"], ix): str(qa["category"]) for item in source
                    for ix, qa in enumerate(item["qa"]) if qa["category"] in (1, 2, 3, 4)}
    else:
        expected = {item["question_id"]: item["question_type"] for item in source}
    require(expected == keys, f"{dataset}: public records differ from the official IDs or categories")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locomo", type=Path, help="optional official locomo10.json")
    parser.add_argument("--longmemeval", type=Path, help="optional official longmemeval_s_cleaned.json")
    args = parser.parse_args()
    summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    for name in FILES:
        stats, keys = summarize(name)
        require(stats == summary[name], f"{name}: record hash, usage or counts differ from summary")
        source = getattr(args, name)
        if source is not None:
            check_dataset(name, source, keys)
        print(f"{name}: {stats['correct']}/{stats['questions']} ({100 * stats['correct'] / stats['questions']:.2f}%), "
              f"{stats['reader_prompt_tokens']} prompt tokens / {stats['reader_calls']} Reader calls")


if __name__ == "__main__":
    main()
