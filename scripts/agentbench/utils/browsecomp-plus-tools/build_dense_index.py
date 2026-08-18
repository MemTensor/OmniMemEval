#!/usr/bin/env python3
"""Build a BrowseComp-Plus dense FAISS pickle index with a configurable embedding provider."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from tqdm import tqdm


_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _DIR.parents[3]
_SEARCHER_DIR = _DIR / "searcher"
if str(_SEARCHER_DIR) not in sys.path:
    sys.path.insert(0, str(_SEARCHER_DIR))

from embedding_client import EmbeddingConfig, create_embedding_client  # noqa: E402


def load_project_dotenv() -> None:
    env_path = _PROJECT_ROOT / ".env.agent"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())


def resolve_env_vars(obj: Any) -> Any:
    if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        return os.environ.get(obj[2:-1], "")
    if isinstance(obj, dict):
        return {k: resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_env_vars(v) for v in obj]
    return obj


def resolve_path(value: str | None, base_dir: Path) -> str | None:
    if not value or "://" in value:
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    project_path = (_PROJECT_ROOT / path).resolve()
    if value.startswith(".") and project_path.exists():
        return str(project_path)
    if value.startswith(".") or "/" in value:
        return str((base_dir / path).resolve())
    return value


def load_config(config_path: str) -> tuple[dict[str, Any], Path]:
    path = Path(config_path).resolve()
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return resolve_env_vars(raw), path.parent


def load_corpus(builder_cfg: dict[str, Any]) -> Iterable[tuple[str, str]]:
    docid_field = builder_cfg.get("docid_field", "docid")
    text_field = builder_cfg.get("text_field", "text")
    corpus_file = builder_cfg.get("corpus_file")

    if corpus_file:
        with open(corpus_file, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                row = json.loads(line)
                docid = str(row.get(docid_field, i))
                text = str(row.get(text_field, ""))
                if text:
                    yield docid, text
        return

    dataset_name = builder_cfg.get("corpus_dataset", "Tevatron/browsecomp-plus-corpus")
    dataset_split = builder_cfg.get("corpus_split", "train")
    dataset_cache = os.getenv("HF_DATASETS_CACHE") or None
    corpus_revision = builder_cfg.get("corpus_revision", "main")

    from datasets import load_dataset

    ds = load_dataset(
        dataset_name,
        split=dataset_split,
        cache_dir=dataset_cache,
        revision=corpus_revision,
    )
    for i, row in enumerate(ds):
        docid = str(row.get(docid_field, i))
        text = str(row.get(text_field, ""))
        if text:
            yield docid, text


def batched(items: Iterable[tuple[str, str]], size: int) -> Iterable[list[tuple[str, str]]]:
    batch: list[tuple[str, str]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def write_shard(output_dir: Path, shard_no: int, reps: list[list[float]], lookup: list[str], dtype: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(reps, dtype=dtype)
    path = output_dir / f"corpus.shard{shard_no:05d}.pkl"
    with open(path, "wb") as f:
        pickle.dump((arr, lookup), f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def main() -> int:
    load_project_dotenv()

    default_config = _DIR.parent.parent / "domains" / "information_retrieval" / "information_retrieval.yaml"
    parser = argparse.ArgumentParser(description="Build a configurable dense FAISS index for BrowseComp-Plus.")
    parser.add_argument("--config", default=str(default_config), help="Information retrieval domain yaml.")
    parser.add_argument("--output-dir", help="Override index_builder.output_dir.")
    parser.add_argument("--batch-size", type=int, help="Override embedding.batch_size.")
    parser.add_argument("--shard-size", type=int, help="Override index_builder.shard_size.")
    parser.add_argument("--corpus-revision", help="Override the corpus dataset revision.")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing corpus.shard*.pkl files first.")
    args = parser.parse_args()

    cfg, base_dir = load_config(args.config)
    embedding_cfg = dict(cfg.get("embedding") or {})
    builder_cfg = dict(cfg.get("index_builder") or {})

    if args.batch_size:
        embedding_cfg["batch_size"] = args.batch_size
    if args.shard_size:
        builder_cfg["shard_size"] = args.shard_size
    if args.corpus_revision:
        builder_cfg["corpus_revision"] = args.corpus_revision

    output_dir_raw = args.output_dir or builder_cfg.get("output_dir")
    if not output_dir_raw:
        raise ValueError("index_builder.output_dir is required, or pass --output-dir")

    output_dir = Path(resolve_path(output_dir_raw, base_dir) or output_dir_raw)
    builder_cfg["output_dir"] = str(output_dir)
    if builder_cfg.get("corpus_file"):
        builder_cfg["corpus_file"] = resolve_path(builder_cfg["corpus_file"], base_dir)

    shard_size = int(builder_cfg.get("shard_size", 50000))
    dtype = str(builder_cfg.get("dtype", "float32"))
    if dtype not in {"float32", "float16"}:
        raise ValueError("index_builder.dtype must be float32 or float16")

    if args.overwrite and output_dir.exists():
        for path in output_dir.glob("corpus.shard*.pkl"):
            path.unlink()
        metadata = output_dir / "metadata.json"
        if metadata.exists():
            metadata.unlink()

    if output_dir.exists() and list(output_dir.glob("corpus.shard*.pkl")) and not args.overwrite:
        raise FileExistsError(f"Index shards already exist in {output_dir}. Pass --overwrite to rebuild.")

    emb_config = EmbeddingConfig.from_dict(embedding_cfg)
    client = create_embedding_client(emb_config)

    total_docs = 0
    shard_no = 1
    shard_reps: list[list[float]] = []
    shard_lookup: list[str] = []
    vector_dim: int | None = None
    written: list[str] = []
    started = time.time()

    corpus = load_corpus(builder_cfg)
    for batch in tqdm(batched(corpus, emb_config.batch_size), desc="Embedding corpus"):
        docids = [docid for docid, _ in batch]
        texts = [text for _, text in batch]
        vectors = client.embed_documents(texts)
        if len(vectors) != len(docids):
            raise RuntimeError(f"embedding provider returned {len(vectors)} vectors for {len(docids)} texts")

        for docid, vec in zip(docids, vectors):
            if vector_dim is None:
                vector_dim = len(vec)
            elif len(vec) != vector_dim:
                raise ValueError(f"inconsistent vector dimension: expected {vector_dim}, got {len(vec)}")
            shard_lookup.append(docid)
            shard_reps.append(vec)
            total_docs += 1

            if len(shard_reps) >= shard_size:
                path = write_shard(output_dir, shard_no, shard_reps, shard_lookup, dtype)
                written.append(str(path))
                shard_no += 1
                shard_reps = []
                shard_lookup = []

    if shard_reps:
        path = write_shard(output_dir, shard_no, shard_reps, shard_lookup, dtype)
        written.append(str(path))

    if total_docs == 0:
        raise RuntimeError("No corpus documents were embedded")

    metadata = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "doc_count": total_docs,
        "dimension": vector_dim,
        "dtype": dtype,
        "metric": builder_cfg.get("metric", cfg.get("mcp_server", {}).get("metric", "inner_product")),
        "embedding": emb_config.redacted(),
        "corpus": {
            "corpus_file": builder_cfg.get("corpus_file"),
            "corpus_dataset": builder_cfg.get("corpus_dataset", "Tevatron/browsecomp-plus-corpus"),
            "corpus_split": builder_cfg.get("corpus_split", "train"),
            "corpus_revision": builder_cfg.get("corpus_revision", "main"),
            "docid_field": builder_cfg.get("docid_field", "docid"),
            "text_field": builder_cfg.get("text_field", "text"),
        },
        "shards": written,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Index written to: {output_dir}")
    print(f"Shards: {len(written)}")
    print(f"Documents: {total_docs}")
    print(f"Dimension: {vector_dim}")
    print("Update information_retrieval.yaml:")
    print(f"  mcp_server.searcher_type: configurable_faiss")
    print(f"  mcp_server.index_path: {output_dir / 'corpus.*.pkl'}")
    print(f"  mcp_server.index_metadata: {output_dir / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
