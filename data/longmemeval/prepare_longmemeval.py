"""Download LongMemEval dataset from Hugging Face.

Source dataset:
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned

The dataset provides three JSON files directly (not Parquet), hosted as
individual files in the Hugging Face repository.

Output files:
  - longmemeval_oracle.json      (~15 MB,  500 questions, evidence sessions only)
  - longmemeval_s_cleaned.json   (~265 MB, 500 questions, ~48 sessions each)
  - longmemeval_m_cleaned.json   (~2.6 GB, 500 questions, ~500 sessions each)

Usage:
    python prepare_longmemeval.py                          # download S variant (used by default)
    python prepare_longmemeval.py --variant oracle s m     # download all variants
    python prepare_longmemeval.py --variant oracle         # download oracle only (~15 MB)
    python prepare_longmemeval.py --force                  # overwrite existing files

License: MIT
"""

import argparse
import json
import os
import sys
import urllib.request

REPO_BASE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"

VARIANT_MAP = {
    "oracle": "longmemeval_oracle.json",
    "s":      "longmemeval_s_cleaned.json",
    "m":      "longmemeval_m_cleaned.json",
}

ALL_VARIANTS = list(VARIANT_MAP.keys())
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def validate_json_array(path: str) -> int:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{path} is not valid JSON: {exc}") from exc

    if not isinstance(data, list) or not data:
        raise RuntimeError(f"{path} should contain a non-empty top-level JSON array")
    return len(data)


def download_url(url: str, output_path: str) -> None:
    tmp_path = f"{output_path}.tmp"
    try:
        urllib.request.urlretrieve(url, tmp_path)
        validate_json_array(tmp_path)
        os.replace(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def download_variant(variant: str, force: bool = False) -> None:
    filename = VARIANT_MAP[variant]
    output_path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(output_path) and not force:
        records = validate_json_array(output_path)
        print(f"  [{variant}] {filename} already exists and has {records} records, skipping (use --force to overwrite)")
        return

    url = f"{REPO_BASE}/{filename}"
    print(f"  [{variant}] Downloading {filename} ...")

    download_url(url, output_path)
    records = validate_json_array(output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  [{variant}] Saved {filename} ({records} records, {size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download LongMemEval dataset from Hugging Face."
    )
    parser.add_argument(
        "--variant",
        nargs="+",
        choices=ALL_VARIANTS,
        default=["s"],
        help="Variant(s) to download (default: s). Evaluation uses S by default. 'm' is ~2.6 GB.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Variants to download: {', '.join(args.variant)}")
    print()

    try:
        for variant in args.variant:
            download_variant(variant, force=args.force)
            print()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()
