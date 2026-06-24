"""Download LoCoMo dataset from GitHub and prepare for evaluation.

Source repository:
  https://github.com/snap-research/locomo

The original dataset contains 10 long conversations with QA pairs,
event summaries, and observations.

Output files:
  - locomo10.json   (~2.8 MB, 10 conversations, original structure)

Usage:
    python prepare_locomo.py              # download dataset
    python prepare_locomo.py --force      # overwrite existing files

License: CC BY-NC 4.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

RAW_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def validate_locomo(path: str) -> None:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{path} is not valid JSON: {exc}") from exc

    if not isinstance(data, list) or len(data) != 10:
        raise RuntimeError(f"{path} should contain a top-level list of 10 conversations")


def download_url(url: str, output_path: str) -> None:
    tmp_path = f"{output_path}.tmp"
    try:
        urllib.request.urlretrieve(url, tmp_path)
        validate_locomo(tmp_path)
        os.replace(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def download_raw(force: bool = False) -> str:
    output_path = os.path.join(OUTPUT_DIR, "locomo10.json")
    if os.path.exists(output_path) and not force:
        validate_locomo(output_path)
        print("  locomo10.json already exists and is valid, skipping (use --force to overwrite)")
        return output_path

    print("  Downloading locomo10.json from GitHub ...")
    download_url(RAW_URL, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved locomo10.json ({size_mb:.1f} MB)")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download LoCoMo dataset from GitHub and prepare for evaluation."
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    print(f"Output directory: {OUTPUT_DIR}")
    print()

    try:
        download_raw(force=args.force)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    print()
    print("Done.")


if __name__ == "__main__":
    main()
