"""
Download and inspect the StoryReasoning dataset.
"""
from __future__ import annotations

import argparse
import os

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download StoryReasoning dataset.")
    parser.add_argument("--cache_dir", type=str, default="hf_cache")
    parser.add_argument("--dataset_id", type=str, default="daniel3303/StoryReasoning")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    ds = load_dataset(args.dataset_id, cache_dir=args.cache_dir)
    print(ds)
    for split in ds.keys():
        print(f"\n--- Split: {split} ---")
        print("Num rows:", len(ds[split]))
        print("Columns:", ds[split].column_names)


if __name__ == "__main__":
    main()
