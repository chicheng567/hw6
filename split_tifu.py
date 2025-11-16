#!/usr/bin/env python3
"""Create train/val/public/private splits for the cleaned TIFU dataset."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

SplitRatios = Dict[str, int]


DEFAULT_RATIOS: SplitRatios = {
    "train": 7,
    "val": 1,
    "public_test": 1,
    "private_test": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        "-i",
        default="/workspace/dlhw/hw6/dataset/tifu/tifu_clean.jsonl",
        help="Input JSONL file produced by clean_tifu.py",
    )
    parser.add_argument(
        "--output-prefix",
        "-o",
        default="/workspace/dlhw/hw6/dataset/tifu/tifu",
        help="Prefix for the generated JSONL files (e.g. prefix_train.jsonl)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for the random number generator"
    )
    return parser.parse_args()


def allocate_counts(total_items: int, ratios: SplitRatios) -> Dict[str, int]:
    """Allocate counts for each split so they sum to total_items."""
    total_ratio = sum(ratios.values())
    raw_counts: List[Tuple[str, float]] = []
    counts: Dict[str, int] = {}

    for split, ratio in ratios.items():
        raw = total_items * ratio / total_ratio
        raw_counts.append((split, raw))
        counts[split] = int(raw)

    remainder = total_items - sum(counts.values())
    if remainder:
        raw_counts.sort(key=lambda item: item[1] - int(item[1]), reverse=True)
        for split, _ in raw_counts[:remainder]:
            counts[split] += 1

    return counts


def split_records(records: List[dict], ratios: SplitRatios, rng: random.Random) -> Dict[str, List[dict]]:
    """Shuffle the dataset and slice it according to the provided ratios."""
    rng.shuffle(records)
    counts = allocate_counts(len(records), ratios)

    split_map: Dict[str, List[dict]] = {}
    cursor = 0
    for split in ratios:
        count = counts[split]
        split_map[split] = records[cursor : cursor + count]
        cursor += count

    return split_map


def read_records(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_split(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            json.dump(record, fh, ensure_ascii=False)
            fh.write("\n")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    prefix = Path(args.output_prefix)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = read_records(input_path)
    rng = random.Random(args.seed)
    splits = split_records(records, DEFAULT_RATIOS, rng)

    for split_name, subset in splits.items():
        output_path = prefix.with_name(f"{prefix.name}_{split_name}.jsonl")
        write_split(output_path, subset)
        print(f"Wrote {len(subset)} records to {output_path}")


if __name__ == "__main__":
    main()
