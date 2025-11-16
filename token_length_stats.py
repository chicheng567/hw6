#!/usr/bin/env python3
"""Measure ModernBERT token-length distributions for the project datasets."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Sequence, Tuple

from transformers import AutoTokenizer

Tokenizer = AutoTokenizer

BOUNDS = [64, 128, 256, 512, 1024, 2048, 4096, 8192]


@dataclass
class LengthStats:
    count: int
    minimum: int
    maximum: int
    mean: float
    median: float
    p90: float
    p95: float
    p99: float
    buckets: Dict[str, int]


def percentile(sorted_vals: Sequence[int], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * pct / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    lower = sorted_vals[f]
    upper = sorted_vals[c]
    return lower + (upper - lower) * (k - f)


def bucketize(lengths: Iterable[int]) -> Dict[str, int]:
    template = [f"<= {bound}" for bound in BOUNDS] + ["> {}".format(BOUNDS[-1])]
    counts = {label: 0 for label in template}
    for length in lengths:
        placed = False
        for bound in BOUNDS:
            if length <= bound:
                counts[f"<= {bound}"] += 1
                placed = True
                break
        if not placed:
            counts[f"> {BOUNDS[-1]}"] += 1
    return counts


def describe(lengths: List[int]) -> LengthStats:
    if not lengths:
        return LengthStats(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, {})
    sorted_vals = sorted(lengths)
    return LengthStats(
        count=len(sorted_vals),
        minimum=sorted_vals[0],
        maximum=sorted_vals[-1],
        mean=mean(sorted_vals),
        median=median(sorted_vals),
        p90=percentile(sorted_vals, 90),
        p95=percentile(sorted_vals, 95),
        p99=percentile(sorted_vals, 99),
        buckets=bucketize(sorted_vals),
    )


def iter_samsun(base_dir: Path) -> Iterable[Tuple[str, str]]:
    for split in ("train", "validation", "test"):
        path = base_dir / f"{split}.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                dialogue = row.get("dialogue", "") or ""
                summary = row.get("summary", "") or ""
                yield dialogue, summary


def iter_tifu(base_dir: Path) -> Iterable[Tuple[str, str]]:
    for split in ("train", "val", "public_test", "private_test"):
        path = base_dir / f"tifu_{split}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                instance = record.get("instance", {})
                source = instance.get("selftext_without_tldr", "") or ""
                summary = instance.get("summary", "") or ""
                yield source, summary


def compute_lengths(records: Iterable[Tuple[str, str]], tokenizer) -> Dict[str, LengthStats]:
    source_lengths: List[int] = []
    summary_lengths: List[int] = []
    for source, summary in records:
        source_lengths.append(len(tokenizer.encode(source, add_special_tokens=True)))
        summary_lengths.append(len(tokenizer.encode(summary, add_special_tokens=True)))
    return {
        "source": describe(source_lengths),
        "summary": describe(summary_lengths),
    }


def main() -> None:
    tokenizer = Tokenizer.from_pretrained("answerdotai/ModernBERT-base")

    samsun_dir = Path("dataset/samsun")
    tifu_dir = Path("dataset/tifu")

    results = {
        "samsun": compute_lengths(iter_samsun(samsun_dir), tokenizer),
        "tifu": compute_lengths(iter_tifu(tifu_dir), tokenizer),
    }

    for dataset, stats in results.items():
        print(f"\nDataset: {dataset}")
        for split_name, detail in stats.items():
            print(f"  {split_name} sequences:")
            print(
                "    count={count} min={minimum} max={maximum} mean={mean:.1f} "
                "median={median:.1f} p90={p90:.1f} p95={p95:.1f} p99={p99:.1f}".format(
                    **detail.__dict__
                )
            )
            print("    buckets:")
            for bucket, value in detail.buckets.items():
                print(f"      {bucket:>7}: {value}")


if __name__ == "__main__":
    main()
