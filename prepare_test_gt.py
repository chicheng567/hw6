#!/usr/bin/env python3
"""Merge test splits into a Kaggle-ready ground-truth CSV."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tifu-dir",
        default="/workspace/dlhw/hw6/dataset/tifu",
        help="Directory containing tifu_public/private_test.jsonl files.",
    )
    parser.add_argument(
        "--samsun-test",
        default="/workspace/dlhw/hw6/dataset/samsun/test.csv",
        help="Path to the samsun test CSV file.",
    )
    parser.add_argument(
        "--output",
        default="/workspace/dlhw/hw6/dataset/gt.csv",
        help="Destination CSV for Kaggle upload.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Samsun public/private assignment.",
    )
    parser.add_argument(
        "--write-samsun-splits",
        action="store_true",
        help="If set, also write samsun test_public/test_private CSVs alongside the GT.",
    )
    return parser.parse_args()


def read_tifu_records(tifu_dir: Path) -> List[Tuple[str, str, str]]:
    records: List[Tuple[str, str, str]] = []
    for split_name, usage in (("public_test", "Public"), ("private_test", "Private")):
        path = tifu_dir / f"tifu_{split_name}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                instance = record.get("instance", {})
                url = instance.get("url") or record.get("id")
                summary = instance.get("summary", "")
                if not url:
                    continue
                records.append((url, usage, summary))
    return records


def read_samsun_rows(test_path: Path) -> List[Dict[str, str]]:
    with test_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def write_gt_csv(path: Path, records: Iterable[Tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "Usage", "summary"])
        for row in records:
            writer.writerow(row)


def write_samsun_split_csv(rows: List[Dict[str, str]], usage_assignments: Dict[int, str], base_path: Path) -> None:
    for usage in ("Public", "Private"):
        path = base_path.parent / f"{base_path.stem}_{usage.lower()}.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["id", "dialogue", "summary"])
            writer.writeheader()
            for idx, row in enumerate(rows):
                if usage_assignments[idx] == usage:
                    writer.writerow(row)


def main() -> None:
    args = parse_args()
    tifu_records = read_tifu_records(Path(args.tifu_dir))

    samsun_path = Path(args.samsun_test)
    samsun_rows = read_samsun_rows(samsun_path)
    rng = random.Random(args.seed)
    indices = list(range(len(samsun_rows)))
    rng.shuffle(indices)
    public_count = len(samsun_rows) // 2
    public_indices = set(indices[:public_count])
    assignments = {idx: ("Public" if idx in public_indices else "Private") for idx in range(len(samsun_rows))}

    samsun_records = [
        (row["id"], assignments[idx], row.get("summary", "")) for idx, row in enumerate(samsun_rows)
    ]

    if args.write_samsun_splits:
        write_samsun_split_csv(samsun_rows, assignments, samsun_path)

    combined = tifu_records + samsun_records
    write_gt_csv(Path(args.output), combined)
    print(f"Wrote {len(combined)} rows to {args.output}")


if __name__ == "__main__":
    main()
