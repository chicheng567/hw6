#!/usr/bin/env python3
"""Unified dataset preparation pipeline for TIFU and Samsung splits."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from clean_tifu import convert_dataset
from split_tifu import DEFAULT_RATIOS, split_records, write_split, read_records

UsageRow = Tuple[str, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean the TIFU dataset, split it into train/val/test, sanitize test splits, "
            "strip Samsung test ground-truth, and write the consolidated gt.csv file."
        )
    )
    parser.add_argument(
        "--tifu-raw",
        type=Path,
        default=Path("dataset/tifu/tifu_all_tokenized_and_filtered.json"),
        help="Path to the raw Reddit TIFU JSON dump.",
    )
    parser.add_argument(
        "--tifu-clean",
        type=Path,
        default=Path("dataset/tifu/tifu_clean.jsonl"),
        help="Destination JSONL file produced after cleaning the TIFU dump.",
    )
    parser.add_argument(
        "--tifu-prefix",
        type=Path,
        default=Path("dataset/tifu/tifu"),
        help="Prefix for the generated TIFU splits (train/val).",
    )
    parser.add_argument(
        "--tifu-test-output",
        type=Path,
        default=Path("dataset/tifu/tifu_test.jsonl"),
        help="Path to the sanitized TIFU test set without ground-truth summaries.",
    )
    parser.add_argument(
        "--samsun-test",
        type=Path,
        default=Path("dataset/samsun/test.csv"),
        help="CSV file containing the Samsung test examples (will be sanitized in-place).",
    )
    parser.add_argument(
        "--gt-output",
        type=Path,
        default=Path("dataset/gt.csv"),
        help="Output CSV that stores ground-truth labels for every test item.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used to deterministically split the cleaned TIFU dataset.",
    )
    return parser.parse_args()


def write_jsonl(records: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            json.dump(record, fh, ensure_ascii=False)
            fh.write("\n")


def prepare_tifu_splits(
    raw_path: Path,
    clean_path: Path,
    prefix: Path,
    test_output: Path,
    seed: int,
) -> List[UsageRow]:
    print(f"Cleaning TIFU raw dump from {raw_path} -> {clean_path}")
    convert_dataset(raw_path, clean_path)
    records = read_records(clean_path)
    rng = random.Random(seed)
    splits = split_records(records, DEFAULT_RATIOS, rng)

    train_path = prefix.with_name(f"{prefix.name}_train.jsonl")
    val_path = prefix.with_name(f"{prefix.name}_val.jsonl")
    print(f"Writing {len(splits['train'])} TIFU train records -> {train_path}")
    write_split(train_path, splits["train"])
    print(f"Writing {len(splits['val'])} TIFU validation records -> {val_path}")
    write_split(val_path, splits["val"])

    usage_rows = sanitize_tifu_test_split(
        test_output,
        public_records=splits["public_test"],
        private_records=splits["private_test"],
    )
    return usage_rows


def sanitize_tifu_test_split(
    output_path: Path,
    public_records: Sequence[Dict],
    private_records: Sequence[Dict],
) -> List[UsageRow]:
    print(
        f"Sanitizing {len(public_records) + len(private_records)} TIFU test samples "
        f"and writing -> {output_path}"
    )
    sanitized_records: List[dict] = []
    usage_rows: List[UsageRow] = []
    counter = 1

    for usage, subset in (("Public", public_records), ("Private", private_records)):
        for record in subset:
            instance = dict(record.get("instance") or {})
            summary = (instance.pop("summary", record.get("summary", "")) or "").strip()
            if not summary:
                continue
            instance.pop("url", None)
            instance.pop("permalink", None)
            sanitized_record = {
                key: value
                for key, value in record.items()
                if key != "summary"
            }
            sanitized_record["id"] = f"tifu-test-{counter:05d}"
            if instance:
                sanitized_record["instance"] = instance
            usage_rows.append((sanitized_record["id"], usage, summary))
            sanitized_records.append(sanitized_record)
            counter += 1

    write_jsonl(sanitized_records, output_path)
    return usage_rows


def sanitize_samsun_test(test_path: Path) -> List[UsageRow]:
    print(f"Sanitizing Samsung test set in-place -> {test_path}")
    with test_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    sanitized_rows: List[Dict[str, str]] = []
    usage_rows: List[UsageRow] = []
    for row in rows:
        sample_id = row.get("id", "").strip()
        dialogue = row.get("dialogue", "")
        summary = (row.get("summary") or "").strip()
        sanitized_rows.append({"id": sample_id, "dialogue": dialogue})
        if summary:
            usage_rows.append((sample_id, "Private", summary))

    with test_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "dialogue"])
        writer.writeheader()
        writer.writerows(sanitized_rows)

    return usage_rows


def write_gt_csv(gt_path: Path, rows: Sequence[UsageRow]) -> None:
    print(f"Writing {len(rows)} ground-truth rows -> {gt_path}")
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    with gt_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "Usage", "summary"])
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    gt_rows: List[UsageRow] = []
    gt_rows.extend(
        prepare_tifu_splits(
            raw_path=args.tifu_raw,
            clean_path=args.tifu_clean,
            prefix=args.tifu_prefix,
            test_output=args.tifu_test_output,
            seed=args.seed,
        )
    )
    gt_rows.extend(sanitize_samsun_test(args.samsun_test))
    write_gt_csv(args.gt_output, gt_rows)


if __name__ == "__main__":
    main()
