#!/usr/bin/env python3
"""Convert the Reddit TIFU dataset to a JSONL subset used for summarization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter Reddit TIFU records down to the fields required for training and "
            "store them as JSONL."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        default="/workspace/dlhw/hw6/dataset/tifu/tifu_all_tokenized_and_filtered.json",
        help="Path to the raw Reddit TIFU JSON file (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="/workspace/dlhw/hw6/dataset/tifu/tifu_clean.jsonl",
        help="Destination JSONL file that will contain the reduced records.",
    )
    return parser.parse_args()


def extract_summary(record: Dict[str, Any]) -> str:
    """Return the TL;DR text if present, otherwise an empty string."""
    return (record.get("tldr") or "").strip()


def build_instance(record: Dict[str, Any]) -> Dict[str, Any]:
    summary = extract_summary(record)
    if not summary:
        return {}

    url = record.get("url") or record.get("permalink") or ""
    if url and url.startswith("/"):
        url = f"https://www.reddit.com{url}"

    return {
        "url": url,
        "selftext_without_tldr": record.get("selftext_without_tldr")
        or record.get("selftext")
        or "",
        "summary": summary,
    }


def convert_dataset(input_path: Path, output_path: Path) -> int:
    """Stream the source JSON and write JSONL objects with the reduced fields."""
    written = 0
    with input_path.open(encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_number, line in enumerate(src, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}") from exc

            reduced_record = build_instance(record)
            if not reduced_record:
                continue

            json.dump({"instance": reduced_record}, dst, ensure_ascii=False)
            dst.write("\n")
            written += 1

    return written


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    written = convert_dataset(input_path, output_path)
    print(f"Wrote {written} records to {output_path}")


if __name__ == "__main__":
    main()
