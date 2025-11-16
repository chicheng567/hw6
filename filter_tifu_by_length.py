#!/usr/bin/env python3
"""Remove TIFU samples whose ModernBERT source length exceeds a threshold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--files",
        "-f",
        nargs="*",
        help="Explicit JSONL files to filter. Defaults to all dataset/tifu/*.jsonl files.",
    )
    parser.add_argument(
        "--directory",
        "-d",
        default="/workspace/dlhw/hw6/dataset/tifu",
        help="Directory that holds the TIFU JSONL files.",
    )
    parser.add_argument(
        "--max_length",
        "-m",
        type=int,
        default=2000,
        help="Maximum allowed ModernBERT token length for the source text.",
    )
    parser.add_argument(
        "--tokenizer",
        "-t",
        default="answerdotai/ModernBERT-base",
        help="Tokenizer name or path used to measure sequence length.",
    )
    return parser.parse_args()


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True))


def filter_records(input_path: Path, output_path: Path, tokenizer, max_len: int) -> tuple[int, int]:
    kept = 0
    dropped = 0
    with input_path.open(encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_num, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            instance = record.get("instance", {})
            source = instance.get("selftext_without_tldr") or ""
            length = count_tokens(tokenizer, source)
            if length <= max_len:
                json.dump(record, dst, ensure_ascii=False)
                dst.write("\n")
                kept += 1
            else:
                dropped += 1
    return kept, dropped


def main() -> None:
    args = parse_args()
    base_dir = Path(args.directory)
    if args.files:
        targets = [Path(path) for path in args.files]
    else:
        targets = sorted(base_dir.glob("*.jsonl"))
        if not targets:
            raise FileNotFoundError(f"No JSONL files found under {base_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    for path in targets:
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue
        temp_path = path.with_suffix(path.suffix + ".tmp")
        kept, dropped = filter_records(path, temp_path, tokenizer, args.max_length)
        temp_path.replace(path)
        total = kept + dropped
        print(f"{path}: kept {kept} / {total} samples (dropped {dropped} > {args.max_length})")


if __name__ == "__main__":
    main()
