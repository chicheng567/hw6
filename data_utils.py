from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import csv
import torch
from transformer.Const import *
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
class SquadSeq2SeqDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer: PreTrainedTokenizerBase,
        max_source_len: int = 384,
        max_target_len: int = 64,
        require_target: bool = True,
    ) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        self.samples: List[Dict[str, str]] = []
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.require_target = require_target
        self.bos_token = SOS
        self.eos_token = EOS
        suffix = path.suffix.lower()
        if suffix == ".csv":
            ds = load_dataset("csv", data_files=str(path), split="train")
        elif suffix in {".json", ".jsonl"}:
            ds = load_dataset("json", data_files=str(path), split="train")
        else:
            raise ValueError(f"Unsupported dataset format: {path}")
        for rec in ds:
            context = self._extract_field(
                rec,
                primary_keys=("context", "dialogue"),
                instance_keys=("selftext_without_tldr", "context", "article"),
            )
            summary = self._extract_field(
                rec,
                primary_keys=("summary", "tldr"),
                instance_keys=("summary", "tldr"),
            )
            if not context or (self.require_target and not summary):
                continue
            identifier = self._extract_id(rec)
            if not identifier:
                identifier = f"sample_{len(self.samples)}"
            self.samples.append({"context": context, "summary": summary, "id": identifier})
        if not self.samples:
            raise ValueError(f"No data found in {path}")

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            for item in value:
                normalized = SquadSeq2SeqDataset._normalize_text(item)
                if normalized:
                    return normalized
        if isinstance(value, dict):
            # prioritize `text` field if present
            text = value.get("text")
            if isinstance(text, str):
                return text.strip()
        return ""

    def _extract_field(
        self,
        record: Dict[str, Any],
        primary_keys: Sequence[str],
        instance_keys: Sequence[str],
    ) -> str:
        for key in primary_keys:
            normalized = self._normalize_text(record.get(key))
            if normalized:
                return normalized
        instance = record.get("instance")
        if isinstance(instance, dict):
            for key in instance_keys:
                normalized = self._normalize_text(instance.get(key))
                if normalized:
                    return normalized
        return ""

    @staticmethod
    def _extract_id(record: Dict[str, Any]) -> str:
        def _sanitize(value: Optional[str]) -> str:
            if isinstance(value, str):
                value = value.strip()
                if value:
                    return value
            return ""

        record_id = _sanitize(record.get("id"))
        if record_id:
            return record_id

        for key in ("url", "permalink"):
            value = _sanitize(record.get(key))
            if value:
                if key == "permalink" and value.startswith("/"):
                    return f"https://www.reddit.com{value}"
                return value

        instance = record.get("instance")
        if isinstance(instance, dict):
            inst_id = _sanitize(instance.get("id"))
            if inst_id:
                return inst_id
            for key in ("url", "permalink"):
                value = _sanitize(instance.get(key))
                if value:
                    if key == "permalink" and value.startswith("/"):
                        return f"https://www.reddit.com{value}"
                    return value
        return ""

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        example = self.samples[idx]
        source_text = example["context"]
        target_text = example.get("summary", "")
        source_tokens = self.tokenizer.encode(
            source_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_source_len
        )
        if not self.require_target:
            target_tokens = None
        else:
            target_tokens = self.tokenizer.encode(
                target_text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_target_len
            )
        return {
            "id": example["id"],
            "input_ids": source_tokens,
            "labels": target_tokens
        }
        
def QACollator(batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("Empty batch provided to collator.")
    src_tokens: List[int] = []
    tgt_tokens: List[int] = []
    src_lens: List[int] = []
    tgt_lens: List[int] = []
    id_s = []
    for item in batch:
        id_s.append(item["id"])
        src_seq = item["input_ids"]
        src_lens.append(len(src_seq))
        src_tokens.extend(src_seq)
        tgt_seq = item.get("labels", None)
        if tgt_seq is not None:
            tgt_lens.append(len(tgt_seq))
            tgt_tokens.extend(tgt_seq)
    return {
        "src": torch.tensor(src_tokens, dtype=torch.long),
        "tgt": torch.tensor(tgt_tokens, dtype=torch.long),
        "src_len": torch.tensor(src_lens, dtype=torch.int32),
        "tgt_len": torch.tensor(tgt_lens, dtype=torch.int32),
        "id": id_s,
    }

def write_predictions_csv(path: Path, predictions: List[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "summary"])
        writer.writerows(predictions)
        