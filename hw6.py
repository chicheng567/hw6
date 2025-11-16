import argparse
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformer.Const import *
from transformer.Models import Seq2SeqModelWithFlashAttn
from datasets import load_dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SquadSeq2SeqDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer: PreTrainedTokenizerBase,
        max_source_len: int = 384,
        max_target_len: int = 64,
    ) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        self.samples: List[Dict[str, str]] = []
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
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
            if not context or not summary:
                continue
            self.samples.append({"context": context, "summary": summary})
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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        example = self.samples[idx]
        source_text = example["context"]
        target_text = example["summary"]
        source_tokens = self.tokenizer.encode(
            source_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_source_len
        )
        target_tokens = self.tokenizer.encode(
            target_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_target_len
        )
        
        return {
            "input_ids": source_tokens,
            "labels": target_tokens
        }

class QACollator:
    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("Empty batch provided to collator.")
        src_tokens: List[int] = []
        tgt_tokens: List[int] = []
        src_lens: List[int] = []
        tgt_lens: List[int] = []

        for item in batch:
            src_seq = item["input_ids"]
            tgt_seq = item["target_ids"]
            if not src_seq or not tgt_seq:
                continue
            src_lens.append(len(src_seq))
            tgt_lens.append(len(tgt_seq))
            src_tokens.extend(src_seq)
            tgt_tokens.extend(tgt_seq)

        if not src_tokens or not tgt_tokens:
            raise ValueError("Batch contains no valid sequences for packing.")

        return {
            "src": torch.tensor(src_tokens, dtype=torch.long),
            "tgt": torch.tensor(tgt_tokens, dtype=torch.long),
            "src_len": torch.tensor(src_lens, dtype=torch.int32),
            "tgt_len": torch.tensor(tgt_lens, dtype=torch.int32),
        }
        
def build_dataset(path:List[Optional[str]],
    tokenizer: PreTrainedTokenizerBase,
) -> Optional[Dataset]:
    if all(p is None for p in path):
        return None
    datasets = []
    for p in path:
        if p is not None:
            dataset = SquadSeq2SeqDataset(
                Path(p), tokenizer, max_source_len=MAX_SOURCE_LEN, max_target_len=MAX_TARGET_LEN
            )
            datasets.append(dataset)
    print(f"Built dataset with {sum(len(ds) for ds in datasets)} samples.")
    print(f"Sample example: {datasets[0][0]}")
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)
    
def build_dataloader(
    source: Union[Optional[Dataset], Optional[str]],
    tokenizer: PreTrainedTokenizerBase,
    max_source_len: int = 384,
    max_target_len: int = 64,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0,
) -> Optional[DataLoader]:
    if source is None:
        return None
    if isinstance(source, Dataset):
        dataset = source
    else:
        dataset = SquadSeq2SeqDataset(
            Path(source), tokenizer, max_source_len=max_source_len, max_target_len=max_target_len
        )
    collator = QACollator()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
    )


def run_epoch(
    dataloader: DataLoader,
    model: Seq2SeqModelWithFlashAttn,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[object],
    pad_id: int,
    max_grad_norm: float,
    train: bool,
) -> float:
    model.train(train)
    total_loss = 0.0
    steps = 0
    for batch in dataloader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        src_seq_len = batch["src_len"].to(device=device, dtype=torch.int32)
        tgt_seq_len = batch["tgt_len"].to(device=device, dtype=torch.int32)
        if torch.any(tgt_seq_len < 2):
            raise ValueError("Each target sequence must contain at least BOS and EOS tokens.")

        tgt_len_long = tgt_seq_len.to(dtype=torch.int64)
        cumulative = torch.cumsum(tgt_len_long, dim=0)
        start_indices = cumulative - tgt_len_long
        end_indices = cumulative - 1

        total_tgt_tokens = tgt.size(0)
        decoder_mask = torch.ones(total_tgt_tokens, dtype=torch.bool, device=device)
        decoder_mask[end_indices] = False
        decoder_input_ids = tgt[decoder_mask]

        label_mask = torch.ones(total_tgt_tokens, dtype=torch.bool, device=device)
        label_mask[start_indices] = False
        labels = tgt[label_mask]
        trg_seq_len = (tgt_seq_len - 1).to(dtype=torch.int32)
        logits = model(
            src_input_ids=src,
            trg_input_ids=decoder_input_ids,
            src_seq_len=src_seq_len,
            trg_seq_len=trg_seq_len,
        )
        loss = F.cross_entropy(logits, labels, ignore_index=pad_id)
        if train:
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        total_loss += loss.item()
        steps += 1
    return total_loss / max(1, steps)


def save_checkpoint(
    model: Seq2SeqModelWithFlashAttn,
    path: Path,
    epoch: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": {
                "pretrained_model": args.pretrained_model,
                "max_target_length": args.max_target_length,
                "freeze_encoder": args.freeze_encoder,
            },
        },
        path,
    )


def load_checkpoint(
    model: Seq2SeqModelWithFlashAttn,
    path: Path,
    device: torch.device,
) -> None:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"])


def main() -> None:
    mode = "train"
    set_seed(42)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise RuntimeError("CUDA is required to run this code.")
    
    # Check if flash attention is available
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        raise ImportError("flash_attn is required to run this code.")
    
    model = Seq2SeqModelWithFlashAttn(
        transformer_model_path="answerdotai/ModernBERT-base",
        freeze_encoder=True,
    ).to(device)
    tokenizer = model.tokenizer
    checkpoint_path = None
    if checkpoint_path is not None and checkpoint_path.exists():
        load_checkpoint(model, checkpoint_path, device)

    if mode == "train":
        train_set = build_dataset(
            ["dataset/tifu/tifu_train.jsonl", "dataset/samsun/train.csv"],
            tokenizer=model.tokenizer,
        )
        train_loader = build_dataloader(
            train_set,
            tokenizer,
            batch_size=8,
            shuffle=True,
            num_workers=4,
        )
        if train_loader is None:
            raise ValueError("Training requires --train-path")
        valid_loader = build_dataloader(
            args.valid_path,
            tokenizer,
            args.max_source_length,
            args.max_target_length,
            args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        total_steps = args.epochs * len(train_loader)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=min(args.warmup_steps, total_steps),
            num_training_steps=total_steps,
        )

        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(
                train_loader,
                model,
                device,
                optimizer,
                scheduler,
                tokenizer.pad_token_id,
                args.max_grad_norm,
                train=True,
            )
            msg = f"Epoch {epoch}/{args.epochs} - train loss: {train_loss:.4f}"
            if valid_loader is not None:
                with torch.no_grad():
                    val_loss = run_epoch(
                        valid_loader,
                        model,
                        device,
                        optimizer=None,
                        scheduler=None,
                        pad_id=tokenizer.pad_token_id,
                        max_grad_norm=args.max_grad_norm,
                        train=False,
                    )
                perplexity = math.exp(min(20, val_loss))
                msg += f" | val loss: {val_loss:.4f} | ppl: {perplexity:.2f}"
            print(msg)
            save_checkpoint(model, checkpoint_path, epoch, args)
            tokenizer.save_pretrained(checkpoint_path.parent)
    else:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint {checkpoint_path} not found for eval mode."
            )
        eval_loader = build_dataloader(
            args.valid_path,
            tokenizer,
            args.max_source_length,
            args.max_target_length,
            args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        if eval_loader is None:
            raise ValueError("Evaluation requires --valid-path")
        with torch.no_grad():
            val_loss = run_epoch(
                eval_loader,
                model,
                device,
                optimizer=None,
                scheduler=None,
                pad_id=tokenizer.pad_token_id,
                max_grad_norm=args.max_grad_norm,
                train=False,
            )
        print(f"Eval loss: {val_loss:.4f}, perplexity: {math.exp(min(20, val_loss)):.2f}")


if __name__ == "__main__":
    main()
