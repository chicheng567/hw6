import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torch.cuda import amp
from transformers import AutoTokenizer, ModernBertModel, ModernBertConfig, get_linear_schedule_with_warmup
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from transformer.Models import Decoder, get_pad_mask, get_subsequent_mask


SPECIAL_TOKENS = {"bos_token": "[BOS]", "eos_token": "[EOS]"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    special_tokens = {}
    for key, value in SPECIAL_TOKENS.items():
        if getattr(tokenizer, key, None) is None:
            special_tokens[key] = value
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = "[PAD]"
    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
    return tokenizer


class SquadSeq2SeqDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer: PreTrainedTokenizerBase,
        max_source_len: int,
        max_target_len: int,
    ) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        self.samples: List[Dict[str, str]] = []
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                answers = record.get("answers", {}).get("text", [])
                answer = answers[0] if answers else ""
                self.samples.append(
                    {
                        "question": record.get("question", ""),
                        "context": record.get("context", ""),
                        "answer": answer,
                    }
                )
        if not self.samples:
            raise ValueError(f"No data found in {path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        example = self.samples[idx]
        encoded = self.tokenizer(
            example["question"],
            example["context"],
            truncation=True,
            max_length=self.max_source_len,
            padding=False,
            return_attention_mask=True,
        )
        target_tokens = self.tokenizer.encode(
            example["answer"],
            add_special_tokens=False,
            truncation=True,
            max_length=max(2, self.max_target_len) - 2,
        )
        target_ids = (
            [self.tokenizer.bos_token_id]
            + target_tokens
            + [self.tokenizer.eos_token_id]
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "target_ids": target_ids,
        }


class QACollator:
    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def _pad_sequences(self, sequences: List[List[int]], pad_value: int) -> torch.Tensor:
        max_len = max(len(seq) for seq in sequences)
        tensor = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)
        for idx, seq in enumerate(sequences):
            tensor[idx, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        return tensor

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_ids = [item["input_ids"] for item in batch]
        attention = [item["attention_mask"] for item in batch]
        target_ids = [item["target_ids"] for item in batch]

        max_src = max(len(seq) for seq in input_ids)
        src_tensor = torch.full((len(batch), max_src), self.pad_id, dtype=torch.long)
        attn_tensor = torch.zeros((len(batch), max_src), dtype=torch.long)
        for idx, (seq, mask) in enumerate(zip(input_ids, attention)):
            seq_len = len(seq)
            src_tensor[idx, :seq_len] = torch.tensor(seq, dtype=torch.long)
            attn_tensor[idx, :seq_len] = torch.tensor(mask, dtype=torch.long)

        tgt_tensor = self._pad_sequences(target_ids, self.pad_id)
        return {
            "input_ids": src_tensor,
            "attention_mask": attn_tensor,
            "target_ids": tgt_tensor,
        }


class ModernBertDecoderModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        tokenizer: PreTrainedTokenizerBase,
        decoder_layers: int,
        decoder_ffn_dim: int,
        decoder_dropout: float,
        max_target_len: int,
        use_flash_attn: bool,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id
        self.use_flash_attn = use_flash_attn
        try:
            self.encoder = ModernBertModel.from_pretrained(model_name)
            self.encoder.resize_token_embeddings(len(tokenizer))
        except OSError:
            config = ModernBertConfig()
            self.encoder = ModernBertModel(config)
            self.encoder.resize_token_embeddings(len(tokenizer))
        hidden_size = self.encoder.config.hidden_size
        n_head = self.encoder.config.num_attention_heads
        head_dim = hidden_size // n_head
        self.decoder = Decoder(
            n_trg_vocab=len(tokenizer),
            d_word_vec=hidden_size,
            n_layers=decoder_layers,
            n_head=n_head,
            d_k=head_dim,
            d_v=head_dim,
            d_model=hidden_size,
            d_inner=decoder_ffn_dim,
            pad_idx=self.pad_id,
            n_position=max_target_len + 5,
            dropout=decoder_dropout,
            scale_emb=True,
            use_flash_attn=use_flash_attn,
        )
        self.generator = nn.Linear(hidden_size, len(tokenizer), bias=False)
        with torch.no_grad():
            encoder_weights = self.encoder.embeddings.tok_embeddings.weight
            self.decoder.trg_word_emb.weight.copy_(encoder_weights)
            self.generator.weight.copy_(encoder_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        encoder_outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        memory = encoder_outputs.last_hidden_state
        src_mask = get_pad_mask(input_ids, self.pad_id)
        trg_pad_mask = get_pad_mask(decoder_input_ids, self.pad_id)
        trg_sub_mask = get_subsequent_mask(decoder_input_ids)
        if self.use_flash_attn and input_ids.is_cuda:
            with amp.autocast(device_type="cuda", dtype=torch.float16):
                dec_output, *_ = self.decoder(
                    decoder_input_ids, trg_pad_mask, trg_sub_mask, memory, src_mask
                )
                logits = self.generator(dec_output)
        else:
            dec_output, *_ = self.decoder(
                decoder_input_ids, trg_pad_mask, trg_sub_mask, memory, src_mask
            )
            logits = self.generator(dec_output)
        return logits.float()


def build_dataloader(
    path: Optional[str],
    tokenizer: PreTrainedTokenizerBase,
    max_source_len: int,
    max_target_len: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> Optional[DataLoader]:
    if path is None:
        return None
    dataset = SquadSeq2SeqDataset(
        Path(path), tokenizer, max_source_len=max_source_len, max_target_len=max_target_len
    )
    collator = QACollator(tokenizer.pad_token_id)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
    )


def run_epoch(
    dataloader: DataLoader,
    model: ModernBertDecoderModel,
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
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_ids = batch["target_ids"].to(device)
        decoder_input_ids = target_ids[:, :-1]
        labels = target_ids[:, 1:]
        logits = model(input_ids, attention_mask, decoder_input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=pad_id,
        )
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
    model: ModernBertDecoderModel,
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
                "decoder_layers": args.decoder_layers,
                "decoder_ffn_dim": args.decoder_ffn_dim,
                "decoder_dropout": args.decoder_dropout,
                "max_target_length": args.max_target_length,
                "use_flash_attn": args.use_flash_attn,
            },
        },
        path,
    )


def load_checkpoint(
    model: ModernBertDecoderModel,
    path: Path,
    device: torch.device,
) -> None:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ModernBERT encoder + custom decoder trainer")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--train-path", type=str, default="dataset/train.jsonl")
    parser.add_argument("--valid-path", type=str, default="dataset/validation.jsonl")
    parser.add_argument("--pretrained-model", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-source-length", type=int, default=384)
    parser.add_argument("--max-target-length", type=int, default=64)
    parser.add_argument("--decoder-layers", type=int, default=6)
    parser.add_argument("--decoder-ffn-dim", type=int, default=3072)
    parser.add_argument("--decoder-dropout", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/bert_decoder.pt")
    parser.add_argument(
        "--use-flash-attn",
        dest="use_flash_attn",
        action="store_true",
        help="Force-enable flash attention (default).",
    )
    parser.add_argument(
        "--no-flash-attn",
        dest="use_flash_attn",
        action="store_false",
        help="Disable flash attention even if the environment supports it.",
    )
    parser.set_defaults(use_flash_attn=True)
    parser.add_argument("--device", type=str, default=None, help="Override device string, e.g. cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    tokenizer = build_tokenizer(args.pretrained_model)
    if args.device is not None:
        device = torch.device(args.device)
        if device.type != "cuda":
            raise RuntimeError("GPU device is required to run this script.")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU device is required but not detected. Please run on CUDA hardware.")
        device = torch.device("cuda")
    model = ModernBertDecoderModel(
        model_name=args.pretrained_model,
        tokenizer=tokenizer,
        decoder_layers=args.decoder_layers,
        decoder_ffn_dim=args.decoder_ffn_dim,
        decoder_dropout=args.decoder_dropout,
        max_target_len=args.max_target_length,
        use_flash_attn=args.use_flash_attn,
    ).to(device)

    checkpoint_path = Path(args.checkpoint_path)
    if checkpoint_path.exists():
        load_checkpoint(model, checkpoint_path, device)

    if args.mode == "train":
        train_loader = build_dataloader(
            args.train_path,
            tokenizer,
            args.max_source_length,
            args.max_target_length,
            args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
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
