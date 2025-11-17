import csv
import math
import random
from pathlib import Path
from typing import Optional, Tuple, Union
from data_utils import *
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformer.Const import *
from transformer.Models import Seq2SeqModelWithFlashAttn
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODE = "train"  # set to "predict" for inference
CHECKPOINT_PATH = Path("checkpoints/latest.pt")
BEST_CHECKPOINT_PATH = Path("checkpoints/best.pt")
PREDICT_CHECKPOINT = Path("checkpoints/best.pt")
TIFU_TEST_PATH = Path("dataset/tifu/tifu_test.jsonl")
SAMSUN_TEST_PATH = Path("dataset/samsun/test.csv")
PREDICTION_OUTPUT = Path("predictions.csv")
MAX_GENERATION_LEN = MAX_TARGET_LEN
TRAIN_EPOCHS = 30
TRAIN_BATCH_SIZE = 100
GLOBAL_SEED = 42
NUM_WORKERS = 4
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_dataset(
    path: List[Optional[str]],
    tokenizer: PreTrainedTokenizerBase,
    require_target: bool = True,
) -> Optional[Dataset]:
    if all(p is None for p in path):
        return None
    datasets = []
    for p in path:
        if p is not None:
            dataset = SquadSeq2SeqDataset(
                Path(p), tokenizer, max_source_len=MAX_SOURCE_LEN, max_target_len=MAX_TARGET_LEN, require_target=require_target
            )
            datasets.append(dataset)
    print(f"Built dataset with {sum(len(ds) for ds in datasets)} samples.")
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)

def build_dataloader(
    source: Union[Optional[Dataset], Optional[str]],
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 8,
) -> Optional[DataLoader]:
    dataset = source
    collator = QACollator
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
    iterator = tqdm(dataloader, desc="train" if train else "eval", leave=False)
    for batch in iterator:
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
        iterator.set_postfix(loss=total_loss / max(1, steps))
    return total_loss / max(1, steps)

def load_checkpoint(
    model: Seq2SeqModelWithFlashAttn,
    path: Path,
    device: torch.device,
) -> None:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"])

def save_checkpoint(
    model: Seq2SeqModelWithFlashAttn,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[object],
    path: Path,
    epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, path)


def main() -> None:
    ### Hyperparameters and arguments ###
    lr = 1e-4
    weight_decay = 0.001
    warmup_steps = 2000
    epochs = TRAIN_EPOCHS
    max_grad_norm = 1.0
    batch_size = TRAIN_BATCH_SIZE
    num_workers = NUM_WORKERS
    #####################################
    mode = MODE
    set_seed(GLOBAL_SEED)
    if torch.cuda.is_available():
        device = torch.device("cuda:1")
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
    checkpoint_path = CHECKPOINT_PATH
    best_checkpoint_path = BEST_CHECKPOINT_PATH

    if mode == "train":
        train_set = build_dataset(
            ["dataset/tifu/tifu_train.jsonl", "dataset/samsun/train.csv"],
            tokenizer=model.tokenizer,
        )
        train_loader = build_dataloader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_set = build_dataset(
            ["dataset/tifu/tifu_val.jsonl", "dataset/samsun/validation.csv"],
            tokenizer=model.tokenizer,
        )
        valid_loader = build_dataloader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        total_steps = epochs * len(train_loader)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=min(warmup_steps, total_steps),
            num_training_steps=total_steps,
        )

        best_val_ppl = float("inf")
        for epoch in range(1, epochs + 1):
            train_loss = run_epoch(
                train_loader,
                model,
                device,
                optimizer,
                scheduler,
                tokenizer.pad_token_id,
                max_grad_norm,
                train=True,
            )
            msg = f"Epoch {epoch}/{epochs} - train loss: {train_loss:.4f}"
            current_val_ppl = None
            with torch.no_grad():
                val_loss = run_epoch(
                    valid_loader,
                    model,
                    device,
                    optimizer=None,
                    scheduler=None,
                    pad_id=tokenizer.pad_token_id,
                    max_grad_norm=max_grad_norm,
                    train=False,
                )
            perplexity = math.exp(min(20, val_loss))
            current_val_ppl = perplexity
            msg += f" | val loss: {val_loss:.4f} | ppl: {perplexity:.2f}"
            print(msg)
            if checkpoint_path is not None:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    path=checkpoint_path,
                    epoch=epoch,
                )
            if (
                current_val_ppl is not None
                and current_val_ppl < best_val_ppl
                and best_checkpoint_path is not None
            ):
                best_val_ppl = current_val_ppl
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    path=best_checkpoint_path,
                    epoch=epoch,
                )
    elif mode == "predict":
        load_checkpoint(model, PREDICT_CHECKPOINT, device)
        model.eval()
        test_set = build_dataset(
            [TIFU_TEST_PATH, SAMSUN_TEST_PATH],
            tokenizer=model.tokenizer,
            require_target=False,
        )
        test_loader = build_dataloader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        predictions: List[Tuple[str, str]] = []
        with torch.no_grad():
            for sample in tqdm(test_loader, desc="predict", leave=False):
                input_ids = sample["src"].to(device)
                src_lens = sample["src_len"].to(device=device, dtype=torch.int32)
                ids = sample["id"] #list of ids
                summaries = model.generate(
                    input_ids=input_ids,
                    src_seq_len=src_lens,
                    generation_limit=MAX_GENERATION_LEN,
                    sampling=True,
                    top_k=50,
                    top_p=0.9,
                )
                predictions.extend(zip(ids, summaries))
        output_path = PREDICTION_OUTPUT
        write_predictions_csv(output_path, predictions)
        print(f"Wrote {len(predictions)} predictions to {output_path}")

if __name__ == "__main__":
    main()
