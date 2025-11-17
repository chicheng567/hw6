''' Define the Transformer model '''
from multiprocessing import context
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional
from transformer.Layers import DecoderLayer_Flash
from transformer.utils import *
from transformer.Const import *
class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x, seq_lens=None):
        if seq_lens is None:
            return x + self.pos_table[:, :x.size(1)].clone().detach()

        seq_lens = seq_lens.to(device=x.device, dtype=torch.long)
        total_seq_len = int(seq_lens.sum().item())
        if total_seq_len != x.size(0):
            raise ValueError(
                f"Packed sequence length mismatch: got {x.size(0)} tokens, "
                f"but seq_lens sum to {total_seq_len}."
            )

        seq_starts = torch.cumsum(seq_lens, dim=0) - seq_lens
        token_seq_ids = torch.arange(seq_lens.size(0), device=x.device).repeat_interleave(seq_lens)
        seq_offsets = seq_starts[token_seq_ids]
        position_ids = torch.arange(total_seq_len, device=x.device) - seq_offsets

        max_pos = int(position_ids.max().item())
        if max_pos >= self.pos_table.size(1):
            raise ValueError(
                f"Requested position {max_pos} exceeds available sinusoid size {self.pos_table.size(1)}."
            )

        pos_emb = self.pos_table[:, position_ids, :].squeeze(0)
        return x + pos_emb
    
class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1, scale_emb=False, flash_attn=True):

        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.flash_attn = flash_attn
        if flash_attn:
            assert d_k == d_v, "For Flash Attention, d_k must be equal to d_v"
            self.layer_stack = nn.ModuleList([
                DecoderLayer_Flash(d_model, d_inner, n_head, d_k, dropout=dropout)
                for _ in range(n_layers)])
        else:
            raise NotImplementedError("Only Flash Attention is implemented in this Decoder.")
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask):
        dec_output = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        if self.flash_attn:
            dec_output = self.position_enc(dec_output, seq_lens=trg_mask)
        else:
            dec_output = self.position_enc(dec_output)
        dec_output = self.dropout(dec_output)
        dec_output = self.layer_norm(dec_output)
        if self.flash_attn:
            # IF flash attention is used, trg_mask and src_mask are actually sequence lengths with shape (batch_size,)
            # so we handle them differently.
            # And we assume under flash attention mode, sequence packing is used.
            # So trg_seq and enc_output are already packed sequences with shape (total_seq_len, d_model)
            trg_seq_len = trg_mask
            enc_seq_len = src_mask
            for dec_layer in self.layer_stack:
                dec_output = dec_layer(dec_output, trg_seq_len, enc_output, enc_seq_len)
            return dec_output
        else:
            raise NotImplementedError("Only Flash Attention is implemented in this Decoder.")
from transformers import ModernBertModel, AutoTokenizer
from transformer.Const import *
class Seq2SeqModelWithFlashAttn(nn.Module):
    def __init__(
        self,
        transformer_model_path: str = "answerdotai/ModernBERT-base",
        freeze_encoder: bool = True,
        weight_dtype: Optional[torch.dtype] = torch.bfloat16,
    ):
        super().__init__()
        encoder_kwargs = {}
        if weight_dtype is not None:
            encoder_kwargs["torch_dtype"] = weight_dtype
        self.encoder = ModernBertModel.from_pretrained(transformer_model_path, **encoder_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model_path)
        self.decoder = Decoder(
            n_trg_vocab=len(self.tokenizer),
            d_word_vec=768,
            n_layers=6,
            n_head=12,
            d_k=768 // 12,
            d_v=768 // 12,
            d_model=768,
            d_inner=768 * 4,
            pad_idx=self.tokenizer.pad_token_id,
            n_position=MAX_TARGET_LEN,
            dropout=0.1,
            scale_emb=False,
            flash_attn=True)
        self.output_projection = nn.Linear(768, len(self.tokenizer), bias=False)
        self._cast_modules_to_dtype(weight_dtype)
        self._tie_decoder_embeddings()
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.weight_dtype = weight_dtype
    def forward(self, src_input_ids, trg_input_ids, src_seq_len, trg_seq_len):
        # src_input and trg_input are assumed to be already tokenized and sequence packed.
        # src_input and trg_input shape should be (total_seq_len, )
        # Encode
        dummy_mask = torch.tensor(1, device=src_input_ids.device)
        bsz = src_seq_len.size(0)
        src_cu_seqlens = seqlen2cu_len(src_seq_len)
        max_src_len = src_seq_len.max().item()
        enc_outputs = self.encoder(
            input_ids=src_input_ids,
            attention_mask=dummy_mask,
            cu_seqlens=src_cu_seqlens,
            max_seqlen=max_src_len,
            batch_size=bsz
        )
        enc_output = enc_outputs["last_hidden_state"] # shape: (total_src_seq_len, d_model)
        assert enc_output.size(0) == src_input_ids.size(0), (enc_output.size(), src_input_ids.size())
        dec_output = self.decoder(
            trg_seq=trg_input_ids,
            trg_mask=trg_seq_len,
            enc_output=enc_output,
            src_mask=src_seq_len
        )
        # Project to vocabulary
        logits = self.output_projection(dec_output)
        return logits
    def top_k_top_p_filtering(
        self,
        logits: torch.Tensor,
        top_k,
        top_p
    ) -> torch.Tensor:
        # logits: (bsz, vocab_size)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        filter_value: float = -float('Inf')
        vocab_size = logits.size(-1)
        if top_k > 0 and top_k < vocab_size:
            k_th_score, indices = torch.topk(logits, top_k)
            min_kth_score = k_th_score[:,-1].view(-1,1)
            logits = torch.where(
                logits < min_kth_score,
                torch.full_like(logits, filter_value),
                logits,
            )
        
        if 0.0 < top_p < 1.0:
            probabilities = torch.softmax(logits.float(), dim=-1)
            sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
            cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
            sorted_indices_to_remove = cumulative_probabilities > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[mask] = filter_value
        return logits
            
    def generate(
        self,
        input_ids: torch.Tensor,
        src_seq_len: torch.Tensor,
        generation_limit: int,
        sampling: bool = False,
        top_k: int = 10,
        top_p: float = 0.9,
    ) -> List[str]:
        device = self.output_projection.weight.device
        src_seq_len = src_seq_len.to(device=device, dtype=torch.int32)
        bsz = src_seq_len.size(0)

        sequences = [
            torch.tensor([self.tokenizer.cls_token_id], device=device, dtype=torch.long)
            for _ in range(bsz)
        ]
        finished = torch.zeros(bsz, dtype=torch.bool, device=device)

        for _ in range(generation_limit):
            trg_seq_len = torch.tensor(
                [seq.numel() for seq in sequences],
                device=device,
                dtype=torch.int32,
            )
            decoder_input_ids = torch.cat(sequences, dim=0)
            logits = self.forward(
                src_input_ids=input_ids,
                trg_input_ids=decoder_input_ids,
                src_seq_len=src_seq_len,
                trg_seq_len=trg_seq_len,
            )
            trg_seq_len_long = trg_seq_len.to(dtype=torch.long)
            end_positions = torch.cumsum(trg_seq_len_long, dim=0) - 1
            next_token_logits = logits[end_positions]

            if finished.any():
                next_token_logits[finished] = -float("inf")
                next_token_logits[
                    finished, self.tokenizer.sep_token_id
                ] = 0.0  # keep SEP sticky

            if sampling:
                filtered_logits = self.top_k_top_p_filtering(
                    next_token_logits,
                    top_k=top_k,
                    top_p=top_p,
                )
                probabilities = torch.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1)

            for idx in range(bsz):
                if finished[idx]:
                    continue
                sequences[idx] = torch.cat(
                    [sequences[idx], next_token[idx].view(1)]
                )
                if next_token[idx].item() == self.tokenizer.sep_token_id:
                    finished[idx] = True

            if bool(torch.all(finished)):
                break

        output_text: List[str] = []
        for seq in sequences:
            tokens = seq.tolist()
            if self.tokenizer.sep_token_id in tokens:
                tokens = tokens[: tokens.index(self.tokenizer.sep_token_id)]
            output_text.append(self.tokenizer.decode(tokens, skip_special_tokens=True))
        return output_text

    def _cast_modules_to_dtype(self, dtype: Optional[torch.dtype]) -> None:
        if dtype is None:
            return
        self.encoder.to(dtype=dtype)
        self.decoder.to(dtype=dtype)
        self.output_projection.to(dtype=dtype)

    def _tie_decoder_embeddings(self) -> None:
        # Initialize decoder embeddings with encoder embeddings and decoder embeddings tie to output projection layer
        with torch.no_grad():
            self.decoder.trg_word_emb.weight.copy_(
                self.encoder.embeddings.tok_embeddings.weight
            )
        self.output_projection.weight = self.decoder.trg_word_emb.weight
