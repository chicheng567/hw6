''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from transformer.Layers import DecoderLayer_Flash
from transformer.utils import *

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

    def _cast_modules_to_dtype(self, dtype: Optional[torch.dtype]) -> None:
        if dtype is None:
            return
        self.encoder.to(dtype=dtype)
        self.decoder.to(dtype=dtype)
        self.output_projection.to(dtype=dtype)

    def _tie_decoder_embeddings(self) -> None:
        with torch.no_grad():
            self.decoder.trg_word_emb.weight.copy_(
                self.encoder.embeddings.tok_embeddings.weight
            )
        self.output_projection.weight = self.decoder.trg_word_emb.weight
