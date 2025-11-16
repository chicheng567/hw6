''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from transformer.Layers import DecoderLayer_Flash

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


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

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()
    
class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1, scale_emb=False, flash_attn=False):

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
        dec_output = self.dropout(self.position_enc(dec_output))
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
        else:
            raise NotImplementedError("Only Flash Attention is implemented in this Decoder.")
from transformers import ModernBertModel, AutoTokenizer
class Seq2SeqModelWithFlashAttn(nn.Module):
    def __init__(self, transformer_model_path="answerdotai/ModernBERT-base", freeze_encoder=True):
        super().__init__()
        self.encoder = ModernBertModel.from_pretrained(transformer_model_path)
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
            n_position=70,
            dropout=0.1,
            scale_emb=False,
            flash_attn=True)
        self.output_projection = nn.Linear(768, len(self.tokenizer), bias=False)
        # Tie weights
        self.decoder.trg_word_emb.weight.copy_(
            self.encoder.embeddings.tok_embeddings.weight
        )
        self.output_projection.weight = self.decoder.trg_word_emb.weight
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
    def forward(self, src_input_ids, trg_input_ids, src_seq_len, trg_seq_len):
        # src_input and trg_input are assumed to be already tokenized and sequence packed.
        # src_input and trg_input shape should be (total_seq_len, )
        # Encode
        enc_outputs = self.encoder(
            input_ids=src_input_ids,
            attention_mask=(src_input_ids != self.tokenizer.pad_token_id).long()
        )
        enc_output = enc_outputs.last_hidden_state
        # Decode
        dec_output, = self.decoder(
            trg_seq=trg_input_ids,
            trg_mask=trg_seq_len,
            enc_output=enc_output,
            src_mask=src_seq_len
        )
        # Project to vocabulary
        logits = self.output_projection(dec_output)
        return logits