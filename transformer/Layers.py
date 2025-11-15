''' Define the Layers '''
import torch.nn as nn
import torch
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward, MultiHeadCrossAttention_Flash, MultiHeadSelfAttention_Flash


__author__ = "Yu-Hsiang Huang"


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn

class DecoderLayer_Flash(nn.Module):
    ''' Compose with three layers using Flash Attention '''

    def __init__(self, d_model, d_inner, n_head, d_qkv, dropout=0.1):
        super(DecoderLayer_Flash, self).__init__()
        self.slf_attn = MultiHeadSelfAttention_Flash(n_head, d_model, d_qkv, dropout=dropout, causal=True)
        self.enc_attn = MultiHeadCrossAttention_Flash(n_head, d_model, d_qkv, dropout=dropout, causal=False)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, dec_seq_lens, enc_output, enc_seq_lens):
        dec_output = self.slf_attn(dec_input, dec_seq_lens)
        dec_output = self.enc_attn(x_q=dec_output, x_kv=enc_output, seq_lens_q=dec_seq_lens, seq_lens_kv=enc_seq_lens)
        dec_output = self.pos_ffn(dec_output)
        return dec_output