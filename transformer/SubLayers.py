''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformer.Modules import ScaledDotProductAttention
import torch
from flash_attn import flash_attn_varlen_qkvpacked_func, flash_attn_varlen_func
__author__ = "Yu-Hsiang Huang"

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn

class MultiHeadSelfAttention_Flash(nn.Module):
    ''' Multi-Head self Attention module with Flash Attention '''

    def __init__(self, n_head, d_model, d_qkv, dropout=0.1, causal=False):
        super().__init__()
        self.n_head = n_head
        self.d_qkv = d_qkv
        self.w_qkv = nn.Linear(d_model, n_head * d_qkv * 3, bias=False)
        self.w_o = nn.Linear(n_head * d_qkv, d_model, bias=False)
        self.dropout_rate = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.causal = causal
    def forward(self, x, seq_lens):
        # x should be of shape (batch_size * seq_len, d_model)
        # seq_lens should be of shape (batch_size,) like (4, 3, 2, 23, 1, ...)
        drop_rate = self.dropout_rate if self.training else 0.0
        residual = x
        qkv_packed = self.w_qkv(x).view(-1, 3, self.n_head, self.d_qkv)
        cu_seqlens_B = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens_B, (1, 0), value=0)
        max_len = seq_lens.max().item()
        # output shape: (total_tokens, n_head, d_qkv)
        output = flash_attn_varlen_qkvpacked_func(qkv_packed, cu_seqlens, max_len, dropout_p=drop_rate, causal=self.causal)
        output = output.reshape(-1, self.n_head * self.d_qkv)
        output = self.dropout_layer(self.w_o(output))
        output += residual
        output = self.layer_norm(output)
        return output

class MultiHeadCrossAttention_Flash(nn.Module):
    def __init__(self, n_head, d_model, d_q, d_kv, dropout=0.1, causal=False):
        super().__init__()
        self.n_head = n_head
        self.d_q = d_q
        self.d_kv = d_kv
        self.w_q = nn.Linear(d_model, n_head * d_q, bias=False)
        self.w_kv = nn.Linear(d_model, n_head * d_kv * 2, bias=False)
        self.w_o = nn.Linear(n_head * d_kv, d_model, bias=False)
        self.dropout_layer = nn.Dropout(dropout)
        self.dropout_rate = dropout
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.causal = causal
    def forward(self, x_q, x_kv, seq_lens_q, seq_lens_kv):
        # x_q should be of shape (batch_size * seq_len_q, d_model)
        # x_kv should be of shape (batch_size * seq_len_kv, d_model)
        # seq_lens_q should be of shape (batch_size,) like (4, 3, 2, 23, 1, ...)
        # seq_lens_kv should be of shape (batch_size,) like (4, 3, 2, 23, 1, ...)
        drop_rate = self.dropout_rate if self.training else 0.0
        residual = x_q
        q = self.w_q(x_q).view(-1, self.n_head, self.d_q)
        kv = self.w_kv(x_kv).view(-1, 2, self.n_head, self.d_kv)
        k = kv[:,0,:,:]
        v = kv[:,1,:,:]
        cu_seqlens_B_q = torch.cumsum(seq_lens_q, dim=0, dtype=torch.int32)
        cu_seqlens_B_kv = torch.cumsum(seq_lens_kv, dim=0, dtype=torch.int32)
        cu_seqlens_q = F.pad(cu_seqlens_B_q, (1, 0), value=0)
        cu_seqlens_kv = F.pad(cu_seqlens_B_kv, (1, 0), value=0)
        max_len_q = seq_lens_q.max().item()
        max_len_kv = seq_lens_kv.max().item()
        # output shape: (total_tokens_q, n_head, d_kv)
        output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_len_q,
            max_seqlen_kv=max_len_kv,
            dropout_p=drop_rate,
            causal=self.causal,
        )
        output = output.reshape(-1, self.n_head * self.d_kv)
        output = self.dropout_layer(self.w_o(output))
        output += residual
        output = self.layer_norm(output)
        return output
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x
