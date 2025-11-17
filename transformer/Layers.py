''' Define the Layers '''
import torch.nn as nn
from transformer.SubLayers import PositionwiseFeedForward, MultiHeadCrossAttention_Flash, MultiHeadSelfAttention_Flash
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