import torch
def seqlen2cu_len(seqlens):
    """
    Convert sequence lengths to cumulative lengths for Flash Attention.

    Args:
        seqlens (torch.Tensor): A 1D tensor containing the lengths of each sequence in the batch.

    Returns:
        torch.Tensor: A 1D tensor containing the cumulative lengths.
    """
    cumsum = torch.cumsum(seqlens, dim=0, dtype=torch.int32)
    cu_lens = torch.nn.functional.pad(cumsum, (1, 0), value=0)
    return cu_lens

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

