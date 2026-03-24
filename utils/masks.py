import torch


def create_stride_mask(seq_len, stride):

    mask = torch.zeros(seq_len, seq_len)

    for i in range(seq_len):
        for j in range(seq_len):

            if abs(i - j) <= stride:
                mask[i][j] = 1

    return mask


def create_global_mask(seq_len):

    return torch.ones(seq_len, seq_len)