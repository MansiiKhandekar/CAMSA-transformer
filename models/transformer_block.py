# models/transformer_block.py
from models.standard_attention import StandardAttention
import torch.nn as nn
from models.camsa_attention import CAMSAAttention
import config


class TransformerBlock(nn.Module):

    def __init__(self):

        super().__init__()

        if config.USE_CAMSA:
         self.attn = CAMSAAttention(config.EMBED_DIM)
        else:
         self.attn = StandardAttention(config.EMBED_DIM)
        self.norm1 = nn.LayerNorm(config.EMBED_DIM)

        self.ff = nn.Sequential(
            nn.Linear(config.EMBED_DIM, config.FF_DIM),
            nn.ReLU(),
            nn.Linear(config.FF_DIM, config.EMBED_DIM)
        )

        self.norm2 = nn.LayerNorm(config.EMBED_DIM)

    def forward(self, x, stride_masks):

        if config.USE_CAMSA:
         attn_out = self.attn(x, stride_masks)
        else:
         attn_out = self.attn(x)

        x = self.norm1(x + attn_out)

        ff_out = self.ff(x)

        x = self.norm2(x + ff_out)

        return x