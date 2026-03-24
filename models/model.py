# models/model.py

import torch
import torch.nn as nn
import config
from models.transformer_block import TransformerBlock


class CAMSATransformer(nn.Module):

    def __init__(self, vocab_size):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, config.EMBED_DIM)

        self.layers = nn.ModuleList(
            [TransformerBlock() for _ in range(config.NUM_LAYERS)]
        )
        
        self.classifier = nn.Linear(config.EMBED_DIM, 2)

    def forward(self, x, stride_masks):

        x = self.embedding(x)

        for layer in self.layers:

            x = layer(x, stride_masks)

        x = x.mean(dim=1)

        return self.classifier(x)