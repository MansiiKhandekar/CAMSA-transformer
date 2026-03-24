import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardAttention(nn.Module):

    def __init__(self, d_model):
        super().__init__()

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.last_attention = None #for visualizing standard model
        self.scale = d_model ** 0.5

    def forward(self, x):

        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        weights = F.softmax(scores, dim=-1)
        self.last_attention = weights

        return torch.matmul(weights, V)