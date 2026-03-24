import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class CAMSAAttention(nn.Module):

    def __init__(self, d_model):
        super().__init__()

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)

        self.scale = d_model ** 0.5
        
    def forward(self, x, stride_masks):

        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        attention_outputs = []

        for mask in stride_masks:

            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

            scores = scores.masked_fill(mask == 0, -1e9)

            weights = F.softmax(scores, dim=-1)
            self.last_attention = weights  #added for visualising
            out = torch.matmul(weights, V)

            attention_outputs.append(out)

        output = torch.mean(torch.stack(attention_outputs), dim=0)

        return self.out(output)