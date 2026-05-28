import torch
import torch.nn as nn


class LearnableDecayHRM(nn.Module):
    """
    Stage 3: Historical Review Module with learnable decay.
    weight = 1 / (distance + beta), beta > 0.
    """

    def __init__(self, seq_len=16):
        super().__init__()
        self.seq_len = seq_len
        self.log_beta = nn.Parameter(torch.zeros(1))

        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        dist = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
        dist = dist * mask
        self.register_buffer('_dist', dist)
        self.register_buffer('_mask', mask)

    def forward(self, x):
        B, M, D = x.shape
        beta = torch.exp(self.log_beta)
        weights = self._mask * (1.0 / (self._dist + beta))
        future_context = torch.bmm(weights.unsqueeze(0).expand(B, -1, -1), x)
        return x + future_context
