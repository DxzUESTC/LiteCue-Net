import torch
import torch.nn as nn
import torch.nn.functional as F


class InterClipModule(nn.Module):
    """Stage 2: Global inconsistency accumulation via Gated-MLP."""

    def __init__(self, dim=256, seq_len=16, expansion_factor=2):
        super().__init__()
        inner_dim = dim * expansion_factor
        self.fc_in = nn.Linear(dim, inner_dim)
        self.norm = nn.LayerNorm(inner_dim // 2)
        self.proj_time = nn.Linear(seq_len, seq_len)
        nn.init.eye_(self.proj_time.weight)
        nn.init.zeros_(self.proj_time.bias)
        self.fc_out = nn.Linear(inner_dim // 2, dim)
        self.act = nn.SiLU()

    def forward(self, x):
        shortcut = x
        x = self.fc_in(x)
        x = self.act(x)
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = v.transpose(1, 2)
        v = self.proj_time(v)
        v = v.transpose(1, 2)
        x = u * v
        x = self.fc_out(x)
        return x + shortcut


class InterClipAttention(nn.Module):
    """Lightweight global temporal interaction via Attention."""

    def __init__(self, dim=256, num_heads=4, seq_len=16, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, dim) * 0.02)

        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        self.register_buffer('causal_mask', mask)

    def forward(self, x):
        x = x + self.pos_embed
        x = x + self.attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            attn_mask=self.causal_mask,
        )[0]
        x = x + self.ffn(self.norm2(x))
        return x
