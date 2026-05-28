import torch
import torch.nn as nn
import torch.nn.functional as F


class IntraClipModule(nn.Module):
    """Stage 1: Local micro-motion modeling within a clip."""

    def __init__(self, input_dim=256, clip_len=4):
        super().__init__()
        self.clip_len = clip_len

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=input_dim,
                kernel_size=3,
                padding=1,
                groups=input_dim,
                bias=False,
            ),
            nn.BatchNorm1d(input_dim),
            nn.SiLU(inplace=True),
        )

        self.attn_pool = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        self.norm = nn.LayerNorm(input_dim)

    def forward(self, spatial_features, M):
        BMK, D = spatial_features.shape
        K = self.clip_len
        B = BMK // (M * K)

        x = spatial_features.view(B * M, K, D)
        x_permuted = x.transpose(1, 2)

        motion = self.temporal_conv(x_permuted)
        fused = x_permuted + motion

        attn_weights = self.attn_pool(fused.transpose(1, 2))
        attn_weights = F.softmax(attn_weights, dim=1)
        clip_tokens_flat = (fused.transpose(1, 2) * attn_weights).sum(dim=1)

        clip_tokens_flat = self.norm(clip_tokens_flat)
        clip_tokens = clip_tokens_flat.view(B, M, D)

        return clip_tokens
