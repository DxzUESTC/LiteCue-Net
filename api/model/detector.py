import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones.mobilenet_v4 import MobileNetV4Backbone
from .components.intra_clip import IntraClipModule
from .components.inter_clip import InterClipModule, InterClipAttention
from .components.reviewer import LearnableDecayHRM


class LiteCueNet(nn.Module):
    """
    LiteCue-Net: Lightweight Forgery Cue Unraveling Network

    Pipeline:
    1. Input: (B, M, K, C, H, W)
    2. Backbone: MobileNetV4 -> (B*M*K, D)
    3. Stage 1 (Intra-Clip): DW-Conv1D -> (B, M, D)
    4. Stage 2 (Inter-Clip): Gated-MLP -> (B, M, D)
    5. Stage 3 (Reviewer): HRM -> (inference only) (B, M, D)
    6. Head: Classifier -> (B, M, 2)
    """

    def __init__(
        self,
        feature_dim=256,
        clip_num=16,
        clip_len=4,
        num_classes=2,
        backbone_name='mobilenetv4_conv_small.e2400_r224_in1k',
        pretrained=True,
        token_dropout=0.0,
        use_temporal_diff=False,
        use_frequency_branch=False,
        frequency_fuse_block=2,
        num_domains=0,
        grl_lambda=1.0,
        temporal_module="attention",
    ):
        super().__init__()
        self.clip_num = clip_num
        self.clip_len = clip_len
        self.token_dropout = float(token_dropout)
        self.use_temporal_diff = use_temporal_diff
        self.use_frequency_branch = use_frequency_branch
        self.frequency_fuse_block = frequency_fuse_block
        self.grl_lambda = grl_lambda
        self.temporal_module = temporal_module

        self.backbone = MobileNetV4Backbone(
            model_name=backbone_name,
            out_dim=feature_dim,
            pretrained=pretrained,
            fuse_block_idx=frequency_fuse_block if use_frequency_branch else -1,
        )

        self.intra_clip = IntraClipModule(
            input_dim=feature_dim,
            clip_len=clip_len,
        )

        if self.use_temporal_diff:
            self.temporal_diff_proj = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, feature_dim),
                nn.SiLU(inplace=True),
            )

        if self.use_frequency_branch:
            self.frequency_branch = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.SiLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.SiLU(inplace=True),
                nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(96),
                nn.SiLU(inplace=True),
                nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(96),
                nn.SiLU(inplace=True),
            )

        if temporal_module == "attention":
            self.inter_clip = InterClipAttention(
                dim=feature_dim,
                seq_len=clip_num,
            )
        else:
            self.inter_clip = InterClipModule(
                dim=feature_dim,
                seq_len=clip_num,
            )

        self.reviewer = LearnableDecayHRM(seq_len=clip_num)

        self.head = nn.Linear(feature_dim, num_classes)

        self.clip_weight_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        # Domain classifier is omitted in the deployment snapshot;
        # it is training-only (num_domains=0 by default) and not needed at inference.
        self.domain_classifier = None

    def forward(self, x, return_features=False, return_domain=False):
        B, M, K, C, H, W = x.shape

        x_flat = x.view(-1, C, H, W)

        if self.use_frequency_branch:
            low_freq = F.avg_pool2d(x_flat, kernel_size=5, stride=1, padding=2)
            high_freq = x_flat - low_freq
            freq_feats = self.frequency_branch(high_freq)
            spatial_feats = self.backbone(x_flat, freq_feats=freq_feats)
        else:
            spatial_feats = self.backbone(x_flat)

        clip_feats = self.intra_clip(spatial_feats, M=M)

        if self.use_temporal_diff:
            frame_feats = spatial_feats.view(B, M, K, -1)
            diff_feats = torch.abs(frame_feats[:, :, 1:, :] - frame_feats[:, :, :-1, :]).mean(dim=2)
            clip_feats = clip_feats + self.temporal_diff_proj(diff_feats)

        if self.training and 0 < self.token_dropout < 1:
            keep = torch.rand(B, M, 1, device=clip_feats.device) > self.token_dropout
            clip_feats = clip_feats * keep / (1.0 - self.token_dropout)

        global_feats = self.inter_clip(clip_feats)

        if not self.training:
            global_feats = self.reviewer(global_feats)

        clip_logits = self.head(global_feats)

        weights = self.clip_weight_net(global_feats)
        weights = F.softmax(weights, dim=1)
        video_logits = (clip_logits * weights).sum(dim=1)

        clip_entropy = None
        if self.training:
            weights_sq = weights.squeeze(-1)
            clip_entropy = -(weights_sq * torch.log(weights_sq.clamp(min=1e-8))).sum(dim=1).mean()

        if return_features or return_domain:
            video_features = global_feats.mean(dim=1)
            outputs = {
                'video_logits': video_logits,
                'clip_logits': clip_logits,
                'features': video_features,
                'clip_entropy': clip_entropy,
            }
            return outputs

        return video_logits, clip_logits
