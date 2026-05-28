import torch
import torch.nn as nn
import timm


class MobileNetV4Backbone(nn.Module):
    """
    timm MobileNetV4 wrapper for deployment inference.
    """
    def __init__(self, model_name='mobilenetv4_conv_small.e2400_r224_in1k',
                 out_dim=256, pretrained=True, freeze=False, fuse_block_idx=-1):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg',
        )

        with torch.no_grad():
            dummy = torch.zeros(2, 3, 224, 224)
            dummy_out = self.backbone(dummy)
            in_features = dummy_out.shape[1]

        self.projector = nn.Sequential(
            nn.Linear(in_features, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.SiLU(inplace=True),
        )

        self.fuse_block_idx = fuse_block_idx
        n_blocks = len(self.backbone.blocks)
        if fuse_block_idx >= 0:
            assert fuse_block_idx < n_blocks, \
                f"fuse_block_idx={fuse_block_idx} out of range, backbone has {n_blocks} blocks (0~{n_blocks-1})"

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x, freq_feats=None):
        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)

        for i, block in enumerate(self.backbone.blocks):
            x = block(x)
            if freq_feats is not None and i == self.fuse_block_idx:
                x = x + freq_feats

        x = self.backbone.global_pool(x)
        x = self.backbone.conv_head(x)
        x = self.backbone.norm_head(x)
        x = self.backbone.act2(x)
        x = x.flatten(1)
        x = self.backbone.classifier(x)

        features = self.projector(x)
        return features
