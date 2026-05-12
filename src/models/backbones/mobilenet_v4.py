import torch
import torch.nn as nn
import timm

class MobileNetV4Backbone(nn.Module):
    """
    基于 timm 的 MobileNetV4 封装。
    功能：
    1. 加载 ImageNet 预训练权重。
    2. 移除最后的 1000 类分类头 (Classifier Head)。
    3. 添加一个投影层 (Projection Head)，将特征维度统一映射到 LiteCue-Net 需要的维度 (如 256)。
    4. 可选：在中间层特征图处融合频域分支的输出（fuse_block_idx >= 0 时启用）。
    """
    def __init__(self, model_name='mobilenetv4_conv_small.e2400_r224_in1k', out_dim=256, pretrained=True, freeze=False, fuse_block_idx=-1):
        """
        Args:
            model_name (str): timm 模型名称，默认使用 MobileNetV4-Small (高精度版)
            out_dim (int): 输出特征维度 D (Stage 1 和 Stage 2 的输入维度)
            pretrained (bool): 是否加载 ImageNet 权重
            freeze (bool): 是否冻结 Backbone 参数 (Transfer Learning 初期常用)
            fuse_block_idx (int): 频域分支融合位置，-1 表示不启用（原始行为），
                                  0~4 分别对应 block_0 ~ block_4 输出后融合。
        """
        super().__init__()

        # -------------------------------------------------------------
        # 1. 加载 Backbone 并移除分类头
        # -------------------------------------------------------------
        # num_classes=0 是 timm 的魔法参数。
        # 设置为 0 后，模型会自动移除最后的 FC 层，并默认应用 Global Average Pooling (GAP)。
        # 输出形状将直接是 (Batch, Num_Features)，例如 (N, 960) 或 (N, 1280)。
        print(f"Loading backbone: {model_name} (Pretrained={pretrained})...")
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,       # <--- 关键：移除分类头
            global_pool='avg'    # <--- 关键：保留全局平均池化，把 (C, 7, 7) 变成 (C)
        )

        # 获取 Backbone原本的输出特征维度
        with torch.no_grad():
            dummy = torch.zeros(2, 3, 224, 224)
            dummy_out = self.backbone(dummy)
            in_features = dummy_out.shape[1]

        print(f"Backbone raw feature dimension: {in_features}")

        # -------------------------------------------------------------
        # 2. 特征投影层 (Projection Head)
        # -------------------------------------------------------------
        self.projector = nn.Sequential(
            nn.Linear(in_features, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.SiLU(inplace=True)
        )

        # -------------------------------------------------------------
        # 3. 中间层融合配置
        # -------------------------------------------------------------
        self.fuse_block_idx = fuse_block_idx
        n_blocks = len(self.backbone.blocks)
        if fuse_block_idx >= 0:
            assert fuse_block_idx < n_blocks, \
                f"fuse_block_idx={fuse_block_idx} out of range, backbone has {n_blocks} blocks (0~{n_blocks-1})"

        # -------------------------------------------------------------
        # 4. 冻结参数 (可选)
        # -------------------------------------------------------------
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone parameters frozen.")

    def forward(self, x, freq_feats=None):
        """
        Args:
            x: 输入张量 (N, 3, H, W)
            freq_feats: 频域分支特征图 (N, C, H, W)，仅在 fuse_block_idx >= 0 时传入
        Returns:
            features: (N, out_dim)
        """
        # 1. Stem: conv_stem + bn1
        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)

        # 2. Blocks with optional intermediate fusion
        for i, block in enumerate(self.backbone.blocks):
            x = block(x)
            if freq_feats is not None and i == self.fuse_block_idx:
                x = x + freq_feats

        # 3. Head: global_pool -> conv_head -> norm_head -> act2 -> flatten -> classifier
        x = self.backbone.global_pool(x)
        x = self.backbone.conv_head(x)
        x = self.backbone.norm_head(x)
        x = self.backbone.act2(x)
        x = x.flatten(1)
        x = self.backbone.classifier(x)

        # 4. 降维投影 -> (N, out_dim)
        features = self.projector(x)

        return features

# ==========================================
# 单元测试 (Unit Test)
# ==========================================
if __name__ == "__main__":
    # 模拟输入：Batch=2, Clips=16, Frames=4 -> 总共 128 张图
    B, M, K = 2, 16, 4
    dummy_input = torch.randn(B * M * K, 3, 224, 224)
    
    # 初始化模型
    model = MobileNetV4Backbone(out_dim=256)
    
    # 前向传播
    output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # 预期: (128, 256)
    
    # 验证是否包含梯度 (如果没有 freeze)
    print(f"Requires grad: {output.requires_grad}")
    
    assert output.shape == (B * M * K, 256), "Shape mismatch!"
    print("Backbone Test Passed.")