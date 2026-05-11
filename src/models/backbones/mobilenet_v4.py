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
    """
    def __init__(self, model_name='mobilenetv4_conv_small.e2400_r224_in1k', out_dim=256, pretrained=True, freeze=False):
        """
        Args:
            model_name (str): timm 模型名称，默认使用 MobileNetV4-Small (高精度版)
            out_dim (int): 输出特征维度 D (Stage 1 和 Stage 2 的输入维度)
            pretrained (bool): 是否加载 ImageNet 权重
            freeze (bool): 是否冻结 Backbone 参数 (Transfer Learning 初期常用)
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
        # 将原始维度 (e.g., 960) 降维到 LiteCue-Net 的隐藏层维度 (e.g., 256)
        # 作用：大幅减少后续 Conv1D 和 Gated-MLP 的参数量，同时起到特征适配的作用。
        self.projector = nn.Sequential(
            nn.Linear(in_features, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.SiLU(inplace=True) # Swish 激活函数
        )
        
        # -------------------------------------------------------------
        # 3. 冻结参数 (可选)
        # -------------------------------------------------------------
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone parameters frozen.")

    def forward(self, x):
        """
        Args:
            x: 输入张量 (N, 3, H, W)
               注意：这里的 N 通常是 B * M * K (把所有视频的所有帧拍扁在一起处理)
        Returns:
            features: (N, out_dim)
        """
        # 1. 提取基础特征 -> (N, 960)
        raw_features = self.backbone(x)
        
        # 2. 降维投影 -> (N, 256)
        features = self.projector(raw_features)
        
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