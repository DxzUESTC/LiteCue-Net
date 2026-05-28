import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入之前实现的组件
from .backbones.mobilenet_v4 import MobileNetV4Backbone
from .components.intra_clip import IntraClipModule
from .components.inter_clip import InterClipModule, InterClipAttention
from .components.reviewer import HistoricalReviewModule, LearnableDecayHRM

class LiteCueNet(nn.Module):
    """
    LiteCue-Net: Lightweight Forgery Cue Unraveling Network
    
    流水线 (Pipeline):
    1. Input: (B, M, K, C, H, W) -> 视频片段张量
    2. Backbone: MobileNetV4 -> 提取每帧的空间特征 (B*M*K, D)
    3. Stage 1 (Intra-Clip): DW-Conv1D -> 捕捉微动并聚合为片段特征 (B, M, D)
    4. Stage 2 (Inter-Clip): Gated-MLP -> 捕捉全局逻辑不一致 (B, M, D)
    5. Stage 3 (Reviewer): HRM -> (仅推理阶段) 利用未来修正过去 (B, M, D)
    6. Head: Classifier -> 输出判别结果 (B, M, 2)
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
        """
        Args:
            feature_dim: 内部特征维度 D (LiteCue设为256以保持轻量)
            clip_num: 全局片段数量 M (16)
            clip_len: 片段内帧数 K (4)
            num_classes: 分类数量 (2: Real/Fake)
        """
        super().__init__()
        self.clip_num = clip_num
        self.clip_len = clip_len
        self.token_dropout = float(token_dropout)
        self.use_temporal_diff = use_temporal_diff
        self.use_frequency_branch = use_frequency_branch
        self.frequency_fuse_block = frequency_fuse_block
        self.grl_lambda = grl_lambda
        self.temporal_module = temporal_module
        
        # ---------------------------------------------------------
        # 1. 空间骨干 (Spatial Backbone)
        # ---------------------------------------------------------
        # 输入: 单帧图像 -> 输出: 空间特征 (D维)
        self.backbone = MobileNetV4Backbone(
            model_name=backbone_name,
            out_dim=feature_dim,
            pretrained=pretrained,
            fuse_block_idx=frequency_fuse_block if use_frequency_branch else -1,
        )
        
        # ---------------------------------------------------------
        # 2. Stage 1: 局部微动模块 (Intra-Clip)
        # ---------------------------------------------------------
        # 对应 TFCU 的 CCM 模块
        # 捕捉 4 帧内的瞬时异常 (DW-Conv1D)
        self.intra_clip = IntraClipModule(
            input_dim=feature_dim,
            clip_len=clip_len
        )

        if self.use_temporal_diff:
            self.temporal_diff_proj = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, feature_dim),
                nn.SiLU(inplace=True),
            )

        if self.use_frequency_branch:
            # 频域分支：输出特征图，在 backbone 中间层融合
            # 输入 (N, 3, 224, 224) -> 输出 (N, 96, 14, 14)
            # 与 backbone block_2 输出 shape 对齐后做残差融合
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
        
        # ---------------------------------------------------------
        # 3. Stage 2: 全局不一致性模块 (Inter-Clip)
        # ---------------------------------------------------------
        # 对应 TFCU 的 FGM 模块
        # 捕捉 16 个片段间的逻辑冲突
        # 支持 "gated_mlp" (默认) 和 "attention" 两种模式
        if temporal_module == "attention":
            self.inter_clip = InterClipAttention(
                dim=feature_dim,
                seq_len=clip_num,
            )
        else:
            self.inter_clip = InterClipModule(
                dim=feature_dim,
                seq_len=clip_num
            )
        
        # ---------------------------------------------------------
        # 4. Stage 3: 历史回顾模块 (Historical Review)
        # ---------------------------------------------------------
        # 对应 TFCU 的 HRM 模块
        # 仅在推理时使用
        self.reviewer = LearnableDecayHRM(seq_len=clip_num)
        
        # ---------------------------------------------------------
        # 5. 分类头与视频级融合权重 (Classifier Head & Fusion)
        # ---------------------------------------------------------
        # 简单的线性层，将特征映射为真伪概率
        self.head = nn.Linear(feature_dim, num_classes)

        # 可学习的 clip 权重网络：自动学习哪些 clip 更可信
        # 用于视频级推理时替代简单的平均池化
        self.clip_weight_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        self.domain_classifier = None
        if num_domains and num_domains > 1:
            from src.losses.generalization import DomainClassifier
            self.domain_classifier = DomainClassifier(feature_dim, num_domains)

    def forward(self, x, return_features=False, return_domain=False):
        """
        Args:
            x: (B, M, K, C, H, W) - 输入视频数据
        Returns:
            logits: (B, Num_Classes) - 视频级别的预测结果 (平均后)
            clip_logits: (B, M, Num_Classes) - 每个片段的独立预测结果 (用于计算 Loss)
        """
        # B: Batch Size
        # M: Clip Num (16)
        # K: Frame Num (4)
        # C, H, W: (3, 224, 224)
        B, M, K, C, H, W = x.shape
        
        # =========================================================
        # Step 1: 空间特征提取 (Spatial Extraction)
        # =========================================================
        # 变形: 把所有 Clip 的所有 Frame 拍扁到一起处理
        # (B, M, K, C, H, W) -> (B*M*K, C, H, W)
        x_flat = x.view(-1, C, H, W)
        
        if self.use_frequency_branch:
            # 像素空间高通残差
            low_freq = F.avg_pool2d(x_flat, kernel_size=5, stride=1, padding=2)
            high_freq = x_flat - low_freq
            # 频域分支提取特征图，在 backbone 中间层融合
            freq_feats = self.frequency_branch(high_freq)
            spatial_feats = self.backbone(x_flat, freq_feats=freq_feats)
        else:
            spatial_feats = self.backbone(x_flat)
        
        # =========================================================
        # Step 2: 局部微动捕捉 (Stage 1)
        # =========================================================
        # 输入: 拍扁的空间特征
        # 内部操作: Reshape回 (B*M, K, D) -> DW-Conv1D -> Pooling
        # 输出: (B, M, D) - 已经是聚合好的 Clip Tokens
        clip_feats = self.intra_clip(spatial_feats, M=M)

        if self.use_temporal_diff:
            frame_feats = spatial_feats.view(B, M, K, -1)
            diff_feats = torch.abs(frame_feats[:, :, 1:, :] - frame_feats[:, :, :-1, :]).mean(dim=2)
            clip_feats = clip_feats + self.temporal_diff_proj(diff_feats)

        if self.training and 0 < self.token_dropout < 1:
            keep = torch.rand(B, M, 1, device=clip_feats.device) > self.token_dropout
            clip_feats = clip_feats * keep / (1.0 - self.token_dropout)
        
        # =========================================================
        # Step 3: 全局不一致性分析 (Stage 2)
        # =========================================================
        # 输入: (B, M, D)
        # 内部操作: Gated-MLP 全局交互
        # 输出: (B, M, D) - 包含上下文信息的特征
        global_feats = self.inter_clip(clip_feats)
        
        # =========================================================
        # Step 4: 历史回顾 (Stage 3) - [TFCU 策略]
        # =========================================================
        # TFCU 论文指出 HRM 是 "post-processing step" 
        # 通常仅在推理阶段 (model.eval()) 启用，以利用未来信息修正历史
        if not self.training:
            global_feats = self.reviewer(global_feats)
            
        # =========================================================
        # Step 5: 分类与聚合 (Head & Aggregation)
        # =========================================================
        # 对每个片段进行独立分类
        # (B, M, D) -> (B, M, Num_Classes)
        clip_logits = self.head(global_feats)
        
        # 视频级预测 (Video-level Prediction)
        # 可学习加权融合：基于 clip 特征动态计算权重
        # 让模型自动抑制噪声 clip、放大可信 clip 的贡献
        # (B, M, D) -> (B, M, 1) -> softmax over M
        weights = self.clip_weight_net(global_feats)
        weights = F.softmax(weights, dim=1)
        # (B, M, 2) * (B, M, 1) -> (B, 2)
        video_logits = (clip_logits * weights).sum(dim=1)

        # 计算 clip 权重熵（防止退化到 one-hot）
        # 仅在训练时有效，用于 entropy regularization
        clip_entropy = None
        if self.training:
            weights_sq = weights.squeeze(-1)  # (B, M)
            clip_entropy = -(weights_sq * torch.log(weights_sq.clamp(min=1e-8))).sum(dim=1).mean()

        if return_features or return_domain:
            video_features = global_feats.mean(dim=1)
            outputs = {
                'video_logits': video_logits,
                'clip_logits': clip_logits,
                'features': video_features,
                'clip_entropy': clip_entropy,
            }
            if return_domain and self.domain_classifier is not None:
                outputs['domain_logits'] = self.domain_classifier(video_features, self.grl_lambda)
            return outputs

        return video_logits, clip_logits

# ==========================================
# 单元测试与模型总览
# ==========================================
if __name__ == "__main__":
    # 配置参数
    CONFIG = {
        'B': 2,
        'M': 16,
        'K': 4,
        'D': 256
    }
    
    print("正在组装 LiteCue-Net...")
    model = LiteCueNet(
        feature_dim=CONFIG['D'],
        clip_num=CONFIG['M'],
        clip_len=CONFIG['K']
    )
    
    # 打印参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Built Successfully.")
    print(f"Total Parameters: {total_params / 1e6:.2f} M") # 预期应远小于 TFCU (ViT-Base ~86M)
    
    # 生成假数据测试前向传播
    dummy_input = torch.randn(CONFIG['B'], CONFIG['M'], CONFIG['K'], 3, 224, 224)
    print(f"\nTesting Forward Pass...")
    print(f"Input Shape: {dummy_input.shape}")
    
    # 1. 训练模式测试
    model.train()
    v_logits, c_logits = model(dummy_input)
    print(f"[Train Mode] Video Logits: {v_logits.shape} (Expect: {CONFIG['B']}, 2)")
    print(f"[Train Mode] Clip Logits:  {c_logits.shape} (Expect: {CONFIG['B']}, {CONFIG['M']}, 2)")
    
    # 2. 推理模式测试 (Stage 3 HRM 激活)
    model.eval()
    with torch.no_grad():
        v_logits_eval, _ = model(dummy_input)
        print(f"[Eval Mode]  Video Logits: {v_logits_eval.shape} (Stage 3 Active)")

    print("\nLiteCue-Net Assembly Complete.")