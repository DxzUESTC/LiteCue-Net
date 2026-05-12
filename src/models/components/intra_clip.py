import torch
import torch.nn as nn
import torch.nn.functional as F

class IntraClipModule(nn.Module):
    """
    Stage 1: 局部微动建模 (Local Micro-Motion Modeling)
    
    核心逻辑:
    1. 恢复维度 (Unfold): 把拍扁的特征恢复出时间维度 K (帧数)。
    2. 时序卷积 (DW-Conv1D): 在 K 帧范围内捕捉高频抖动或伪影。
    3. 残差融合 (Residual): 将捕捉到的微动线索加回到原始空间特征上。
    4. 聚合 (Aggregate): 将 K 帧特征平均池化为 1 个 Clip Token，供下一阶段使用。
    """
    def __init__(self, input_dim=256, clip_len=4):
        super().__init__()
        self.clip_len = clip_len
        
        # 定义时序卷积: 捕捉微动线索
        # 使用 Depth-wise (DW) 卷积，groups=input_dim 意味着通道独立
        # 就像派了 256 个独立的侦探，每个侦探只盯一个特征点的时间变化
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=input_dim,  # 输入通道数 (D)
                out_channels=input_dim, # 输出通道数 (D)
                kernel_size=3,          # 卷积核大小: 3 (看3帧的关联)
                padding=1,              # 填充: 1 (保证输出的时间长度不变，还是 K)
                groups=input_dim,       # [关键] 分组卷积: 极大的减少参数量，实现通道独立建模
                bias=False
            ),
            nn.BatchNorm1d(input_dim),  # 批归一化，稳定训练
            nn.SiLU(inplace=True)       # 激活函数
        )
        
        # 可学习注意力池化 (Learnable Attention Pooling)
        # 替代 mean pool，让模型学会关注 K 帧中"最可疑"的那一帧
        self.attn_pool = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        # 层归一化 (LayerNorm)
        # 在进入 Stage 2 (Gated-MLP) 之前对特征进行标准化，防止梯度爆炸
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, spatial_features, M):
        """
        Args:
            spatial_features: (B*M*K, D) - 来自 Backbone 的拍扁后的空间特征
            M: int - 视频被切分的片段数量 (Clip Num, 例如 16)
            
        Returns:
            clip_tokens: (B, M, D) - 聚合后的片段特征，准备送入 Stage 2
        """
        # 获取输入形状: BMK 是总图片数，D 是特征维度 (256)
        BMK, D = spatial_features.shape
        K = self.clip_len
        
        # 计算真实的 Batch Size
        # 公式: 总图数 / (片段数 * 每段帧数)
        B = BMK // (M * K)

        # -----------------------------------------------------------
        # 第一步：维度变形 (Reshape & Permute)
        # -----------------------------------------------------------
        # 目前数据是散乱的 (B*M*K, D)，我们需要让模型知道哪 K 帧是一组
        # 变形为: (总片段数, 帧数, 特征) -> (B*M, K, D)
        x = spatial_features.view(B * M, K, D)
        
        # 适配 Conv1D 的输入要求: (Batch, Channels, Length)
        # 变形为: (总片段数, 特征通道, 时间长度) -> (B*M, D, K)
        x_permuted = x.transpose(1, 2)
        
        # -----------------------------------------------------------
        # 第二步：提取微动线索 (Micro-Motion Extraction)
        # -----------------------------------------------------------
        # 卷积核在 K (时间轴) 上滑动，计算相邻帧的差异
        # 输出形状保持不变: (B*M, D, K)
        motion = self.temporal_conv(x_permuted)
        
        # -----------------------------------------------------------
        # 第三步：残差融合 (Residual Fusion)
        # -----------------------------------------------------------
        # 将 "微动线索" 加回到 "原始特征" 上
        # fused = x_permuted + motion
        fused = x_permuted + motion
        
        # -----------------------------------------------------------
        # 第四步：时序聚合 (Temporal Aggregation)
        # -----------------------------------------------------------
        # 既然已经提取完 K 帧内的关联了，我们不再需要保留每一帧
        # 对时间维度求平均 (Mean Pooling)，把 4 帧浓缩成 1 个代表性的向量
        # 可学习注意力池化：让模型自己决定 K 帧中哪些帧更重要
        # fused: (B*M, D, K) -> transpose -> (B*M, K, D)
        # attn_weights: (B*M, K, 1) softmax over K
        attn_weights = self.attn_pool(fused.transpose(1, 2))  # (B*M, K, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        clip_tokens_flat = (fused.transpose(1, 2) * attn_weights).sum(dim=1)  # (B*M, D)
        
        # -----------------------------------------------------------
        # 第五步：整理输出 (Final Reshape)
        # -----------------------------------------------------------
        # 归一化
        clip_tokens_flat = self.norm(clip_tokens_flat)
        
        # 恢复 Batch 和 Clip 维度，供下一阶段做全局分析
        # (B*M, D) -> (B, M, D)
        clip_tokens = clip_tokens_flat.view(B, M, D)
        
        return clip_tokens

if __name__ == "__main__":
    # 单元测试
    # 假设 Batch=2, M=16 Clips, K=4 Frames, Dim=256
    B, M, K, D = 2, 16, 4, 256
    module = IntraClipModule(input_dim=D, clip_len=K)
    
    # 模拟 Backbone 的输出 (拍扁的)
    dummy_spatial = torch.randn(B * M * K, D)
    
    out = module(dummy_spatial, M=M)
    
    print(f"Intra-Clip 输入形状: {dummy_spatial.shape}")
    print(f"Intra-Clip 输出形状: {out.shape}") # 预期: (2, 16, 256)
    
    assert out.shape == (B, M, D)
    print("Stage 1 测试通过。")