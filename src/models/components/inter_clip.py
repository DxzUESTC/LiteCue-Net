import torch
import torch.nn as nn

class InterClipModule(nn.Module):
    """
    Stage 2: 全局不一致性累积 (Global Inconsistency Accumulation)
    
    架构: Gated-MLP (Based on gMLP / MoGa)
    作用: 让 M 个时序片段之间进行全局交互，捕捉长程逻辑矛盾。
    核心: 不使用沉重的 Self-Attention，而是使用 "门控(Gating) + 线性投影" 实现信息混合。
    """
    def __init__(self, dim=256, seq_len=16, expansion_factor=2):
        """
        Args:
            dim: 输入特征维度 D (256)
            seq_len: 时序长度 M (16) - GatedMLP 需要知道序列长度来进行全局投影
            expansion_factor: MLP 中间层放大倍数
        """
        super().__init__()
        inner_dim = dim * expansion_factor
        
        # 1. 通道混合 (Channel Mixing) - 进门先放大
        # 类似于 Transformer FFN 的前半部分
        self.fc_in = nn.Linear(dim, inner_dim)
        
        # 2. 空间门控单元 (Spatial Gating Unit, SGU) - 核心魔法
        # 这里的 "空间" 其实指 "时间序列(Sequence)" 维度
        self.norm = nn.LayerNorm(inner_dim // 2)
        
        # 这个投影层负责让 16 个片段互相"看见"
        # 输入长度 M -> 输出长度 M
        # 初始化极其重要：接近 0 意味着初始状态下 Gate 是透传的，容易训练
        self.proj_time = nn.Linear(seq_len, seq_len)
        # nn.init.dirac_(self.proj_time.weight) # 错误：Linear 层不能用 Dirac
        nn.init.eye_(self.proj_time.weight)
        nn.init.zeros_(self.proj_time.bias)
        
        # 3. 输出投影 - 恢复维度
        self.fc_out = nn.Linear(inner_dim // 2, dim)
        
        # 激活函数
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Args:
            x: (B, M, D) - 来自 Stage 1 的 Clip Tokens
        Returns:
            x: (B, M, D) - 注入了全局一致性信息的特征
        """
        # 残差连接的跳跃分支
        shortcut = x
        
        # -----------------------------------------------------------
        # 第一步：通道膨胀与激活
        # -----------------------------------------------------------
        # (B, M, D) -> (B, M, 2*D)
        x = self.fc_in(x)
        x = self.act(x)
        
        # -----------------------------------------------------------
        # 第二步：门控分割 (Split for Gating)
        # -----------------------------------------------------------
        # 将特征一分为二：
        # u: 内容流 (Content)
        # v: 门控流 (Gate) - 用来计算权重
        # shape: (B, M, D)
        u, v = x.chunk(2, dim=-1)
        
        # -----------------------------------------------------------
        # 第三步：全局时序交互 (Global Temporal Interaction)
        # -----------------------------------------------------------
        # 我们要对 v 在 "M" (时间) 维度上进行线性投影
        
        # 归一化门控流
        v = self.norm(v)
        
        # 维度转置: Linear 默认只处理最后一个维度
        # 我们想处理 M，所以把 M 换到最后
        # (B, M, D) -> (B, D, M)
        v = v.transpose(1, 2)
        
        # 全局投影
        # 矩阵乘法让 Clip 1 的信息流向 Clip 16
        # (B, D, M) * (M, M) -> (B, D, M)
        v = self.proj_time(v)
        
        # 转回正常维度
        # (B, D, M) -> (B, M, D)
        v = v.transpose(1, 2)
        
        # -----------------------------------------------------------
        # 第四步：门控融合 (Gating Operation)
        # -----------------------------------------------------------
        # 逐元素相乘
        # 如果 v (Gate) 的某个值很小，说明对应的 u (Content) 是噪音/冲突，被抑制
        # 如果 v 很大，说明这个特征符合全局逻辑，被通过/放大
        x = u * v
        
        # -----------------------------------------------------------
        # 第五步：输出投影与残差
        # -----------------------------------------------------------
        # (B, M, D) -> (B, M, D)
        x = self.fc_out(x)
        
        return x + shortcut

# ==========================================
# 单元测试
# ==========================================
if __name__ == "__main__":
    B, M, D = 2, 16, 256
    model = InterClipModule(dim=D, seq_len=M)
    
    # 模拟 Stage 1 的输出
    dummy_input = torch.randn(B, M, D)
    
    output = model(dummy_input)
    
    print(f"Stage 2 Input: {dummy_input.shape}")
    print(f"Stage 2 Output: {output.shape}")
    
    assert output.shape == (B, M, D)
    print("Stage 2 (Gated-MLP) 测试通过。")