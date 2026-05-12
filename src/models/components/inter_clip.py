import torch
import torch.nn as nn
import torch.nn.functional as F


class InterClipModule(nn.Module):
    """
    Stage 2: 全局不一致性累积 (Global Inconsistency Accumulation)

    架构: Gated-MLP (Based on gMLP / MoGa)
    作用: 让 M 个时序片段之间进行全局交互，捕捉长程逻辑矛盾。
    核心: 使用 "门控(Gating) + 线性投影" 实现信息混合。
    """
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
    """
    轻量级全局时序交互模块 (替代 Gated-MLP).

    用 Attention 替代 Gated-MLP 的固定线性混合，
    使模型能根据帧内容动态调整 clip 间注意力。

    参数量: ~795K (dim=256, heads=4, seq_len=16)
    相比 Gated-MLP (~394K) 增加约 400K 参数。
    """
    def __init__(self, dim=256, num_heads=4, seq_len=16, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, dim) * 0.02)

        # Causal mask: 防止 future clip 信息泄露，确保训练时 clip i 只能看到 clip 1~i
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        self.register_buffer('causal_mask', mask)

    def forward(self, x):
        """
        Args:
            x: (B, M, D) - 来自 Stage 1 的 Clip Tokens
        Returns:
            x: (B, M, D) - 注入了动态全局上下文信息的特征（因果约束）
        """
        x = x + self.pos_embed
        x = x + self.attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            attn_mask=self.causal_mask,
        )[0]
        x = x + self.ffn(self.norm2(x))
        return x


# ==========================================
# 单元测试
# ==========================================
if __name__ == "__main__":
    B, M, D = 2, 16, 256

    # Test Gated-MLP
    model_gmlp = InterClipModule(dim=D, seq_len=M)
    dummy = torch.randn(B, M, D)
    out = model_gmlp(dummy)
    params_gmlp = sum(p.numel() for p in model_gmlp.parameters())
    print(f"Gated-MLP Params: {params_gmlp:,}")
    assert out.shape == (B, M, D), f"Gated-MLP output shape mismatch: {out.shape}"
    print("Gated-MLP OK")

    # Test Attention
    model_attn = InterClipAttention(dim=D, seq_len=M)
    out = model_attn(dummy)
    params_attn = sum(p.numel() for p in model_attn.parameters())
    print(f"Attention Params: {params_attn:,}")
    assert out.shape == (B, M, D), f"Attention output shape mismatch: {out.shape}"
    print("Attention OK")
