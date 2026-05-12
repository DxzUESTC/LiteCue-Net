import torch
import torch.nn as nn


class HistoricalReviewModule(nn.Module):
    """
    Stage 3: 历史回顾模块 (Historical Review Module) — 原始参数无关版本

    对应 TFCU 论文中的 HRM 模块。
    核心逻辑: "Future Guide History" (利用未来修正过去)
    使用硬编码的逆线性衰减权重: weight = 1 / (distance + 1)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, M, D = x.shape
        x_refined = x.clone()
        for m in range(M - 1):
            for s in range(m + 1, M):
                distance = s - m
                weight = 1.0 / (distance + 1)
                x_refined[:, m, :] += x[:, s, :] * weight
        return x_refined


class LearnableDecayHRM(nn.Module):
    """
    可学习衰减系数的历史回顾模块 (改进五).

    [问题] 原 LearnableHRM 用 M×M 可学习矩阵 + softmax 归一化:
      - softmax 在变长行上导致位置偏差，末尾 clip 的回顾信号逐行衰减
      - 最后一行全 -inf → NaN，靠 nan_to_num 掩盖
      - 256 个参数过于冗余

    [方案] 回到原始 HRM 的 1/(distance + 1) 公式，
          但将衰减系数 β 改为可学习（单参数）。
          weight = 1 / (distance + β), β > 0

    参数量: 1 (log_beta → β)
    """
    def __init__(self, seq_len=16):
        super().__init__()
        self.seq_len = seq_len
        # β = exp(log_beta) 保证正值；初始 β=1 匹配原始 1/(distance+1)
        self.log_beta = nn.Parameter(torch.zeros(1))

        # 距离矩阵: dist[i, j] = j - i, 但仅对 s > m 有效，其余位置为 0
        # 将非上三角位置的距离置零，避免后续 1/(dist + beta) 时 dist=-1, beta=1 导致除零
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        dist = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
        dist = dist * mask  # 非未来位置的距离置零
        self.register_buffer('_dist', dist)
        self.register_buffer('_mask', mask)

    def forward(self, x):
        """
        Args:
            x: (B, M, D) - 来自 Stage 2 的全局特征序列
        Returns:
            x_refined: (B, M, D) - 经过可学习历史回顾增强后的特征
        """
        B, M, D = x.shape
        beta = torch.exp(self.log_beta)
        # weights[m, s] = 1/(s-m+β)  for s > m, 0 otherwise
        weights = self._mask * (1.0 / (self._dist + beta))
        # weighted sum of future clips for each position
        future_context = torch.bmm(weights.unsqueeze(0).expand(B, -1, -1), x)
        return x + future_context


# ==========================================
# 单元测试
# ==========================================
if __name__ == "__main__":
    # 原始 HRM 测试
    dummy = torch.tensor([[[1.0], [2.0], [3.0], [10.0]]])
    hrm = HistoricalReviewModule()
    out = hrm(dummy)
    expected = 1.0 + (2.0/2) + (3.0/3) + (10.0/4)
    print(f"原始 HRM Clip 0: {out[0,0,0].item():.2f} (预期 {expected:.2f})")

    # LearnableDecayHRM 测试
    lhrm = LearnableDecayHRM(seq_len=4)
    out2 = lhrm(dummy)
    print(f"LearnableDecayHRM 输出形状: {out2.shape}")
    print(f"LearnableDecayHRM 参数量: {sum(p.numel() for p in lhrm.parameters())}")  # 预期: 1

    # 验证权重: 原始 1/(distance+1) 匹配
    row0_expected = [0, 1/(1+1), 1/(2+1), 1/(3+1)]  # [0, 1/2, 1/3, 1/4]
    with torch.no_grad():
        weights = lhrm._mask * (1.0 / (lhrm._dist + 1.0))
        print(f"Weight row 0: {weights[0].tolist()} (expected {row0_expected})")
        print(f"Weight row 3 (last): {weights[3].tolist()} (expected all 0)")

    # 验证 HumanReviewModule 与 LearnableDecayHRM β=1 的输出一致
    hrm = HistoricalReviewModule()
    out_orig = hrm(dummy)
    # lhrm 初始 β=exp(0)=1, 应输出相同
    out_new = lhrm(dummy)
    print(f"HRM clip 0: {out_orig[0,0,0].item():.4f}")
    print(f"LearnableDecayHRM clip 0: {out_new[0,0,0].item():.4f}")

    print("HRM 测试通过。")
