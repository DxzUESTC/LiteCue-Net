import torch
import torch.nn as nn

class HistoricalReviewModule(nn.Module):
    """
    Stage 3: 历史回顾模块 (Historical Review Module)
    
    对应 TFCU 论文中的 HRM 模块。
    核心逻辑: "Future Guide History" (利用未来修正过去)
    
    原理:
    后面的片段 (Future) 往往包含了更明显的伪造线索（因为深伪的漏洞通常随时间累积）。
    我们将未来的特征，按照距离衰减的权重，累积加回到过去的历史特征上。
    
    公式 (Inverse Linear Decay):
    Feature_m' = Feature_m + sum( Feature_s * Weight )
    Weight = 1 / (s - m + 1)
    其中 m 是当前片段，s 是未来片段 (s > m)。
    """
    def __init__(self):
        super().__init__()
        # 该模块没有可学习的参数 (Parameter-free)
        # 它是一个纯数学的后处理操作

    def forward(self, x):
        """
        Args:
            x: (B, M, D) - 来自 Stage 2 的全局特征序列
        Returns:
            x_refined: (B, M, D) - 经过历史回顾增强后的特征
        """
        B, M, D = x.shape
        
        # 为了不破坏原数据，先克隆一份
        x_refined = x.clone()
        
        # 双重循环实现动量累积
        # 复杂度 O(M^2)。因为 M 通常很小 (16)，所以计算开销几乎可以忽略不计。
        
        # 遍历每一个"历史"片段 m (从 0 到 M-2)
        # 最后一个片段没有"未来"，所以不用处理
        for m in range(M - 1):
            
            # 遍历该片段之后的所有"未来"片段 s
            for s in range(m + 1, M):
                
                # 计算衰减权重 (Inverse Linear Decay)
                # TFCU 原文公式: alpha = 1 / (s - m + 1)
                # 距离越远，权重越小
                distance = s - m
                weight = 1.0 / (distance + 1)
                
                # 执行累积: Future (s) -> History (m)
                # 将 s 的特征加权叠加到 m 上
                # 注意：这里用的是 x (原始输入) 还是 x_refined (更新后的)？
                # TFCU 原文语境暗示是基于 Stage 2 的输出进行累积，通常指原始输出。
                x_refined[:, m, :] += x[:, s, :] * weight

        return x_refined

# ==========================================
# 单元测试
# ==========================================
if __name__ == "__main__":
    # 模拟数据: Batch=1, Clips=4, Dim=1 (方便肉眼观察数值变化)
    # 假设特征值随时间递增，代表"伪造线索越来越明显"
    dummy_input = torch.tensor([[[1.0], [2.0], [3.0], [10.0]]]) # (1, 4, 1)
    
    model = HistoricalReviewModule()
    output = model(dummy_input)
    
    print("原始特征:", dummy_input.squeeze().numpy())
    print("回顾特征:", output.squeeze().numpy())
    
    # 手算验证第 0 个片段 (值=1.0) 的更新:
    # + Clip 1 (2.0) * 1/(1+1) = 2.0 * 0.5 = 1.0
    # + Clip 2 (3.0) * 1/(2+1) = 3.0 * 0.33 = 1.0
    # + Clip 3 (10.0)* 1/(3+1) = 10.0 * 0.25 = 2.5
    # 新值应该是: 1.0 + 1.0 + 1.0 + 2.5 = 5.5
    
    expected_val = 1.0 + (2.0/2) + (3.0/3) + (10.0/4)
    print(f"手算验证 Clip 0: {expected_val}")
    
    assert torch.isclose(output[0,0,0], torch.tensor(expected_val)), "计算逻辑有误！"
    print("Stage 3 (HRM) 测试通过。")