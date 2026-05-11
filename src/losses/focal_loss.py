import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard/easy sample imbalance.
    Formula: Loss(pt) = - alpha_t * (1 - pt)^gamma * log(pt)
    """
    def __init__(self, alpha=1.0, gamma=2.0, num_classes=2, reduction='mean'):
        """
        Args:
            alpha (float or list): 类别权重。
                                   如果是 float，则对 Class 1 (Fake) 使用 alpha，Class 0 (Real) 使用 1-alpha。
                                   如果是 list，则对应 [weight_class0, weight_class1]。
            gamma (float): 聚焦参数 (Focusing parameter)。Gamma > 0 会减少易分类样本的损失贡献。
                           默认 2.0 是论文推荐值。
            num_classes (int): 类别数量 (LiteCue 为 2)。
            reduction (str): 'mean', 'sum' or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes

        # 处理 alpha
        if isinstance(alpha, (float, int)):
            # 二分类下的简便写法: alpha 是正类(Fake)的权重
            self.alpha = torch.tensor([1 - alpha, alpha])
        elif isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = None # 不使用 alpha 平衡

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, Num_Classes) - 模型输出的 Logits (未经过 Softmax)
            targets: (B,) - 真实标签 (0 或 1)
        """
        # 确保 alpha 在正确的设备上
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)

        # 1. 计算 Softmax 概率
        # inputs: Logits -> Probs
        prob = F.softmax(inputs, dim=1)
        
        # 2. 获取对应真实标签的概率 pt
        # gather 选取 target 对应的概率值
        # log_prob: (B, 2) -> (B, 1)
        # 相当于 log(pt)
        log_prob = F.log_softmax(inputs, dim=1)
        log_pt = log_prob.gather(1, targets.view(-1, 1))
        
        # pt = exp(log_pt)
        pt = log_pt.exp()

        # 3. 计算 Focal Term: (1 - pt) ^ gamma
        focal_term = (1 - pt).pow(self.gamma)

        # 4. 计算 Alpha Term (类别平衡)
        if self.alpha is not None:
            # 选取 target 对应的 alpha
            alpha_t = self.alpha.gather(0, targets.view(-1))
            alpha_t = alpha_t.view(-1, 1)
            # 最终损失公式
            loss = -alpha_t * focal_term * log_pt
        else:
            loss = -1.0 * focal_term * log_pt

        # 5. Reduction (Mean/Sum)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss