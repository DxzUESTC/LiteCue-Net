import torch
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

class AverageMeter(object):
    """
    计算并存储平均值和当前值。
    用于平滑打印 Loss, Time, DataTime 等。
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def _safe_percent(value):
    if value is None or np.isnan(value):
        return 0.0
    return float(value) * 100.0


def _calculate_eer(y_true, fake_probs):
    try:
        fpr, tpr, thresholds = roc_curve(y_true, fake_probs)
    except ValueError:
        return 0.5, 0.5
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    threshold = thresholds[idx]
    return float(eer), float(threshold)


def _tpr_at_fpr(y_true, fake_probs, target_fpr):
    try:
        fpr, tpr, _ = roc_curve(y_true, fake_probs)
    except ValueError:
        return 0.0
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return 0.0
    return float(np.max(tpr[valid]))


def logits_to_fake_probs(y_pred_logits):
    y_pred_logits = np.array(y_pred_logits)
    probs = torch.softmax(torch.tensor(y_pred_logits), dim=1).numpy()
    return probs[:, 1], probs


def calculate_metrics(y_true, y_pred_logits):
    """
    计算 Deepfake 检测的核心指标: AUC 和 Accuracy
    
    Args:
        y_true (list or np.array): 真实标签 [0, 1, 1, 0, ...]
        y_pred_logits (list or np.array): 模型输出的 Logits (未经过 Softmax) [ [2.1, -0.5], ... ]
    
    Returns:
        dict: {'auc': float, 'acc': float}
    """
    # 转换为 numpy
    y_true = np.array(y_true)
    y_pred_logits = np.array(y_pred_logits)
    
    # 1. 计算概率 (Softmax)
    # 这是一个二分类问题，我们通常取 Class 1 (Fake) 的概率来计算 AUC
    fake_probs, probs = logits_to_fake_probs(y_pred_logits)
    
    # 2. 计算预测标签 (用于 Accuracy)
    preds = np.argmax(probs, axis=1)
    
    # 3. 计算 AUC (Area Under ROC Curve)
    # 处理特殊情况：如果 batch 里只有一个类别 (全是真或全是假)，AUC 无法计算
    try:
        auc = roc_auc_score(y_true, fake_probs)
    except ValueError:
        auc = 0.5 # 无法计算时返回随机猜的水平
        
    # 4. 计算 Accuracy
    acc = accuracy_score(y_true, preds)
    balanced_acc = balanced_accuracy_score(y_true, preds)

    try:
        ap = average_precision_score(y_true, fake_probs)
    except ValueError:
        ap = 0.0

    eer, eer_threshold = _calculate_eer(y_true, fake_probs)
    tpr_fpr_1 = _tpr_at_fpr(y_true, fake_probs, 0.01)
    tpr_fpr_01 = _tpr_at_fpr(y_true, fake_probs, 0.001)

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    except ValueError:
        tn = fp = fn = tp = 0
    
    return {
        "auc": auc * 100.0, # 转换为百分比
        "acc": acc * 100.0,
        "balanced_acc": _safe_percent(balanced_acc),
        "ap": _safe_percent(ap),
        "eer": _safe_percent(eer),
        "eer_threshold": eer_threshold,
        "tpr_at_fpr_1": _safe_percent(tpr_fpr_1),
        "tpr_at_fpr_0_1": _safe_percent(tpr_fpr_01),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

# ==========================================
# 单元测试
# ==========================================
if __name__ == "__main__":
    # 模拟数据
    labels = [0, 1, 0, 1]
    # Logits: 第一个维度大代表0(Real)，第二个维度大代表1(Fake)
    logits = [
        [2.0, 0.5], # 预测0 (Real) -> 对
        [0.2, 1.8], # 预测1 (Fake) -> 对
        [1.5, 1.6], # 预测1 (Fake) -> 错 (标签是0)
        [0.1, 3.0]  # 预测1 (Fake) -> 对
    ]
    
    metrics = calculate_metrics(labels, logits)
    print(f"Test Metrics: {metrics}")
    # 预期: Acc应该比较高, AUC应该也还可以