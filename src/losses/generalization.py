import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradientReverseFunction.apply(x, lambd)


def supervised_contrastive_loss(features, labels, temperature=0.2):
    """
    Supervised contrastive loss for video-level features.
    features: (B, D)
    labels: (B,)
    """
    if features.size(0) < 2:
        return features.new_tensor(0.0)

    features = F.normalize(features, dim=1)
    logits = torch.matmul(features, features.t()) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.t()).float().to(features.device)
    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=features.device)
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_count = mask.sum(dim=1)
    valid = positive_count > 0
    if valid.sum() == 0:
        return features.new_tensor(0.0)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1)[valid] / positive_count[valid]
    return -mean_log_prob_pos.mean()


def coral_loss(features, domains):
    """
    CORAL loss: align covariance statistics across domains in a mini-batch.
    """
    unique_domains = torch.unique(domains)
    if unique_domains.numel() < 2:
        return features.new_tensor(0.0)

    covariances = []
    for domain in unique_domains:
        domain_features = features[domains == domain]
        if domain_features.size(0) < 2:
            continue
        centered = domain_features - domain_features.mean(dim=0, keepdim=True)
        cov = centered.t().matmul(centered) / (domain_features.size(0) - 1)
        covariances.append(cov)

    if len(covariances) < 2:
        return features.new_tensor(0.0)

    loss = features.new_tensor(0.0)
    pairs = 0
    for i in range(len(covariances)):
        for j in range(i + 1, len(covariances)):
            loss = loss + F.mse_loss(covariances[i], covariances[j])
            pairs += 1
    return loss / max(1, pairs)


def mmd_loss(features, domains, sigma=1.0):
    """
    RBF-kernel MMD across domains. Intended as a lightweight optional DG loss.
    """
    unique_domains = torch.unique(domains)
    if unique_domains.numel() < 2:
        return features.new_tensor(0.0)

    def kernel(x, y):
        dist = torch.cdist(x, y).pow(2)
        return torch.exp(-dist / (2 * sigma * sigma))

    groups = [features[domains == d] for d in unique_domains if (domains == d).sum() > 1]
    if len(groups) < 2:
        return features.new_tensor(0.0)

    loss = features.new_tensor(0.0)
    pairs = 0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            x, y = groups[i], groups[j]
            loss = loss + kernel(x, x).mean() + kernel(y, y).mean() - 2 * kernel(x, y).mean()
            pairs += 1
    return loss / max(1, pairs)


def group_dro_loss(per_sample_losses, groups, group_weights=None, eta=0.1):
    """
    GroupDRO over domain groups. Returns loss and updated group weights.
    """
    unique_groups = torch.unique(groups)
    if unique_groups.numel() == 0:
        return per_sample_losses.mean(), group_weights

    if group_weights is None or group_weights.numel() != unique_groups.numel():
        group_weights = torch.ones(unique_groups.numel(), device=per_sample_losses.device)
        group_weights = group_weights / group_weights.sum()

    group_losses = []
    for group in unique_groups:
        mask = groups == group
        group_losses.append(per_sample_losses[mask].mean())
    group_losses = torch.stack(group_losses)

    with torch.no_grad():
        group_weights = group_weights * torch.exp(eta * group_losses.detach())
        group_weights = group_weights / group_weights.sum().clamp_min(1e-12)

    return (group_weights.detach() * group_losses).sum(), group_weights.detach()


class DomainClassifier(nn.Module):
    def __init__(self, input_dim, num_domains, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, features, grl_lambda=1.0):
        return self.net(grad_reverse(features, grl_lambda))
