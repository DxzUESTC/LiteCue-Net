import torch
import torch.nn as nn
import torch.nn.functional as F
from .focal_loss import FocalLoss
from .generalization import coral_loss, group_dro_loss, mmd_loss, supervised_contrastive_loss

class LiteCueLoss(nn.Module):
    """
    LiteCue-Net 混合损失函数 (Focal Loss 版本)
    
    结构:
    1. Video-level Loss: 针对整体视频判决的 Focal Loss
    2. Clip-level Loss: 针对每个片段判决的 Focal Loss
    
    解决痛点:
    - 数据不平衡 (Real vs Fake)
    - 难易样本不平衡 (Hard Mining)
    """
    def __init__(
        self,
        video_loss_weight=1.0,
        clip_loss_weight=1.0,
        alpha=0.25,
        gamma=2.0,
        label_smoothing=0.0,
        supcon_weight=0.0,
        supcon_temperature=0.2,
        coral_weight=0.0,
        mmd_weight=0.0,
        domain_adv_weight=0.0,
        groupdro_weight=0.0,
        groupdro_eta=0.1,
    ):
        """
        Args:
            video_loss_weight: 视频级损失权重
            clip_loss_weight: 片段级损失权重
            alpha: Focal Loss 的 alpha 参数 (通常 0.25 用于降低背景/负类权重)
                   FaceForensics++ 中 Fake(1) 很多，Real(0) 较少，
                   但为了强调少数类，也可以根据实际比例调整。
            gamma: Focal Loss 的 gamma 参数 (通常 2.0)
        """
        super().__init__()
        self.video_loss_weight = video_loss_weight
        self.clip_loss_weight = clip_loss_weight
        self.label_smoothing = label_smoothing
        self.supcon_weight = supcon_weight
        self.supcon_temperature = supcon_temperature
        self.coral_weight = coral_weight
        self.mmd_weight = mmd_weight
        self.domain_adv_weight = domain_adv_weight
        self.groupdro_weight = groupdro_weight
        self.groupdro_eta = groupdro_eta
        self.groupdro_state = None
        
        # 使用 Focal Loss 替代 CE Loss
        # alpha=0.25 是 RetinaNet 论文中的推荐值，
        # 意味着: weight_class0 (Real) = 0.75, weight_class1 (Fake) = 0.25
        # (注：如果你的数据集中 Fake 很多，降低 Fake 的权重有助于平衡)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, num_classes=2)

    def _classification_loss(self, logits, targets, reduction='mean'):
        focal = self.focal_loss(logits, targets)
        if self.label_smoothing <= 0:
            return focal
        ce = F.cross_entropy(
            logits,
            targets,
            label_smoothing=self.label_smoothing,
            reduction=reduction,
        )
        return 0.5 * focal + 0.5 * ce

    def forward(
        self,
        video_logits,
        clip_logits,
        targets,
        features=None,
        domain_labels=None,
        domain_logits=None,
    ):
        """
        Args:
            video_logits: (B, 2)
            clip_logits: (B, M, 2)
            targets: (B,)
        """
        # 1. Video-level Focal Loss
        loss_video = self._classification_loss(video_logits, targets)

        # 2. Clip-level Focal Loss
        # 将 Video 标签广播给所有 Clip
        B, M, C = clip_logits.shape
        
        # (B,) -> (B*M,)
        clip_targets = targets.view(B, 1).expand(B, M).reshape(-1)
        # (B, M, 2) -> (B*M, 2)
        clip_logits_flat = clip_logits.view(-1, C)
        
        loss_clip = self._classification_loss(clip_logits_flat, clip_targets)

        # 3. 加权总和
        total_loss = (self.video_loss_weight * loss_video) + \
                     (self.clip_loss_weight * loss_clip)

        loss_dict = {
            "loss_total": total_loss.item(),
            "loss_video": loss_video.item(),
            "loss_clip": loss_clip.item()
        }

        if features is not None and self.supcon_weight > 0:
            loss_supcon = supervised_contrastive_loss(
                features,
                targets,
                temperature=self.supcon_temperature,
            )
            total_loss = total_loss + self.supcon_weight * loss_supcon
            loss_dict["loss_supcon"] = loss_supcon.item()

        if features is not None and domain_labels is not None and self.coral_weight > 0:
            loss_coral = coral_loss(features, domain_labels)
            total_loss = total_loss + self.coral_weight * loss_coral
            loss_dict["loss_coral"] = loss_coral.item()

        if features is not None and domain_labels is not None and self.mmd_weight > 0:
            loss_mmd = mmd_loss(features, domain_labels)
            total_loss = total_loss + self.mmd_weight * loss_mmd
            loss_dict["loss_mmd"] = loss_mmd.item()

        if domain_logits is not None and domain_labels is not None and self.domain_adv_weight > 0:
            loss_domain = F.cross_entropy(domain_logits, domain_labels)
            total_loss = total_loss + self.domain_adv_weight * loss_domain
            loss_dict["loss_domain_adv"] = loss_domain.item()

        if domain_labels is not None and self.groupdro_weight > 0:
            per_sample = F.cross_entropy(video_logits, targets, reduction='none')
            loss_groupdro, self.groupdro_state = group_dro_loss(
                per_sample,
                domain_labels,
                self.groupdro_state,
                eta=self.groupdro_eta,
            )
            total_loss = total_loss + self.groupdro_weight * loss_groupdro
            loss_dict["loss_groupdro"] = loss_groupdro.item()

        loss_dict["loss_total"] = total_loss.item()
        return total_loss, loss_dict