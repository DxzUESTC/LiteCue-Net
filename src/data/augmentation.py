# -*- coding: utf-8 -*-
"""
LiteCue-Net 训练期遮挡增强：对整段 clip 施加同一块空间遮挡，
强制模型从边缘与未遮挡区域学习，提升泛化。
"""
import random
import torch
import torch.nn.functional as F


# ImageNet 归一化后的均值，用于遮挡填充（与 transforms 一致）
IMAGENET_MEAN = [0.485, 0.456, 0.406]


def apply_clip_occlusion(video_tensor, cfg):
    """
    对整段 clip (M, K, C, H, W) 施加同一块随机矩形遮挡。
    同一空间位置在所有帧上被遮住，迫使模型依赖边缘等未遮挡区域。

    Args:
        video_tensor: torch.Tensor, shape (M, K, C, H, W)，已归一化
        cfg: dict，例如:
            - enabled (bool): 是否启用
            - prob (float): 应用遮挡的概率，默认 0.5
            - scale_range (tuple): 遮挡面积相对图像比例范围，如 (0.1, 0.35)
            - fill (str or float): 'mean' 或数值，填充值
            - num_patches (int): 每段 clip 遮挡块数，默认 1

    Returns:
        video_tensor: 原地修改或复制后修改的 (M, K, C, H, W)
    """
    if not cfg.get('enabled', True):
        return video_tensor
    if random.random() > cfg.get('prob', 0.5):
        return video_tensor

    M, K, C, H, W = video_tensor.shape
    scale_range_raw = cfg.get('scale_range', (0.1, 0.35))
    if not scale_range_raw or len(scale_range_raw) < 2:
        scale_range = (0.1, 0.35)
    else:
        low, high = min(scale_range_raw[0], scale_range_raw[1]), max(scale_range_raw[0], scale_range_raw[1])
        scale_range = (low, high)
    num_patches = max(0, min(int(cfg.get('num_patches', 1)), int(cfg.get('max_patches', 8))))
    fill = cfg.get('fill', 'mean')
    edge_bias = float(cfg.get('edge_bias', 0.0))

    if fill == 'mean':
        # (C,) 与 (1,1,C,1,1) 便于广播
        fill_val = torch.tensor(IMAGENET_MEAN, dtype=video_tensor.dtype, device=video_tensor.device)
        fill_val = fill_val.view(1, 1, C, 1, 1)
    else:
        fill_val = float(fill)

    out = video_tensor.clone()
    for _ in range(num_patches):
        # 随机遮挡面积比例
        scale = random.uniform(scale_range[0], scale_range[1])
        h_len = max(1, int(H * (scale ** 0.5)))
        w_len = max(1, int(W * (scale ** 0.5)))
        # 随机左上角，保证不越界
        if edge_bias > 0 and random.random() < edge_bias:
            side = random.choice(["top", "bottom", "left", "right"])
            if side == "top":
                top = 0
                left = random.randint(0, max(0, W - w_len))
            elif side == "bottom":
                top = max(0, H - h_len)
                left = random.randint(0, max(0, W - w_len))
            elif side == "left":
                top = random.randint(0, max(0, H - h_len))
                left = 0
            else:
                top = random.randint(0, max(0, H - h_len))
                left = max(0, W - w_len)
        else:
            top = random.randint(0, max(0, H - h_len))
            left = random.randint(0, max(0, W - w_len))
        # 对所有 M*K 帧同一位置遮挡
        if isinstance(fill_val, torch.Tensor):
            out[:, :, :, top:top + h_len, left:left + w_len] = fill_val
        else:
            out[:, :, :, top:top + h_len, left:left + w_len] = fill_val

    return out


def apply_temporal_perturbation(video_tensor, cfg):
    """
    对 (M, K, C, H, W) 做轻量时序扰动：clip token 丢帧/重复帧、局部运动模糊。
    这些扰动模拟平台抽帧、帧率变化和短时间模糊，默认保持输出形状不变。
    """
    if not cfg or not cfg.get('enabled', False):
        return video_tensor

    out = video_tensor.clone()
    M, K, C, H, W = out.shape

    drop_prob = float(cfg.get('drop_prob', 0.0))
    repeat_prob = float(cfg.get('repeat_prob', 0.0))
    blur_prob = float(cfg.get('motion_blur_prob', 0.0))

    for m in range(M):
        for k in range(K):
            if drop_prob > 0 and random.random() < drop_prob:
                src = max(0, k - 1)
                out[m, k] = out[m, src]
            elif repeat_prob > 0 and random.random() < repeat_prob:
                src = random.randint(0, K - 1)
                out[m, k] = out[m, src]

    if blur_prob > 0 and random.random() < blur_prob:
        # 用一维平均核近似水平运动模糊，逐帧逐通道处理。
        kernel_size = int(cfg.get('motion_blur_kernel', 5))
        kernel_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
        kernel = torch.ones(C, 1, 1, kernel_size, dtype=out.dtype, device=out.device) / kernel_size
        flat = out.view(M * K, C, H, W)
        flat = F.pad(flat, (kernel_size // 2, kernel_size // 2, 0, 0), mode='reflect')
        blurred = F.conv2d(flat, kernel, groups=C)
        out = blurred.view(M, K, C, H, W)

    return out


def apply_domain_randomization(video_tensor, cfg):
    """训练期 tensor 级域随机化入口，组合遮挡和时序扰动。"""
    if not cfg or not cfg.get('enabled', False):
        return video_tensor

    out = video_tensor
    occlusion_cfg = cfg.get('occlusion')
    if occlusion_cfg and occlusion_cfg.get('enabled', False):
        out = apply_clip_occlusion(out, occlusion_cfg)

    temporal_cfg = cfg.get('temporal')
    if temporal_cfg and temporal_cfg.get('enabled', False):
        out = apply_temporal_perturbation(out, temporal_cfg)

    return out
