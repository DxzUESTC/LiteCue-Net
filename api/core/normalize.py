"""Shared image normalization constants and utilities for inference."""

import numpy as np

# ImageNet normalization (matches training transforms)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_frames(frames: np.ndarray) -> np.ndarray:
    """Normalize uint8 RGB frames to float32 with ImageNet stats.

    Args:
        frames: (N, H, W, 3) uint8 RGB array in [0, 255].

    Returns:
        (N, 3, H, W) float32 in CHW layout, standardized per channel.
    """
    arr = frames.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(0, 3, 1, 2)


def denormalize_frame(chw: np.ndarray) -> np.ndarray:
    """Convert a single normalized CHW frame back to HWC RGB uint8.

    Args:
        chw: (3, H, W) float32, standardized.

    Returns:
        (H, W, 3) uint8 RGB in [0, 255].
    """
    hwc = chw.transpose(1, 2, 0)
    hwc = hwc * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(hwc * 255.0, 0, 255).astype(np.uint8)
