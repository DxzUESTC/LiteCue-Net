"""Face alignment (norm_crop) using OpenCV only — no insightface dependency.

Replicates insightface.utils.face_align.norm_crop behaviour.
"""

import cv2
import numpy as np

# Reference 5-point landmarks from ArcFace (112×112 base).
# Shifted up by 10 px to leave more room for the chin in the output crop.
# Original ArcFace y-coords: [51.7, 51.5, 71.7, 92.4, 92.2] give only ~20 px
# below the mouth at 112 scale — insufficient for many faces.
_CHIN_OFFSET = 10.0  # px at 112 base (→ +20 px at 224 output)

_ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963 - _CHIN_OFFSET],   # left eye
        [73.5318, 51.5014 - _CHIN_OFFSET],   # right eye
        [56.0252, 71.7366 - _CHIN_OFFSET],   # nose
        [41.5493, 92.3655 - _CHIN_OFFSET],   # left mouth corner
        [70.7299, 92.2041 - _CHIN_OFFSET],   # right mouth corner
    ],
    dtype=np.float64,
)


def estimate_norm(lmk: np.ndarray, image_size: int = 224) -> np.ndarray:
    """Estimate similarity transform from detected landmarks to reference.

    Args:
        lmk: (5, 2) detected face landmarks.
        image_size: Target crop size (must be multiple of 112).

    Returns:
        2×3 affine transform matrix.
    """
    assert lmk.shape == (5, 2), f"Expected (5,2) landmarks, got {lmk.shape}"
    assert image_size % 112 == 0 or image_size % 128 == 0

    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0.0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio

    dst = _ARCFACE_DST * ratio
    dst[:, 0] += diff_x

    # cv2.estimateAffinePartial2D computes a similarity transform
    # (rotation + scaling + translation, no shear)
    tform, _ = cv2.estimateAffinePartial2D(lmk, dst, method=cv2.LMEDS)
    return tform


def norm_crop(img: np.ndarray, landmark: np.ndarray, image_size: int = 224) -> np.ndarray:
    """Align and crop face using 5-point landmarks (same as insightface).

    Args:
        img: (H, W, 3) RGB uint8 image.
        landmark: (5, 2) detected face landmarks.
        image_size: Output crop size in pixels.

    Returns:
        (image_size, image_size, 3) aligned face crop.
    """
    M = estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped
