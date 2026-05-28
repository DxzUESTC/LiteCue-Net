"""Face detection result helpers — pick largest face, extract landmarks.

These helpers work with the common output format:
    [x, y, w, h, confidence, l0x, l0y, ..., l4x, l4y]
"""

from typing import List, Optional

import numpy as np


def pick_largest_face(
    results: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Pick the largest face (by bbox area) from detection results.

    Args:
        results: (N, 15) array from a face detector, or None.

    Returns:
        (15,) array for the largest face, or None.
    """
    if results is None or len(results) == 0:
        return None
    if len(results) == 1:
        return results[0]
    areas = results[:, 2] * results[:, 3]  # width * height
    return results[int(np.argmax(areas))]


def get_landmarks(face: np.ndarray) -> np.ndarray:
    """Extract 5-point landmarks from a detection result.

    Returns:
        (5, 2) float64 array of landmark coordinates.
    """
    return face[5:15].reshape(5, 2).astype(np.float64)
