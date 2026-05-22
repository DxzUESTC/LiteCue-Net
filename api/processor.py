"""Face detection, alignment, and video preprocessing pipeline."""

import logging
from typing import Dict, List, Tuple

import cv2
import insightface
import numpy as np
from insightface.utils import face_align

from api.config import settings

logger = logging.getLogger(__name__)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FaceProcessor:
    """Detect, align, and crop faces from video frames, then build the (1, M, K, 3, H, W) input tensor."""

    def __init__(self):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            root=settings.INSIGHTFACE_ROOT,
            providers=providers,
        )
        ctx_id = 0 if settings.DEVICE == "cuda" else -1
        self.app.prepare(ctx_id=ctx_id, det_size=settings.DET_SIZE)

    def process_video(self, video_path: str) -> Tuple[np.ndarray, Dict]:
        """Load a video, sample frames, detect+align faces, build the input tensor.

        Returns:
            tensor: (1, M, K, 3, H, W) float32 array, normalized.
            metadata: dict with video info and face-detection stats.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0.0

        M = settings.CLIP_NUM
        K = settings.CLIP_LEN
        required = M * K  # 64

        indices = _sample_indices(total_frames, M, K)
        faces, face_ok = _extract_faces(self.app, cap, indices, required)

        cap.release()

        tensor = _build_tensor(faces, M, K)

        metadata = {
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "duration_sec": round(duration, 2),
            "faces_detected": int(sum(face_ok)),
            "total_sampled": len(faces),
        }
        return tensor, metadata

    def extract_face_mid(self, video_path: str) -> np.ndarray:
        """Extract a single aligned face from roughly the middle of the video (for Grad-CAM overlay)."""
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid = total // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = self.app.get(rgb)
        if not dets:
            return None
        areas = [(d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]) for d in dets]
        best = dets[np.argmax(areas)]
        return face_align.norm_crop(rgb, landmark=best.kps, image_size=settings.FACE_SIZE)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sample_indices(total_frames: int, M: int, K: int) -> np.ndarray:
    """Global-sparse + local-dense sampling (same strategy as training).

    Divides the video into M segments and takes K consecutive frames from each.
    """
    required = M * K
    if total_frames <= required:
        return np.linspace(0, max(total_frames - 1, 0), required, dtype=int)

    seg_size = total_frames / M
    indices = []
    half_k = K // 2
    for i in range(M):
        center = int((i + 0.5) * seg_size)
        for j in range(K):
            idx = center - half_k + j
            idx = max(0, min(total_frames - 1, idx))
            indices.append(idx)
    return np.array(indices, dtype=int)


def _extract_faces(
    app: insightface.app.FaceAnalysis,
    cap: cv2.VideoCapture,
    indices: np.ndarray,
    required: int,
) -> Tuple[List[np.ndarray], List[bool]]:
    """Run face detection + alignment on sampled frame indices.

    Returns (aligned_faces, face_ok_flags).
    When no face is found, repeats the last successful face (or centre-crop)."""
    faces: List[np.ndarray] = []
    face_ok: List[bool] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            _pad_last(faces, face_ok)
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = app.get(rgb)

        if dets:
            # Pick the largest face
            areas = [(d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]) for d in dets]
            best = dets[np.argmax(areas)]
            aligned = face_align.norm_crop(rgb, landmark=best.kps, image_size=settings.FACE_SIZE)
            faces.append(aligned)
            face_ok.append(True)
        else:
            _pad_or_centrecrop(faces, face_ok, rgb)

    # Ensure exactly `required` frames
    while len(faces) < required:
        if faces:
            faces.append(faces[-1].copy())
            face_ok.append(face_ok[-1])
        else:
            # No faces at all — create a blank fallback
            blank = np.zeros((settings.FACE_SIZE, settings.FACE_SIZE, 3), dtype=np.uint8)
            faces.append(blank)
            face_ok.append(False)

    return faces[:required], face_ok[:required]


def _pad_last(faces: list, face_ok: list) -> None:
    if faces:
        faces.append(faces[-1].copy())
        face_ok.append(face_ok[-1])


def _pad_or_centrecrop(faces: list, face_ok: list, rgb: np.ndarray) -> None:
    if faces:
        faces.append(faces[-1].copy())
        face_ok.append(face_ok[-1])
        return
    h, w = rgb.shape[:2]
    size = min(h, w)
    y = (h - size) // 2
    x = (w - size) // 2
    crop = cv2.resize(rgb[y : y + size, x : x + size], (settings.FACE_SIZE, settings.FACE_SIZE))
    faces.append(crop)
    face_ok.append(False)


def _build_tensor(faces: List[np.ndarray], M: int, K: int) -> np.ndarray:
    """Normalise and reshape face list into model input.

    faces: list of (224, 224, 3) uint8 RGB arrays.
    returns (1, M, K, 3, 224, 224) float32.
    """
    arr = np.stack(faces, axis=0).astype(np.float32) / 255.0  # (N, 224, 224, 3)
    arr = (arr - _MEAN) / _STD
    arr = arr.transpose(0, 3, 1, 2)  # (N, 3, 224, 224)
    arr = arr.reshape(1, M, K, 3, settings.FACE_SIZE, settings.FACE_SIZE)
    return arr
