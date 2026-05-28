"""Face detection, alignment, and video preprocessing pipeline."""

import logging
from typing import Dict, List, Tuple

import cv2
import numpy as np

from api.config import settings
from api.core.face_align import norm_crop
from api.core.face_detector import get_landmarks, pick_largest_face
from api.core.retinaface_detector import RetinaFaceDetector
from api.core.normalize import normalize_frames

logger = logging.getLogger(__name__)


class FaceProcessor:
    """Detect, align, and crop faces from video frames, then build the (1, M, K, 3, H, W) input tensor."""

    def __init__(self):
        self._detector = RetinaFaceDetector(
            model_path=settings.RETINA_MODEL_PATH,
            input_size=settings.DET_SIZE,
        )

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
        faces, face_ok = _extract_faces(self._detector, cap, indices, required)

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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sample_indices(total_frames: int, M: int, K: int) -> np.ndarray:
    """Global-sparse + local-dense sampling, aligned with training validation mode.

    Divides the video into M segments and takes K consecutive frames from
    the centre of each segment.  Falls back to np.linspace for short videos.
    """
    required = M * K
    if total_frames <= required:
        return np.linspace(0, max(total_frames - 1, 0), required, dtype=int)

    interval = total_frames // M
    indices = []

    for i in range(M):
        seg_start = i * interval
        seg_end = (i + 1) * interval if i < M - 1 else total_frames
        seg_len = seg_end - seg_start

        if seg_len >= K:
            offset = (seg_len - K) // 2  # centre of segment
        else:
            offset = 0
        for j in range(K):
            idx = seg_start + offset + j
            idx = max(0, min(total_frames - 1, idx))
            indices.append(idx)

    return np.array(indices, dtype=int)


def _extract_faces(
    detector: RetinaFaceDetector,
    cap: cv2.VideoCapture,
    indices: np.ndarray,
    required: int,
) -> Tuple[List[np.ndarray], List[bool]]:
    """Run face detection + alignment on sampled frame indices.

    Returns (aligned_faces, face_ok_flags).
    When no face is found, repeats the last successful face (or centre-crop).
    """
    faces: List[np.ndarray] = []
    face_ok: List[bool] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            _pad_last(faces, face_ok)
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect(rgb)
        face = pick_largest_face(results)

        if face is not None:
            landmarks = get_landmarks(face)
            aligned = norm_crop(rgb, landmarks, image_size=settings.FACE_SIZE)
            faces.append(aligned)
            face_ok.append(True)
        else:
            _pad_or_centrecrop(faces, face_ok, rgb)

    while len(faces) < required:
        if faces:
            faces.append(faces[-1].copy())
            face_ok.append(face_ok[-1])
        else:
            blank = np.zeros((settings.FACE_SIZE, settings.FACE_SIZE, 3), dtype=np.uint8)
            faces.append(blank)
            face_ok.append(False)

    return faces[:required], face_ok[:required]


def _build_tensor(faces: List[np.ndarray], M: int, K: int) -> np.ndarray:
    """Normalise and reshape face list into model input.

    faces: list of (224, 224, 3) uint8 RGB arrays.
    returns (1, M, K, 3, 224, 224) float32.
    """
    arr = np.stack(faces, axis=0)
    arr = normalize_frames(arr)
    arr = arr.reshape(1, M, K, 3, settings.FACE_SIZE, settings.FACE_SIZE)
    return arr


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
