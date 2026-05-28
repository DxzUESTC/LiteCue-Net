"""Profile each stage of the inference pipeline to identify bottlenecks.

Usage:
    conda activate litecue-api-gpu
    python tools/profile_api.py --video <path-to-video>

    # compare with CPU:
    conda activate litecue-api
    python tools/profile_api.py --video <path-to-video>
"""

import argparse
import time
import sys
from pathlib import Path

import numpy as np

# add project root
_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

from api.config import settings


def check_environment():
    """Check which compute backend is actually active."""
    print("=" * 60)
    print("Environment check")
    print("=" * 60)

    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Compute Capability: {torch.cuda.get_device_capability(0)}")

    import onnxruntime as ort
    print(f"ONNX Runtime: {ort.__version__}")
    print(f"  Available providers: {ort.get_available_providers()}")
    print(f"  CUDA provider: {'CUDAExecutionProvider' in ort.get_available_providers()}")
    print(f"  TensorRT provider: {'TensorrtExecutionProvider' in ort.get_available_providers()}")

    import cv2
    print(f"OpenCV: {cv2.__version__}")
    print()


def profile_pipeline(video_path: str):
    """Time each pipeline stage using a single pass through the video."""
    import cv2
    import torch
    from api.processor import _sample_indices, _extract_faces, _build_tensor
    from api.core.retinaface_detector import RetinaFaceDetector

    # --- Video info ---
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames @ {fps:.1f} fps ({total_frames / fps:.1f}s)")
    cap.release()

    M = settings.CLIP_NUM  # 16
    K = settings.CLIP_LEN  # 4
    required = M * K       # 64

    indices = _sample_indices(total_frames, M, K)
    detector = RetinaFaceDetector(settings.RETINA_MODEL_PATH, settings.DET_SIZE)

    # ---------- Stage 1+2: Seek-efficient face detection + alignment ----------
    cap = cv2.VideoCapture(video_path)
    t0 = time.perf_counter()
    faces, face_ok = _extract_faces(detector, cap, indices, required)
    t_detect = time.perf_counter() - t0
    cap.release()
    faces_detected = sum(face_ok[:len(faces)])
    print(f"Stage 1+2 — Seek + detection + alignment ({faces_detected}/{required} faces): "
          f"{t_detect*1000:.0f} ms  "
          f"(detection: {'GPU' if 'CUDAExecutionProvider' in __import__('onnxruntime').get_available_providers() else 'CPU'} ONNX)")

    # ---------- Stage 3: Normalization + tensor build ----------
    t0 = time.perf_counter()
    tensor = _build_tensor(faces, M, K)
    t_norm = time.perf_counter() - t0
    print(f"Stage 3 — Normalize → tensor:             {t_norm*1000:.0f} ms  (CPU)")

    # ---------- Stage 4: Model inference (PyTorch) ----------
    from api.backends.litecuenet import LiteCueNetBackend

    backend = LiteCueNetBackend(
        checkpoint_path=settings.CHECKPOINT_PATH,
        device=settings.DEVICE,
        clip_num=settings.CLIP_NUM,
        clip_len=settings.CLIP_LEN,
    )
    inp = torch.from_numpy(tensor).to(backend._device)

    # warmup
    with torch.no_grad():
        _ = backend._model(inp)
    if backend._device_str == "cuda":
        torch.cuda.synchronize()

    # timed (average 10 runs)
    N = 10
    t0 = time.perf_counter()
    for _ in range(N):
        with torch.no_grad():
            _ = backend._model(inp)
        if backend._device_str == "cuda":
            torch.cuda.synchronize()
    t_infer = (time.perf_counter() - t0) / N
    print(f"Stage 4 — Model inference (avg {N}x):     {t_infer*1000:.1f} ms  ({backend._device_str.upper()})")

    backend.cleanup()

    # ---------- Summary ----------
    print()
    print("=" * 60)
    print("Bottleneck summary")
    print("=" * 60)
    total = t_detect + t_norm + t_infer
    print(f"  Face pipeline (seek+detect+align): {t_detect*1000:>7.0f} ms  ({t_detect/total*100:>5.1f}%)")
    print(f"  Normalize → tensor:                {t_norm*1000:>7.0f} ms  ({t_norm/total*100:>5.1f}%)")
    print(f"  Model inference:                   {t_infer*1000:>7.0f} ms  ({t_infer/total*100:>5.1f}%)")
    print(f"  ──────────────────────────────────────────────────")
    print(f"  Total per video:                   {total*1000:>7.0f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to a test video")
    args = parser.parse_args()

    check_environment()
    profile_pipeline(args.video)
