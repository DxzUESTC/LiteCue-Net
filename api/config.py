"""API configuration — service & preprocessing params only.

Model architecture parameters are auto-discovered from the checkpoint
by the backend at load time (see api/backends/litecuenet.py).
"""

import os
import sys
from pathlib import Path

# ONNX Runtime GPU (onnxruntime-gpu) needs cuDNN 9 DLLs at load time.
# PyTorch bundles cuDNN 9 in its own lib directory, so we expose it via PATH
# so that RetinaFace (ONNX) can also use GPU acceleration.
_torch_cudnn = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
if os.path.isdir(_torch_cudnn) and os.path.isfile(os.path.join(_torch_cudnn, "cudnn64_9.dll")):
    os.environ["PATH"] = _torch_cudnn + os.pathsep + os.environ.get("PATH", "")
    _found = True
else:
    _found = False
del _torch_cudnn

import logging
logging.getLogger("api.config").info(
    "PyTorch cuDNN 9 %s, added to PATH: %s",
    "found" if _found else "not found",
    _found,
)
del _found


class Settings:
    # --- Paths ---
    BASE_DIR = Path(__file__).resolve().parent.parent
    CHECKPOINT_PATH = os.getenv(
        "CHECKPOINT_PATH",
        str(BASE_DIR / "checkpoints" / "exp_20260511" / "best_model.pth"),
    )
    RETINA_MODEL_PATH = str(
        BASE_DIR / "models" / "buffalo_l" / "det_10g.onnx"
    )

    # --- Input protocol (must match what the model expects) ---
    CLIP_NUM = 16   # M
    CLIP_LEN = 4    # K

    # --- Face detection (RetinaFace via ONNX Runtime) ---
    DET_SIZE = (640, 640)
    FACE_SIZE = 224

    # --- API server ---
    HOST = os.getenv("API_HOST", "0.0.0.0")
    PORT = int(os.getenv("API_PORT", "8001"))
    MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB

    # --- Device ---
    DEVICE = os.getenv("DEVICE", "cuda")

    # --- Grad-CAM: top keyframes to return ---
    TOP_K_FRAMES = 6


settings = Settings()


def verify_api_assets() -> None:
    """Ensure bundled checkpoint and face detection model are present."""
    ckpt = Path(settings.CHECKPOINT_PATH)
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"LiteCue-Net checkpoint not found: {ckpt}\n"
            "Ensure the repo is fully cloned (checkpoints/exp_20260511/best_model.pth)."
        )

    det = Path(settings.RETINA_MODEL_PATH)
    if not det.is_file():
        raise FileNotFoundError(
            f"Face detection model not found: {det}\n"
            "Restore from git:\n"
            "  git checkout -- models/buffalo_l/det_10g.onnx"
        )
