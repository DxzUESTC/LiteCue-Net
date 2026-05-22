"""API configuration, matching the checkpoint at checkpoints/exp_20260511/best_model.pth."""

import os
from pathlib import Path


class Settings:
    # Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    CHECKPOINT_PATH = os.getenv(
        "CHECKPOINT_PATH",
        str(BASE_DIR / "checkpoints" / "exp_20260511" / "best_model.pth"),
    )
    INSIGHTFACE_ROOT = os.getenv(
        "INSIGHTFACE_ROOT",
        os.path.expanduser("~/.insightface/models"),
    )

    # --- LiteCueNet architecture (must match checkpoint) ---
    FEATURE_DIM = 256
    CLIP_NUM = 16  # M
    CLIP_LEN = 4  # K
    NUM_CLASSES = 2
    BACKBONE_NAME = "mobilenetv4_conv_small.e2400_r224_in1k"
    USE_FREQUENCY_BRANCH = True
    FREQUENCY_FUSE_BLOCK = 2
    TEMPORAL_MODULE = "attention"

    # Face detection
    DET_SIZE = (640, 640)
    FACE_SIZE = 224

    # API server
    HOST = os.getenv("API_HOST", "0.0.0.0")
    PORT = int(os.getenv("API_PORT", "8000"))
    MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB

    # Device
    DEVICE = os.getenv("DEVICE", "cuda")

    # Grad-CAM: number of top keyframes to return when video is fake
    TOP_K_FRAMES = 6


settings = Settings()
