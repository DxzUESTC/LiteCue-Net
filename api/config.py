"""API configuration, matching the checkpoint at checkpoints/exp_20260511/best_model.pth."""

import os
from pathlib import Path

# InsightFace buffalo_l pack (bundled under models/buffalo_l/, tracked in repo / Git LFS)
BUFFALO_L_ONNX_FILES = (
    "det_10g.onnx",
    "w600k_r50.onnx",
    "1k3d68.onnx",
    "2d106det.onnx",
    "genderage.onnx",
)


class Settings:
    # Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    CHECKPOINT_PATH = os.getenv(
        "CHECKPOINT_PATH",
        str(BASE_DIR / "checkpoints" / "exp_20260511" / "best_model.pth"),
    )
    # Note: insightface.app.FaceAnalysis(name="buffalo_l", root=X) internally
    # resolves to X/models/buffalo_l, so INSIGHTFACE_ROOT must point to the
    # parent of models/ (the project root), not models/ itself.
    INSIGHTFACE_ROOT = os.getenv(
        "INSIGHTFACE_ROOT",
        str(BASE_DIR),
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


def insightface_model_dir() -> Path:
    return Path(settings.INSIGHTFACE_ROOT) / "models" / "buffalo_l"


def verify_api_assets() -> None:
    """Ensure bundled checkpoint and InsightFace ONNX files are present."""
    ckpt = Path(settings.CHECKPOINT_PATH)
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"LiteCue-Net checkpoint not found: {ckpt}\n"
            "Ensure the repo is fully cloned (checkpoints/exp_20260511/best_model.pth)."
        )

    model_dir = insightface_model_dir()
    missing = [f for f in BUFFALO_L_ONNX_FILES if not (model_dir / f).is_file()]
    if missing:
        lfs_hint = (
            " If ONNX files are only a few KB, run: git lfs install && git lfs pull"
        )
        raise FileNotFoundError(
            f"InsightFace buffalo_l models missing under {model_dir}: {missing}\n"
            f"Clone the repo with Git LFS, or run: python scripts/download_models.py"
            f"{lfs_hint}"
        )

    # Detect Git LFS pointer stubs (clone without `git lfs pull`)
    for name in BUFFALO_L_ONNX_FILES:
        path = model_dir / name
        if path.stat().st_size < 1024:
            raise FileNotFoundError(
                f"{path} looks like a Git LFS pointer ({path.stat().st_size} bytes).\n"
                "Run: git lfs install && git lfs pull"
            )
