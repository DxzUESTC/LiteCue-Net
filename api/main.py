"""LiteCue-Net FastAPI inference server.

Endpoints
---------
GET  /api/v1/health  –  health check
POST /api/v1/detect  –  upload a face video, get deepfake score (+ Grad-CAM if fake)
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

# Ensure project root is on sys.path so that `from src.models.detector import …` works.
_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("api")

# ---------------------------------------------------------------------------
# Global resources (initialised on startup)
# ---------------------------------------------------------------------------
_face_processor = None
_inference_engine = None


def _lazy_face():
    if _face_processor is None:
        raise RuntimeError("Face processor not initialised.")
    return _face_processor


def _lazy_engine():
    if _inference_engine is None:
        raise RuntimeError("Inference engine not initialised.")
    return _inference_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _face_processor, _inference_engine
    logger.info("Starting up — loading face processor & inference engine ...")
    t0 = time.time()
    from api.processor import FaceProcessor  # late import
    from api.engine import InferenceEngine  # late import

    _face_processor = FaceProcessor()
    logger.info("Face processor ready (%.1fs)", time.time() - t0)

    _inference_engine = InferenceEngine()
    logger.info("Inference engine ready (%.1fs)", time.time() - t0)
    yield
    # Shutdown: clean up Grad-CAM hooks
    if _inference_engine is not None:
        _inference_engine.gradcam.remove_hooks()
    logger.info("API shutdown complete.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LiteCue-Net Deepfake Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class HeatmapFrame(BaseModel):
    frame_index: int
    clip_index: int
    clip_fake_probability: float
    heatmap_base64: str


class VideoInfo(BaseModel):
    total_frames: int
    fps: float
    duration_sec: float
    faces_detected: int
    total_sampled: int


class DetectResponse(BaseModel):
    is_fake: bool
    fake_probability: float
    real_probability: float
    processing_time_ms: float
    video_info: VideoInfo
    heatmap_frames: Optional[List[HeatmapFrame]] = None


class HealthResponse(BaseModel):
    status: str
    device: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/v1/health", response_model=HealthResponse)
async def health():
    eng = _lazy_engine()
    return HealthResponse(status="ok", device=str(eng.device))


@app.post("/api/v1/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...)):
    """Upload a face video. Returns deepfake score and, for fake videos, Grad-CAM heatmaps."""
    # --- Validate ---
    if file.content_type and not file.content_type.startswith("video/"):
        raise HTTPException(400, detail="Only video files are supported.")

    # --- Save upload ---
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    import tempfile
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    try:
        content = await file.read()
        if len(content) > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(413, detail="File too large (max 100 MB).")
        with open(tmp_path, "wb") as f:
            f.write(content)

        # --- Process & infer ---
        t0 = time.time()
        processor = _lazy_face()
        tensor, meta = processor.process_video(tmp_path)

        engine = _lazy_engine()
        result = engine.predict_with_heatmap(tensor)

        elapsed_ms = round((time.time() - t0) * 1000, 2)

        return DetectResponse(
            is_fake=result["is_fake"],
            fake_probability=result["fake_probability"],
            real_probability=result["real_probability"],
            processing_time_ms=elapsed_ms,
            video_info=VideoInfo(**meta),
            heatmap_frames=result.get("heatmap_frames"),
        )

    except RuntimeError as exc:
        logger.error("Processing error: %s", exc)
        raise HTTPException(422, detail=str(exc)) from exc
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        log_level="info",
    )
