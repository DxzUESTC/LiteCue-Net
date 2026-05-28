"""LiteCueNet model backend: build, load checkpoint, infer, and explain."""

import base64
import logging
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from api.backends.base import ModelBackend
from api.backends.gradcam import GradCAM
from api.core.normalize import denormalize_frame

logger = logging.getLogger(__name__)


def _infer_model_config(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Infer model architecture parameters from a checkpoint state_dict.

    This avoids hardcoding architecture params in config.py.
    """
    config: Dict[str, Any] = {}

    # Detect feature_dim from backbone projector weight
    for key in ("backbone.projector.0.weight", "backbone.projector.0.bias"):
        if key in state_dict:
            config["feature_dim"] = state_dict[key].shape[0]
            break

    # Detect clip_num from inter_clip position embedding or reviewer mask
    for key in ("inter_clip.pos_embed", "reviewer._mask"):
        if key in state_dict:
            config["clip_num"] = state_dict[key].shape[0] if "_mask" in key else state_dict[key].shape[1]
            break

    # clip_len (K) cannot be inferred from state_dict alone; it is an
    # input-protocol parameter provided by the caller / config.

    # Detect num_classes from head
    if "head.weight" in state_dict:
        config["num_classes"] = state_dict["head.weight"].shape[0]

    # Detect temporal_module type
    if "inter_clip.pos_embed" in state_dict:
        config["temporal_module"] = "attention"
    elif "inter_clip.proj_time.weight" in state_dict:
        config["temporal_module"] = "gated_mlp"

    # Detect frequency branch
    config["use_frequency_branch"] = "frequency_branch.0.weight" in state_dict

    if "frequency_branch.0.weight" in state_dict:
        config["frequency_fuse_block"] = 2  # default, no reliable way to infer

    # Detect temporal diff
    config["use_temporal_diff"] = "temporal_diff_proj.1.weight" in state_dict

    return config


def _normalize_checkpoint(ckpt: Any) -> Dict[str, torch.Tensor]:
    """Normalise a checkpoint into a flat state_dict."""
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in ckpt:
                ckpt = ckpt[key]
                break
    if isinstance(ckpt, dict) and any(k.startswith("module.") for k in ckpt):
        ckpt = {k[7:]: v for k, v in ckpt.items()}
    return ckpt


class LiteCueNetBackend(ModelBackend):
    """Model backend for LiteCueNet."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        clip_num: int = 16,
        clip_len: int = 4,
    ):
        self._checkpoint_path = checkpoint_path
        self._device_str = device if torch.cuda.is_available() else "cpu"
        self._device = torch.device(self._device_str)
        self._provided_clip_num = clip_num
        self._provided_clip_len = clip_len

        self._model: Optional[torch.nn.Module] = None
        self._gradcam: Optional[GradCAM] = None
        self._model_config: Dict[str, Any] = {}

        self.load()

    # ------------------------------------------------------------------
    # ModelBackend interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        logger.info("Loading checkpoint: %s", self._checkpoint_path)
        ckpt = torch.load(self._checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = _normalize_checkpoint(ckpt)

        self._model_config = _infer_model_config(state_dict)
        # Merge input-protocol params (shared with processor)
        self._model_config["clip_num"] = self._provided_clip_num
        self._model_config["clip_len"] = self._provided_clip_len
        logger.info("Model config (auto-discovered + provided): %s", self._model_config)

        self._model = self._build_model(self._model_config)
        missing, unexpected = self._model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing keys in checkpoint: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys in checkpoint: %s", unexpected)

        self._model.to(self._device)
        self._model.eval()

        # Attach Grad-CAM to the last backbone block
        target_block = self._model.backbone.backbone.blocks[-1]
        self._gradcam = GradCAM(target_block)
        logger.info("LiteCueNetBackend ready on %s", self._device)

    def predict(self, tensor: np.ndarray) -> Dict:
        """Quick prediction without Grad-CAM."""
        inp = torch.from_numpy(tensor).to(self._device)
        with torch.no_grad():
            outputs = self._model(inp)
        if isinstance(outputs, dict):
            video_logits = outputs["video_logits"]
        else:
            video_logits = outputs[0]

        probs = F.softmax(video_logits, dim=1)
        fake_prob = float(probs[0, 1].item())
        real_prob = float(probs[0, 0].item())
        return {
            "is_fake": fake_prob > real_prob,
            "fake_probability": fake_prob,
            "real_probability": real_prob,
            "heatmap_frames": [],
        }

    def predict_with_explain(self, tensor: np.ndarray) -> Dict:
        """Run inference and generate Grad-CAM heatmaps."""
        assert self._gradcam is not None, "Grad-CAM not initialised"
        M = self._model_config.get("clip_num", 16)
        K = self._model_config.get("clip_len", 4)
        top_k = 6

        try:
            inp = torch.from_numpy(tensor).to(self._device)
            inp.requires_grad = True
            cam_maps, clip_logits, video_logits = self._gradcam.generate(
                self._model, inp, target_class=1
            )
        except RuntimeError as exc:
            logger.warning("Grad-CAM failed, falling back: %s", exc)
            return self.predict(tensor)

        video_probs = F.softmax(video_logits, dim=1)
        fake_prob = float(video_probs[0, 1].item())
        real_prob = float(video_probs[0, 0].item())
        is_fake = fake_prob > real_prob

        clip_fake = F.softmax(clip_logits, dim=2)[0, :, 1].detach().cpu().numpy()

        cam_maps = cam_maps.reshape(M, K, *cam_maps.shape[1:])

        top_indices = np.argsort(clip_fake)[::-1][:top_k]

        frames: List[Dict] = []
        for ci in top_indices:
            clip_cams = cam_maps[ci]
            mean_scores = clip_cams.reshape(K, -1).mean(axis=1)
            best_k = int(np.argmax(mean_scores))
            frame_cam = clip_cams[best_k]
            frame_idx = int(ci * K + best_k)
            face_rgb = _face_rgb_from_tensor(tensor, frame_idx, M, K)

            b64 = _overlay_cam_to_base64(face_rgb, frame_cam)
            frames.append({
                "frame_index": frame_idx,
                "clip_index": int(ci),
                "clip_fake_probability": float(clip_fake[ci]),
                "heatmap_base64": b64,
            })

        return {
            "is_fake": is_fake,
            "fake_probability": fake_prob,
            "real_probability": real_prob,
            "heatmap_frames": frames,
        }

    @property
    def device(self) -> str:
        return self._device_str

    @property
    def model_config(self) -> Dict:
        return dict(self._model_config)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self, cfg: Dict[str, Any]) -> torch.nn.Module:
        from api.model.detector import LiteCueNet

        return LiteCueNet(
            feature_dim=cfg.get("feature_dim", 256),
            clip_num=cfg.get("clip_num", 16),
            clip_len=cfg.get("clip_len", 4),
            num_classes=cfg.get("num_classes", 2),
            backbone_name="mobilenetv4_conv_small.e2400_r224_in1k",
            pretrained=False,
            use_frequency_branch=cfg.get("use_frequency_branch", True),
            frequency_fuse_block=cfg.get("frequency_fuse_block", 2),
            temporal_module=cfg.get("temporal_module", "attention"),
        )

    def cleanup(self):
        """Remove Grad-CAM hooks."""
        if self._gradcam is not None:
            self._gradcam.remove_hooks()


# ---------------------------------------------------------------------------
# Utilities (moved from the old api/engine.py)
# ---------------------------------------------------------------------------


def _face_rgb_from_tensor(
    video_tensor: np.ndarray, frame_idx: int, M: int, K: int
) -> np.ndarray:
    """Denormalise one aligned face crop from model input (1, M, K, 3, H, W)."""
    ci, ki = divmod(frame_idx, K)
    chw = video_tensor[0, ci, ki]
    return denormalize_frame(chw)


def _overlay_cam_to_base64(face_rgb: np.ndarray, cam: np.ndarray) -> str:
    """Overlay CAM on face crop and return base64 JPEG."""
    face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
    dsize = (face_rgb.shape[1], face_rgb.shape[0])
    cam_resized = cv2.resize(cam, dsize, interpolation=cv2.INTER_LINEAR)
    cam_uint8 = np.clip(cam_resized * 255, 0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    blended_bgr = cv2.addWeighted(face_bgr, 0.75, heatmap_bgr, 0.25, 0)
    blended_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)

    buf = BytesIO()
    Image.fromarray(blended_rgb).save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
