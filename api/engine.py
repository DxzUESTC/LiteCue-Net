"""Model loading, inference, and Grad-CAM heatmap generation."""

import base64
import logging
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from api.config import settings

logger = logging.getLogger(__name__)


# ===================================================================
# Grad-CAM
# ===================================================================


class GradCAM:
    """Grad-CAM heatmap generator attached to the last backbone block."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fwd_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._bwd_handle: Optional[torch.utils.hooks.RemovableHandle] = None

        target = model.backbone.backbone.blocks[-1]
        self._fwd_handle = target.register_forward_hook(self._save_activation)

    def _save_activation(self, _module, _inputs, output):
        self.activations = output.detach()
        if self._bwd_handle is not None:
            self._bwd_handle.remove()
            self._bwd_handle = None
        # 在 torch.no_grad() 下的前向传播也会触发 hook，此时 output 不 require grad
        if output.requires_grad:
            self._bwd_handle = output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad.detach()

    def generate(
        self, video_tensor: torch.Tensor
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Forward + backward from the fake class, return per-frame CAM maps.

        Returns:
            cam_maps: (M*K, H, W) float32 in [0, 1].
            clip_logits: (B, M, 2).
        """
        self.model.zero_grad(set_to_none=True)

        outputs = self.model(video_tensor)
        if isinstance(outputs, dict):
            video_logits = outputs["video_logits"]
            clip_logits = outputs["clip_logits"]
        else:
            video_logits, clip_logits = outputs

        video_logits[:, 1].sum().backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM: failed to capture activations or gradients.")

        # activations/gradients: (B*M*K, C, H', W')
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (N, C, 1, 1)
        cams = F.relu((weights * self.activations).sum(dim=1))  # (N, H', W')

        # Per-frame min-max normalisation
        N, H, W = cams.shape
        flat = cams.view(N, -1)
        cmin = flat.min(dim=1, keepdim=True).values
        cmax = flat.max(dim=1, keepdim=True).values
        cams_norm = ((flat - cmin) / (cmax - cmin + 1e-8)).view(N, H, W)

        return cams_norm.detach().cpu().numpy(), clip_logits

    def remove_hooks(self):
        if self._fwd_handle is not None:
            self._fwd_handle.remove()
        if self._bwd_handle is not None:
            self._bwd_handle.remove()


# ===================================================================
# Inference engine
# ===================================================================


class InferenceEngine:
    """LiteCueNet wrapper: load checkpoint, run inference, generate Grad-CAM."""

    def __init__(self):
        device_str = settings.DEVICE if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)
        logger.info("Inference engine device: %s", self.device)

        self.model = self._build_model()
        self._load_checkpoint(settings.CHECKPOINT_PATH)
        self.model.eval()
        self.gradcam = GradCAM(self.model)
        logger.info("Inference engine ready (checkpoint: %s)", settings.CHECKPOINT_PATH)

    def _build_model(self) -> torch.nn.Module:
        # Late import so the API can start even when the training package is not fully built
        from src.models.detector import LiteCueNet  # type: ignore[import-untyped]

        model = LiteCueNet(
            feature_dim=settings.FEATURE_DIM,
            clip_num=settings.CLIP_NUM,
            clip_len=settings.CLIP_LEN,
            num_classes=settings.NUM_CLASSES,
            backbone_name=settings.BACKBONE_NAME,
            pretrained=False,  # weights come from our checkpoint
            use_frequency_branch=settings.USE_FREQUENCY_BRANCH,
            frequency_fuse_block=settings.FREQUENCY_FUSE_BLOCK,
            temporal_module=settings.TEMPORAL_MODULE,
        )
        return model.to(self.device)

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in ckpt:
                    ckpt = ckpt[key]
                    break
        if isinstance(ckpt, dict) and any(k.startswith("module.") for k in ckpt):
            ckpt = {k[7:]: v for k, v in ckpt.items()}
        missing, unexpected = self.model.load_state_dict(ckpt, strict=False)
        if missing:
            logger.warning("Missing keys in checkpoint: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys in checkpoint: %s", unexpected)

    # ------------------------------------------------------------------
    # Public inference methods
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, video_tensor: np.ndarray) -> Dict:
        """Quick prediction without Grad-CAM."""
        inp = torch.from_numpy(video_tensor).to(self.device)
        outputs = self.model(inp)
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
        }

    def predict_with_heatmap(
        self,
        video_tensor: np.ndarray,
    ) -> Dict:
        """Run inference; if fake, also generate Grad-CAM keyframe heatmaps."""
        # --- First forward (no_grad) for fast prediction ---
        with torch.no_grad():
            inp = torch.from_numpy(video_tensor).to(self.device)
            outputs = self.model(inp)
            if isinstance(outputs, dict):
                video_logits = outputs["video_logits"]
                clip_logits = outputs.get("clip_logits")
            else:
                video_logits, clip_logits = outputs

            probs = F.softmax(video_logits, dim=1)
            fake_prob = float(probs[0, 1].item())
            real_prob = float(probs[0, 0].item())

        is_fake = fake_prob > real_prob
        result: Dict = {
            "is_fake": is_fake,
            "fake_probability": fake_prob,
            "real_probability": real_prob,
        }

        if not is_fake:
            return result

        # --- Second forward (with grad) for Grad-CAM ---
        try:
            inp = torch.from_numpy(video_tensor).to(self.device)
            inp.requires_grad = True
            cam_maps, clip_logits = self.gradcam.generate(inp)
        except RuntimeError as exc:
            logger.warning("Grad-CAM generation failed: %s", exc)
            return result

        M = settings.CLIP_NUM
        K = settings.CLIP_LEN
        # clip_logits: (1, M, 2) -> (M,)
        clip_fake = F.softmax(clip_logits, dim=2)[0, :, 1].detach().cpu().numpy()
        # cam_maps: (M*K, H, W) -> (M, K, H, W)
        cam_maps = cam_maps.reshape(M, K, *cam_maps.shape[1:])

        # Rank clips by fake probability, pick the best frame within each
        top_indices = np.argsort(clip_fake)[::-1][: settings.TOP_K_FRAMES]

        frames: List[Dict] = []
        for ci in top_indices:
            clip_cams = cam_maps[ci]  # (K, H, W)
            mean_scores = clip_cams.reshape(K, -1).mean(axis=1)
            best_k = int(np.argmax(mean_scores))
            frame_cam = clip_cams[best_k]
            frame_idx = int(ci * K + best_k)

            b64 = _cam_to_base64(frame_cam, dsize=(settings.FACE_SIZE, settings.FACE_SIZE))
            frames.append(
                {
                    "frame_index": frame_idx,
                    "clip_index": int(ci),
                    "clip_fake_probability": float(clip_fake[ci]),
                    "heatmap_base64": b64,
                }
            )

        result["heatmap_frames"] = frames
        return result


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _cam_to_base64(cam: np.ndarray, dsize: Tuple[int, int]) -> str:
    """Resize CAM to *dsize*, apply JET colormap, return base64 JPEG (RGB)."""
    cam_resized = cv2.resize(cam, dsize, interpolation=cv2.INTER_LINEAR)
    cam_uint8 = np.clip(cam_resized * 255, 0, 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)  # (H, W, 3) BGR
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    buf = BytesIO()
    Image.fromarray(colored_rgb).save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
