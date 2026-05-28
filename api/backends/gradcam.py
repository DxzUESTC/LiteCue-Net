"""Grad-CAM heatmap generation, independent of any specific model backend."""

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GradCAM:
    """Grad-CAM heatmap generator.

    Attaches a forward hook to the last block of a backbone to capture
    activations and gradients, then computes class-activation maps.
    """

    def __init__(self, backbone_block: torch.nn.Module):
        """Attach hook to the target backbone block.

        Args:
            backbone_block: The module whose output will be used as the
                            activation map (e.g., model.backbone.blocks[-1]).
        """
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fwd_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._bwd_handle: Optional[torch.utils.hooks.RemovableHandle] = None

        self._fwd_handle = backbone_block.register_forward_hook(self._save_activation)

    def _save_activation(self, _module, _inputs, output):
        self.activations = output.detach()
        if self._bwd_handle is not None:
            self._bwd_handle.remove()
            self._bwd_handle = None
        if output.requires_grad:
            self._bwd_handle = output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad.detach()

    @torch.enable_grad()
    def generate(
        self,
        model: torch.nn.Module,
        video_tensor: torch.Tensor,
        target_class: int = 1,
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Forward + backward, returning per-frame CAM maps.

        Args:
            model: The model to run.
            video_tensor: (1, M, K, 3, H, W) input.
            target_class: Class index for backward (default 1 = fake).

        Returns:
            cam_maps: (M*K, H', W') float32 in [0, 1].
            clip_logits: (B, M, 2).
            video_logits: (B, 2).
        """
        model.zero_grad(set_to_none=True)

        outputs = model(video_tensor)
        if isinstance(outputs, dict):
            video_logits = outputs["video_logits"]
            clip_logits = outputs["clip_logits"]
        else:
            video_logits, clip_logits = outputs

        video_logits[:, target_class].sum().backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM: failed to capture activations or gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cams = F.relu((weights * self.activations).sum(dim=1))

        N, Hp, Wp = cams.shape
        flat = cams.view(N, -1)
        cmin = flat.min(dim=1, keepdim=True).values
        cmax = flat.max(dim=1, keepdim=True).values
        cams_norm = ((flat - cmin) / (cmax - cmin + 1e-8)).view(N, Hp, Wp)

        return cams_norm.detach().cpu().numpy(), clip_logits, video_logits

    def remove_hooks(self):
        if self._fwd_handle is not None:
            self._fwd_handle.remove()
        if self._bwd_handle is not None:
            self._bwd_handle.remove()
