"""Abstract base class for model backends."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class ModelBackend(ABC):
    """Interface for model inference backends.

    Each backend wraps a specific model architecture and provides
    prediction and explainability methods.
    """

    @abstractmethod
    def load(self) -> None:
        """Load model weights and prepare for inference."""

    @abstractmethod
    def predict(self, tensor: Any) -> Dict:
        """Run inference without explainability.

        Args:
            tensor: Preprocessed input (e.g. np.ndarray).

        Returns:
            Dict with keys: is_fake, fake_probability, real_probability.
        """

    @abstractmethod
    def predict_with_explain(self, tensor: Any) -> Dict:
        """Run inference with explainability (Grad-CAM).

        Args:
            tensor: Preprocessed input (e.g. np.ndarray).

        Returns:
            Dict with keys: is_fake, fake_probability, real_probability,
                            heatmap_frames (list).
        """

    @property
    @abstractmethod
    def device(self) -> str:
        """Device string (e.g. 'cuda', 'cpu')."""

    @property
    @abstractmethod
    def model_config(self) -> Dict:
        """Model architecture configuration as a dict."""
