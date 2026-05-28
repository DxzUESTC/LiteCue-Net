"""Model backend registry and factory."""

from typing import Any, Dict, Type

from api.backends.base import ModelBackend
from api.backends.litecuenet import LiteCueNetBackend

_REGISTRY: Dict[str, Type[ModelBackend]] = {
    "litecuenet": LiteCueNetBackend,
}


def register_backend(name: str, backend_cls: Type[ModelBackend]) -> None:
    """Register a new model backend."""
    _REGISTRY[name] = backend_cls


def create_backend(
    name: str = "litecuenet",
    checkpoint_path: str = "",
    device: str = "cuda",
    **kwargs: Any,
) -> ModelBackend:
    """Factory: instantiate a registered model backend.

    Extra keyword arguments are passed to the backend constructor.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown backend '{name}'. Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[name](checkpoint_path=checkpoint_path, device=device, **kwargs)


__all__ = ["ModelBackend", "create_backend", "register_backend"]
