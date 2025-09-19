from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def _coerce_args_kwargs(features: Any | None) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Map arbitrary feature payloads to positional/keyword tensors."""

    if features is None:
        return tuple(), {}

    if isinstance(features, Mapping):
        args = features.get("args", ())
        kwargs = dict(features.get("kwargs", {}))
        if not isinstance(args, (tuple, list)):
            args = (args,)
        else:
            args = tuple(args)
        return args, kwargs

    if isinstance(features, (tuple, list)):
        return tuple(features), {}

    return (features,), {}


class MGHDAdapter:
    """Expose MGHD predictions as per-bit error probabilities for clustering."""

    def __init__(self, n_bits: int, model_loader: Any | None = None, *, model: Any | None = None):
        self.n_bits = int(n_bits)
        self.model = model

        if self.model is None:
            if isinstance(model_loader, str):
                try:
                    from .mghd_loader import load_mghd_model

                    self.model = load_mghd_model(model_loader)
                except Exception:
                    self.model = None
            elif callable(model_loader):
                try:
                    self.model = model_loader()
                except Exception:
                    self.model = None

        if self.model is not None and hasattr(self.model, "eval"):
            self.model.eval()

    def predict_error_probs(self, features: Any | None) -> np.ndarray:
        if self.model is None:
            return np.full(self.n_bits, 0.5, dtype=np.float64)

        import torch

        args, kwargs = _coerce_args_kwargs(features)

        try:
            with torch.no_grad():
                logits = self.model(*args, **kwargs)
        except Exception:
            return np.full(self.n_bits, 0.5, dtype=np.float64)

        if isinstance(logits, torch.Tensor):
            if logits.ndim > 1:
                logits = logits.squeeze(0)
            probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)
        else:
            probs = np.asarray(logits, dtype=np.float64)

        if probs.shape[0] != self.n_bits:
            return np.full(self.n_bits, 0.5, dtype=np.float64)

        return np.clip(probs, 1e-6, 1 - 1e-6)
