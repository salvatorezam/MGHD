from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _ensure_tensor(vec: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(vec, torch.Tensor):
        return vec.clone().detach()
    return torch.from_numpy(np.asarray(vec, dtype=np.float32))


def build_features_surface(syndrome: np.ndarray | torch.Tensor, **_: Any) -> dict[str, Any]:
    """Construct placeholder features for surface-code MGHD models.

    Replace this stub with the exact preprocessing pipeline used during MGHD
    training (node inputs, graph indices, etc.). The default implementation
    simply returns the syndrome vector as a single positional argument.
    """

    tensor = _ensure_tensor(syndrome).reshape(1, -1)
    return {"args": (tensor,), "kwargs": {}}


def build_features_bb(syndrome: np.ndarray | torch.Tensor, *_: Any, **__: Any) -> dict[str, Any]:
    """Construct placeholder features for the [[144,12,12]] BB MGHD models.

    Customize this helper to mirror the MGHD training input signature for BB
    codes. The default returns the syndrome vector only.
    """

    tensor = _ensure_tensor(syndrome).reshape(1, -1)
    return {"args": (tensor,), "kwargs": {}}
