from __future__ import annotations

import importlib
from typing import Any, Tuple, Dict

import torch


def load_from_spec(spec: str):
    """Import a class from "module.path:ClassName" specification."""
    if ":" not in spec:
        raise ValueError(f"Model spec must be module:ClassName, got {spec!r}")
    module_path, class_name = spec.split(":", 1)
    module = importlib.import_module(module_path)
    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:  # pragma: no cover
        raise ImportError(f"Class {class_name!r} not found in module {module_path!r}") from exc
    return cls


def load_mghd(ckpt: str, model_spec: str, init_kwargs: Dict[str, Any] | None = None):
    """Instantiate an MGHD model from checkpoint with robust fallbacks."""
    init_kwargs = dict(init_kwargs or {})
    cls = load_from_spec(model_spec)

    try:
        model = cls.load_from_checkpoint(ckpt, map_location="cpu", **init_kwargs)  # type: ignore[attr-defined]
    except Exception:
        model = cls(**init_kwargs)
        state = torch.load(ckpt, map_location="cpu")
        state_dict = state.get("state_dict", state)
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _coerce_args_kwargs(features: Any | None) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    if features is None:
        return tuple(), {}
    if isinstance(features, dict):
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


@torch.no_grad()
def forward_logits(model: Any, features: Any) -> torch.Tensor:
    args, kwargs = _coerce_args_kwargs(features)
    logits = model(*args, **kwargs)
    if isinstance(logits, torch.Tensor):
        return logits.squeeze(0) if logits.ndim > 1 else logits
    return torch.as_tensor(logits, dtype=torch.float32).squeeze(0)


@torch.no_grad()
def forward_probs(model: Any, features: Any) -> torch.Tensor:
    logits = forward_logits(model, features)
    return torch.sigmoid(logits).clamp_(1e-6, 1 - 1e-6)
