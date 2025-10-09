from __future__ import annotations
from typing import Dict, Callable

_REGISTRY: Dict[str, Callable[..., object]] = {}

def register_sampler(name: str, factory: Callable[..., object]) -> None:
    _REGISTRY[name] = factory

def get_sampler(name: str, **kwargs):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown sampler '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
