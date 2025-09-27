"""Clean-core shim exposing MGHD building blocks."""
from __future__ import annotations

from mghd_public.blocks import AstraGNN, ChannelSE
from panq_functions import GNNDecoder

try:
    from mamba_ssm import Mamba  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "Mamba SSM backend is unavailable; archived implementations live under archive/poc_my_models.py"
    ) from exc


__all__ = ["ChannelSE", "AstraGNN", "GNNDecoder", "Mamba"]
