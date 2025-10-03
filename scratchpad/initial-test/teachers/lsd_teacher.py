"""
LSD/BP primary teacher using ldpc's BP+LSD on parity-check matrices.
We run X and Z bases independently for CSS codes.

Refs:
- ldpc docs (BP+LSD, BP+OSD, belief-find): https://software.roffe.eu/ldpc/quantum_decoder.html
- LSD paper (parallel local inversions / cluster solves): https://arxiv.org/abs/2406.18655
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import warnings

import numpy as np

import core

try:  # pragma: no cover - optional dependency
    from ldpc.bplsd_decoder import BpLsdDecoder  # type: ignore

    _HAVE_LDPC = True
    _LDPC_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - executed when ldpc missing
    BpLsdDecoder = None  # type: ignore
    _HAVE_LDPC = False
    _LDPC_IMPORT_ERROR = exc


@dataclass
class LSDConfig:
    error_rate: float = 0.05
    max_iter: int = 3
    bp_method: str = "product_sum"
    schedule: str = "serial"
    lsd_method: str = "lsd_cs"  # cluster-search
    lsd_order: int = 2


class LSDTeacher:
    """BP+LSD teacher with a numpy fallback when ldpc is unavailable."""

    def __init__(self, Hx: np.ndarray, Hz: np.ndarray, *, cfg: Optional[LSDConfig] = None):
        self.cfg = cfg or LSDConfig()
        self.Hx = np.asarray(Hx, dtype=np.uint8)
        self.Hz = np.asarray(Hz, dtype=np.uint8)

        if _HAVE_LDPC:
            self.dec_x = BpLsdDecoder(  # type: ignore[operator]
                self.Hx,
                error_rate=self.cfg.error_rate,
                bp_method=self.cfg.bp_method,
                max_iter=self.cfg.max_iter,
                schedule=self.cfg.schedule,
                osd_method=self.cfg.lsd_method,
                osd_order=self.cfg.lsd_order,
            )
            self.dec_z = BpLsdDecoder(  # type: ignore[operator]
                self.Hz,
                error_rate=self.cfg.error_rate,
                bp_method=self.cfg.bp_method,
                max_iter=self.cfg.max_iter,
                schedule=self.cfg.schedule,
                osd_method=self.cfg.lsd_method,
                osd_order=self.cfg.lsd_order,
            )
        else:
            self.dec_x = self.dec_z = None
            warnings.warn(
                "ldpc not available – LSDTeacher will use GF(2) projection fallback.",
                RuntimeWarning,
                stacklevel=2,
            )

    def decode_batch_xz(
        self,
        syndromes_x: np.ndarray,
        syndromes_z: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (ex, ez) corrections in data-qubit space: uint8 [B, n]."""

        syndromes_x = np.asarray(syndromes_x, dtype=np.uint8)
        syndromes_z = np.asarray(syndromes_z, dtype=np.uint8)
        if syndromes_x.ndim != 2 or syndromes_z.ndim != 2:
            raise ValueError("Syndromes must be 2-D batches")
        if syndromes_x.shape[0] != syndromes_z.shape[0]:
            raise ValueError("Batch sizes of X and Z syndromes must match")

        if _HAVE_LDPC:
            ex_list = [
                self.dec_x.decode(syndromes_x[b].astype(np.uint8)).astype(np.uint8)  # type: ignore[union-attr]
                for b in range(syndromes_x.shape[0])
            ]
            ez_list = [
                self.dec_z.decode(syndromes_z[b].astype(np.uint8)).astype(np.uint8)  # type: ignore[union-attr]
                for b in range(syndromes_z.shape[0])
            ]
            return np.stack(ex_list, axis=0), np.stack(ez_list, axis=0)

        # Fallback: exact ML projection using the GF(2) helper from `core`.
        B = syndromes_x.shape[0]
        ex = np.zeros((B, self.Hx.shape[1]), dtype=np.uint8)
        ez = np.zeros((B, self.Hz.shape[1]), dtype=np.uint8)
        for b in range(B):
            ex[b] = core.ml_parity_project(self.Hx, syndromes_x[b])
            ez[b] = core.ml_parity_project(self.Hz, syndromes_z[b])
        return ex, ez


__all__ = ["LSDTeacher", "LSDConfig", "_HAVE_LDPC"]


if not _HAVE_LDPC and _LDPC_IMPORT_ERROR is not None:  # pragma: no cover - info
    warnings.warn(
        "ldpc package not found – LSDTeacher fallback will be used.",
        RuntimeWarning,
        stacklevel=2,
    )
