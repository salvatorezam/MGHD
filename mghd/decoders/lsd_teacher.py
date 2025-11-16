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

from mghd.decoders.lsd.clustered import ml_parity_project
import torch as _torch  # Enforce CUDA availability for GPU projector

try:  # pragma: no cover - optional dependency
    # ldpc 0.1.x has bposd_decoder, not bplsd_decoder
    # BP+OSD with osd_method can approximate LSD behavior
    import ldpc

    BpLsdDecoder = ldpc.bposd_decoder  # type: ignore
    _HAVE_LDPC = True
    _LDPC_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - executed when ldpc missing
    BpLsdDecoder = None  # type: ignore
    _HAVE_LDPC = False
    _LDPC_IMPORT_ERROR = exc


@dataclass
class LSDConfig:
    """Configuration for BP+LSD teacher (and NumPy fallback).

    error_rate: Prior error probability used by BP/OSD.
    max_iter:   BP iterations per decode.
    bp_method:  Update rule (e.g., product_sum).
    schedule:   Message schedule (serial/parallel).
    lsd_method: OSD variant; 'OSD_CS' approximates LSD with cluster-search.
    lsd_order:  OSD order parameter.
    """

    error_rate: float = 0.05
    max_iter: int = 3
    bp_method: str = "product_sum"
    schedule: str = "serial"
    lsd_method: str = "OSD_CS"  # cluster-search; valid options: 'OSD_0', 'OSD_E', 'OSD_CS'
    lsd_order: int = 2


class LSDTeacher:
    """BP+LSD teacher with a NumPy fallback when ldpc is unavailable.

    When ldpc is present, we instantiate per-basis BP+OSD decoders (bposd).
    When absent or when overrides/erasure are provided, we project via GF(2)
    parity using ml_parity_project (or the torch-accelerated variant when
    available) to produce data-qubit corrections ex/ez per batch.
    """

    def __init__(self, Hx: np.ndarray, Hz: np.ndarray, *, cfg: Optional[LSDConfig] = None):
        """Create a per-basis decoder for CSS codes.

        Parameters
        - Hx, Hz: parity-check matrices (uint8) for X and Z bases.
        - cfg: optional LSDConfig controlling BP/OSD behavior.
        """
        # Enforce GPU availability: this teacher runs exclusively on CUDA for performance.
        if not _torch.cuda.is_available():  # pragma: no cover - fail fast on misconfigured envs
            raise RuntimeError("LSDTeacher requires CUDA (torch.cuda.is_available() == True).")
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
        *,
        llr_overrides: Optional[np.ndarray] = None,
        erase_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (ex, ez) corrections in data-qubit space: uint8 [B, n]."""

        syndromes_x = np.asarray(syndromes_x, dtype=np.uint8)
        syndromes_z = np.asarray(syndromes_z, dtype=np.uint8)
        if syndromes_x.ndim != 2 or syndromes_z.ndim != 2:
            raise ValueError("Syndromes must be 2-D batches")
        if syndromes_x.shape[0] != syndromes_z.shape[0]:
            raise ValueError("Batch sizes of X and Z syndromes must match")

        llr_arr = None
        if llr_overrides is not None:
            llr_arr = np.asarray(llr_overrides, dtype=np.float64)
            if llr_arr.ndim == 1:
                llr_arr = np.broadcast_to(llr_arr, (syndromes_x.shape[0], llr_arr.shape[0]))
            if llr_arr.shape != (syndromes_x.shape[0], self.Hx.shape[1]):
                raise ValueError("llr_overrides must broadcast to (B, n)")

        erase_arr = None
        if erase_mask is not None:
            erase_arr = np.asarray(erase_mask, dtype=np.uint8)
            if erase_arr.ndim == 1:
                erase_arr = np.broadcast_to(erase_arr, (syndromes_x.shape[0], erase_arr.shape[0]))
            if erase_arr.shape != (syndromes_x.shape[0], self.Hx.shape[1]):
                raise ValueError("erase_mask must broadcast to (B, n)")

        if llr_arr is None and erase_arr is None and _HAVE_LDPC:
            ex_list = [
                self.dec_x.decode(syndromes_x[b].astype(np.uint8)).astype(np.uint8)  # type: ignore[union-attr]
                for b in range(syndromes_x.shape[0])
            ]
            ez_list = [
                self.dec_z.decode(syndromes_z[b].astype(np.uint8)).astype(np.uint8)  # type: ignore[union-attr]
                for b in range(syndromes_z.shape[0])
            ]
            return np.stack(ex_list, axis=0), np.stack(ez_list, axis=0)

        # Weighted GF(2) projection for overrides / erasure-aware path
        B = syndromes_x.shape[0]
        ex = np.zeros((B, self.Hx.shape[1]), dtype=np.uint8)
        ez = np.zeros((B, self.Hz.shape[1]), dtype=np.uint8)
        # Use the torch-accelerated projector on CUDA unconditionally
        from mghd.decoders.lsd.clustered import ml_parity_project_torch as _ml_t

        for b in range(B):
            probs_x = None
            probs_z = None
            if llr_arr is not None:
                llr = llr_arr[b]
                probs = 1.0 / (1.0 + np.exp(llr))
                probs_x = probs.copy()
                probs_z = probs.copy()
            if erase_arr is not None:
                mask = erase_arr[b].astype(bool)
                if probs_x is None:
                    probs_x = np.full(self.Hx.shape[1], 0.5, dtype=np.float64)
                if probs_z is None:
                    probs_z = np.full(self.Hz.shape[1], 0.5, dtype=np.float64)
                probs_x[mask] = 0.5
                probs_z[mask] = 0.5

            ex[b] = _ml_t(self.Hx, syndromes_x[b], probs_x, device="cuda")
            ez[b] = _ml_t(self.Hz, syndromes_z[b], probs_z, device="cuda")
        return ex, ez


__all__ = ["LSDTeacher", "LSDConfig", "_HAVE_LDPC"]


if not _HAVE_LDPC and _LDPC_IMPORT_ERROR is not None:  # pragma: no cover - info
    warnings.warn(
        "ldpc package not found – LSDTeacher fallback will be used.",
        RuntimeWarning,
        stacklevel=2,
    )
