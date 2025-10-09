"""Erasure-aware surface-code ML decoder (Delfosse–Zémor style)."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from core import solve_on_erasure


class ErasureSurfaceMLTeacher:
    """Linear-time ML decoder for erasures on rotated surface codes.

    References
    ----------
    M. Delfosse and P. Zémor, Phys. Rev. Research 2, 033042 (2020).
    """

    def __init__(self, code: Any) -> None:
        if not hasattr(code, "Hx") or not hasattr(code, "Hz"):
            raise ValueError("ErasureSurfaceMLTeacher requires CSS code with Hx/Hz")
        self.code = code
        self.Hx = np.asarray(code.Hx, dtype=np.uint8)
        self.Hz = np.asarray(code.Hz, dtype=np.uint8)
        self.n = self.Hx.shape[1]
        self.mx = self.Hx.shape[0]
        self.mz = self.Hz.shape[0]

    def decode_batch(
        self,
        syndromes_x: np.ndarray,
        syndromes_z: np.ndarray,
        erase_data_mask: Optional[np.ndarray],
        erase_det_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        if erase_data_mask is None:
            raise ValueError("erase_data_mask is required for erasure decoding")
        syndromes_x = np.asarray(syndromes_x, dtype=np.uint8)
        syndromes_z = np.asarray(syndromes_z, dtype=np.uint8)
        erase_data_mask = np.asarray(erase_data_mask, dtype=np.uint8)
        if syndromes_x.ndim != 2 or syndromes_z.ndim != 2:
            raise ValueError("Syndromes must be 2-D arrays")
        B = syndromes_x.shape[0]
        if syndromes_z.shape[0] != B or erase_data_mask.shape[0] != B:
            raise ValueError("Batch dimensions must match")
        if erase_data_mask.shape[1] != self.n:
            raise ValueError("erase_data_mask must have length equal to number of data qubits")

        if erase_det_mask is not None:
            erase_det_mask = np.asarray(erase_det_mask, dtype=np.uint8)
            if erase_det_mask.shape[0] != B:
                raise ValueError("erase_det_mask batch size mismatch")
            if erase_det_mask.shape[1] < self.mx + self.mz:
                raise ValueError("erase_det_mask must cover X and Z checks")

        ex = np.zeros((B, self.n), dtype=np.uint8)
        ez = np.zeros((B, self.n), dtype=np.uint8)
        for b in range(B):
            cols_mask = erase_data_mask[b]
            if not np.any(cols_mask):
                continue
            rows_x = None
            rows_z = None
            if erase_det_mask is not None:
                rows_mask = erase_det_mask[b]
                rows_x = rows_mask[: self.mx]
                rows_z = rows_mask[self.mx : self.mx + self.mz]
            ex[b] = solve_on_erasure(self.Hx, syndromes_x[b], cols_mask, rows_x)
            ez[b] = solve_on_erasure(self.Hz, syndromes_z[b], cols_mask, rows_z)
        return {"which": "erasure_surface_ml", "ex": ex, "ez": ez}
