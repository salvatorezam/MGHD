"""
MWPM fallback using PyMatching v2.
- Can build from a parity-check matrix (columns = faults) or directly from a DEM.
- We avoid DEM for training; build from H for CSS codes and use ``decode_batch``.

Refs:
- PyMatching API (from_check_matrix, decode_batch): https://pymatching.readthedocs.io/en/latest/api.html
"""
from __future__ import annotations

from typing import Optional
import warnings

import numpy as np

import core

try:  # pragma: no cover - optional dependency
    import pymatching as pm  # type: ignore

    _HAVE_PM = True
    _PM_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - executed without pymatching
    pm = None  # type: ignore
    _HAVE_PM = False
    _PM_IMPORT_ERROR = exc


class MWPMFallback:
    """PyMatching-backed MWPM decoder with a GF(2) fallback."""

    def __init__(self, H: np.ndarray, *, weights: Optional[np.ndarray] = None):
        self.H = np.asarray(H, dtype=np.uint8)
        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        self.m = None
        if _HAVE_PM:
            try:
                self.m = pm.Matching.from_check_matrix(self.H, weights=self.weights)  # type: ignore[union-attr]
            except ValueError as exc:  # pragma: no cover - dependent on optional library
                warnings.warn(
                    "PyMatching could not load the parity-check matrix; falling back to GF(2) projection.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._pm_failure = exc
            else:
                self._pm_failure = None
        else:
            warnings.warn(
                "PyMatching not available – MWPMFallback will use GF(2) projection fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._pm_failure = _PM_IMPORT_ERROR

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        """Return corrections in 'fault id' space aligned with columns of H."""

        syndromes = np.asarray(syndromes, dtype=np.uint8)
        if syndromes.ndim != 2:
            raise ValueError("syndromes must be rank-2 array [B, #checks]")

        if _HAVE_PM and self.m is not None:
            return self.m.decode_batch(syndromes).astype(np.uint8)  # type: ignore[union-attr]

        # Fallback: solve Hx=s exactly via GF(2) projection for each sample.
        B = syndromes.shape[0]
        cols = self.H.shape[1]
        out = np.zeros((B, cols), dtype=np.uint8)
        for b in range(B):
            out[b] = core.ml_parity_project(self.H, syndromes[b])
        return out


__all__ = ["MWPMFallback", "_HAVE_PM"]


if not _HAVE_PM and _PM_IMPORT_ERROR is not None:  # pragma: no cover - info
    warnings.warn(
        "pymatching package not found – MWPMFallback fallback will be used.",
        RuntimeWarning,
        stacklevel=2,
    )
