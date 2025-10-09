"""
MWPM fallback using PyMatching v2.
- Can build from a parity-check matrix (columns = faults) or directly from a DEM.
- We avoid DEM for training; build from H for CSS codes and use ``decode_batch``.

Refs:
- PyMatching API (from_check_matrix, decode_batch): https://pymatching.readthedocs.io/en/latest/api.html
"""
from __future__ import annotations

from typing import Any, Iterable, Optional
import warnings

import numpy as np
import scipy.sparse as sp

from mghd.decoders.lsd.cluster_core import ml_parity_project
from mghd.utils.graphlike import is_graphlike

try:  # pragma: no cover - optional dependency
    import pymatching as pm  # type: ignore

    _HAVE_PM = True
    _PM_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - executed without pymatching
    pm = None  # type: ignore
    _HAVE_PM = False
    _PM_IMPORT_ERROR = exc


class MwpmNotGraphlike(RuntimeError):
    """Raised when PyMatching cannot operate due to non-graphlike structure."""

    pass


def _safe_matching_from_check(H: np.ndarray, **kwargs):
    if not _HAVE_PM:
        raise MwpmNotGraphlike("pymatching unavailable")
    try:
        return pm.Matching.from_check_matrix(H, **kwargs)  # type: ignore[union-attr]
    except ValueError as exc:
        raise MwpmNotGraphlike(str(exc)) from exc
    except BaseException as exc:  # catches pyo3 PanicException
        raise MwpmNotGraphlike(f"PyMatching panic: {exc}") from exc


def _coerce_weights_to_float(weights: Optional[Iterable[Any]], ncols: int) -> Optional[np.ndarray]:
    """Best-effort conversion of MWPF-style weights to float32 for PyMatching."""

    if weights is None:
        return None
    try:
        arr = np.asarray(list(weights), dtype=object)
    except Exception:
        # Some iterables (e.g., numpy arrays) support direct np.asarray but not list().
        try:
            arr = np.asarray(weights, dtype=object)
        except Exception:
            return None

    if arr.size not in (0, ncols):
        return None

    try:
        float_arr = arr.astype(np.float64)
    except Exception:
        return None

    if not np.all(np.isfinite(float_arr)):
        return None

    return float_arr.astype(np.float32, copy=False)


class MWPMFallback:
    """PyMatching-backed MWPM decoder with optional graphlike enforcement."""

    def __init__(
        self,
        code: Any,
        *,
        basis: str,
        weights: Optional[Iterable[Any]] = None,
        require_graphlike: bool = False,
    ):
        basis_norm = basis.lower()
        if basis_norm not in {"x", "z"}:
            raise ValueError("basis must be 'x' or 'z'")
        self.code = code
        self.basis = basis_norm
        matrix = getattr(code, "Hx" if basis_norm == "x" else "Hz")
        if matrix is None:
            raise ValueError("Code object missing required parity-check matrix")
        self.H = np.asarray(matrix, dtype=np.uint8) & 1
        self._gf2_cols = self.H.shape[1]
        self.weights = _coerce_weights_to_float(weights, self._gf2_cols)
        self.require_graphlike = bool(require_graphlike)
        self._graphlike = is_graphlike(self.H)
        self.m = None
        if self.require_graphlike and not self._graphlike:
            raise MwpmNotGraphlike("mwpm_not_graphlike")

        if self._graphlike and _HAVE_PM:
            try:
                self.m = _safe_matching_from_check(self.H, weights=self.weights)
            except MwpmNotGraphlike as exc:
                self.m = None
                self._graphlike = False
                if self.require_graphlike:
                    raise
                warnings.warn(
                    f"PyMatching unavailable for MWPM ({exc}); using GF(2) fallback.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        elif _HAVE_PM and not self._graphlike:
            warnings.warn(
                "PyMatching graphlike requirement not met; using GF(2) projection fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif not _HAVE_PM:
            warnings.warn(
                "PyMatching not available – MWPMFallback will use GF(2) projection fallback.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _gf2_decode(self, syndromes: np.ndarray) -> np.ndarray:
        B = syndromes.shape[0]
        out = np.zeros((B, self._gf2_cols), dtype=np.uint8)
        # ml_parity_project requires p_flip and sparse matrix; use uniform 0.5 for unweighted case
        p_uniform = np.full(self.H.shape[1], 0.5, dtype=np.float32)
        H_sparse = sp.csr_matrix(self.H) if not sp.issparse(self.H) else self.H
        for b in range(B):
            out[b] = ml_parity_project(H_sparse, syndromes[b], p_uniform)
        return out

    def decode_batch(
        self,
        syndromes: np.ndarray,
        *,
        column_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return corrections in 'fault id' space aligned with columns of H."""

        syndromes = np.asarray(syndromes, dtype=np.uint8) & 1
        if syndromes.ndim != 2:
            raise ValueError("syndromes must be rank-2 array [B, #checks]")

        if _HAVE_PM and self._graphlike:
            if self.require_graphlike and not self._graphlike:
                raise ValueError("mwpm_not_graphlike")
            matcher = self.m
            if column_weights is not None:
                weights_arr = np.asarray(column_weights, dtype=np.float32)
                if weights_arr.shape[0] != self._gf2_cols:
                    raise ValueError("column_weights must match number of columns in H")
                try:
                    matcher = _safe_matching_from_check(self.H, weights=weights_arr)
                except MwpmNotGraphlike as exc:
                    warnings.warn(
                        f"Column weight rebuild failed ({exc}); using GF(2) fallback.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    matcher = None
            if matcher is not None:
                if hasattr(matcher, "decode_batch"):
                    decoded = matcher.decode_batch(syndromes)  # type: ignore[union-attr]
                else:
                    decoded = np.stack(
                        [np.asarray(matcher.decode(s), dtype=np.uint8) for s in syndromes],  # type: ignore[union-attr]
                        axis=0,
                    )
                return np.asarray(decoded, dtype=np.uint8)

        return self._gf2_decode(syndromes)


__all__ = ["MWPMFallback", "MwpmNotGraphlike", "_HAVE_PM", "_safe_matching_from_check"]


if not _HAVE_PM and _PM_IMPORT_ERROR is not None:  # pragma: no cover - info
    warnings.warn(
        "pymatching package not found – MWPMFallback fallback will be used.",
        RuntimeWarning,
        stacklevel=2,
    )
