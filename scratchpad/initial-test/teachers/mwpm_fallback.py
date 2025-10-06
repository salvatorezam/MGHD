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

import core

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


def _is_graphlike(H: np.ndarray) -> bool:
    """Return True if every column has at most two non-zero entries."""

    try:
        col_sums = np.asarray(H.sum(axis=0)).ravel()
    except Exception:
        col_sums = np.asarray(np.sum(H, axis=0)).ravel()
    return bool(np.all(col_sums <= 2))


class MWPMFallback:
    """PyMatching-backed MWPM decoder with optional graphlike enforcement."""

    def __init__(
        self,
        H: np.ndarray,
        *,
        weights: Optional[Iterable[Any]] = None,
        require_graphlike: bool = False,
    ):
        self.H = np.asarray(H, dtype=np.uint8) & 1
        self._gf2_cols = self.H.shape[1]
        self.weights = _coerce_weights_to_float(weights, self._gf2_cols)
        self.require_graphlike = bool(require_graphlike)
        self._graphlike = _is_graphlike(self.H)
        self.m = None
        if self.require_graphlike and not self._graphlike:
            raise MwpmNotGraphlike("mwpm_not_graphlike")

        if _HAVE_PM and self._graphlike:
            try:
                self.m = pm.Matching.from_check_matrix(self.H, weights=self.weights)  # type: ignore[union-attr]
            except ValueError as exc:  # pragma: no cover - communicated to caller
                raise MwpmNotGraphlike(str(exc)) from exc
            except BaseException as exc:  # pyo3 panic surfaces here
                raise MwpmNotGraphlike(f"PyMatching panic: {exc}") from exc
            else:
                self._pm_failure = None
        elif _HAVE_PM:
            warnings.warn(
                "PyMatching graphlike requirement not met; using GF(2) projection fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._pm_failure = MwpmNotGraphlike("mwpm_not_graphlike")
        else:
            warnings.warn(
                "PyMatching not available – MWPMFallback will use GF(2) projection fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._pm_failure = _PM_IMPORT_ERROR

    def _gf2_decode(self, syndromes: np.ndarray) -> np.ndarray:
        B = syndromes.shape[0]
        out = np.zeros((B, self._gf2_cols), dtype=np.uint8)
        for b in range(B):
            out[b] = core.ml_parity_project(self.H, syndromes[b])
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

        if _HAVE_PM:
            if self.require_graphlike and not self._graphlike:
                raise ValueError("mwpm_not_graphlike")
            matcher = self.m
            if column_weights is not None:
                weights_arr = np.asarray(column_weights, dtype=np.float32)
                if weights_arr.shape[0] != self._gf2_cols:
                    raise ValueError("column_weights must match number of columns in H")
                try:
                    matcher = pm.Matching.from_check_matrix(self.H, weights=weights_arr)
                except ValueError as exc:
                    raise MwpmNotGraphlike(str(exc)) from exc
                except BaseException as exc:
                    raise MwpmNotGraphlike(f"PyMatching panic: {exc}") from exc
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


__all__ = ["MWPMFallback", "MwpmNotGraphlike", "_HAVE_PM", "_is_graphlike"]


if not _HAVE_PM and _PM_IMPORT_ERROR is not None:  # pragma: no cover - info
    warnings.warn(
        "pymatching package not found – MWPMFallback fallback will be used.",
        RuntimeWarning,
        stacklevel=2,
    )
