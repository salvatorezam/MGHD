"""Detector-error-model based teacher using PyMatching."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import pymatching as _pm  # type: ignore

    _HAVE_PM = True
    _PM_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - executed when pymatching missing
    _pm = None  # type: ignore
    _HAVE_PM = False
    _PM_IMPORT_ERROR = exc


class DEMMatchingTeacher:
    """Decode detector streams using a Stim DetectorErrorModel and PyMatching."""

    def __init__(self, dem: Any, *, correlated: bool = False) -> None:
        if not _HAVE_PM:
            raise RuntimeError(
                "pymatching is required for DEMMatchingTeacher"
            ) from _PM_IMPORT_ERROR

        self.pm = _pm
        try:
            self.matching = self.pm.Matching.from_detector_error_model(  # type: ignore[union-attr]
                dem,
                correlated=bool(correlated),
            )
        except TypeError:
            self.matching = self.pm.Matching.from_detector_error_model(dem)  # type: ignore[union-attr]
        self.num_detectors = getattr(dem, "num_detectors", None)
        self.num_observables = getattr(dem, "num_observables", None)
        self.correlated = bool(correlated)

    def decode_batch(self, dets: np.ndarray) -> Dict[str, Any]:
        dets = np.asarray(dets, dtype=np.uint8) & 1
        if dets.ndim != 2:
            raise ValueError("dets must be rank-2 array [B, num_detectors]")
        pred_obs = self.matching.decode_batch(dets)  # type: ignore[union-attr]
        return {
            "which": "dem_matching",
            "pred_obs": np.asarray(pred_obs, dtype=np.uint8),
        }


__all__ = ["DEMMatchingTeacher", "_HAVE_PM"]


if not _HAVE_PM and _PM_IMPORT_ERROR is not None:  # pragma: no cover - info
    import warnings

    warnings.warn(
        "pymatching package not found – DEMMatchingTeacher unavailable.",
        RuntimeWarning,
        stacklevel=2,
    )
