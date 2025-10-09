from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class LEResult:
    ler_per_logical: Optional[np.ndarray]
    ler_mean: Optional[float]
    n_shots: int
    notes: str = ""


def logical_error_rate(true_obs: Optional[np.ndarray],
                       pred_obs: Optional[np.ndarray]) -> LEResult:
    """Compute logical error rate given true and predicted logical flips."""
    if true_obs is None or pred_obs is None:
        return LEResult(None, None, 0, notes="obs unavailable")
    true_arr = np.asarray(true_obs, dtype=np.uint8)
    pred_arr = np.asarray(pred_obs, dtype=np.uint8)
    if true_arr.shape != pred_arr.shape:
        return LEResult(None, None, 0, notes="shape mismatch")
    B, _ = true_arr.shape
    mism = (true_arr ^ pred_arr)
    per_logical = mism.mean(axis=0)
    return LEResult(per_logical, float(per_logical.mean()), int(B))


def throughput(shots: int, elapsed_s: float) -> float:
    return float(shots / max(elapsed_s, 1e-9))


def summary_line(family: str,
                 distance: int,
                 batches: int,
                 shots_per_batch: int,
                 ler: LEResult,
                 elapsed_s: float,
                 teacher_usage: Dict[str, int]) -> str:
    shots = batches * shots_per_batch
    tps = throughput(shots, elapsed_s)
    if ler.ler_mean is None:
        note = f" ({ler.notes})" if ler.notes else ""
        return (f"[done] family={family} d={distance} shots={shots} in {elapsed_s:.2f}s "
                f"({tps:.0f} shots/s), LER=NA{note}, teacher-usage={teacher_usage}")
    per_str = np.array2string(ler.ler_per_logical, precision=2)
    return (f"[done] family={family} d={distance} shots={shots} in {elapsed_s:.2f}s "
            f"({tps:.0f} shots/s), LER={ler.ler_mean:.3e}, per-logical={per_str}, "
            f"teacher-usage={teacher_usage}")
