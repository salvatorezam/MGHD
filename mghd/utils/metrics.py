from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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


def _wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Parameters
    - k: number of successes
    - n: number of trials
    - z: z-score for desired confidence (1.96 ≈ 95%)
    """
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + (z ** 2) / n
    center = (p + (z ** 2) / (2 * n)) / denom
    half = (z * np.sqrt((p * (1 - p) / n) + (z ** 2) / (4 * n ** 2))) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (lo, hi)


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
    per_str = np.array2string(ler.ler_per_logical, formatter={"float_kind": lambda x: f"{x:.3e}"})
    # Approximate CI by aggregating flips across all logicals
    if ler.ler_per_logical is not None:
        k_total = int(round(float(ler.ler_per_logical.sum() * ler.n_shots)))
        lo, hi = _wilson_interval(k_total, max(ler.n_shots * len(ler.ler_per_logical), 1))
        pm = 0.5 * (hi - lo)
        ci_text = f"{ler.ler_mean:.3e}±{pm:.3e}"
    else:
        ci_text = f"{ler.ler_mean:.3e}"
    return (f"[done] family={family} d={distance} shots={shots} in {elapsed_s:.2f}s "
            f"({tps:.0f} shots/s), LER={ci_text} per-logical={per_str}, "
            f"teacher-usage={teacher_usage}")
