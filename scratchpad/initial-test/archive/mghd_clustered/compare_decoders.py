from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np

from .adapter import MGHDAdapter
from .decoder import MGHDClusteredDecoder
from .mghd_loader import mghd_forward_probs


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def latency_stats(samples: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(samples), dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(arr.mean()),
    }


def _summarize_stats(stats_list: List[Dict[str, Any]]) -> Dict[str, float]:
    keys = (
        "cluster_count",
        "largest_cluster_size",
        "steps",
        "validated_clusters",
        "merged_clusters",
        "elapsed_time",
        "bp_iterations",
    )
    summary: Dict[str, float] = {}
    for key in keys:
        values = [s.get(key) for s in stats_list if s.get(key) is not None]
        if values:
            summary[key] = float(np.mean(values))
    return summary


def decode_bp_only(pcm, syndrome: np.ndarray, error_rate: float, n_bits: int) -> tuple[np.ndarray, float, Dict[str, Any]]:
    decoder = MGHDClusteredDecoder(
        pcm,
        n_bits=n_bits,
        mghd_adapter=None,
        error_rate=error_rate,
        bp_method="minimum_sum",
        max_iter=8,
        schedule="serial",
        lsd_method="LSD_E",
        lsd_order=0,
        bits_per_step=n_bits,
    )
    t0 = time.perf_counter()
    recovery = decoder.decode(syndrome, features=None, use_mghd_priors=False)
    dt_ms = (time.perf_counter() - t0) * 1e3
    return recovery, dt_ms, decoder.get_stats()


def decode_lsd(pcm, syndrome: np.ndarray, error_rate: float, n_bits: int, bits_per_step: int) -> tuple[np.ndarray, float, Dict[str, Any]]:
    decoder = MGHDClusteredDecoder(
        pcm,
        n_bits=n_bits,
        mghd_adapter=None,
        error_rate=error_rate,
        bp_method="minimum_sum",
        max_iter=1,
        schedule="serial",
        lsd_method="LSD_E",
        lsd_order=0,
        bits_per_step=bits_per_step,
    )
    t0 = time.perf_counter()
    recovery = decoder.decode(syndrome, features=None, use_mghd_priors=False)
    dt_ms = (time.perf_counter() - t0) * 1e3
    return recovery, dt_ms, decoder.get_stats()


def decode_mghd_guided(
    pcm,
    syndrome: np.ndarray,
    error_rate: float,
    n_bits: int,
    bits_per_step: int,
    mghd_model,
    feature_builder: Callable[[np.ndarray], Any],
) -> tuple[np.ndarray, float, Dict[str, Any]]:
    features = feature_builder(syndrome)
    probs = mghd_forward_probs(mghd_model, features).detach().cpu().numpy()

    adapter = MGHDAdapter(n_bits=n_bits, model=None)
    adapter.predict_error_probs = lambda _: probs  # type: ignore[assignment]

    decoder = MGHDClusteredDecoder(
        pcm,
        n_bits=n_bits,
        mghd_adapter=adapter,
        error_rate=error_rate,
        bp_method="minimum_sum",
        max_iter=1,
        schedule="serial",
        lsd_method="LSD_E",
        lsd_order=0,
        bits_per_step=bits_per_step,
    )
    t0 = time.perf_counter()
    recovery = decoder.decode(syndrome, features=features, use_mghd_priors=True)
    dt_ms = (time.perf_counter() - t0) * 1e3
    return recovery, dt_ms, decoder.get_stats()


def decode_mghd_end_to_end(
    pcm,
    syndrome: np.ndarray,
    feature_builder: Callable[[np.ndarray], Any],
    mghd_model,
) -> tuple[np.ndarray, float, Dict[str, Any]]:
    import torch

    features = feature_builder(syndrome)

    args, kwargs = _coerce_args_kwargs(features)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = mghd_model(*args, **kwargs)
        if logits.ndim > 1:
            logits = logits.squeeze(0)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).to(torch.uint8).cpu().numpy()
    dt_ms = (time.perf_counter() - t0) * 1e3

    return preds.astype(np.uint8, copy=False), dt_ms, {"bp_converged": None}


def _coerce_args_kwargs(features: Any | None) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if features is None:
        return tuple(), {}
    if isinstance(features, dict):
        args = features.get("args", ())
        kwargs = dict(features.get("kwargs", {}))
        if not isinstance(args, (tuple, list)):
            args = (args,)
        else:
            args = tuple(args)
        return args, kwargs
    if isinstance(features, (tuple, list)):
        return tuple(features), {}
    return (features,), {}


__all__ = [
    "decode_bp_only",
    "decode_lsd",
    "decode_mghd_guided",
    "decode_mghd_end_to_end",
    "latency_stats",
    "wilson_ci",
]
