"""Pre-cache hard surface-code syndromes that meet specified cluster filters."""

from __future__ import annotations

import argparse
import os
import time
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp

from mghd_clustered.pcm_real import rotated_surface_pcm
from mghd_clustered.cluster_core import (
    active_components,
    extract_subproblem,
    gf2_nullspace,
)
from tools.bench_clustered_sweep_surface import ensure_dir


def sample_bsc(H: sp.csr_matrix, p: float, rng: np.random.Generator) -> np.ndarray:
    """Sample a full-code syndrome under a binary symmetric channel."""
    n = H.shape[1]
    e = (rng.random(n) < p).astype(np.uint8)
    s = (H @ e) % 2
    return np.asarray(s, dtype=np.uint8)


def _format_prob_tag(d: int, p: float, side: str) -> str:
    return f"d{d}_p{p:.6f}_{side.upper()}"


def _collect_slice(
    *,
    H: sp.csr_matrix,
    d: int,
    p: float,
    side: str,
    rng: np.random.Generator,
    min_nullity: int | None,
    min_size: int | None,
    target_mghd: int,
    time_budget_min: float | None,
    halo: int,
) -> Tuple[np.ndarray, dict]:
    """Collect accepted syndromes for a single (d,p,side)."""

    accepted: List[np.ndarray] = []
    attempts = 0
    start = time.perf_counter()

    def _elapsed_min() -> float:
        return (time.perf_counter() - start) / 60.0

    while len(accepted) < target_mghd:
        if time_budget_min is not None and _elapsed_min() >= time_budget_min:
            break
        s = sample_bsc(H, p, rng)
        attempts += 1
        checks_list, qubits_list = active_components(H, s, halo=halo)
        sizes: List[int] = []
        nullities: List[int] = []
        for ci, qi in zip(checks_list, qubits_list):
            H_sub, s_sub, _, _ = extract_subproblem(H, s, ci, qi)
            sizes.append(int(qi.size))
            nullities.append(int(gf2_nullspace(H_sub).shape[1]))
        passes_size = True if min_size is None else any(val >= min_size for val in sizes)
        passes_nullity = True if min_nullity is None else any(val >= min_nullity for val in nullities)
        if passes_size and passes_nullity:
            accepted.append(np.array(s, copy=True))
    meta = dict(attempts=attempts, collected=len(accepted), min_nullity=min_nullity, min_size=min_size)
    return (np.stack(accepted, axis=0) if accepted else np.empty((0, H.shape[0]), dtype=np.uint8), meta)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-cache hard crops for perf-only MGHD runs")
    ap.add_argument("--dists", type=int, nargs="+", required=True)
    ap.add_argument("--ps", type=float, nargs="+", required=True)
    ap.add_argument("--min-nullity", type=int, default=None)
    ap.add_argument("--min-size", type=int, default=None)
    ap.add_argument("--target-mghd", type=int, default=2000)
    ap.add_argument("--time-budget-min", type=float, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--halo", type=int, default=0)
    args = ap.parse_args()

    ensure_dir(args.out)
    rng_global = np.random.default_rng(args.seed)

    for d in args.dists:
        print(f"=== Pre-caching d={d} ===")
        Hx = rotated_surface_pcm(d, "X")
        Hz = rotated_surface_pcm(d, "Z")
        for p in args.ps:
            print(f"  p={p:.3f}")
            for side, H in [("X", Hx), ("Z", Hz)]:
                rng = np.random.default_rng(rng_global.integers(0, 2**32 - 1))
                syndromes, meta = _collect_slice(
                    H=H,
                    d=d,
                    p=p,
                    side=side,
                    rng=rng,
                    min_nullity=args.min_nullity,
                    min_size=args.min_size,
                    target_mghd=args.target_mghd,
                    time_budget_min=args.time_budget_min,
                    halo=args.halo,
                )
                tag = _format_prob_tag(d, p, side)
                path = os.path.join(args.out, f"{tag}.npz")
                np.savez_compressed(
                    path,
                    syndromes=syndromes,
                    metadata=dict(
                        d=d,
                        p=p,
                        side=side,
                        min_nullity=args.min_nullity,
                        min_size=args.min_size,
                        target_mghd=args.target_mghd,
                        time_budget_min=args.time_budget_min,
                        attempts=meta["attempts"],
                        collected=meta["collected"],
                    ),
                )
                print(
                    f"    {side}: cached {meta['collected']} crops (attempts={meta['attempts']}) -> {path}"
                )


if __name__ == "__main__":
    main()
