#!/usr/bin/env python
"""
MGHD training entrypoint (teacher-supervised). CUDA-Q trajectories are the default sampler.

Examples:
  # Surface code d=3..31, CUDA-Q sampler, 64 shots per batch, 10 batches per distance
  python -m tools.train_core --family surface --distances 3-31:2 --sampler cudaq \
      --shots-per-batch 64 --batches 10

Notes:
  - Teachers:
      * MWPF (hypergraph) takes detector streams; Python API used directly (no DEM).
        (See mwpf README / examples.)  [Ref]
      * LSD (BP+LSD) runs per-basis on Hx/Hz.                           [Ref]
      * MWPM fallback uses PyMatching v2 decode_batch from H.           [Ref]
  - CUDA-Q trajectories simulate general Kraus/coherent noise at circuit level;
    we keep this as the gold training path vs Pauli-only DEM approximations.   [Ref]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from samplers import get_sampler
from teachers.mix import MixConfig, TeacherMix

from .code_loader import load_code
from .curriculum import parse_distances


def _resolve_syndromes(code_obj, dets):
    """
    BEST-EFFORT mapping from det-streams to CSS syndromes for LSD/MWPM teachers.
    If code_obj exposes a helper, use it. Else return zeros with correct shapes.
    """

    mx = getattr(code_obj, "Hx", None)
    mz = getattr(code_obj, "Hz", None)
    B = dets.shape[0]
    if hasattr(code_obj, "detectors_to_syndromes"):
        sx, sz = code_obj.detectors_to_syndromes(dets)
        return sx.astype(np.uint8), sz.astype(np.uint8)
    if mx is not None and getattr(mx, "ndim", 2) == 2:
        sx = np.zeros((B, mx.shape[0]), dtype=np.uint8)
    else:
        sx = np.zeros((B, 0), dtype=np.uint8)
    if mz is not None and getattr(mz, "ndim", 2) == 2:
        sz = np.zeros((B, mz.shape[0]), dtype=np.uint8)
    else:
        sz = np.zeros((B, 0), dtype=np.uint8)
    return sx, sz


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--family",
        default="surface",
        help="code family id in codes_registry (e.g., surface, bb, rm, steane, repetition, color)",
    )
    p.add_argument("--distances", default="3-31:2", help="e.g., '3,5,7' or '3-31:2'")
    p.add_argument("--sampler", default="cudaq", choices=["cudaq", "stim"])
    p.add_argument("--shots-per-batch", type=int, default=128)
    p.add_argument("--batches", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--p-mwpf", type=float, default=0.5)
    p.add_argument("--p-lsd", type=float, default=0.4)
    p.add_argument("--p-mwpm", type=float, default=0.1)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    distances = parse_distances(args.distances)

    # Sampler (CUDA-Q is the default; actual CLN via trajectories)  [Ref CUDA-Q]
    sampler = get_sampler(args.sampler)

    for d in distances:
        print(f"\n=== Family={args.family}  d={d}  sampler={args.sampler} ===")
        # Load code object (Hx/Hz, detector metadata, hypergraph mapping if available)
        code = load_code(args.family, d)
        Hx = getattr(code, "Hx", None)
        Hz = getattr(code, "Hz", None)

        # Teacher stack
        mix = TeacherMix(
            code,
            Hx,
            Hz,
            mix_cfg=MixConfig(
                p_mwpf=args.p_mwpf,
                p_lsd=args.p_lsd,
                p_mwpm=args.p_mwpm,
            ),
        )

        totals = {"mwpf": 0, "lsd": 0, "mwpm": 0, "mwpm_fallback": 0}
        t0 = time.time()
        for _ in range(args.batches):
            batch = sampler.sample(
                code,
                n_shots=args.shots_per_batch,
                seed=int(rng.integers(1 << 32) - 1),
            )
            sx, sz = _resolve_syndromes(code, batch.dets)
            out = mix.route_batch(
                dets=batch.dets,
                syndromes_x=sx,
                syndromes_z=sz,
                rng=rng,
            )
            totals[out["which"]] = totals.get(out["which"], 0) + 1

        dt = time.time() - t0
        print(
            f"[done] batches={args.batches} shots/batch={args.shots_per_batch} "
            f"teacher-usage={totals} elapsed={dt:.2f}s"
        )
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
