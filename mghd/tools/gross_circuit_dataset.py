#!/usr/bin/env python
"""
Generate circuit-level gross BB crops using Relay-BP testdata circuits.

Workflow:
1) Load the 12+1 gross memory circuit (Z basis) from Relay-BP testdata.
2) Strip baked-in noise, re-insert standard circuit depolarizing noise at p
   via `insert_uniform_academic_circuit_noise`.
3) Build DEM, extract detector×error matrix.
4) Sample the DEM with `return_errors=True` to get detector bits, logical obs,
   and per-shot error vectors (length = num_errors).
5) Treat DEM error variables as “qubits” for MGHD:
      H_sub := check_matrix (detectors × errors)
      synd_Z_then_X_bits := detector sample
      y_bits_local := error vector
      side := "Z"
6) Pack crops with `pack_cluster` and save as .npz shards compatible with
   `mghd/cli/train.py --data-root`.

Usage:
  PYTHONPATH="/u/home/kulp/MGHD-data/external-projects/relay-main/tests:$PYTHONPATH" \\
  python -m mghd.tools.gross_circuit_dataset \\
    --out data/gross_circuit_crops \\
    --shots 1000 \\
    --p-values 0.0005,0.0010,0.0015,0.0020,0.0025,0.0030
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from relay_bp.stim.noise import insert_uniform_academic_circuit_noise
import beliefmatching as bm
import stim

from mghd.core.core import pack_cluster


def _import_testdata(relay_tests_path: Path):
    import sys

    if str(relay_tests_path) not in sys.path:
        sys.path.insert(0, str(relay_tests_path))
    from testdata import get_test_circuit, filter_detectors_by_basis  # type: ignore

    return get_test_circuit, filter_detectors_by_basis


def _pack_shot(
    H: np.ndarray,
    dets: np.ndarray,
    err: np.ndarray,
    *,
    p: float,
    seed: int,
    d: int,
    n_errors: int,
    priors_sum: float,
    N_max: int = 4096,
    E_max: int = 20000,
    S_max: int = 4096,
) -> dict:
    """Pack one shot into a dict with packed tensors."""
    nC, nQ = H.shape  # detectors, errors-as-qubits
    # Simple line coords: data (errors) on x-axis, checks on x-axis offset
    coords_q = np.stack([np.arange(nQ), np.zeros(nQ)], axis=1).astype(np.float32)
    coords_c = np.stack([np.arange(nC), np.ones(nC)], axis=1).astype(np.float32)
    bbox = (0.0, 0.0, float(max(nQ, nC)), 1.0)

    # Debug parity check: H @ err mod 2 should equal dets
    s_pred = (H @ err.astype(np.uint8)) & 1
    if not np.array_equal(s_pred, dets.astype(np.uint8)):
        raise ValueError("Parity check failed: H @ err != dets")

    packed = pack_cluster(
        H_sub=H,
        xy_qubit=coords_q,
        xy_check=coords_c,
        synd_Z_then_X_bits=dets,
        k=nQ,  # use nQ/nC as meta
        r=nC,
        bbox_xywh=bbox,
        kappa_stats={},
        y_bits_local=err,
        side="Z",
        d=d,
        p=p,
        seed=seed,
        N_max=max(N_max, nQ + nC),
        E_max=max(E_max, int(H.sum()) * 4),
        S_max=max(S_max, nC),
        add_jump_edges=False,
    )
    # Attach useful meta for downstream validation
    packed.meta.n_errors = int(n_errors)
    packed.meta.error_weight = int(err.sum())
    packed.meta.priors_sum = float(priors_sum)
    packed.meta.stim_det_bits = dets.astype(np.uint8)
    return packed


def _load_base_circuit(relay_tests_path: Path, distance: int) -> stim.Circuit:
    """Load a gross Z-basis circuit from testdata, ignoring baked error_rate."""
    get_test_circuit, filter_detectors_by_basis = _import_testdata(relay_tests_path)
    # First try with explicit 0.0, fall back to wildcard and pick first.
    try:
        base = get_test_circuit(
            circuit="bicycle_bivariate_144_12_12_memory_Z",
            distance=distance,
            rounds=12,
            error_rate=0.0,
        )
    except Exception:
        from testdata import get_all_test_circuits  # type: ignore

        all_circs = get_all_test_circuits(
            circuit="bicycle_bivariate_144_12_12_memory_Z",
            distance=distance,
            rounds=12,
            error_rate="*",
        )
        if not all_circs:
            raise
        # pick the first deterministically
        base = next(iter(all_circs.values()))
    base = filter_detectors_by_basis(base, "Z")
    return base


def generate_for_p(
    relay_tests_path: Path,
    p: float,
    shots: int,
    distance: int = 12,
) -> tuple[np.ndarray, list[object], dict]:
    """Generate packed crops for a single p."""
    base = _load_base_circuit(relay_tests_path, distance)
    noisy = insert_uniform_academic_circuit_noise(base.without_noise(), p)

    dem = noisy.detector_error_model(decompose_errors=False, ignore_decomposition_failures=True)
    dm = bm.detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True)
    H = dm.check_matrix.astype(np.uint8).toarray()  # shape (num_det, num_errors)
    n_errors = H.shape[1]
    priors_sum = float(dm.priors.sum())

    sampler = dem.compile_sampler()
    dets, obs, err = sampler.sample(shots=shots, return_errors=True, bit_packed=False)

    crops = []
    rng = np.random.default_rng(1234)
    for b in range(shots):
        seed = int(rng.integers(0, 2**31 - 1))
        crops.append(
            _pack_shot(
                H,
                dets[b],
                err[b],
                p=p,
                seed=seed,
                d=distance,
                n_errors=n_errors,
                priors_sum=priors_sum,
            )
        )

    meta = {
        "p": p,
        "distance": distance,
        "num_detectors": H.shape[0],
        "num_errors": n_errors,
        "shots": shots,
        "priors_sum": priors_sum,
    }
    return np.array(crops, dtype=object), meta


def save_shard(out_dir: Path, p: float, packed_crops: np.ndarray, meta: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / f"gross_d12_p{p:.6f}.npz"
    np.savez(shard_path, packed=packed_crops)
    return shard_path, meta


def main():
    parser = argparse.ArgumentParser(description="Generate circuit-level gross BB crops (DEM error variables).")
    parser.add_argument("--out", type=str, required=True, help="Output directory for shards/manifest")
    parser.add_argument("--shots", type=int, default=1000, help="Shots per p value")
    parser.add_argument(
        "--p-values",
        type=str,
        default="0.0005,0.0010,0.0015,0.0020,0.0025,0.0030",
        help="Comma-separated list of p values",
    )
    parser.add_argument(
        "--relay-testdata",
        type=str,
        default="/u/home/kulp/MGHD-data/external-projects/relay-main/tests",
        help="Path to relay-main/tests (contains testdata/ with circuits)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    relay_tests_path = Path(args.relay_testdata)
    p_vals = [float(x) for x in args.p_values.split(",") if x.strip()]

    manifest = []
    for p in p_vals:
        print(f"Generating p={p} shots={args.shots} ...", flush=True)
        packed, meta = generate_for_p(relay_tests_path, p, args.shots)
        shard_path, meta = save_shard(out_dir, p, packed, meta)
        manifest.append({"p": p, "path": shard_path.name, **meta})
        print(f"  -> {shard_path}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nDone. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
