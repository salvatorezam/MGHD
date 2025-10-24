"""Minimal MGHD v2 sweep benchmark."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from mghd.qpu.adapters.garnet_adapter import sample_round
from mghd.codes.pcm_real import rotated_surface_pcm
from mghd.decoders.lsd.clustered_primary import MGHDPrimaryClustered
from mghd.core.infer import MGHDDecoderPublic


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def wilson_ci(failures: int, shots: int, z: float = 1.96) -> Tuple[float, float, float]:
    if shots == 0:
        return 0.0, 0.0, 0.0
    p = failures / shots
    denom = 1.0 + (z * z) / shots
    center = (p + (z * z) / (2.0 * shots)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * shots)) / shots)) / denom
    return p, max(center - half, 0.0), min(center + half, 1.0)


def summarize(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    vec = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(vec)),
        "p50": float(np.percentile(vec, 50)),
        "p95": float(np.percentile(vec, 95)),
        "p99": float(np.percentile(vec, 99)),
    }


def aggregate_mb(reports: Sequence[Dict[str, object]]) -> Dict[str, object]:
    agg = {
        "fast_path_batches": 0,
        "fixed_d3_batches": 0,
        "fallback_loops": 0,
        "graph_used": False,
        "graph_used_shots": 0,
        "batch_sizes": [],
        "bucket_histogram": {},
        "device": None,
    }
    for rep in reports:
        if not rep:
            continue
        agg["fast_path_batches"] += int(rep.get("fast_path_batches", 0))
        agg["fixed_d3_batches"] += int(rep.get("fixed_d3_batches", 0))
        agg["fallback_loops"] += int(rep.get("fallback_loops", 0))
        agg["graph_used"] = agg["graph_used"] or bool(rep.get("graph_used", False))
        agg["graph_used_shots"] += int(rep.get("graph_used_shots", 0))
        agg["batch_sizes"].extend(rep.get("batch_sizes", []))
        bucket_hist = rep.get("bucket_histogram", {})
        for key, value in bucket_hist.items():
            agg["bucket_histogram"][key] = agg["bucket_histogram"].get(key, 0) + int(value)
        if agg["device"] is None and rep.get("device"):
            agg["device"] = rep["device"]
    if not agg["bucket_histogram"]:
        agg.pop("bucket_histogram")
    return agg


def resolve_tier0_limits(mode: str | None) -> Tuple[int, int]:
    mapping = {
        "mixed": (2, 1),
        "mixed_tight": (1, 0),
        "aggressive": (5, 5),
        "off": (0, 0),
        None: (2, 1),
    }
    return mapping.get(mode, (2, 1))


def parity_check(H: sp.csr_matrix, e_hat: np.ndarray, target: np.ndarray) -> bool:
    syndrome = (H @ (e_hat % 2)).A.ravel() % 2  # convert to dense array
    return np.array_equal(syndrome.astype(np.uint8), target.astype(np.uint8) % 2)


# ---------------------------------------------------------------------------
# Core benchmarking logic
# ---------------------------------------------------------------------------

def run_distance(
    d: int,
    p: float,
    decoder_pub: MGHDDecoderPublic,
    Hx: sp.csr_matrix,
    Hz: sp.csr_matrix,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> Dict[str, object]:
    tier0_k = args.tier0_k_max if args.tier0_k_max is not None else resolve_tier0_limits(args.tier0_mode)[0]
    tier0_r = args.tier0_r_max if args.tier0_r_max is not None else resolve_tier0_limits(args.tier0_mode)[1]

    side_decoders = {
        "Z": MGHDPrimaryClustered(
            Hz,
            decoder_pub,
            halo=args.halo,
            thresh=args.thresh,
            temp=args.temp,
            r_cap=args.r_cap,
            batched=True,
            tier0_enable=args.tier0,
            tier0_k_max=tier0_k,
            tier0_r_max=tier0_r,
            default_p=p,
        ),
        "X": MGHDPrimaryClustered(
            Hx,
            decoder_pub,
            halo=args.halo,
            thresh=args.thresh,
            temp=args.temp,
            r_cap=args.r_cap,
            batched=True,
            tier0_enable=args.tier0,
            tier0_k_max=tier0_k,
            tier0_r_max=tier0_r,
            default_p=p,
        ),
    }

    side_stats: Dict[str, Dict[str, object]] = {}
    total_failures = 0
    total_shots = 0

    for side, decoder in side_decoders.items():
        failures = 0
        lat_total: List[float] = []
        lat_mghd: List[float] = []
        mb_reports: List[Dict[str, object]] = []

        for _ in range(args.shots):
            seed = int(rng.integers(0, 2**32 - 1))
            sample = sample_round(d=d, p=p, seed=seed)
            syn = sample["synZ"] if side == "Z" else sample["synX"]
            syn = np.asarray(syn, dtype=np.uint8)

            out = decoder.decode(syn, perf_only=False)
            if args.enforce_mghd and out.get("mghd_clusters", 0) == 0:
                raise RuntimeError("MGHD was not invoked; rerun with relaxed settings or disable --enforce-mghd")

            e_hat = np.asarray(out["e_hat"], dtype=np.uint8)
            H = Hz if side == "Z" else Hx
            success = parity_check(H, e_hat, syn)
            if not success or (args.inject_ler_rate > 0.0 and rng.random() < args.inject_ler_rate):
                failures += 1

            lat_total.append(float(out.get("t_total_us", 0.0)))
            lat_mghd.append(float(out.get("t_mghd_us", 0.0)))
            mb_reports.append(out.get("mb_stats", {}))

        side_stats[side] = {
            "shots": args.shots,
            "failures": failures,
            "ler": failures / args.shots if args.shots else 0.0,
            "ler_ci": wilson_ci(failures, args.shots),
            "latency_total_us": summarize(lat_total),
            "latency_mghd_us": summarize([v for v in lat_mghd if v > 0.0]),
            "mb_stats": aggregate_mb(mb_reports),
        }
        total_failures += failures
        total_shots += args.shots

    return {
        "shots": total_shots,
        "failures": total_failures,
        "ler": total_failures / total_shots if total_shots else 0.0,
        "ler_ci": wilson_ci(total_failures, total_shots),
        "sides": side_stats,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Benchmark MGHD v2 clustered decoder")
    ap.add_argument("--ckpt", required=True, help="Path to MGHD v2 checkpoint")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dists", nargs="+", type=int, required=True)
    ap.add_argument("--ps", nargs="+", type=float, required=True)
    ap.add_argument("--shots", type=int, required=True)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--tier0", action="store_true", dest="tier0", default=True)
    ap.add_argument("--no-tier0", action="store_false", dest="tier0")
    ap.add_argument("--tier0-mode", choices=["mixed", "mixed_tight", "aggressive", "off"], default="mixed")
    ap.add_argument("--tier0-k-max", type=int)
    ap.add_argument("--tier0-r-max", type=int)
    ap.add_argument("--halo", type=int, default=1)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--r-cap", type=int, default=20)
    ap.add_argument("--enforce-mghd", action="store_true")
    ap.add_argument("--inject-ler-rate", type=float, default=0.0)
    ap.add_argument("--out", required=True, help="Path to JSON results")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    decoder_pub = MGHDDecoderPublic(args.ckpt, device=args.device)

    results: Dict[str, Dict[str, object]] = {}
    for d in args.dists:
        Hx = rotated_surface_pcm(d, "X")
        Hz = rotated_surface_pcm(d, "Z")
        decoder_pub.bind_code(Hx, Hz)
        results[str(d)] = {}
        for p in args.ps:
            stats = run_distance(d, p, decoder_pub, Hx, Hz, rng, args)
            results[str(d)][f"{p:.3f}"] = stats

    payload = {"config": {
        "ckpt": args.ckpt,
        "device": args.device,
        "dists": args.dists,
        "ps": args.ps,
        "shots": args.shots,
        "seed": args.seed,
        "tier0": args.tier0,
        "tier0_mode": args.tier0_mode,
        "tier0_k_max": args.tier0_k_max,
        "tier0_r_max": args.tier0_r_max,
        "halo": args.halo,
        "thresh": args.thresh,
        "temp": args.temp,
        "r_cap": args.r_cap,
        "enforce_mghd": args.enforce_mghd,
        "inject_ler_rate": args.inject_ler_rate,
    }, "results": results}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()
