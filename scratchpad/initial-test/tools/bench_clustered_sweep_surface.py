from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from mghd_public.config import MGHDConfig
from mghd_public.infer import MGHDDecoderPublic, warmup_and_capture
from mghd_clustered.clustered_primary import MGHDPrimaryClustered
from mghd_clustered.cluster_core import (
    active_components,
    extract_subproblem,
    gf2_nullspace,
)
from mghd_clustered.pcm_real import rotated_surface_pcm


def wilson_ci_upper(failures: int, shots: int, confidence: float = 0.95) -> float:
    """Compute Wilson confidence interval upper bound for binomial proportion."""
    if shots == 0:
        return 1.0
    
    # Use the wilson_ci function and return the upper bound
    _, _, upper = wilson_ci(failures, shots, z=1.96 if confidence == 0.95 else 2.576)
    return upper


def wilson_ci(k, n, z=1.96):
    """Wilson confidence interval for binomial proportion."""
    if n == 0: return (0.0, 0.0, 0.0)
    p = k/n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n))/denom
    half = (z*math.sqrt((p*(1-p) + z*z/(4*n))/n))/denom
    return (p, max(center-half,0.0), min(center+half,1.0))


def summarize(arr: List[float]) -> Dict[str, float]:
    vec = np.asarray(arr, dtype=np.float64)
    if vec.size == 0:
        return dict(mean=0.0, p50=0.0, p95=0.0, p99=0.0, count_nonzero=0, mean_nonzero=0.0)
    
    # Fix mean_nonzero calculation
    nz = vec[vec > 0]
    mean_nonzero = float(nz.mean()) if nz.size else 0.0
    count_nonzero = int(nz.size)
    
    return dict(
        mean=float(np.mean(vec)),
        p50=float(np.percentile(vec, 50)),
        p95=float(np.percentile(vec, 95)),
        p99=float(np.percentile(vec, 99)),
        count_nonzero=count_nonzero,
        mean_nonzero=mean_nonzero,
    )


def _bucket_size(size: int) -> str:
    bucket = 1
    while size > bucket:
        bucket *= 2
    return str(bucket)


def _parse_bucket_spec(spec: str) -> List[Tuple[int, int, int]]:
    spec = (spec or "").strip()
    if not spec:
        return []
    buckets: List[Tuple[int, int, int]] = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        fields = [int(x) for x in part.split(",")]
        if len(fields) != 3:
            raise ValueError(f"Invalid bucket spec fragment '{part}'")
        buckets.append(tuple(fields))
    buckets.sort()
    return buckets


def _aggregate_mb_stats(reports):
    agg = dict(
        shots=len(reports),
        fast_path_batches=0,
        fixed_d3_batches=0,
        fallback_loops=0,
        graph_used_shots=0,
    )
    hist: Dict[str, int] = {}
    total = 0.0
    count = 0
    device = None
    graph_flag = False
    for rep in reports:
        if not rep:
            continue
        agg["fast_path_batches"] += int(rep.get("fast_path_batches", 0))
        agg["fixed_d3_batches"] += int(rep.get("fixed_d3_batches", 0))
        agg["fallback_loops"] += int(rep.get("fallback_loops", 0))
        graph_used_shots = int(rep.get("graph_used_shots", 0))
        agg["graph_used_shots"] += graph_used_shots
        graph_flag = graph_flag or bool(rep.get("graph_used")) or graph_used_shots > 0
        sizes = rep.get("batch_sizes", [])
        total += float(np.sum(sizes)) if len(sizes) else 0.0
        count += len(sizes)
        for size in sizes:
            bucket = _bucket_size(int(size))
            hist[bucket] = hist.get(bucket, 0) + 1
        if device is None and rep.get("device"):
            device = rep["device"]
    agg["avg_batch_size"] = (total / count) if count else 0.0
    agg["batch_histogram"] = {k: int(v) for k, v in sorted(hist.items(), key=lambda kv: int(kv[0]))}
    agg["graph_used"] = graph_flag
    agg["device"] = device or {}
    return agg


def sample_bsc(H: sp.csr_matrix, p: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    n = H.shape[1]
    e = (rng.random(n) < p).astype(np.uint8)
    s = (H @ e) % 2
    return e, np.asarray(s, dtype=np.uint8)


def cluster_stats_from_checks(
    H: sp.csr_matrix,
    s: np.ndarray,
) -> Tuple[int, List[int], Counter, Counter]:
    checks_list, qubits_list = active_components(H, s, halo=0)
    cluster_sizes = []
    size_hist: Counter = Counter()
    nullity_hist: Counter = Counter()
    for ci, qi in zip(checks_list, qubits_list):
        cluster_sizes.append(int(qi.size))
        size_key = str(cluster_sizes[-1]) if cluster_sizes[-1] <= 8 else "9+"
        size_hist[size_key] += 1
        H_sub, s_sub, _, _ = extract_subproblem(H, s, ci, qi)
        nullity = gf2_nullspace(H_sub).shape[1]
        null_key = str(nullity) if nullity <= 8 else "9+"
        nullity_hist[null_key] += 1
    return len(checks_list), cluster_sizes, size_hist, nullity_hist


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--dists", type=int, nargs="+", default=[3, 5, 9, 11])
    ap.add_argument("--ps", type=float, nargs="+", default=[0.002, 0.003, 0.005, 0.007, 0.010, 0.015])
    
    # Tier-0 configuration with modes
    ap.add_argument("--tier0-mode", choices=["aggressive", "mixed", "mixed_tight", "off"], default=None,
                    help="Preset tier-0 configurations: aggressive (k=15,r=20), mixed (k=5,r=6), mixed_tight (k=1,r=0), off (disabled)")
    ap.add_argument("--tier0", dest="tier0", action="store_true", default=True)
    ap.add_argument("--no-tier0", dest="tier0", action="store_false")
    ap.add_argument("--tier0-k-max", type=int, default=3)
    ap.add_argument("--tier0-r-max", type=int, default=6)
    
    # Channel parameters with auto option
    ap.add_argument("--p-channel", default="auto",
                    help="Channel p for Tier-0 solver: float value or 'auto' to match sweep p")
    ap.add_argument("--halo", type=int, default=0)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--r-cap", type=int, default=20)
    
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--graph-capture", action="store_true", default=True)
    ap.add_argument("--no-graph-capture", dest="graph_capture", action="store_false")
    
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/sweeps")
    ap.add_argument("--min-nullity", type=int, default=None,
                    help="Resample until at least one cluster has nullity ≥ value (up to limits)")
    ap.add_argument("--min-size", type=int, default=None,
                    help="Resample until at least one cluster has ≥ value qubits (up to limits)")
    
    # Expert model selection
    ap.add_argument("--expert", type=str, default="v1", choices=["v1","v2","auto"], 
                    help="Use distance-agnostic v2 path (auto = auto-detect)")
    ap.add_argument("--log-kappa-hist", action="store_true", 
                    help="Log cluster size (kappa) histograms")
    ap.add_argument("--log-nullity-hist", action="store_true",
                    help="Log nullity (r) histograms")
    ap.add_argument("--out-json", type=str, default=None,
                    help="Output results to JSON file")
    
    # LER sanity injection for validation
    ap.add_argument("--inject-ler", "--inject-logical-rate", action="store_true",
                    help="Inject logical errors at specified rate for sanity testing")
    ap.add_argument("--inject-ler-rate", type=float, default=0.1,
                    help="Rate of logical error injection when --inject-ler is enabled")
    ap.add_argument("--bucket-spec", type=str, default="32,64,32;64,128,64",
                    help="Semicolon-separated bucket ceilings (nodes,edges,seq) for v2 batching.")
    ap.add_argument("--microbatch", type=int, default=64,
                    help="Target microbatch size for cross-shot batching.")
    ap.add_argument("--flush-ms", type=float, default=1.0,
                    help="Flush partial microbatch after this many milliseconds of waiting.")
    ap.add_argument("--target-ler-ci", type=float, default=1e-4,
                    help="Early-stop a (d,p,side) once the 95% Wilson upper bound is below this value.")

    # CI enforcement
    ap.add_argument("--enforce-mghd", action="store_true",
                    help="Fail if MGHD was never invoked in mixed/mixed_tight modes")
    
    args = ap.parse_args()

    bucket_spec = _parse_bucket_spec(args.bucket_spec)
    args.microbatch = max(1, int(args.microbatch))
    args.flush_ms = max(0.0, float(args.flush_ms))
    if args.target_ler_ci is not None and args.target_ler_ci < 0:
        args.target_ler_ci = None

    # Process tier-0 mode presets
    if args.tier0_mode == "aggressive":
        args.tier0 = True
        args.tier0_k_max = 15
        args.tier0_r_max = 20
    elif args.tier0_mode == "mixed":
        args.tier0 = True
        args.tier0_k_max = 2
        args.tier0_r_max = 1
    elif args.tier0_mode == "off":
        args.tier0 = False

    # Process p_channel auto
    if args.p_channel != "auto":
        try:
            args.p_channel = float(args.p_channel)
        except ValueError:
            raise ValueError(f"Invalid p_channel value: {args.p_channel}. Use 'auto' or a float.")

    ensure_dir(args.out)

    cfg = MGHDConfig(
        gnn={
            "dist": 3,
            "n_node_inputs": 9,
            "n_node_outputs": 9,
            "n_iters": 7,
            "n_node_features": 128,
            "n_edge_features": 128,
            "msg_net_size": 96,
            "msg_net_dropout_p": 0.04,
            "gru_dropout_p": 0.11,
        },
        mamba={
            "d_model": 192,
            "d_state": 32,
            "d_conv": 2,
            "expand": 3,
            "attention_mechanism": "channel_attention",
            "se_reduction": 4,
            "post_mamba_ln": False,
        },
        n_checks=8,
        n_qubits=9,
        n_node_inputs=9,
        n_node_outputs=2,
    )

    results: Dict[str, Dict[str, Dict[str, Dict[str, object]]]] = {"grid": [], "meta": {}}
    results["meta"]["expert"] = args.expert
    results["meta"]["graph_capture"] = bool(getattr(args, "graph_capture", False))
    results["meta"]["min_nullity"] = args.min_nullity
    results["meta"]["min_size"] = args.min_size
    results["meta"]["bucket_spec"] = [list(b) for b in bucket_spec]
    results["meta"]["microbatch"] = args.microbatch
    results["meta"]["flush_ms"] = args.flush_ms
    results["meta"]["target_ler_ci"] = args.target_ler_ci
    base_rng = np.random.default_rng(args.seed)

    # Create MGHD decoder with expert selection
    if args.expert == "v2":
        # For v2, we'll create a compatible decoder that can handle distance-agnostic crops
        print(f"Using distance-agnostic expert v2")
        dec_pub = MGHDDecoderPublic(
            args.ckpt,
            cfg,
            device=args.device,
            expert="v2",
            graph_capture=args.graph_capture,
        )
    elif args.expert == "auto":
        # Auto-detect v1 vs v2 from checkpoint
        print(f"Using auto-detection for expert model")
        dec_pub = MGHDDecoderPublic(
            args.ckpt,
            cfg,
            device=args.device,
            expert="auto",
            graph_capture=args.graph_capture,
        )
    else:
        # Standard v1 path
        dec_pub = MGHDDecoderPublic(
            args.ckpt,
            cfg,
            device=args.device,
            expert="v1",
            graph_capture=args.graph_capture,
        )
    
    # Only bind d=3 matrices - for other distances, MGHD won't be used
    Hx_d3 = rotated_surface_pcm(3, "X")
    Hz_d3 = rotated_surface_pcm(3, "Z")
    dec_pub.bind_code(Hx_d3, Hz_d3)

    for d in args.dists:
        print(f"=== Distance d={d} ===")
        Hx = rotated_surface_pcm(d, "X")
        Hz = rotated_surface_pcm(d, "Z")

        results[str(d)] = {}

        for p in args.ps:
            print(f"  p={p:.3f}")
            if d != 3:
                print(f"    Note: Using tier0_k_max={min(15, d * 3)}, tier0_r_max={min(20, d * 4)} for d={d}")
            results[str(d)][f"{p:.3f}"] = {}
            rng_side = np.random.default_rng(base_rng.integers(0, 2**32 - 1))

            for side, H in [("X", Hx), ("Z", Hz)]:
                rng = np.random.default_rng(rng_side.integers(0, 2**32 - 1))
                
                # Warmup MGHD with CUDA graph capture for this side (only for d=3)
                warmup_info = None
                if d == 3:
                    try:
                        warmup_info = warmup_and_capture(dec_pub, args.device, side, use_fixed_d3=True)
                        print(f"    {side}: Warmup {warmup_info['warmup_us']:.1f}μs, graph={warmup_info['graph_used']}, path={warmup_info['path']}")
                    except Exception as e:
                        print(f"    {side}: Warmup failed: {e}")
                        warmup_info = {'warmup_us': 0.0, 'graph_used': False, 'path': 'failed'}
                
                # Configure tier-0 limits based on mode and distance
                # Start with mode defaults or CLI args
                tier0_k_max = args.tier0_k_max
                tier0_r_max = args.tier0_r_max
                
                # Only override with mode presets if using defaults and no explicit CLI override
                cli_k_explicit = "--tier0-k-max" in ' '.join(sys.argv)
                cli_r_explicit = "--tier0-r-max" in ' '.join(sys.argv)
                
                if not cli_k_explicit and not cli_r_explicit:
                    # Use mode presets when no explicit CLI overrides
                    if args.tier0_mode == "aggressive":
                        tier0_k_max = 15
                        tier0_r_max = 20
                    elif args.tier0_mode == "mixed":
                        tier0_k_max = 5
                        tier0_r_max = 6
                    elif args.tier0_mode == "mixed_tight":
                        if d == 3:
                            tier0_k_max = 1
                            tier0_r_max = 0
                        else:
                            tier0_k_max = 15
                            tier0_r_max = 20
                    elif args.tier0_mode == "off":
                        if d == 3:
                            tier0_k_max = 0
                            tier0_r_max = 0
                        else:
                            tier0_k_max = 15
                            tier0_r_max = 20
                    elif args.tier0_mode is None:
                        # Default behavior for no mode: scale for larger distances
                        if d != 3:
                            tier0_k_max = min(15, d * 3)
                            tier0_r_max = min(20, d * 4)
                
                # Configure p_channel
                if args.p_channel == "auto":
                    p_channel = p  # Use the current sweep p value
                else:
                    p_channel = args.p_channel
                
                if d != 3 and args.tier0_mode is None:
                    print(f"    Note: Using tier0_k_max={tier0_k_max}, tier0_r_max={tier0_r_max} for d={d}")
                
                dec = MGHDPrimaryClustered(
                    H,
                    dec_pub,
                    halo=args.halo,
                    thresh=args.thresh,
                    temp=args.temp,
                    r_cap=args.r_cap,
                    batched=True,
                    tier0_enable=args.tier0,
                    tier0_k_max=tier0_k_max,
                    tier0_r_max=tier0_r_max,
                    p_channel=p_channel,
                    default_p=p,
                    bucket_spec=bucket_spec,
                    microbatch=args.microbatch,
                    flush_ms=args.flush_ms,
                )

                lat_total = []
                lat_cluster = []
                lat_tier0 = []
                lat_mghd = []
                lat_proj = []
                mghd_invoke_times = []  # Collect all non-zero MGHD invoke times
                mb_reports = []
                tier0_clusters = []
                tier0_qubits = []
                mghd_clusters = []
                mghd_invoked_flags = []
                cluster_counts = []
                max_cluster_sizes = []
                size_hist = Counter()
                nullity_hist = Counter()

                failures = 0
                shots_taken = 0
                early_stopped = False
                check_interval = max(1, min(1000, args.shots))

                while shots_taken < args.shots:
                    attempts = 0
                    while True:
                        _, s = sample_bsc(H, p, rng)
                        checks_list, qubits_list = active_components(H, s, halo=args.halo)
                        sizes = []
                        nullities = []
                        sz_hist_local: Counter = Counter()
                        null_hist_local: Counter = Counter()
                        for ci, qi in zip(checks_list, qubits_list):
                            H_sub, s_sub, _, _ = extract_subproblem(H, s, ci, qi)
                            n_qubits_local = int(qi.size)
                            nullity_local = int(gf2_nullspace(H_sub).shape[1])
                            sizes.append(n_qubits_local)
                            nullities.append(nullity_local)
                            size_key = str(n_qubits_local) if n_qubits_local <= 8 else "9+"
                            sz_hist_local[size_key] += 1
                            null_key = str(nullity_local) if nullity_local <= 8 else "9+"
                            null_hist_local[null_key] += 1

                        passes_size = True if args.min_size is None else any(size >= args.min_size for size in sizes)
                        passes_nullity = True if args.min_nullity is None else any(null_val >= args.min_nullity for null_val in nullities)

                        if passes_size and passes_nullity:
                            break
                        attempts += 1
                        if attempts >= 64:
                            break

                    cluster_counts.append(len(sizes))
                    max_cluster_sizes.append(max(sizes) if sizes else 0)
                    size_hist.update(sz_hist_local)
                    nullity_hist.update(null_hist_local)

                    out = dec.decode(s)
                    e_hat = out["e_hat"]

                    if args.inject_ler:
                        if rng.random() < args.inject_ler_rate:
                            is_correct = np.array_equal((H @ e_hat) % 2, s)
                            failures += int(is_correct)
                        else:
                            failures += int(not np.array_equal((H @ e_hat) % 2, s))
                    else:
                        failures += int(not np.array_equal((H @ e_hat) % 2, s))

                    lat_total.append(out["t_total_us"])
                    lat_cluster.append(out["t_cluster_us"])
                    lat_tier0.append(out.get("t_tier0_us", 0.0))
                    lat_mghd.append(out["t_mghd_us"])
                    lat_proj.append(out["t_project_us"])

                    if out.get("mghd_invoked", False) and out["t_mghd_us"] > 0:
                        mghd_invoke_times.append(out["t_mghd_us"])

                    mb_reports.append(out.get("mb_stats", {}))
                    tier0_clusters.append(out.get("tier0_clusters", 0))
                    tier0_qubits.append(out.get("tier0_qubits", 0))
                    mghd_clusters.append(out.get("mghd_clusters", 0))
                    mghd_invoked_flags.append(bool(out.get("mghd_invoked", False)))

                    shots_taken += 1
                    if args.target_ler_ci is not None and (shots_taken % check_interval == 0):
                        ub = wilson_ci_upper(failures, shots_taken)
                        if ub <= args.target_ler_ci:
                            early_stopped = True
                            break

                if shots_taken == 0:
                    continue

                tier0_total = int(np.sum(tier0_clusters))
                total_clusters = int(np.sum(cluster_counts))
                mghd_total = int(np.sum(mghd_clusters))
                tier0_pct = (100.0 * tier0_total / total_clusters) if total_clusters else 0.0

                cluster_stats = dict(
                    clusters_per_shot_mean=float(np.mean(cluster_counts)),
                    clusters_per_shot_p95=float(np.percentile(cluster_counts, 95)),
                    max_cluster_mean=float(np.mean(max_cluster_sizes)),
                    max_cluster_p95=float(np.percentile(max_cluster_sizes, 95)),
                    size_hist={k: int(v) for k, v in sorted(size_hist.items(), key=lambda kv: (len(kv[0]), kv[0]))},
                    nullity_hist={k: int(v) for k, v in sorted(nullity_hist.items(), key=lambda kv: (len(kv[0]), kv[0]))},
                )

                # Calculate LER with Wilson CI
                ler = failures / shots_taken
                wilson_hi = wilson_ci_upper(failures, shots_taken)
                ler_point, ler_lo, ler_hi = wilson_ci(failures, shots_taken)
                
                # Latency quantiles
                lat_total_arr = np.array(lat_total)
                lat_p50_us = float(np.percentile(lat_total_arr, 50)) if lat_total_arr.size else None
                lat_p95_us = float(np.percentile(lat_total_arr, 95)) if lat_total_arr.size else None  
                lat_p99_us = float(np.percentile(lat_total_arr, 99)) if lat_total_arr.size else None
                
                # Tier-0 fraction and MGHD clusters per shot
                tier0_frac = tier0_pct / 100.0
                mghd_clusters_per_shot = float(mghd_total / shots_taken)
                
                # Calculate non-zero MGHD invoke statistics
                mghd_invoke_times_arr = np.array(mghd_invoke_times) if mghd_invoke_times else np.array([])
                mghd_nonzero_stats = {
                    "count_invokes": int(mghd_invoke_times_arr.size),
                    "p50_nonzero_us": float(np.quantile(mghd_invoke_times_arr, 0.5)) if mghd_invoke_times_arr.size else 0.0,
                    "p95_nonzero_us": float(np.quantile(mghd_invoke_times_arr, 0.95)) if mghd_invoke_times_arr.size else 0.0,
                    "mean_nonzero_us": float(mghd_invoke_times_arr.mean()) if mghd_invoke_times_arr.size else 0.0
                }
                
                side_result = dict(
                    shots=shots_taken,
                    shots_target=args.shots,
                    stopped_early=bool(early_stopped),
                    failures=int(failures),
                    ler=float(ler_point),
                    ler_lo=float(ler_lo),
                    ler_hi=float(ler_hi),
                    wilson_ci_upper=float(wilson_hi),  # backward compatibility
                    lat_p50_us=lat_p50_us,
                    lat_p95_us=lat_p95_us,
                    lat_p99_us=lat_p99_us,
                    tier0_frac=float(tier0_frac),
                    mghd_clusters_per_shot=float(mghd_clusters_per_shot),
                    latency_total_us=summarize(lat_total),
                    t_cluster_us=summarize(lat_cluster),
                    t_tier0_us=summarize(lat_tier0),
                    t_mghd_us=summarize(lat_mghd),
                    t_project_us=summarize(lat_proj),
                    t_mghd_nonzero_stats=mghd_nonzero_stats,
                    mb_stats=_aggregate_mb_stats(mb_reports),
                    tier0_stats=dict(
                        total_clusters=total_clusters,
                        tier0_clusters=tier0_total,
                        tier0_qubits=int(np.sum(tier0_qubits)),
                        tier0_pct=tier0_pct,
                        mghd_clusters=mghd_total,
                        mghd_clusters_per_shot=float(mghd_total / shots_taken),
                        mghd_invoked_shots=int(np.sum(mghd_invoked_flags)),
                        mghd_invoked=bool(mghd_total),
                        p_channel_used=float(p_channel),
                    ),
                    cluster_stats=cluster_stats,
                )
                
                # Add histograms if requested
                if args.log_kappa_hist: 
                    side_result["kappa_hist"] = cluster_stats["size_hist"]
                if args.log_nullity_hist: 
                    side_result["nullity_hist"] = cluster_stats["nullity_hist"]
                
                # Add warmup info if available
                if warmup_info is not None:
                    side_result["warmup"] = warmup_info
                
                # Add histogram data if requested
                if args.log_kappa_hist:
                    side_result["kappa_hist"] = dict(size_hist)
                if args.log_nullity_hist:
                    side_result["nullity_hist"] = dict(nullity_hist)

                # Store in grid format for new enhanced output
                entry = {
                    "d": d, "p": p,
                    "shots": shots_taken,
                    "shots_target": args.shots,
                    "stopped_early": bool(early_stopped),
                    "ler": float(ler_point), "ler_lo": float(ler_lo), "ler_hi": float(ler_hi),
                    "lat_p50_us": lat_p50_us, "lat_p95_us": lat_p95_us, "lat_p99_us": lat_p99_us,
                    "tier0_frac": float(tier0_frac),
                    "mghd_clusters_per_shot": float(mghd_clusters_per_shot),
                }
                if args.min_size is not None:
                    entry["min_size"] = int(args.min_size)
                if args.min_nullity is not None:
                    entry["min_nullity"] = int(args.min_nullity)
                if args.log_kappa_hist: 
                    entry["kappa_hist"] = cluster_stats["size_hist"]
                if args.log_nullity_hist: 
                    entry["nullity_hist"] = cluster_stats["nullity_hist"]
                results["grid"].append(entry)
                
                results[str(d)][f"{p:.3f}"][side] = side_result
                total_p95_us = side_result["latency_total_us"]["p95"]
                mghd_rate = mghd_total / shots_taken if shots_taken else 0.0
                print(
                    f"    {side}: p95={total_p95_us:.1f}μs, LER={ler:.2e}, "
                    f"Tier0={tier0_pct:.1f}%, MGHD={mghd_rate:.3f}/shot, "
                    f"shots={shots_taken}{' (early stop)' if early_stopped else ''}, "
                    f"max_κ_p95={cluster_stats['max_cluster_p95']:.1f}"
                )

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out, f"clustered_surface_sweep_{timestamp}.json")
    payload = dict(
        metadata=dict(
            ckpt=args.ckpt,
            shots=args.shots,
            tier0=args.tier0,
            tier0_mode=args.tier0_mode,
            tier0_k_max=args.tier0_k_max,
            tier0_r_max=args.tier0_r_max,
            p_channel=args.p_channel,
            halo=args.halo,
            thresh=args.thresh,
            temp=args.temp,
            r_cap=args.r_cap,
            device=args.device,
            graph_capture=args.graph_capture,
            expert=args.expert,
            min_nullity=args.min_nullity,
            min_size=args.min_size,
            bucket_spec=[list(b) for b in bucket_spec],
            microbatch=args.microbatch,
            flush_ms=args.flush_ms,
            target_ler_ci=args.target_ler_ci,
        ),
        results=results,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"WROTE {out_path}")
    
    # Also write to --out-json if specified
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"WROTE {args.out_json}")
    
    # CI guard: require MGHD to be exercised when Tier-0 is not 'off' or 'aggressive'
    if args.enforce_mghd and args.tier0_mode in ("mixed", "mixed_tight"):
        # Check aggregate MGHD usage across all sides and conditions
        total_mghd_invoked = 0
        max_tier0_frac = 0.0
        total_shots = 0

        def _walk(node: object) -> None:
            nonlocal total_mghd_invoked, max_tier0_frac, total_shots
            if isinstance(node, dict):
                for key, value in node.items():
                    if key in ("X", "Z") and isinstance(value, dict):
                        invoked = value.get("mghd_invoked_shots")
                        if invoked is None:
                            invoked = value.get("tier0_stats", {}).get("mghd_invoked_shots", 0)
                        total_mghd_invoked += int(invoked or 0)
                        total_shots += int(value.get("shots", 0) or 0)
                        max_tier0_frac = max(max_tier0_frac, value.get("tier0_frac", 1.0))
                    else:
                        _walk(value)
            elif isinstance(node, list):
                for item in node:
                    _walk(item)

        _walk(payload)

        invoked_frac = float(total_mghd_invoked) / float(total_shots or 1)

        if total_mghd_invoked <= 0 or (max_tier0_frac >= 0.99 and invoked_frac < 0.01):
            raise SystemExit("MGHD was not exercised under mixed gating — adjust k_max/r_max or investigate.")
    
    return payload


if __name__ == "__main__":
    out = main()
    if hasattr(out, '__dict__'):
        # Return value for programmatic use
        pass
