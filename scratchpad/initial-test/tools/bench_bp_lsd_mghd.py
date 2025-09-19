from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

import numpy as np

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mghd_clustered.compare_decoders import (
    decode_bp_only,
    decode_lsd,
    decode_mghd_end_to_end,
    decode_mghd_guided,
    latency_stats,
    wilson_ci,
)
from mghd_clustered.features import build_features_bb, build_features_surface
from mghd_clustered.mghd_loader import load_mghd_model
from mghd_clustered.pcm_real import bb_144_12_12_pcm, rotated_surface_pcm


def _syndrome_matches(pcm, recovery: np.ndarray, target: np.ndarray) -> bool:
    check = pcm.dot(recovery) % 2
    return np.array_equal(np.asarray(check).astype(np.uint8).ravel(), target.ravel())


def _summarize(stats_list: List[Dict[str, Any]]) -> Dict[str, float]:
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


def _aggregate(latencies: List[float], failures: int, stats_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    shots = len(latencies)
    result: Dict[str, Any] = {
        "shots": shots,
        "failures": int(failures),
        "latency": latency_stats(latencies),
        "ler": float(failures / shots) if shots else 0.0,
    }
    ci_low, ci_high = wilson_ci(failures, shots) if shots else (0.0, 1.0)
    result["wilson_ci"] = [ci_low, ci_high]
    stats_summary = _summarize(stats_list)
    if stats_summary:
        result["stats_mean"] = stats_summary
    return result


def run_suite_surface(distance: int, physical_error: float, shots: int, mghd_ckpt: str | None) -> Dict[str, Any]:
    Hx, Hz = rotated_surface_pcm(distance)
    out: Dict[str, Any] = {}
    matrices = {
        "X": Hx,
        "Z": Hz,
    }

    mghd_model = load_mghd_model(mghd_ckpt) if mghd_ckpt else None

    for label, pcm in matrices.items():
        n_bits = pcm.shape[1]
        rng = np.random.default_rng(42)
        errors = (rng.random((shots, n_bits)) < physical_error).astype(np.uint8)
        syndromes = (pcm.dot(errors.T) % 2).astype(np.uint8)

        aggregates = {
            "A_bp": [],
            "B_lsd_clustered": [],
            "B_lsd_monolithic": [],
            "C_mghd_guided": [],
            "D_mghd": [],
        }
        failure_counts = {k: 0 for k in aggregates}
        stats_accum: Dict[str, List[Dict[str, Any]]] = {k: [] for k in aggregates}

        for idx in range(shots):
            syndrome = syndromes[:, idx].astype(np.uint8)

            rec, latency, stats = decode_bp_only(pcm, syndrome, physical_error, n_bits)
            aggregates["A_bp"].append(latency)
            stats_accum["A_bp"].append(stats)
            if not _syndrome_matches(pcm, rec, syndrome):
                failure_counts["A_bp"] += 1

            rec, latency, stats = decode_lsd(pcm, syndrome, physical_error, n_bits, bits_per_step=16)
            aggregates["B_lsd_clustered"].append(latency)
            stats_accum["B_lsd_clustered"].append(stats)
            if not _syndrome_matches(pcm, rec, syndrome):
                failure_counts["B_lsd_clustered"] += 1

            rec, latency, stats = decode_lsd(pcm, syndrome, physical_error, n_bits, bits_per_step=n_bits)
            aggregates["B_lsd_monolithic"].append(latency)
            stats_accum["B_lsd_monolithic"].append(stats)
            if not _syndrome_matches(pcm, rec, syndrome):
                failure_counts["B_lsd_monolithic"] += 1

            if mghd_model is not None:
                rec, latency, stats = decode_mghd_guided(
                    pcm,
                    syndrome,
                    physical_error,
                    n_bits,
                    bits_per_step=16,
                    mghd_model=mghd_model,
                    feature_builder=lambda syn: build_features_surface(syn),
                )
                aggregates["C_mghd_guided"].append(latency)
                stats_accum["C_mghd_guided"].append(stats)
                if not _syndrome_matches(pcm, rec, syndrome):
                    failure_counts["C_mghd_guided"] += 1

                rec, latency, stats = decode_mghd_end_to_end(
                    pcm,
                    syndrome,
                    feature_builder=lambda syn: build_features_surface(syn),
                    mghd_model=mghd_model,
                )
                aggregates["D_mghd"].append(latency)
                stats_accum["D_mghd"].append(stats)
                if not _syndrome_matches(pcm, rec, syndrome):
                    failure_counts["D_mghd"] += 1

        summary: Dict[str, Any] = {}
        for key, samples in aggregates.items():
            if not samples:
                continue
            summary[key] = _aggregate(samples, failure_counts[key], stats_accum[key])
        out[label] = summary

    return out


def run_suite_bb(physical_error: float, shots: int, mghd_ckpt: str | None) -> Dict[str, Any]:
    pcm = bb_144_12_12_pcm()
    n_bits = pcm.shape[1]
    rng = np.random.default_rng(42)
    errors = (rng.random((shots, n_bits)) < physical_error).astype(np.uint8)
    syndromes = (pcm.dot(errors.T) % 2).astype(np.uint8)

    mghd_model = load_mghd_model(mghd_ckpt) if mghd_ckpt else None

    aggregates = {
        "A_bp": [],
        "B_lsd_clustered": [],
        "B_lsd_monolithic": [],
        "C_mghd_guided": [],
        "D_mghd": [],
    }
    failure_counts = {k: 0 for k in aggregates}
    stats_accum: Dict[str, List[Dict[str, Any]]] = {k: [] for k in aggregates}

    for idx in range(shots):
        syndrome = syndromes[:, idx].astype(np.uint8)

        rec, latency, stats = decode_bp_only(pcm, syndrome, physical_error, n_bits)
        aggregates["A_bp"].append(latency)
        stats_accum["A_bp"].append(stats)
        if not _syndrome_matches(pcm, rec, syndrome):
            failure_counts["A_bp"] += 1

        rec, latency, stats = decode_lsd(pcm, syndrome, physical_error, n_bits, bits_per_step=16)
        aggregates["B_lsd_clustered"].append(latency)
        stats_accum["B_lsd_clustered"].append(stats)
        if not _syndrome_matches(pcm, rec, syndrome):
            failure_counts["B_lsd_clustered"] += 1

        rec, latency, stats = decode_lsd(pcm, syndrome, physical_error, n_bits, bits_per_step=n_bits)
        aggregates["B_lsd_monolithic"].append(latency)
        stats_accum["B_lsd_monolithic"].append(stats)
        if not _syndrome_matches(pcm, rec, syndrome):
            failure_counts["B_lsd_monolithic"] += 1

        if mghd_model is not None:
            rec, latency, stats = decode_mghd_guided(
                pcm,
                syndrome,
                physical_error,
                n_bits,
                bits_per_step=16,
                mghd_model=mghd_model,
                feature_builder=lambda syn: build_features_bb(syn, pcm),
            )
            aggregates["C_mghd_guided"].append(latency)
            stats_accum["C_mghd_guided"].append(stats)
            if not _syndrome_matches(pcm, rec, syndrome):
                failure_counts["C_mghd_guided"] += 1

            rec, latency, stats = decode_mghd_end_to_end(
                pcm,
                syndrome,
                feature_builder=lambda syn: build_features_bb(syn, pcm),
                mghd_model=mghd_model,
            )
            aggregates["D_mghd"].append(latency)
            stats_accum["D_mghd"].append(stats)
            if not _syndrome_matches(pcm, rec, syndrome):
                failure_counts["D_mghd"] += 1

    summary: Dict[str, Any] = {}
    for key, samples in aggregates.items():
        if not samples:
            continue
        summary[key] = _aggregate(samples, failure_counts[key], stats_accum[key])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare BP, LSD, and MGHD decoders on identical shots.")
    parser.add_argument("--mghd-ckpt", type=str, default=None, help="Path to MGHD checkpoint (optional)")
    parser.add_argument("--shots", type=int, default=1000, help="Number of shots per configuration")
    parser.add_argument("--p-surface", type=float, default=0.002, help="Physical error rate for surface codes")
    parser.add_argument("--p-bb", type=float, default=0.001, help="Physical error rate for BB code")
    args = parser.parse_args()

    results: Dict[str, Any] = {}
    for d in (3, 5, 9):
        results[f"surface_d{d}"] = run_suite_surface(d, args.p_surface, args.shots, args.mghd_ckpt)
    results["bb_144_12_12"] = run_suite_bb(args.p_bb, args.shots, args.mghd_ckpt)

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"compare_bp_lsd_mghd_{int(time.time())}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("WROTE", out_path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
