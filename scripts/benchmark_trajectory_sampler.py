#!/usr/bin/env python
"""Trajectory sampler throughput smoke benchmark.

Measures syndrome-generation throughput (shots/sec) for CUDA-Q sampler paths
under canonical noise models.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mghd.codes.registry import get_code
from mghd.samplers.cudaq_sampler import CudaQSampler


def _parse_csv_int(spec: str) -> list[int]:
    return [int(x) for x in str(spec).split(",") if x.strip()]


def _parse_csv_float(spec: str) -> list[float]:
    return [float(x) for x in str(spec).split(",") if x.strip()]


def _parse_csv_str(spec: str) -> list[str]:
    return [str(x).strip() for x in str(spec).split(",") if str(x).strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CUDA-Q trajectory sampler throughput.")
    parser.add_argument("--family", default="surface")
    parser.add_argument("--distances", default="3,5,7")
    parser.add_argument("--p-values", default="0.003,0.005,0.01")
    parser.add_argument(
        "--noise-models",
        default="code_capacity,phenomenological,circuit_standard,circuit_augmented",
        help="Comma-separated canonical model names.",
    )
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    distances = _parse_csv_int(args.distances)
    p_values = _parse_csv_float(args.p_values)
    noise_models = _parse_csv_str(args.noise_models)

    results: list[dict] = []

    for d in distances:
        code = get_code(args.family, distance=d)
        for p in p_values:
            for nm in noise_models:
                sampler = CudaQSampler(
                    device_profile="garnet",
                    profile_kwargs={
                        "rounds": int(args.rounds),
                        "phys_p": float(p),
                        "noise_model": str(nm),
                    },
                )
                for w in range(max(0, int(args.warmup))):
                    _ = sampler.sample(code, n_shots=int(args.shots), seed=int(args.seed + w))
                elapsed = 0.0
                for r in range(max(1, int(args.repeats))):
                    t0 = time.perf_counter()
                    _ = sampler.sample(code, n_shots=int(args.shots), seed=int(args.seed + 100 + r))
                    elapsed += time.perf_counter() - t0
                total_shots = int(args.shots) * max(1, int(args.repeats))
                shots_per_sec = total_shots / max(elapsed, 1e-9)
                results.append(
                    {
                        "distance": int(d),
                        "p": float(p),
                        "noise_model": str(nm),
                        "shots": int(args.shots),
                        "repeats": int(args.repeats),
                        "elapsed_sec": float(elapsed),
                        "shots_per_sec": float(shots_per_sec),
                    }
                )
                print(
                    f"d={d} p={p:.4g} noise={nm}: {shots_per_sec:.2f} shots/s "
                    f"({total_shots} shots in {elapsed:.3f}s)"
                )

    payload = {
        "family": args.family,
        "distances": distances,
        "p_values": p_values,
        "noise_models": noise_models,
        "shots": int(args.shots),
        "repeats": int(args.repeats),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved throughput benchmark to {args.output}")


if __name__ == "__main__":
    main()
