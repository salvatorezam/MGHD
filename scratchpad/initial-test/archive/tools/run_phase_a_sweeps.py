#!/usr/bin/env python3
"""Phase-A validation sweeps orchestration helper.

Runs the large-shot MGHD validation sweeps described in the roadmap, including:
  • d=3, p∈{0.02,0.03,0.04}, shots=100k, Tier0 off (MGHD-only)
  • d=5, p∈{0.01,0.02}, shots=20k, Tier0 off
  • Mixed-mode κ/nullity stress with --min-nullity=2
  • LER injection smoke tests (1% and 5%) to keep guard rails active

The script simply forwards to `tools.bench_clustered_sweep_surface` with the
appropriate arguments. Use `--dry-run` to just print commands.
"""

from __future__ import annotations

import argparse
import datetime
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

DEFAULT_SWEEPS: List[Dict[str, object]] = [
    dict(
        name="d3_high_p_mghd_only",
        dists=[3],
        ps=[0.02, 0.03, 0.04],
        shots=100_000,
        tier0_mode="off",
        tier0_k_max=None,
        tier0_r_max=None,
        min_nullity=None,
        min_size=None,
        inject_ler=False,
        inject_ler_rate=0.0,
    ),
    dict(
        name="d5_high_p_mghd_only",
        dists=[5],
        ps=[0.01, 0.02],
        shots=20_000,
        tier0_mode="off",
        tier0_k_max=None,
        tier0_r_max=None,
        min_nullity=None,
        min_size=None,
        inject_ler=False,
        inject_ler_rate=0.0,
    ),
    dict(
        name="d3_mixed_min_nullity2",
        dists=[3],
        ps=[0.005, 0.010, 0.015],
        shots=20_000,
        tier0_mode="mixed",
        tier0_k_max=2,
        tier0_r_max=1,
        min_nullity=2,
        min_size=None,
        inject_ler=False,
        inject_ler_rate=0.0,
    ),
    dict(
        name="guardrail_inject_1pct",
        dists=[3],
        ps=[0.005],
        shots=5_000,
        tier0_mode="mixed",
        tier0_k_max=2,
        tier0_r_max=1,
        min_nullity=None,
        min_size=None,
        inject_ler=True,
        inject_ler_rate=0.01,
    ),
    dict(
        name="guardrail_inject_5pct",
        dists=[3],
        ps=[0.005],
        shots=5_000,
        tier0_mode="mixed",
        tier0_k_max=2,
        tier0_r_max=1,
        min_nullity=None,
        min_size=None,
        inject_ler=True,
        inject_ler_rate=0.05,
    ),
]


def build_command(args: argparse.Namespace, sweep: Dict[str, object], timestamp: str) -> List[str]:
    shots = max(1, int(round(float(sweep["shots"]) * args.shots_scale)))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{sweep['name']}_{timestamp}.json"

    cmd: List[str] = [
        sys.executable,
        "-m",
        "tools.bench_clustered_sweep_surface",
        "--ckpt",
        args.ckpt,
        "--expert",
        args.expert,
        "--device",
        args.device,
        "--shots",
        str(shots),
        "--tier0-mode",
        str(sweep["tier0_mode"]),
        "--graph-capture",
    ]

    if sweep.get("tier0_k_max") is not None:
        cmd.extend(["--tier0-k-max", str(sweep["tier0_k_max"])])
    if sweep.get("tier0_r_max") is not None:
        cmd.extend(["--tier0-r-max", str(sweep["tier0_r_max"])])

    dists = [str(dist) for dist in sweep["dists"]]  # type: ignore[arg-type]
    cmd.extend(["--dists", *dists])
    ps_values = [f"{phys_p:.6f}" for phys_p in sweep["ps"]]  # type: ignore[arg-type]
    cmd.extend(["--ps", *ps_values])

    if sweep.get("min_nullity") is not None:
        cmd.extend(["--min-nullity", str(sweep["min_nullity"])])
    if sweep.get("min_size") is not None:
        cmd.extend(["--min-size", str(sweep["min_size"])])

    if sweep.get("inject_ler"):
        cmd.append("--inject-ler")
        cmd.extend(["--inject-ler-rate", str(sweep["inject_ler_rate"])])

    if args.enforce_mghd:
        cmd.append("--enforce-mghd")

    cmd.extend(["--out-json", str(out_json)])

    if args.extra_args:
        cmd.extend(shlex.split(args.extra_args))
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase-A MGHD validation sweeps")
    parser.add_argument("--ckpt", required=True, help="Path to MGHD checkpoint")
    parser.add_argument("--expert", default="auto", choices=["v1", "v2", "auto"],
                        help="Which expert mode to use for inference")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/phase_a",
                        help="Directory to store JSON outputs")
    parser.add_argument("--shots-scale", type=float, default=1.0,
                        help="Scale factor applied to each sweep's shot count")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--enforce-mghd", action="store_true",
                        help="Propagate --enforce-mghd to the bench harness")
    parser.add_argument("--extra-args", default="",
                        help="Additional CLI args passed verbatim to bench_clustered_sweep_surface")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Sweep names to skip")
    args = parser.parse_args()

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    env = os.environ.copy()
    # Ensure we are using real CUDA-Q noise unless user explicitly overrides.
    env.pop("MGHD_SYNTHETIC", None)

    for sweep in DEFAULT_SWEEPS:
        if sweep["name"] in args.skip:
            print(f"[SKIP] {sweep['name']}")
            continue
        cmd = build_command(args, sweep, timestamp)
        print(f"[RUN] {sweep['name']}\n      {shlex.join(cmd)}")
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)


if __name__ == "__main__":
    main()
