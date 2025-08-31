#!/usr/bin/env python3
"""
Auto Post-Run Evaluator

Watches a training outdir for process completion (via pid.txt), then runs the
forward-path LER evaluation (N=10k per p) using the best checkpoint found in
the outdir. Writes results to ler_<profile>_forward.json and a small summary.

Usage:
  python tools/auto_post_eval.py --outdir /path/to/results_dir [--device cuda]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import json
from pathlib import Path
from typing import Optional
import subprocess


def _ps_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        # Fallback to ps for non-POSIX environments
        try:
            out = subprocess.run(["ps", "-p", str(pid), "-o", "pid="], capture_output=True, text=True)
            return out.returncode == 0 and out.stdout.strip() != ""
        except Exception:
            return False


def _infer_profile(outdir: Path) -> str:
    args_json = outdir / "args.json"
    if args_json.exists():
        try:
            with open(args_json) as f:
                aj = json.load(f)
            prof = aj.get("profile", None)
            if prof:
                return str(prof)
        except Exception:
            pass
    # Fallback: scan for checkpoint name
    for p in outdir.glob("step11_garnet_*_best.pt"):
        name = p.name
        try:
            mid = name.split("step11_garnet_")[1]
            prof = mid.split("_best.pt")[0]
            return prof
        except Exception:
            continue
    return "L"


def main():
    ap = argparse.ArgumentParser(description="Auto post-run evaluator")
    ap.add_argument("--outdir", required=True, help="Training run outdir with pid.txt and checkpoint")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--p-grid", default="0.02,0.03,0.05,0.08")
    ap.add_argument("--N-per-p", type=int, default=10000)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    pid_file = outdir / "pid.txt"
    log_file = outdir / "post_eval.log"
    with open(log_file, "a") as log:
        def _log(msg: str):
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            print(f"[{ts}] {msg}", file=log, flush=True)

        _log(f"Auto post-eval starting for outdir={outdir}")
        if not pid_file.exists():
            _log("pid.txt not found; proceeding to eval immediately if checkpoint exists")
        else:
            try:
                pid = int(pid_file.read_text().strip())
                _log(f"Watching PID {pid}...")
                while _ps_alive(pid):
                    time.sleep(15)
                _log(f"PID {pid} exited; continuing to evaluation")
            except Exception as e:
                _log(f"Failed reading pid.txt ({e}); continuing to evaluation")

        prof = _infer_profile(outdir)
        ckpt = outdir / f"step11_garnet_{prof}_best.pt"
        if not ckpt.exists():
            _log(f"Checkpoint not found: {ckpt}")
            return 1

        out_json = outdir / f"ler_{prof}_forward.json"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parents[0] / "eval_ler.py"),
            "--decoder", "mghd_forward",
            "--checkpoint", str(ckpt),
            "--N-per-p", str(args.N_per_p),
            "--p-grid", args.p_grid,
            "--out", str(out_json),
            "--device", args.device,
        ]
        _log(f"Running eval: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, cwd=str(outdir.parents[0]), check=False)
            _log(f"Eval complete. Wrote {out_json}")
        except Exception as e:
            _log(f"Eval failed: {e}")
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

