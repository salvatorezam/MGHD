#!/usr/bin/env python
"""Run logical-memory scaling experiments over syndrome-cycle depth T.

Outputs:
- memory_scaling_report.json (per decoder, per T, with CIs and eps/cycle)
- memory_scaling_report.png/.pdf
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch

from mghd.codes.registry import get_code
from mghd.core.core import MGHDDecoderPublic
from mghd.decoders.lsd.clustered import MGHDPrimaryClustered
from mghd.decoders.lsd_teacher import LSDTeacher
from mghd.decoders.mwpm_ctx import MWPMatchingContext
from mghd.samplers.cudaq_sampler import CudaQSampler
from mghd.samplers.stim_sampler import StimSampler
from mghd.utils.metrics import logical_error_rate


def _align_preds(preds: np.ndarray, obs_true: np.ndarray) -> np.ndarray:
    if preds.ndim == 3 and preds.shape[1] == 1:
        preds = preds[:, 0, :]
    if preds.shape == obs_true.shape:
        return preds
    if obs_true.shape[1] == 1 and preds.shape[1] == 2:
        return preds[:, 1:]
    return preds


def _resolve_syndromes(code_obj, dets):
    mx = getattr(code_obj, "Hx", None)
    mz = getattr(code_obj, "Hz", None)
    B = dets.shape[0]
    if hasattr(code_obj, "detectors_to_syndromes"):
        sx, sz = code_obj.detectors_to_syndromes(dets)
        return sx.astype(np.uint8), sz.astype(np.uint8)
    sx = np.zeros((B, mx.shape[0] if mx is not None else 0), dtype=np.uint8)
    sz = np.zeros((B, mz.shape[0] if mz is not None else 0), dtype=np.uint8)
    return sx, sz


def _effective_cycle_rate(ler: float, rounds: int) -> float | None:
    if rounds <= 0:
        return None
    if ler <= 0.0:
        return 0.0
    if ler >= 1.0:
        return None
    return float(-np.log(1.0 - ler) / rounds)


def _build_sampler(sampler: str, rounds: int, p: float):
    sampler = sampler.lower()
    if sampler == "stim":
        return StimSampler(rounds=rounds, dep=p)
    if sampler == "cudaq":
        return CudaQSampler(
            device_profile="garnet",
            profile_kwargs={"phys_p": p, "rounds": rounds},
        )
    raise ValueError(f"Unsupported sampler {sampler}")


def run_memory_experiment(args) -> list[dict]:
    device = "cuda" if args.cuda else "cpu"
    code = get_code(args.family, distance=args.distance)

    if sp.issparse(code.Hx):
        Hx_sparse = code.Hx
        Hx_dense = code.Hx.toarray()
    else:
        Hx_sparse = sp.csr_matrix(code.Hx)
        Hx_dense = np.asarray(code.Hx)

    if sp.issparse(code.Hz):
        Hz_sparse = code.Hz
        Hz_dense = code.Hz.toarray()
    else:
        Hz_sparse = sp.csr_matrix(code.Hz)
        Hz_dense = np.asarray(code.Hz)

    decoder_public = MGHDDecoderPublic(
        args.checkpoint,
        device=device,
        profile=args.profile,
        node_feat_dim=args.node_feat_dim,
    )
    decoder_public.bind_code(Hx_sparse, Hz_sparse)
    mghd_Z = MGHDPrimaryClustered(Hx_sparse, decoder_public, batched=False)
    mghd_X = MGHDPrimaryClustered(Hz_sparse, decoder_public, batched=False)

    mwpm_ctx = MWPMatchingContext()
    lsd = LSDTeacher(Hx_dense, Hz_dense)

    rounds_list = [int(x) for x in args.rounds.split(",") if x.strip()]
    p_values = [float(x) for x in args.p_values.split(",") if x.strip()]
    results = []

    for p in p_values:
        for rounds in rounds_list:
            sampler = _build_sampler(args.sampler, rounds, p)
            total = 0
            fail_mghd = 0.0
            fail_mwpm = 0.0
            fail_lsd = 0.0
            mghd_decode_errors = 0

            n_batches = (args.shots + args.batch_size - 1) // args.batch_size
            for batch_id in range(n_batches):
                this_batch = min(args.batch_size, args.shots - total)
                if this_batch <= 0:
                    break
                batch = sampler.sample(code, n_shots=this_batch, seed=args.seed + batch_id)
                obs_true = batch.obs
                if hasattr(batch, "synX") and hasattr(batch, "synZ"):
                    sx = batch.synX
                    sz = batch.synZ
                elif args.sampler == "cudaq":
                    mz = int(code.Hz.shape[0])
                    mx = int(code.Hx.shape[0])
                    dets = np.asarray(batch.dets, dtype=np.uint8)
                    sz = dets[:, :mz].astype(np.uint8)
                    sx = dets[:, mz : mz + mx].astype(np.uint8)
                else:
                    sx, sz = _resolve_syndromes(code, batch.dets)

                preds_mghd = []
                for i in range(this_batch):
                    try:
                        ez = mghd_Z.decode(sx[i])["e"]
                        ex = mghd_X.decode(sz[i])["e"]
                        obs_pred = code.data_to_observables(ex, ez)
                    except Exception:
                        mghd_decode_errors += 1
                        ex_zero = np.zeros(Hz_dense.shape[1], dtype=np.uint8)
                        ez_zero = np.zeros(Hx_dense.shape[1], dtype=np.uint8)
                        obs_pred = code.data_to_observables(ex_zero, ez_zero)
                    preds_mghd.append(obs_pred)
                preds_mghd = _align_preds(np.asarray(preds_mghd, dtype=np.uint8), obs_true)
                fail_mghd += float(logical_error_rate(obs_true, preds_mghd).ler_mean) * this_batch

                preds_mwpm = []
                for i in range(this_batch):
                    ez_pm, _ = mwpm_ctx.decode(Hx_dense, sx[i], "Z")
                    ex_pm, _ = mwpm_ctx.decode(Hz_dense, sz[i], "X")
                    preds_mwpm.append(code.data_to_observables(ex_pm, ez_pm))
                preds_mwpm = _align_preds(np.asarray(preds_mwpm, dtype=np.uint8), obs_true)
                fail_mwpm += float(logical_error_rate(obs_true, preds_mwpm).ler_mean) * this_batch

                ez_lsd, ex_lsd = lsd.decode_batch_xz(sx, sz)
                preds_lsd = []
                for i in range(this_batch):
                    preds_lsd.append(code.data_to_observables(ex_lsd[i], ez_lsd[i]))
                preds_lsd = _align_preds(np.asarray(preds_lsd, dtype=np.uint8), obs_true)
                fail_lsd += float(logical_error_rate(obs_true, preds_lsd).ler_mean) * this_batch

                total += this_batch

            ler_mghd = fail_mghd / max(1, total)
            ler_mwpm = fail_mwpm / max(1, total)
            ler_lsd = fail_lsd / max(1, total)
            results.append(
                {
                    "distance": int(args.distance),
                    "sampler": args.sampler,
                    "p": float(p),
                    "rounds": int(rounds),
                    "shots": int(total),
                    "ler_mghd": ler_mghd,
                    "ler_mwpm": ler_mwpm,
                    "ler_lsd": ler_lsd,
                    "eps_cycle_mghd": _effective_cycle_rate(ler_mghd, rounds),
                    "eps_cycle_mwpm": _effective_cycle_rate(ler_mwpm, rounds),
                    "eps_cycle_lsd": _effective_cycle_rate(ler_lsd, rounds),
                    "mghd_decode_errors": int(mghd_decode_errors),
                    "decoder_output_classes": {
                        "mghd": "per_qubit",
                        "mwpm": "per_qubit",
                        "lsd": "per_qubit",
                    },
                }
            )
            print(
                f"p={p:.6f} T={rounds}: "
                f"MGHD={ler_mghd:.6f} MWPM={ler_mwpm:.6f} LSD={ler_lsd:.6f} "
                f"MGHD_err={mghd_decode_errors}"
            )

    return results


def plot_memory_results(results: list[dict], output_json: Path) -> None:
    grouped = defaultdict(list)
    for row in results:
        grouped[row["p"]].append(row)

    plt.style.use("seaborn-v0_8-paper")
    plt.figure(figsize=(10, 7))
    for p, rows in sorted(grouped.items(), key=lambda kv: kv[0]):
        rows = sorted(rows, key=lambda r: r["rounds"])
        rounds = np.array([r["rounds"] for r in rows], dtype=float)
        mghd = np.array([r["ler_mghd"] for r in rows], dtype=float)
        mwpm = np.array([r["ler_mwpm"] for r in rows], dtype=float)
        lsd = np.array([r["ler_lsd"] for r in rows], dtype=float)
        plt.semilogy(rounds, np.clip(mghd, 1e-9, None), marker="o", label=f"MGHD p={p:g}")
        plt.semilogy(rounds, np.clip(mwpm, 1e-9, None), marker="s", linestyle="--", alpha=0.6)
        plt.semilogy(rounds, np.clip(lsd, 1e-9, None), marker="^", linestyle=":", alpha=0.6)

    plt.xlabel("Memory Rounds (T)")
    plt.ylabel("Logical Error Rate")
    plt.title("Memory Experiment: LER vs Syndrome Rounds")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    png_path = output_json.with_suffix(".png")
    pdf_path = output_json.with_suffix(".pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved memory plots to {png_path} and {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MGHD memory experiment sweep.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--family", default="surface")
    parser.add_argument("--distance", type=int, default=5)
    parser.add_argument("--p-values", default="0.001,0.003,0.005")
    parser.add_argument("--rounds", default="5,10,20,40")
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampler", choices=["stim", "cudaq"], default="stim")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--profile", default="S")
    parser.add_argument("--node-feat-dim", type=int, default=9)
    parser.add_argument("--output", default="memory_scaling_report.json")
    args = parser.parse_args()

    out_path = Path(args.output)
    results = run_memory_experiment(args)
    payload = {
        "config": vars(args),
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote memory report to {out_path}")
    plot_memory_results(results, out_path)


if __name__ == "__main__":
    main()
