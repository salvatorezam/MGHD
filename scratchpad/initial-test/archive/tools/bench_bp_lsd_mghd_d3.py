#!/usr/bin/env python3
"""Benchmark BP vs LSD vs MGHD-guided LSD on rotated d=3."""

from __future__ import annotations

import json
import os
import time
from argparse import ArgumentParser

import numpy as np
import scipy.sparse as sp
from ldpc.bplsd_decoder import BpLsdDecoder

from mghd_public.cluster_proxy import kappa_nu_proxy
from mghd_public.config import MGHDConfig
from mghd_public.infer import MGHDDecoderPublic


def load_d3_pack(path: str) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    pack = np.load(path)
    return sp.csr_matrix(pack["Hx"]), sp.csr_matrix(pack["Hz"])


def sample_bsc(H: sp.csr_matrix, p: float, rng: np.random.Generator) -> np.ndarray:
    n = H.shape[1]
    e = (rng.random(n) < p).astype(np.uint8)
    s = (H @ e) % 2
    return s.astype(np.uint8)


def make_decoder(H: sp.csr_matrix, p: float, bits_per_step: int, max_iter_bp: int) -> BpLsdDecoder:
    return BpLsdDecoder(
        H,
        error_rate=p,
        bp_method="minimum_sum",
        max_iter=max_iter_bp,
        schedule="serial",
        lsd_method="LSD_E",
        lsd_order=0,
        bits_per_step=bits_per_step,
    )


def run_side(H: sp.csr_matrix, decoder: MGHDDecoderPublic, *, side: str, p: float, shots: int, seed: int):
    rng = np.random.default_rng(seed)
    n = H.shape[1]

    dec_A = make_decoder(H, p, bits_per_step=n, max_iter_bp=8)   # BP-only
    dec_Bc = make_decoder(H, p, bits_per_step=16, max_iter_bp=1)  # LSD clustered
    dec_Bm = make_decoder(H, p, bits_per_step=n, max_iter_bp=1)   # LSD monolithic
    dec_C = make_decoder(H, p, bits_per_step=16, max_iter_bp=1)   # MGHD-guided clustered

    acc = {k: dict(times=[], fails=0, kappa=[], nu=[]) for k in ["A", "B_cluster", "B_mono", "C"]}

    for _ in range(shots):
        s = sample_bsc(H, p, rng)
        pri = decoder.priors_from_syndrome(s, side=side)

        t0 = time.perf_counter()
        r = dec_A.decode(s)
        acc["A"]["times"].append((time.perf_counter() - t0) * 1e3)
        acc["A"]["fails"] += int(not np.array_equal((H @ r) % 2, s))
        k, nu = kappa_nu_proxy(H, s)
        acc["A"]["kappa"].append(k)
        acc["A"]["nu"].append(nu)

        t0 = time.perf_counter()
        r = dec_Bc.decode(s)
        acc["B_cluster"]["times"].append((time.perf_counter() - t0) * 1e3)
        acc["B_cluster"]["fails"] += int(not np.array_equal((H @ r) % 2, s))
        k, nu = kappa_nu_proxy(H, s)
        acc["B_cluster"]["kappa"].append(k)
        acc["B_cluster"]["nu"].append(nu)

        t0 = time.perf_counter()
        r = dec_Bm.decode(s)
        acc["B_mono"]["times"].append((time.perf_counter() - t0) * 1e3)
        acc["B_mono"]["fails"] += int(not np.array_equal((H @ r) % 2, s))
        k, nu = kappa_nu_proxy(H, s)
        acc["B_mono"]["kappa"].append(k)
        acc["B_mono"]["nu"].append(nu)

        dec_C.update_channel_probs(pri)
        t0 = time.perf_counter()
        r = dec_C.decode(s)
        acc["C"]["times"].append((time.perf_counter() - t0) * 1e3)
        acc["C"]["fails"] += int(not np.array_equal((H @ r) % 2, s))
        k, nu = kappa_nu_proxy(H, s)
        acc["C"]["kappa"].append(k)
        acc["C"]["nu"].append(nu)

    def summarize(key: str) -> dict:
        times = np.array(acc[key]["times"], dtype=float)
        kappa = np.array(acc[key]["kappa"], dtype=float)
        nu = np.array(acc[key]["nu"], dtype=float)

        pct = lambda q: float(np.percentile(times, q))
        return dict(
            shots=len(times),
            failures=int(acc[key]["fails"]),
            latency_ms=dict(mean=float(times.mean()), p50=pct(50), p95=pct(95), p99=pct(99)),
            kappa=dict(mean=float(kappa.mean()), p50=float(np.percentile(kappa, 50)), p95=float(np.percentile(kappa, 95))),
            nu=dict(mean=float(nu.mean()), p50=float(np.percentile(nu, 50)), p95=float(np.percentile(nu, 95))),
        )

    return {
        "A_bp": summarize("A"),
        "B_lsd_cluster": summarize("B_cluster"),
        "B_lsd_mono": summarize("B_mono"),
        "C_mghd_guided": summarize("C"),
    }


def main() -> None:
    ap = ArgumentParser(description="Benchmark MGHD-guided LSD on rotated d=3")
    ap.add_argument("--ckpt", required=True, help="Path to MGHD state_dict checkpoint")
    ap.add_argument("--shots", type=int, default=5000)
    ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--pack", default="student_pack_p003.npz")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    Hx, Hz = load_d3_pack(args.pack)

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
    )

    decoder = MGHDDecoderPublic(args.ckpt, cfg, device=args.device)
    decoder.bind_code(Hx, Hz)

    results = dict(
        X=run_side(Hx, decoder, side="X", p=args.p, shots=args.shots, seed=123),
        Z=run_side(Hz, decoder, side="Z", p=args.p, shots=args.shots, seed=456),
    )

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "compare_bp_lsd_mghd_d3.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("WROTE", out_path)


if __name__ == "__main__":
    main()
