#!/usr/bin/env python3
"""Probe MGHD public inference path on rotated d=3."""

from __future__ import annotations

import argparse
import json
import numpy as np
import scipy.sparse as sp

from mghd_public.config import MGHDConfig
from mghd_public.infer import MGHDDecoderPublic


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe MGHD public priors on d=3")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pack", default="student_pack_p003.npz")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    pack = np.load(args.pack)
    Hx = sp.csr_matrix(pack["Hx"])
    Hz = sp.csr_matrix(pack["Hz"])

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

    zero_x = np.zeros(Hx.shape[0], dtype=np.uint8)
    zero_z = np.zeros(Hz.shape[0], dtype=np.uint8)

    px = decoder.priors_from_syndrome(zero_x, side="X")
    pz = decoder.priors_from_syndrome(zero_z, side="Z")

    out = {
        "len_px": int(px.shape[0]),
        "len_pz": int(pz.shape[0]),
        "px_minmax": [float(px.min()), float(px.max())],
        "pz_minmax": [float(pz.min()), float(pz.max())],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
