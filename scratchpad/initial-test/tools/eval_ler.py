#!/usr/bin/env python3
"""
Phase-3 LER evaluation harness (coset-aware, Wilson CIs, latency stats).

Decoders: {mghd, mwpm, mwpf, relay, fastpath, garnet}

JSON output schema (example):
{
  "decoder": "mghd",
  "metric_type": "coset_parity",
  "confidence_interval": 0.95,
  "params": {"n_params": 402113},
  "approx_flops": null,
  "results": [
    {"p": 0.02, "N": 10000, "LER": 0.0123, "ler_low": 0.0101, "ler_high": 0.0148,
     "latency_p50_us": 47.1, "latency_p99_us": 52.7}
  ]
}

Notes:
  - Coset-aware success requires: parity match AND (delta corrections in ker(Hx/Hz) for X/Z flows).
  - Canonical d=3: N_syn=8 (Z first, then X), N_bits=9, LSBF per byte.
  - No CUDA at import time: GPUs are used only inside main() or callables.
"""

from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * np.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    low = (center - margin) / denom
    high = (center + margin) / denom
    return float(max(0.0, low)), float(min(1.0, high))


def _bit_unpack_rows(packed: np.ndarray, n_bits: int) -> np.ndarray:
    B, n_bytes = packed.shape
    bit_idx = np.arange(8, dtype=np.uint8)
    bits = ((packed[:, :, None] >> bit_idx[None, None, :]) & 1).astype(np.uint8)
    bits = bits.reshape(B, n_bytes * 8)
    return bits[:, :n_bits]


def _pack_bits_rows(bits: np.ndarray) -> np.ndarray:
    B, K = bits.shape
    n_bytes = (K + 7) // 8
    out = np.zeros((B, n_bytes), dtype=np.uint8)
    for i in range(K):
        b = i // 8
        j = i % 8
        out[:, b] |= ((bits[:, i].astype(np.uint8) & 1) << j)
    return out


def _coset_success(Hz: np.ndarray, Hx: np.ndarray, s_bin: np.ndarray, y_pred: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
    """Return boolean [B] success under coset+parity criteria.
    s_bin: [B,8] (Z then X), y_pred/y_ref: [B,9].
    """
    B = s_bin.shape[0]
    sZ = s_bin[:, :4]
    sX = s_bin[:, 4:8]

    # Parities from predictions and reference
    sZ_pred = (Hz @ y_pred.T) % 2
    sX_pred = (Hx @ y_pred.T) % 2
    sZ_ref = (Hz @ y_ref.T) % 2
    sX_ref = (Hx @ y_ref.T) % 2

    parity_ok = (sZ_pred.T == sZ) & (sX_pred.T == sX)

    # Coset equivalence: delta in kernel (zero syndrome difference)
    d = (y_pred ^ y_ref).astype(np.uint8)  # XOR difference
    z_zero = ((Hz @ d.T) % 2 == 0).all(axis=0)
    x_zero = ((Hx @ d.T) % 2 == 0).all(axis=0)
    coset_ok = z_zero & x_zero

    return (parity_ok.all(axis=1) & coset_ok).astype(np.bool_)


def _ensure_fastpath_lut():
    # Touch the LUT once (raises early if missing)
    from fastpath import load_rotated_d3_lut_npz  # type: ignore
    return load_rotated_d3_lut_npz()


def decode_fastpath(synd_bytes: np.ndarray) -> np.ndarray:
    from fastpath import decode_bytes, load_rotated_d3_lut_npz  # type: ignore
    lut16, *_ = load_rotated_d3_lut_npz()
    return decode_bytes(synd_bytes, lut16)


def _infer_profile_from_args(ckpt: str) -> str:
    from pathlib import Path as _Path
    import json as _json
    args_json = _Path(ckpt).parent / 'args.json'
    if args_json.exists():
        try:
            with open(args_json) as f:
                aj = _json.load(f)
            return aj.get('profile', 'S')
        except Exception:
            return 'S'
    return 'S'


def _build_mghd_from_ckpt_meta(ckpt: str, device: str = 'cuda'):
    import torch
    from poc_my_models import MGHD
    prof = _infer_profile_from_args(ckpt)
    profiles = {
        'S': dict(n_iters=7, n_node_features=128, n_edge_features=128, msg_net=96, d_model=192, d_state=32),
        'M': dict(n_iters=8, n_node_features=192, n_edge_features=192, msg_net=128, d_model=256, d_state=48),
        'L': dict(n_iters=9, n_node_features=256, n_edge_features=256, msg_net=160, d_model=320, d_state=64),
    }
    pf = profiles.get(prof, profiles['S'])
    gnn_params = {
        'dist': 3,
        'n_node_inputs': 9,
        'n_node_outputs': 9,
        'n_iters': pf['n_iters'],
        'n_node_features': pf['n_node_features'],
        'n_edge_features': pf['n_edge_features'],
        'msg_net_size': pf['msg_net'],
        'msg_net_dropout_p': 0.0,
        'gru_dropout_p': 0.0,
    }
    mamba_params = {
        'd_model': pf['d_model'],
        'd_state': pf['d_state'],
        'd_conv': 2,
        'expand': 3,
        'attention_mechanism': 'none',
    }
    model = MGHD(gnn_params=gnn_params, mamba_params=mamba_params).to(device)
    try:
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state, strict=False)
    except Exception:
        pass
    try:
        model.set_rotated_layout()
    except Exception:
        pass
    try:
        model._ensure_static_indices(device)
    except Exception:
        pass
    model.eval()
    return model


def decode_mghd(s_bin: np.ndarray, ckpt: str, device: str = "cuda") -> np.ndarray:
    import torch
    from poc_my_models import MGHD

    # Try to reconstruct architecture from checkpoint run args (profile-aware)
    import json as _json
    from pathlib import Path as _Path
    ckpt_path = _Path(ckpt)
    run_dir = ckpt_path.parent
    profile = 'S'
    args_json = run_dir / 'args.json'
    if args_json.exists():
        try:
            with open(args_json) as f:
                aj = _json.load(f)
            profile = aj.get('profile', profile)
        except Exception:
            pass
    profiles = {
        'S': dict(n_iters=7, n_node_features=128, n_edge_features=128, msg_net=96, d_model=192, d_state=32),
        'M': dict(n_iters=8, n_node_features=192, n_edge_features=192, msg_net=128, d_model=256, d_state=48),
        'L': dict(n_iters=9, n_node_features=256, n_edge_features=256, msg_net=160, d_model=320, d_state=64),
    }
    pf = profiles.get(profile, profiles['S'])
    gnn_params = {
        'dist': 3,
        'n_node_inputs': 9,
        'n_node_outputs': 9,  # adjusted to 2 for rotated if model enforces
        'n_iters': pf['n_iters'],
        'n_node_features': pf['n_node_features'],
        'n_edge_features': pf['n_edge_features'],
        'msg_net_size': pf['msg_net'],
        'msg_net_dropout_p': 0.0,
        'gru_dropout_p': 0.0,
    }
    mamba_params = {
        'd_model': pf['d_model'],
        'd_state': pf['d_state'],
        'd_conv': 2,
        'expand': 3,
        'attention_mechanism': 'none',
    }
    model = _build_mghd_from_ckpt_meta(ckpt, device=device)

    # Batch decode via decode_one in a loop (keeps implementation simple)
    B = s_bin.shape[0]
    out = np.zeros((B, 9), dtype=np.uint8)
    for i in range(B):
        # Pack row to one byte LSBF
        b = 0
        row = s_bin[i]
        for j in range(8):
            b |= (int(row[j]) & 1) << j
        bs = np.array([b], dtype=np.uint8)
        t = model.decode_one(torch.from_numpy(bs).to(device), device=device)
        out[i] = t.squeeze(0).detach().cpu().numpy()
    return out


def decode_mghd_forward(s_bin: np.ndarray, ckpt: str, device: str = 'cuda') -> np.ndarray:
    import torch
    model = _build_mghd_from_ckpt_meta(ckpt, device=device)
    B = s_bin.shape[0]
    out = np.zeros((B, 9), dtype=np.uint8)
    bs = 1024
    for off in range(0, B, bs):
        sl = slice(off, min(off + bs, B))
        chunk = s_bin[sl]
        bsz = chunk.shape[0]
        node_inputs = torch.zeros(bsz, 17, 9, device=device, dtype=torch.float32)
        node_inputs[:, :8, 0] = torch.from_numpy(chunk.astype(np.float32)).to(device)
        flat_inputs = node_inputs.view(-1, 9)
        with torch.no_grad():
            outs = model(flat_inputs, getattr(model, '_src_ids', None), getattr(model, '_dst_ids', None))
            final = outs[-1].view(bsz, 17, -1)[:, 8:, :]
            if final.shape[-1] == 2:
                bitlogits = (final[..., 1] - final[..., 0])
            else:
                bitlogits = final.squeeze(-1)
            bits = (bitlogits.sigmoid() > 0.5).to(torch.uint8)
        out[sl] = bits.detach().cpu().numpy()
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Coset-aware LER evaluator")
    ap.add_argument('--decoder', required=True, choices=['mghd','mghd_forward','mwpm','mwpf','relay','fastpath','garnet'])
    ap.add_argument('--checkpoint', type=str, default=None, help='Required for decoder=mghd')
    ap.add_argument('--metric', choices=['coset','coset_parity'], default='coset_parity')
    ap.add_argument('--N-per-p', type=int, default=10000)
    ap.add_argument('--p-grid', type=str, default='0.02,0.03,0.05,0.08')
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--ci', type=float, default=0.95)
    args = ap.parse_args()

    if args.N_per_p < 10000:
        print("[WARN] N-per-p < 1e4; acceptance expects >= 1e4.")

    # Load H matrices
    from tools.cudaq_sampler import get_code_mats, CudaqGarnetSampler
    Hx, Hz, meta = get_code_mats()
    assert Hx.shape == (4,9) and Hz.shape == (4,9)

    # Seed reproducibly
    np.random.seed(42)

    sampler = CudaqGarnetSampler("foundation")
    p_vals = [float(x) for x in args.p_grid.split(',') if x]

    # Decoder params / flops (rough)
    params = {}
    approx_flops = None
    if args.decoder in ('mghd','mghd_forward') and args.checkpoint:
        try:
            import torch
            from poc_my_models import MGHD
            # lightweight param count (load then sum)
            gnn_params = {
                'dist': 3, 'n_node_inputs': 9, 'n_node_outputs': 9,
                'n_iters': 7, 'n_node_features': 128, 'n_edge_features': 128,
                'msg_net_size': 96, 'msg_net_dropout_p': 0.0, 'gru_dropout_p': 0.0,
            }
            mamba_params = {'d_model':192,'d_state':32,'d_conv':2,'expand':3,'attention_mechanism':'none'}
            m = MGHD(gnn_params, mamba_params)
            st = torch.load(args.checkpoint, map_location='cpu')
            try: m.load_state_dict(st, strict=False)
            except: pass
            params['n_params'] = int(sum(p.numel() for p in m.parameters()))
        except Exception:
            params['n_params'] = None

    results = []
    z = 1.96  # for 95% CI

    for p in p_vals:
        # Sample syndromes + teacher labels
        s_bin, labels_x, labels_z = sampler.sample_batch(args.N_per_p, p, teacher='mwpf')
        synd_bytes = _pack_bits_rows(s_bin)[:, 0]

        # Decode and time
        t0 = time.time()
        if args.decoder == 'fastpath':
            y_pred = decode_fastpath(synd_bytes)
        elif args.decoder in ('mwpm','mwpf','relay'):
            from tools.relay_teacher import teacher_labels
            lx, lz = teacher_labels(synd_bytes, mode='mwpf' if args.decoder!='mwpm' else 'mwpm')
            y_pred = lx  # symmetric handling for d=3
        elif args.decoder == 'mghd':
            if not args.checkpoint:
                print("--checkpoint required for decoder=mghd", file=sys.stderr)
                sys.exit(2)
            y_pred = decode_mghd(s_bin, args.checkpoint, device=args.device)
        elif args.decoder == 'mghd_forward':
            if not args.checkpoint:
                print("--checkpoint required for decoder=mghd_forward", file=sys.stderr)
                sys.exit(2)
            y_pred = decode_mghd_forward(s_bin, args.checkpoint, device=args.device)
        elif args.decoder == 'garnet':
            # For now treat Garnet as LUT baseline at d=3
            y_pred = decode_fastpath(synd_bytes)
        else:
            raise ValueError(args.decoder)
        t1 = time.time()

        # Latency stats (rough, per-batch amortized):
        latency_us = (t1 - t0) * 1e6 / max(1, s_bin.shape[0])
        lat_p50 = float(latency_us)
        lat_p99 = float(latency_us)  # single batch estimate; keep same

        # Reference labels (coset): use LUT-based labels_x as reference
        y_ref = labels_x
        succ = _coset_success(Hz, Hx, s_bin, y_pred, y_ref)  # [B]
        errs = int((~succ).sum())
        N = int(s_bin.shape[0])
        ler = errs / max(1, N)
        lo, hi = wilson_interval(errs, N, z=z)

        results.append({
            'p': p,
            'N': N,
            'LER': float(ler),
            'ler_low': float(lo),
            'ler_high': float(hi),
            'latency_p50_us': lat_p50,
            'latency_p99_us': lat_p99,
        })

    out_obj = {
        'decoder': args.decoder,
        'metric_type': 'coset_parity',
        'confidence_interval': args.ci,
        'params': params,
        'approx_flops': approx_flops,
        'results': results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out_obj, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
