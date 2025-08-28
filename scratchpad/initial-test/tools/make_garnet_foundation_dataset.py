#!/usr/bin/env python3
"""
Generate a rotated d=3 Garnet foundation dataset using CUDA-Q sampler and MWPF/MWPM teachers.

Outputs NPZ with:
  - syndromes: uint8 [B, 8]
  - labels_x: uint8 [B, 9]
  - labels_z: uint8 [B, 9]
  - meta: dict with Hx/Hz ordering and hashes (JSON-serialized)
"""
import argparse, json, hashlib
import numpy as np
from pathlib import Path

from tools.cudaq_sampler import CudaqGarnetSampler, get_code_mats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-B', '--batch', type=int, default=20000)
    ap.add_argument('--mode', choices=['foundation','student'], default='foundation')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('-o', '--out', type=str, required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sam = CudaqGarnetSampler(args.mode)
    # Sample p randomly per-batch draw near 0.02..0.08
    p = float(rng.choice([0.02,0.03,0.05,0.08]))
    s, lx, lz = sam.sample_batch(args.batch, p=p, teacher='mwpf', rng=rng)

    # Build meta
    Hx, Hz, meta0 = get_code_mats()
    meta = dict(meta0)
    meta['surface_layout'] = 'rotated'
    meta['distance'] = 3
    meta_json = json.dumps(meta)
    
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Save labels file
    np.savez_compressed(out, syndromes=s, labels_x=lx, labels_z=lz, meta=meta_json)
    # Save canonical pack alongside
    hx_hash = hashlib.sha256(Hx.tobytes()).hexdigest()
    hz_hash = hashlib.sha256(Hz.tobytes()).hexdigest()
    packed = np.zeros((s.shape[0], 1), dtype=np.uint8)
    for i in range(s.shape[0]):
        val = 0
        for k in range(8):
            val |= (int(s[i, k]) & 1) << k
        packed[i, 0] = val
    pack_path = out.with_suffix('').with_name(out.stem + '_pack.npz')
    np.savez_compressed(pack_path, syndromes=packed, Hx=Hx, Hz=Hz, meta=meta_json,
                        Hx_hash=hx_hash, Hz_hash=hz_hash)
    print(f"Saved labels {out} and pack {pack_path} with B={args.batch} p={p}")

if __name__ == '__main__':
    main()
