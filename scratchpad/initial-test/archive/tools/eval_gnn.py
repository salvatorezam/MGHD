#!/usr/bin/env python3
"""
Evaluate GNN baseline checkpoint on a p-grid (circuit-level, rotated d=3) with MWPF teacher.
Outputs JSON with LER ± 95% CI per p.
"""
from __future__ import annotations
import json, argparse
from pathlib import Path
import numpy as np
import torch

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from panq_functions import GNNDecoder
from tools.cudaq_sampler import CudaqGarnetSampler, get_code_mats
from tools.eval_ler import _coset_success, wilson_interval


def build_indices_from_H(Hx: np.ndarray, Hz: np.ndarray, device: torch.device):
    num_z, num_data = Hz.shape
    num_x, _ = Hx.shape
    num_checks = num_z + num_x
    src, dst = [], []
    # Z then X edges
    for i in range(num_z):
        for j in range(num_data):
            if Hz[i, j]: src.append(i); dst.append(num_checks + j)
    for i in range(num_x):
        for j in range(num_data):
            if Hx[i, j]: src.append(num_z + i); dst.append(num_checks + j)
    # reverse
    for i in range(num_z):
        for j in range(num_data):
            if Hz[i, j]: src.append(num_checks + j); dst.append(i)
    for i in range(num_x):
        for j in range(num_data):
            if Hx[i, j]: src.append(num_checks + j); dst.append(num_z + i)
    return (torch.tensor(src, device=device, dtype=torch.long),
            torch.tensor(dst, device=device, dtype=torch.long), num_checks, num_data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--p-grid', required=True, help='comma-separated p list')
    ap.add_argument('--N-per-p', type=int, default=10000)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Hx, Hz, _ = get_code_mats()
    src, dst, num_checks, num_data = build_indices_from_H(Hx, Hz, device)

    gnn = GNNDecoder(dist=3, n_iters=7, n_node_features=128, n_node_inputs=9, n_edge_features=128, n_node_outputs=2,
                     msg_net_size=96, msg_net_dropout_p=0.04, gru_dropout_p=0.11).to(device).eval()
    st = torch.load(args.checkpoint, map_location=device)
    try: gnn.load_state_dict(st, strict=False)
    except Exception: pass

    sampler = CudaqGarnetSampler('foundation')
    pvals = [float(x) for x in args.p_grid.split(',') if x]
    out = []
    for p in pvals:
        s, lx, lz = sampler.sample_batch(args.N_per_p, p, teacher='mwpf')
        B = s.shape[0]
        vin = torch.zeros(B, num_checks+num_data, 9, device=device, dtype=torch.float32)
        vin[:, :num_checks, 0] = torch.from_numpy(s.astype(np.float32)).to(device)
        with torch.no_grad():
            vout = gnn(vin.view(-1,9), src, dst)[-1].view(B, num_checks+num_data, -1)[:, num_checks:, :]
            logits = (vout[...,1]-vout[...,0]) if vout.shape[-1]==2 else vout.squeeze(-1)
            pred = (logits.sigmoid()>0.5).to(torch.uint8).cpu().numpy()
        succ = _coset_success(Hz, Hx, s, pred, lx)
        err = int((~succ).sum()); N = B
        ler = err/N; lo, hi = wilson_interval(err, N)
        out.append(dict(p=p, N=N, LER=float(ler), ler_low=lo, ler_high=hi, latency_p50_us=0.0, latency_p99_us=0.0))

    obj = dict(decoder='gnn_baseline', metric_type='coset_parity', confidence_interval=0.95, params={}, results=out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out,'w') as f: json.dump(obj, f, indent=2)
    print('Saved', args.out)

if __name__ == '__main__':
    main()

