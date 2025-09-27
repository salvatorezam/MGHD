#!/usr/bin/env python3
"""
Baseline GNN (Astra-style) trainer on CUDA-Q Garnet (rotated d=3), circuit-level p curriculum.

Trains only the GNNDecoder (no Mamba), using the same sampler/teacher and loss setup as MGHD.
Writes best checkpoint and per-grid LER JSON (N=10k per p) on completion.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from panq_functions import GNNDecoder
from tools.cudaq_sampler import CudaqGarnetSampler, get_code_mats
from tools.eval_ler import _coset_success, wilson_interval


def build_indices_from_H(Hx: np.ndarray, Hz: np.ndarray, device: torch.device):
    # Canonical ordering: Z checks first, then X checks
    num_z_checks, num_data = Hz.shape
    num_x_checks, _ = Hx.shape
    num_checks = num_z_checks + num_x_checks
    total_nodes = num_checks + num_data
    src_ids, dst_ids = [], []
    # Z checks
    for i in range(num_z_checks):
        for j in range(num_data):
            if Hz[i, j]:
                src_ids.append(i)
                dst_ids.append(num_checks + j)
    # X checks
    for i in range(num_x_checks):
        for j in range(num_data):
            if Hx[i, j]:
                src_ids.append(num_z_checks + i)
                dst_ids.append(num_checks + j)
    # Reverse edges
    for i in range(num_z_checks):
        for j in range(num_data):
            if Hz[i, j]:
                src_ids.append(num_checks + j)
                dst_ids.append(i)
    for i in range(num_x_checks):
        for j in range(num_data):
            if Hx[i, j]:
                src_ids.append(num_checks + j)
                dst_ids.append(num_z_checks + i)
    return (torch.tensor(src_ids, device=device, dtype=torch.long),
            torch.tensor(dst_ids, device=device, dtype=torch.long),
            num_checks, num_data)


def train_baseline(outdir: Path, epochs=20, steps_per_epoch=1200, batch_size=512, lr=1e-4, wd=1e-4,
                   ema_decay=0.999, parity_lambda=0.03, smoothing=0.09, teacher='mwpf', curriculum=True, seed=42):
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(outdir/'args.json','w') as f:
        json.dump(dict(epochs=epochs, steps_per_epoch=steps_per_epoch, batch_size=batch_size, lr=lr, wd=wd,
                       ema_decay=ema_decay, parity_lambda=parity_lambda, smoothing=smoothing, teacher=teacher,
                       curriculum=curriculum, seed=seed), f, indent=2)
    (outdir/'cmd.txt').write_text(' '.join(map(str, sys.argv)))
    # H matrices and indices
    Hx, Hz, meta = get_code_mats()
    Hx_t = torch.from_numpy(Hx.astype(np.uint8)); Hz_t = torch.from_numpy(Hz.astype(np.uint8))
    src_ids, dst_ids, num_checks, num_data = build_indices_from_H(Hx, Hz, device)
    # Model
    gnn = GNNDecoder(dist=3, n_iters=7, n_node_features=128, n_node_inputs=9, n_edge_features=128, n_node_outputs=2,
                     msg_net_size=96, msg_net_dropout_p=0.04, gru_dropout_p=0.11).to(device)
    opt = optim.AdamW(gnn.parameters(), lr=lr, weight_decay=wd)
    # Sampler
    sampler = CudaqGarnetSampler('foundation')
    rng = np.random.default_rng(seed)
    # EMA
    use_ema = ema_decay > 0.0
    ema_state = {k: v.detach().clone() for k, v in gnn.state_dict().items() if v.dtype.is_floating_point} if use_ema else None
    # Training curriculum grid
    grid = [0.001,0.002,0.003,0.004,0.005,0.006,0.008,0.010,0.012,0.015]
    def bucket_w(p):
        return 0.40/4.0 if 0.001 <= p <= 0.004 else 0.45/4.0 if 0.004 < p <= 0.008 else 0.15/3.0
    def stage_boost(p, ep):
        return 3.0 if ep < 2 and (0.008 <= p <= 0.010) else (2.0 if (2 <= ep < 7 and 0.005 <= p <= 0.007) or (ep>=7 and 0.003 <= p <= 0.006) else 1.0)
    # Metrics CSV
    (outdir/'metrics.csv').write_text('epoch,train_loss_mean,val_ler,samples_epoch\n')
    best_val = 1.0
    for ep in range(epochs):
        gnn.train(); losses=[]; samples=0
        for step in range(steps_per_epoch):
            if curriculum:
                weights = np.array([bucket_w(p)*stage_boost(p, ep) for p in grid], dtype=np.float64);
                weights/=weights.sum()
                p_train = float(rng.choice(grid, p=weights))
            else:
                p_train = 0.05
            s_bin, labels_x, labels_z = sampler.sample_batch(batch_size, p_train, teacher=teacher, rng=rng)
            samples += int(s_bin.shape[0])
            B = s_bin.shape[0]
            # Node inputs
            node_inputs = torch.zeros(B, num_checks+num_data, 9, device=device, dtype=torch.float32)
            node_inputs[:, :num_checks, 0] = torch.from_numpy(s_bin.astype(np.float32)).to(device)
            flat = node_inputs.view(-1, 9)
            y = torch.from_numpy(labels_x.astype(np.float32)).to(device)
            # p-aware smoothing
            s_eff = smoothing * (0.6 if p_train >= 0.05 else 1.0)
            y_eff = y * (1.0 - s_eff) + 0.5 * s_eff
            # Forward
            outs = gnn(flat, src_ids, dst_ids)
            final = outs[-1].view(B, num_checks+num_data, -1)[:, num_checks:, :]
            bitlogits = (final[...,1]-final[...,0]) if final.shape[-1]==2 else final.squeeze(-1)
            bitlogits = torch.nan_to_num(bitlogits, nan=0.0, posinf=30.0, neginf=-30.0)
            loss_main = nn.functional.binary_cross_entropy_with_logits(bitlogits, y_eff, reduction='mean')
            loss = loss_main
            # Parity aux
            if parity_lambda>0:
                pprob = bitlogits.sigmoid(); q = 1.0 - 2.0*pprob; qx=q.unsqueeze(1)
                mz= (Hz_t.to(torch.float32)>0).to(device).unsqueeze(0); mx=(Hx_t.to(torch.float32)>0).to(device).unsqueeze(0)
                z_prod = torch.where(mz, qx, torch.ones_like(qx)).prod(dim=2); x_prod = torch.where(mx, qx, torch.ones_like(qx)).prod(dim=2)
                z_par = 0.5*(1.0 - z_prod); x_par = 0.5*(1.0 - x_prod)
                sZ_t = torch.from_numpy(s_bin[:,:4].astype(np.float32)).to(device); sX_t = torch.from_numpy(s_bin[:,4:8].astype(np.float32)).to(device)
                loss_par = nn.functional.binary_cross_entropy(z_par.float(), sZ_t.float()) + nn.functional.binary_cross_entropy(x_par.float(), sX_t.float())
                loss = loss + parity_lambda*loss_par
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(gnn.parameters(), 1.0); opt.step()
            if use_ema:
                for k,v in gnn.state_dict().items():
                    if v.dtype.is_floating_point and k in ema_state:
                        ema_state[k].mul_(ema_decay).add_(v.detach(), alpha=(1.0-ema_decay))
            losses.append(float(loss.detach().cpu().item()))
        # Validation at p=0.05 for tracking
        gnn.eval();
        with torch.no_grad():
            vs, vlx, vlz = sampler.sample_batch(2048, 0.05, teacher=teacher, rng=rng)
            Bv = vs.shape[0]
            vin = torch.zeros(Bv, num_checks+num_data, 9, device=device, dtype=torch.float32)
            vin[:, :num_checks, 0] = torch.from_numpy(vs.astype(np.float32)).to(device)
            vout = gnn(vin.view(-1,9), src_ids, dst_ids)[-1].view(Bv, num_checks+num_data, -1)[:, num_checks:, :]
            logits = (vout[...,1]-vout[...,0]) if vout.shape[-1]==2 else vout.squeeze(-1)
            pred = (logits.sigmoid()>0.5).to(torch.uint8).cpu().numpy()
            succ = _coset_success(Hz, Hx, vs, pred, vlx); val = 1.0 - float(succ.mean())
        (outdir/'metrics.csv').open('a').write(f"{ep+1},{np.mean(losses):.6f},{val:.6f},{samples}\n")
        if val < best_val:
            best_val = val; torch.save(gnn.state_dict(), outdir/'gnn_baseline_best.pt')
    # Final per-grid LER
    results=[]
    for p in [0.02,0.03,0.05,0.08]:
        s, lx, lz = sampler.sample_batch(10000, p, teacher=teacher)
        B = s.shape[0]
        vin = torch.zeros(B, num_checks+num_data, 9, device=device, dtype=torch.float32)
        vin[:, :num_checks, 0] = torch.from_numpy(s.astype(np.float32)).to(device)
        vout = gnn(vin.view(-1,9), src_ids, dst_ids)[-1].view(B, num_checks+num_data, -1)[:, num_checks:, :]
        logits = (vout[...,1]-vout[...,0]) if vout.shape[-1]==2 else vout.squeeze(-1)
        pred = (logits.sigmoid()>0.5).to(torch.uint8).cpu().numpy()
        succ = _coset_success(Hz, Hx, s, pred, lx); err = int((~succ).sum()); N=B
        ler = err/N; lo, hi = wilson_interval(err, N)
        results.append(dict(p=p, N=N, LER=float(ler), ler_low=lo, ler_high=hi, latency_p50_us=0.0, latency_p99_us=0.0))
    with open(outdir/'ler_gnn_baseline.json','w') as f:
        json.dump(dict(decoder='gnn_baseline', metric_type='coset_parity', confidence_interval=0.95, params={}, results=results), f, indent=2)
    print('Saved', outdir/'ler_gnn_baseline.json')


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Baseline GNN trainer (rotated d=3)')
    ap.add_argument('--outdir', type=str, required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--steps-per-epoch', type=int, default=1200)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--ema', type=float, default=0.999)
    ap.add_argument('--parity-lambda', type=float, default=0.03)
    ap.add_argument('--smoothing', type=float, default=0.09)
    ap.add_argument('--teacher', type=str, default='mwpf')
    ap.add_argument('--no-curriculum', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    train_baseline(Path(args.outdir), epochs=args.epochs, steps_per_epoch=args.steps_per_epoch, batch_size=args.batch_size,
                   lr=args.lr, wd=args.wd, ema_decay=args.ema, parity_lambda=args.parity_lambda,
                   smoothing=args.smoothing, teacher=args.teacher, curriculum=(not args.no_curriculum), seed=args.seed)

if __name__ == '__main__':
    main()

