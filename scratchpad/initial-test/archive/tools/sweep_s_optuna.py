#!/usr/bin/env python3
"""
Optuna sweep for S-profile MGHD on CUDA-Q Garnet (p=0.05 focus).

Objective: minimize forward-path LER@p=0.05 after a short run.
Default: 6 epochs, 800 steps/epoch, batch 512, bf16 AMP, channel attention on.

Writes trial logs and best params under an outdir.
"""
from __future__ import annotations

import os
import sys
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.cudaq_sampler import CudaqGarnetSampler, get_code_mats
from tools.eval_ler import _coset_success
from poc_my_models import MGHD


@dataclass
class SweepCfg:
    trials: int = 24
    epochs: int = 6
    steps_per_epoch: int = 800
    batch_size: int = 512
    seed: int = 42
    outdir: str = 'results/optuna_S'


def build_model(params: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    # S profile skeleton; override by params
    gnn = dict(
        dist=3,
        n_node_inputs=9,
        n_node_outputs=9,
        n_iters=int(params.get('n_iters', 7)),
        n_node_features=int(params.get('n_node_features', 128)),
        n_edge_features=int(params.get('n_edge_features', 128)),
        msg_net_size=int(params.get('msg_net_size', 96)),
        msg_net_dropout_p=float(params.get('msg_net_dropout_p', 0.04)),
        gru_dropout_p=float(params.get('gru_dropout_p', 0.11)),
    )
    mmb = dict(
        d_model=int(params.get('mamba_d_model', 192)),
        d_state=int(params.get('mamba_d_state', 32)),
        d_conv=2,
        expand=int(params.get('mamba_expand', 3)),
        attention_mechanism='channel_attention',
        se_reduction=int(params.get('se_reduction', 4)),
    )
    model = MGHD(gnn_params=gnn, mamba_params=mmb).to(device)
    try:
        model.set_rotated_layout()
        model._ensure_static_indices(device)
    except Exception:
        pass
    return model


def trial_objective(trial: optuna.Trial, cfg: SweepCfg, device: torch.device, Hx: np.ndarray, Hz: np.ndarray) -> float:
    # Search space anchored on prior good S values
    params = {
        'lr': trial.suggest_float('lr', 3e-5, 1.2e-4, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 3e-5, 3e-4, log=True),
        'n_iters': trial.suggest_int('n_iters', 6, 8),
        'n_node_features': trial.suggest_categorical('n_node_features', [96, 128, 160]),
        'n_edge_features': trial.suggest_categorical('n_edge_features', [128, 192, 256, 384]),
        'msg_net_size': trial.suggest_categorical('msg_net_size', [80, 96, 112, 128]),
        'msg_net_dropout_p': trial.suggest_float('msg_net_dropout_p', 0.02, 0.08),
        'gru_dropout_p': trial.suggest_float('gru_dropout_p', 0.06, 0.14),
        'mamba_d_model': trial.suggest_categorical('mamba_d_model', [160, 192, 224]),
        'mamba_d_state': trial.suggest_int('mamba_d_state', 32, 80, step=16),
        'mamba_expand': trial.suggest_categorical('mamba_expand', [3, 4]),
        'se_reduction': trial.suggest_categorical('se_reduction', [4, 8]),
        'label_smoothing': trial.suggest_float('label_smoothing', 0.08, 0.16),
        'gradient_clip': trial.suggest_float('gradient_clip', 0.5, 1.5),
        'noise_injection': trial.suggest_float('noise_injection', 0.0, 0.01),
    }

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(cfg.seed)

    model = build_model(params, device)
    opt = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    use_amp = True if device.type == 'cuda' else False
    amp_dtype = torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == 'cuda')

    sampler = CudaqGarnetSampler('foundation')
    label_smoothing = float(params['label_smoothing'])
    grad_clip = float(params['gradient_clip'])
    # Implement BCE label smoothing manually: y' = (1-s)*y + 0.5*s

    best = 1.0
    for ep in range(cfg.epochs):
        model.train()
        for step in range(cfg.steps_per_epoch):
            s_bin, labels_x, labels_z = sampler.sample_batch(cfg.batch_size, p=0.05)
            y = torch.from_numpy(labels_x.astype(np.float32)).to(device)
            B = s_bin.shape[0]
            node_inputs = torch.zeros(B, 17, 9, device=device, dtype=torch.float32)
            node_inputs[:, :8, 0] = torch.from_numpy(s_bin.astype(np.float32)).to(device)
            flat_inputs = node_inputs.view(-1, 9)
            if params['noise_injection'] > 0:
                flat_inputs += params['noise_injection'] * torch.randn_like(flat_inputs)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                outs = model(flat_inputs, getattr(model, '_src_ids', None), getattr(model, '_dst_ids', None))
                final = outs[-1].view(B, 17, -1)[:, 8:, :]
                bitlogits = (final[..., 1] - final[..., 0]) if final.shape[-1] == 2 else final.squeeze(-1)
                bitlogits = torch.nan_to_num(bitlogits, nan=0.0, posinf=30.0, neginf=-30.0)
                y_smooth = y * (1.0 - label_smoothing) + 0.5 * label_smoothing
                loss = F.binary_cross_entropy_with_logits(bitlogits, y_smooth, reduction='mean')
            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()

        # Validation at p=0.05
        model.eval()
        with torch.no_grad():
            v_s, v_lx, v_lz = sampler.sample_batch(4096, p=0.05)
            Bv = v_s.shape[0]
            node_inputs = torch.zeros(Bv, 17, 9, device=device, dtype=torch.float32)
            node_inputs[:, :8, 0] = torch.from_numpy(v_s.astype(np.float32)).to(device)
            final = model(node_inputs.view(-1, 9), getattr(model, '_src_ids', None), getattr(model, '_dst_ids', None))[-1]
            final = final.view(Bv, 17, -1)[:, 8:, :]
            bitlogits = (final[..., 1] - final[..., 0]) if final.shape[-1] == 2 else final.squeeze(-1)
            y_pred = (bitlogits.sigmoid() > 0.5).to(torch.uint8).cpu().numpy()
            succ = _coset_success(Hz, Hx, v_s, y_pred, v_lx)
            val_ler = 1.0 - float(succ.mean())
            best = min(best, val_ler)
        trial.report(best, ep)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Optuna sweep for S-profile MGHD')
    ap.add_argument('--trials', type=int, default=24)
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--steps-per-epoch', type=int, default=800)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--outdir', type=str, default='results/optuna_S')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = SweepCfg(trials=args.trials, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
                   batch_size=args.batch_size, seed=args.seed, outdir=str(outdir))
    with open(outdir / 'cfg.json', 'w') as f:
        json.dump(cfg.__dict__, f, indent=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Hx, Hz, meta = get_code_mats()
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=2))
    obj = lambda tr: trial_objective(tr, cfg, device, Hx, Hz)
    study.optimize(obj, n_trials=cfg.trials)

    with open(outdir / 'best_params.json', 'w') as f:
        json.dump(study.best_params, f, indent=2)
    with open(outdir / 'study_summary.json', 'w') as f:
        json.dump({'best_value': study.best_value, 'best_trial': study.best_trial.number}, f, indent=2)
    print('Best LER:', study.best_value)
    print('Best params saved to', outdir / 'best_params.json')


if __name__ == '__main__':
    main()
