#!/usr/bin/env python3
"""
Optuna sweep for S_core MGHD (size-locked) on CUDA-Q Garnet at p=0.05.

Goal: keep S footprint small (baseline S architecture) and sweep only
non-size hyperparameters that affect optimization and calibration.

Objective: minimize forward-path LER@p=0.05 after a short run (6 epochs).
"""
from __future__ import annotations

import os
import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
import optuna

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.cudaq_sampler import CudaqGarnetSampler, get_code_mats
from tools.eval_ler import _coset_success
from poc_my_models import MGHD


@dataclass
class Cfg:
    trials: int = 32
    epochs: int = 6
    steps_per_epoch: int = 800
    batch_size: int = 512
    seed: int = 42
    outdir: str = 'results/optuna_S_core'


def build_s_core(device: torch.device, post_ln: bool) -> torch.nn.Module:
    # Baseline S sizes (size-locked)
    gnn = dict(
        dist=3,
        n_node_inputs=9,
        n_node_outputs=9,
        n_iters=7,
        n_node_features=128,
        n_edge_features=128,
        msg_net_size=96,
        msg_net_dropout_p=0.04,
        gru_dropout_p=0.11,
    )
    mmb = dict(
        d_model=192,
        d_state=32,
        d_conv=2,
        expand=3,
        attention_mechanism='channel_attention',
        se_reduction=4,
        post_mamba_ln=bool(post_ln),
    )
    m = MGHD(gnn_params=gnn, mamba_params=mmb).to(device)
    try:
        m.set_rotated_layout()
        m._ensure_static_indices(device)
    except Exception:
        pass
    return m


def objective(trial: optuna.Trial, cfg: Cfg, device: torch.device, Hx: np.ndarray, Hz: np.ndarray) -> float:
    # Sweep ONLY non-size hyperparameters
    lr = trial.suggest_float('lr', 3e-5, 1.5e-4, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 2e-4, log=True)
    label_smoothing = trial.suggest_float('label_smoothing', 0.05, 0.15)
    grad_clip = trial.suggest_float('grad_clip', 0.5, 1.5)
    msg_drop = trial.suggest_float('msg_dropout', 0.02, 0.08)
    gru_drop = trial.suggest_float('gru_dropout', 0.06, 0.14)
    ema_decay = trial.suggest_categorical('ema_decay', [0.0, 0.998, 0.999, 0.9995])
    parity_lambda = trial.suggest_categorical('parity_lambda', [0.0, 0.025, 0.05, 0.1])
    post_ln = trial.suggest_categorical('post_mamba_ln', [False, True])
    lr_schedule = trial.suggest_categorical('lr_schedule', ['constant', 'cosine'])

    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(cfg.seed)

    model = build_s_core(device, post_ln)
    # apply dropouts
    model.gnn.msg_net_dropout_p = float(msg_drop)
    model.gnn.gru_dropout_p = float(gru_drop)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_epochs = cfg.epochs
    if lr_schedule == 'cosine':
        warm = max(1, int(0.05 * total_epochs))
        base_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_epochs)
        def lr_lambda(ep):
            if ep < warm: return (ep + 1) / float(max(1, warm))
            return 1.0
        warm_sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    else:
        base_sched = None
        warm_sched = None

    use_amp = device.type == 'cuda'
    amp_dtype = torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    sampler = CudaqGarnetSampler('foundation')
    ema_state = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point} if ema_decay else None

    best = 1.0
    for ep in range(cfg.epochs):
        model.train()
        for step in range(cfg.steps_per_epoch):
            s_bin, labels_x, _ = sampler.sample_batch(cfg.batch_size, p=0.05)
            y = torch.from_numpy(labels_x.astype(np.float32)).to(device)
            B = s_bin.shape[0]
            node_inputs = torch.zeros(B, 17, 9, device=device, dtype=torch.float32)
            node_inputs[:, :8, 0] = torch.from_numpy(s_bin.astype(np.float32)).to(device)
            flat = node_inputs.view(-1, 9)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                outs = model(flat, getattr(model, '_src_ids', None), getattr(model, '_dst_ids', None))
                final = outs[-1].view(B, 17, -1)[:, 8:, :]
                bitlogits = (final[..., 1] - final[..., 0]) if final.shape[-1] == 2 else final.squeeze(-1)
                bitlogits = torch.nan_to_num(bitlogits, nan=0.0, posinf=30.0, neginf=-30.0)
                if label_smoothing > 0.0:
                    y_eff = y * (1.0 - label_smoothing) + 0.5 * label_smoothing
                else:
                    y_eff = y
                loss_main = F.binary_cross_entropy_with_logits(bitlogits, y_eff, reduction='mean')
                if parity_lambda and parity_lambda > 0.0:
                    p = bitlogits.sigmoid(); q = 1.0 - 2.0 * p; q_exp = q.unsqueeze(1)
                    Hx_t = torch.from_numpy(Hx.astype(np.uint8)).to(device)
                    Hz_t = torch.from_numpy(Hz.astype(np.uint8)).to(device)
                    mz = (Hz_t.to(torch.float32) > 0).unsqueeze(0)
                    mx = (Hx_t.to(torch.float32) > 0).unsqueeze(0)
                    z_prod = torch.where(mz, q_exp, torch.ones_like(q_exp)).prod(dim=2)
                    x_prod = torch.where(mx, q_exp, torch.ones_like(q_exp)).prod(dim=2)
                    z_par = 0.5 * (1.0 - z_prod); x_par = 0.5 * (1.0 - x_prod)
                    sZ = torch.from_numpy(s_bin[:, :4].astype(np.float32)).to(device)
                    sX = torch.from_numpy(s_bin[:, 4:8].astype(np.float32)).to(device)
                    with torch.cuda.amp.autocast(enabled=False):
                        loss_par = F.binary_cross_entropy(z_par.float(), sZ.float()) + F.binary_cross_entropy(x_par.float(), sX.float())
                    loss = loss_main + float(parity_lambda) * loss_par
                else:
                    loss = loss_main
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
            if ema_state is not None:
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point and k in ema_state:
                        ema_state[k].mul_(ema_decay).add_(v.detach(), alpha=(1.0 - ema_decay))

        # Validation at p=0.05 using EMA if available
        model.eval()
        with torch.no_grad():
            save_buf = {}
            if ema_state is not None:
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point and k in ema_state:
                        save_buf[k] = v.detach().clone(); v.data.copy_(ema_state[k])
            v_s, v_lx, _ = sampler.sample_batch(4096, p=0.05)
            Bv = v_s.shape[0]
            node_inputs = torch.zeros(Bv, 17, 9, device=device, dtype=torch.float32)
            node_inputs[:, :8, 0] = torch.from_numpy(v_s.astype(np.float32)).to(device)
            final = model(node_inputs.view(-1, 9), getattr(model, '_src_ids', None), getattr(model, '_dst_ids', None))[-1]
            final = final.view(Bv, 17, -1)[:, 8:, :]
            bitlogits = (final[..., 1] - final[..., 0]) if final.shape[-1] == 2 else final.squeeze(-1)
            y_pred = (bitlogits.sigmoid() > 0.5).to(torch.uint8).cpu().numpy()
            succ = _coset_success(Hz, Hx, v_s, y_pred, v_lx)
            val_ler = 1.0 - float(succ.mean())
            if ema_state is not None:
                for k, v in model.state_dict().items():
                    if k in save_buf: v.data.copy_((save_buf[k]))
            best = min(best, val_ler)

        if warm_sched is not None: warm_sched.step(ep)
        if base_sched is not None: base_sched.step()

        trial.report(best, ep)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Optuna sweep for S_core (size-locked)')
    ap.add_argument('--trials', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--steps-per-epoch', type=int, default=800)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--outdir', type=str, default='results/optuna_S_core')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = Cfg(trials=args.trials, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch, batch_size=args.batch_size, seed=args.seed, outdir=str(outdir))
    with open(outdir / 'cfg.json', 'w') as f: json.dump(cfg.__dict__, f, indent=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Hx, Hz, _ = get_code_mats()
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=2))
    study.optimize(lambda tr: objective(tr, cfg, device, Hx, Hz), n_trials=cfg.trials)
    with open(outdir / 'best_params.json', 'w') as f: json.dump(study.best_params, f, indent=2)
    with open(outdir / 'study_summary.json', 'w') as f: json.dump({'best_value': study.best_value, 'best_trial': study.best_trial.number}, f, indent=2)
    print('Best LER:', study.best_value)
    print('Best params saved to', outdir / 'best_params.json')


if __name__ == '__main__':
    main()

