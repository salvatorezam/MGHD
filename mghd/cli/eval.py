"""Standalone circuit-level DEM evaluation for trained MGHD checkpoints.

Loads a checkpoint, sweeps a (distance, p) grid, samples Stim detection
events, runs **batched** model inference with active-component decomposition
(matching training), computes observable corrections via L_obs, and compares
to ground truth + MWPM teacher.

Key design: each shot is decomposed into small connected components (just
like training with ``--component-scope active``).  Per-component predictions
are reassembled into the full edge vector before computing the observable
correction.  Zero-syndrome shots short-circuit to all-zeros.

Outputs structured JSON with per-(d, p) LER, Wilson 95% CIs, and teacher
LER for direct comparison.

Usage example
-------------
    python -m mghd.cli.eval \\
        --checkpoint data/.../best.pt \\
        --distances 3,5,7 \\
        --p-values 0.001,0.003,0.005,0.01 \\
        --shots 10000 \\
        --output eval_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix as _csr
from scipy.sparse.csgraph import connected_components as _cc

from mghd.core.core import MGHDv2, PackedCrop
from mghd.cli.train import (
    _build_dem_info,
    _teacher_labels_from_matching,
    collate_packed,
    compute_observable_correction,
    move_to,
    pack_dem_cluster,
)
from mghd.utils.metrics import _wilson_interval


# ── CLI ───────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MGHD checkpoint on circuit-level DEM tasks.",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to checkpoint (.pt) with 'model' and 'model_config' keys.",
    )
    parser.add_argument(
        "--distances", type=str, required=True,
        help="Comma-separated code distances to evaluate, e.g. '3,5,7,9'.",
    )
    parser.add_argument(
        "--p-values", type=str, required=True,
        help="Comma-separated physical error rates, e.g. '0.001,0.003,0.005,0.01'.",
    )
    parser.add_argument(
        "--shots", type=int, default=10_000,
        help="Number of Stim shots per (d, p) point (default: 10000).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Inference batch size (default: 64).",
    )
    parser.add_argument(
        "--noise-model", type=str, default=None,
        help="Noise model override (default: use checkpoint's noise_model).",
    )
    parser.add_argument(
        "--edge-prune-thresh", type=float, default=1e-6,
        help="Prune DEM edges with probability below this threshold (default: 1e-6).",
    )
    parser.add_argument(
        "--rounds", type=int, default=0,
        help="Number of QEC rounds (0 = use distance).",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'auto', 'cpu', 'cuda', or 'cuda:N'.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path (default: print to stdout).",
    )
    parser.add_argument(
        "--amp", type=str, default="off", choices=["off", "bf16", "fp16"],
        help="Automatic mixed-precision mode (default: off).",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="CPU workers for parallel shot decomposition (0=auto, uses ~75%% of cores).",
    )
    return parser


# ── Model loading ─────────────────────────────────────────────────────────


def load_model(ckpt_path: str, device: torch.device) -> tuple[MGHDv2, dict]:
    """Load checkpoint and build MGHDv2 from saved model_config.

    When ``model_config`` is missing (older checkpoints), infer dimensions
    from the saved weight shapes.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_config", {})
    sd = ckpt["model"]

    # Infer dimensions from weight shapes when config is absent/incomplete
    node_feat_dim = int(cfg.get("node_feat_dim", sd["node_in.weight"].shape[1]))
    edge_feat_dim = int(cfg.get("edge_feat_dim", sd["edge_in.weight"].shape[1]))
    g_dim = int(cfg.get("g_token_dim", sd["g_proj.weight"].shape[1]))
    d_model = int(cfg.get("d_model", sd["node_in.weight"].shape[0]))

    profile = cfg.get("profile", "S")
    model = MGHDv2(
        profile=profile,
        d_model=d_model,
        d_state=int(cfg.get("d_state", 80)),
        n_iters=int(cfg.get("n_iters", 8)),
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        g_dim=g_dim,
    )
    model.load_state_dict(sd, strict=False)
    model = model.to(device)
    model.eval()

    # Store inferred dims back into cfg for downstream use
    cfg.setdefault("node_feat_dim", node_feat_dim)
    cfg.setdefault("edge_feat_dim", edge_feat_dim)
    cfg.setdefault("g_token_dim", g_dim)
    cfg.setdefault("d_model", d_model)

    print(
        f"Loaded checkpoint: epoch={ckpt.get('epoch', '?')}, "
        f"loss={ckpt.get('loss', '?')}, "
        f"profile={profile}, d_model={d_model}, "
        f"node_feat={node_feat_dim}, g_dim={g_dim}, "
        f"params={sum(p.numel() for p in model.parameters()):,}",
        file=sys.stderr,
    )
    return model, cfg


# ── Batched evaluation ────────────────────────────────────────────────────


def _extract_edge_predictions(
    logits: torch.Tensor,
    node_type: torch.Tensor,
    node_mask: torch.Tensor,
    N_max: int,
    batch_size: int,
) -> list[np.ndarray]:
    """Extract per-sample edge-node predictions using node_type == 0 mask.

    Parameters
    ----------
    logits : (B * N_max, 2) or (N_max, 2) raw model output
    node_type : (B, N_max) or (N_max,) int8
    node_mask : (B, N_max) or (N_max,) bool
    N_max : padding width
    batch_size : number of samples in batch

    Returns
    -------
    List of (n_edge_nodes_i,) uint8 arrays, one per sample.
    """
    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    pred_bits = (probs > 0.5).astype(np.uint8)

    if node_type.dim() == 1:
        # Unbatched: single sample
        nt = node_type.cpu().numpy()
        nm = node_mask.cpu().numpy().astype(bool)
        edge_mask = (nt == 0) & nm
        return [pred_bits[edge_mask]]

    results = []
    nt_np = node_type.cpu().numpy()
    nm_np = node_mask.cpu().numpy().astype(bool)
    for i in range(batch_size):
        start = i * N_max
        nt_i = nt_np[i]
        nm_i = nm_np[i]
        edge_mask_i = (nt_i == 0) & nm_i
        results.append(pred_bits[start:start + N_max][edge_mask_i])
    return results


# ── Active component decomposition (mirrors training) ────────────────────


def _decompose_shot_active(
    det_bits: np.ndarray,
    info: dict,
    d: int,
    n_rounds: int,
    p_val: float,
    N_max: int,
    E_max: int,
    S_max: int,
) -> list[tuple[PackedCrop, np.ndarray]]:
    """Decompose one shot into active connected components.

    Mirrors the training decomposition in ``OnlineSurfaceDataset`` with
    ``--component-scope active``.

    Returns list of (PackedCrop, comp_edge_indices) tuples.  Each
    ``comp_edge_indices`` is an array of *global* edge indices so the
    caller can reassemble per-component predictions into the full edge
    vector.
    """
    H_edge_full = info["H_edge"]
    det_adj = info["det_adj"]
    n_edges = info["n_edges"]

    fired = np.where(det_bits > 0)[0]
    if len(fired) == 0:
        # Zero-syndrome: no corrections needed
        return []

    # 1-hop expansion on detector adjacency graph
    active_set = set(fired.tolist())
    for f in fired:
        active_set.update(np.where(det_adj[f] > 0)[0].tolist())
    active_dets = np.array(sorted(active_set), dtype=np.intp)

    # Connected components among active detectors
    adj_sub = det_adj[np.ix_(active_dets, active_dets)]
    n_comp, labels = _cc(_csr(adj_sub.astype(np.float32)), directed=False)

    results = []
    for comp_id in range(n_comp):
        comp_det_local = np.where(labels == comp_id)[0]
        comp_dets = active_dets[comp_det_local]
        comp_det_set = set(comp_dets.tolist())

        # TIGHT: only edge-nodes where ALL connected detectors
        # are within this component
        H_comp = H_edge_full[comp_dets, :]
        candidate_edges = np.where(H_comp.sum(axis=0) > 0)[0]
        tight_edges = []
        for eidx in candidate_edges:
            all_dets_of_edge = set(
                np.where(H_edge_full[:, eidx] > 0)[0].tolist()
            )
            if all_dets_of_edge.issubset(comp_det_set):
                tight_edges.append(eidx)
        if len(tight_edges) == 0:
            continue
        comp_edges = np.array(tight_edges, dtype=np.intp)

        pack = pack_dem_cluster(
            H_edge=H_edge_full,
            det_coords=info["det_coords"],
            edge_coords=info["edge_coords"],
            det_bits=det_bits,
            y_bits_edge=np.zeros(n_edges, dtype=np.uint8),
            edge_indices=comp_edges,
            det_indices=comp_dets,
            d=d, rounds=n_rounds, p=p_val,
            N_max=N_max, E_max=E_max, S_max=S_max,
        )
        if pack is not None:
            results.append((pack, comp_edges))
    return results


# ── Parallel worker for shot decomposition + teacher labeling ─────────


def _eval_grid_point(
    model_sd: dict,
    model_kwargs: dict,
    d: int,
    p_val: float,
    shots: int,
    batch_size: int,
    noise_model: str,
    edge_prune_thresh: float,
    rounds_override: int,
    threads_per_worker: int = 4,
) -> dict:
    """Evaluate one (d, p) grid point entirely on CPU in a worker process.

    Loads its own model copy, samples shots, decomposes, runs inference,
    and compares to MWPM teacher.  Returns only lightweight statistics —
    no tensors cross process boundaries.
    """
    import torch, numpy as np, time as _t

    # Limit per-worker threads to avoid contention across workers
    torch.set_num_threads(threads_per_worker)
    t0 = _t.time()
    n_rounds = rounds_override if rounds_override > 0 else d

    # Build DEM info (worker-local, not picklable)
    info = _build_dem_info(d, n_rounds, p_val, noise_model, edge_prune_thresh)
    L_obs = info["L_obs"]
    n_edges = info["n_edges"]
    n_det = info["n_det"]

    N_max = max(512, n_edges + n_det + 16)
    E_max = max(4096, int(np.count_nonzero(info["H_edge"])) + 64)
    S_max = max(512, n_det + 16)

    # Build model on CPU
    model = MGHDv2(**model_kwargs)
    model.load_state_dict(model_sd, strict=False)
    model.eval()

    # Sample shots
    det_all, obs_all = info["sampler"].sample(
        shots=shots, separate_observables=True,
    )

    mghd_errors = 0
    teacher_errors = 0
    n_valid = 0
    n_zero_syndrome = 0
    n_components = 0

    for j in range(shots):
        det_bits = det_all[j].astype(np.uint8)
        true_obs = obs_all[j].astype(np.uint8)

        comps = _decompose_shot_active(
            det_bits, info, d, n_rounds, p_val,
            N_max, E_max, S_max,
        )

        # Teacher labels (MWPM) — always compute
        teacher_edges = _teacher_labels_from_matching(
            det_bits, info["matching"],
            info["det_pair_to_edge"], n_edges,
        )
        teacher_obs = compute_observable_correction(L_obs, teacher_edges)
        teacher_err = int(np.any(teacher_obs != true_obs))

        if len(comps) == 0:
            # Zero-syndrome → predict all-zeros
            mghd_obs = np.zeros_like(true_obs)
            mghd_err = int(np.any(mghd_obs != true_obs))
            mghd_errors += mghd_err
            teacher_errors += teacher_err
            n_valid += 1
            n_zero_syndrome += 1
            continue

        # Batched CPU inference on components of this shot
        packs = [pack for pack, _ce in comps]
        comp_edges_list = [ce for _pack, ce in comps]
        n_components += len(packs)

        pred_full = np.zeros(n_edges, dtype=np.uint8)

        for b_start in range(0, len(packs), batch_size):
            b_end = min(b_start + batch_size, len(packs))
            batch_packs = packs[b_start:b_end]
            batch_ce = comp_edges_list[b_start:b_end]

            batched = collate_packed(batch_packs)
            if batched.x_nodes.dim() == 2:
                batched.x_nodes = batched.x_nodes.unsqueeze(0)
                batched.node_mask = batched.node_mask.unsqueeze(0)
                batched.node_type = batched.node_type.unsqueeze(0)
                batched.g_token = batched.g_token.unsqueeze(0)
                batched.y_bits = batched.y_bits.unsqueeze(0)
                batched.s_sub = batched.s_sub.unsqueeze(0)
                batched.seq_idx = batched.seq_idx.unsqueeze(0)
                batched.seq_mask = batched.seq_mask.unsqueeze(0)

            actual_bs = batched.x_nodes.shape[0]
            N_pad = batched.x_nodes.shape[1]

            with torch.no_grad():
                logits, _nm = model(packed=batched)

            edge_preds = _extract_edge_predictions(
                logits, batched.node_type, batched.node_mask,
                N_pad, actual_bs,
            )

            for k, ce in enumerate(batch_ce):
                pred_comp = edge_preds[k]
                n_comp = min(len(pred_comp), len(ce))
                pred_full[ce[:n_comp]] = pred_comp[:n_comp]

        mghd_obs = compute_observable_correction(L_obs, pred_full)
        mghd_err = int(np.any(mghd_obs != true_obs))
        mghd_errors += mghd_err
        teacher_errors += teacher_err
        n_valid += 1

    elapsed = _t.time() - t0
    return {
        "distance": d,
        "p": p_val,
        "mghd_errors": mghd_errors,
        "teacher_errors": teacher_errors,
        "n_valid": n_valid,
        "n_zero_syndrome": n_zero_syndrome,
        "n_components": n_components,
        "n_edges": n_edges,
        "elapsed_s": round(elapsed, 2),
    }


@torch.no_grad()
def evaluate_grid(
    model: nn.Module,
    distances: list[int],
    p_values: list[float],
    shots: int,
    batch_size: int,
    noise_model: str,
    edge_prune_thresh: float,
    rounds_override: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    n_workers: int = 0,
) -> list[dict[str, Any]]:
    """Evaluate MGHD vs MWPM teacher across a (d, p) grid.

    Each (d, p) grid point runs as a separate CPU worker process that owns
    its own model copy and performs sampling, decomposition, inference, and
    comparison entirely on CPU.  Only lightweight statistics cross process
    boundaries — no tensor serialisation.
    """
    if n_workers <= 0:
        n_workers = max(1, int(os.cpu_count() * 0.75))

    # Prepare model state dict on CPU for workers
    model_sd = {k: v.cpu() for k, v in model.state_dict().items()}
    # Build model kwargs for reconstruction in workers
    model_kwargs = {
        "profile": getattr(model, "profile", "S"),
        "d_model": getattr(model, "d_model", 192),
        "d_state": getattr(model, "d_state", 80),
        "n_iters": getattr(model, "n_iters", 8),
        "node_feat_dim": model.node_in.in_features,
        "edge_feat_dim": model.edge_in.in_features,
        "g_dim": model.g_proj.in_features,
    }

    # Build grid
    grid = [(d, p) for d in distances for p in p_values]
    n_points = len(grid)

    # Limit workers to grid size — no point having more workers than points
    effective_workers = min(n_workers, n_points)
    threads_per_worker = max(1, os.cpu_count() // max(1, effective_workers))
    print(
        f"Dispatching {n_points} grid points across {effective_workers} CPU workers "
        f"({threads_per_worker} threads each)",
        file=sys.stderr,
    )

    # Submit all grid points
    results_map: dict[tuple[int, float], dict] = {}
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=effective_workers, mp_context=ctx) as pool:
        future_to_key = {}
        for d_val, p_val in grid:
            fut = pool.submit(
                _eval_grid_point,
                model_sd, model_kwargs,
                d_val, p_val,
                shots, batch_size,
                noise_model, edge_prune_thresh,
                rounds_override,
                threads_per_worker,
            )
            future_to_key[fut] = (d_val, p_val)

        done_count = 0
        for fut in as_completed(future_to_key):
            key = future_to_key[fut]
            try:
                r = fut.result()
            except Exception as exc:
                print(f"  FAILED d={key[0]} p={key[1]}: {exc}", file=sys.stderr)
                continue
            results_map[key] = r
            done_count += 1

            # Progress
            n_valid = r["n_valid"]
            ler = r["mghd_errors"] / n_valid if n_valid > 0 else 0.0
            lo, hi = _wilson_interval(r["mghd_errors"], n_valid) if n_valid > 0 else (0.0, 0.0)
            teacher_ler = r["teacher_errors"] / n_valid if n_valid > 0 else 0.0
            ratio = ler / teacher_ler if teacher_ler > 0 else float("inf")
            print(
                f"[{done_count}/{n_points}] d={r['distance']} p={r['p']:.4f}  "
                f"MGHD={ler:.4e} [{lo:.4e}, {hi:.4e}]  "
                f"MWPM={teacher_ler:.4e}  "
                f"ratio={ratio:.3f}  "
                f"({n_valid} shots, {r['n_zero_syndrome']} zero-synd, "
                f"{r['n_components']} comps, total={r['elapsed_s']:.1f}s)",
                file=sys.stderr,
            )

    # Assemble results in grid order
    results = []
    for d_val, p_val in grid:
        if (d_val, p_val) not in results_map:
            continue
        r = results_map[(d_val, p_val)]
        n_valid = r["n_valid"]
        if n_valid > 0:
            ler = r["mghd_errors"] / n_valid
            lo, hi = _wilson_interval(r["mghd_errors"], n_valid)
            teacher_ler = r["teacher_errors"] / n_valid
            teacher_lo, teacher_hi = _wilson_interval(r["teacher_errors"], n_valid)
        else:
            ler, lo, hi = 0.0, 0.0, 0.0
            teacher_ler, teacher_lo, teacher_hi = 0.0, 0.0, 0.0

        results.append({
            "distance": d_val,
            "p": p_val,
            "n_shots": n_valid,
            "ler_mghd": ler,
            "ler_mwpm": teacher_ler,
            "confidence_intervals_95": {
                "mghd": {"lo": lo, "hi": hi},
                "mwpm": {"lo": teacher_lo, "hi": teacher_hi},
            },
            "n_edges": r["n_edges"],
            "n_zero_syndrome": r["n_zero_syndrome"],
            "n_components": r["n_components"],
            "elapsed_s": r["elapsed_s"],
        })

    return results


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Parse grid
    distances = [int(x) for x in args.distances.split(",") if x.strip()]
    p_values = [float(x) for x in args.p_values.split(",") if x.strip()]

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # AMP
    amp_dtype = None
    if args.amp == "bf16":
        amp_dtype = torch.bfloat16
    elif args.amp == "fp16":
        amp_dtype = torch.float16

    # Load model
    model, cfg = load_model(args.checkpoint, device)
    noise_model = args.noise_model or cfg.get("noise_model", "SI1000")

    print(
        f"Evaluating: distances={distances}, p={p_values}, "
        f"shots={args.shots}, batch_size={args.batch_size}, "
        f"noise_model={noise_model}, device={device}",
        file=sys.stderr,
    )

    n_workers = args.workers if args.workers > 0 else max(1, int(os.cpu_count() * 0.75))
    print(f"Using {n_workers} CPU workers for parallel decomposition", file=sys.stderr)

    # Run evaluation
    t_start = time.time()
    results = evaluate_grid(
        model=model,
        distances=distances,
        p_values=p_values,
        shots=args.shots,
        batch_size=args.batch_size,
        noise_model=noise_model,
        edge_prune_thresh=args.edge_prune_thresh,
        rounds_override=args.rounds,
        device=device,
        amp_dtype=amp_dtype,
        n_workers=n_workers,
    )
    t_total = time.time() - t_start

    # Wrap output
    output = {
        "checkpoint": str(args.checkpoint),
        "noise_model": noise_model,
        "total_elapsed_s": round(t_total, 2),
        "results": results,
    }

    json_str = json.dumps(output, indent=2)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json_str, encoding="utf-8")
        print(f"Results saved to {out_path}", file=sys.stderr)
    else:
        print(json_str)

    # Summary table
    print("\n── Summary ──", file=sys.stderr)
    print(f"{'d':>3}  {'p':>8}  {'MGHD LER':>12}  {'MWPM LER':>12}  {'ratio':>8}  {'shots':>7}", file=sys.stderr)
    for r in results:
        ratio = r["ler_mghd"] / r["ler_mwpm"] if r["ler_mwpm"] > 0 else float("inf")
        print(
            f"{r['distance']:>3}  {r['p']:>8.4f}  {r['ler_mghd']:>12.4e}  "
            f"{r['ler_mwpm']:>12.4e}  {ratio:>8.3f}  {r['n_shots']:>7}",
            file=sys.stderr,
        )
    print(f"\nTotal time: {t_total:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
