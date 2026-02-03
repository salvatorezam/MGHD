"""Circuit-level MGHD trainer using DEM-based architecture.

This trainer operates in DEM (Detector Error Model) space rather than code-level
Hz/Hx space. Key differences from train.py:

1. Input: Detection events (temporal XORs), not syndrome bits
2. Graph: DEM structure (detectors as nodes, error mechanisms as edges)
3. Output: Observable prediction, not per-qubit error labels
4. Teacher: Ground truth observable from Stim (no teacher decoder needed)

Shares infrastructure with train.py:
- DDP multi-GPU support
- Mixed precision (AMP)
- Resume from checkpoint
- Early stopping
- Curriculum over distance/rounds/p

Usage:
    mghd-train-circuit --distance 3 --rounds 5 --dep 0.001 --epochs 50
    mghd-train-circuit --distance-curriculum 3,5,7 --dep-curriculum 0.001,0.003
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from mghd.core.core import MGHDCircuit, DEMGraph, build_dem_graph
from mghd.samplers.stim_sampler import StimCircuitSampler


def compute_ler(pred_obs: np.ndarray, true_obs: np.ndarray) -> float:
    """Compute logical error rate."""
    pred = (pred_obs > 0.5).astype(np.uint8)
    true = true_obs.astype(np.uint8)
    return float(np.mean(pred != true))


def train_epoch(
    model: MGHDCircuit,
    sampler: StimCircuitSampler,
    optimizer: optim.Optimizer,
    *,
    shots_per_batch: int,
    n_batches: int,
    device: torch.device,
    grad_clip: float = 1.0,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_ler = 0.0
    total_samples = 0

    for _ in range(n_batches):
        # Sample detection events and build DEM graph
        graph = sampler.sample_with_dem_graph(shots_per_batch)
        graph = _move_graph_to_device(graph, device)

        # Forward pass
        obs_logits, edge_probs = model(graph)

        # Loss: BCE on observable prediction
        loss = F.binary_cross_entropy_with_logits(
            obs_logits.squeeze(-1),
            graph.y_obs.squeeze(-1),
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Metrics
        with torch.no_grad():
            pred_obs = torch.sigmoid(obs_logits).cpu().numpy()
            true_obs = graph.y_obs.cpu().numpy()
            ler = compute_ler(pred_obs, true_obs)

        total_loss += loss.item() * shots_per_batch
        total_ler += ler * shots_per_batch
        total_samples += shots_per_batch

    return {
        "loss": total_loss / total_samples,
        "ler": total_ler / total_samples,
        "samples": total_samples,
    }


@torch.no_grad()
def evaluate(
    model: MGHDCircuit,
    sampler: StimCircuitSampler,
    *,
    n_shots: int,
    device: torch.device,
) -> dict:
    """Evaluate model on fresh samples."""
    model.eval()

    graph = sampler.sample_with_dem_graph(n_shots)
    graph = _move_graph_to_device(graph, device)

    obs_logits, _ = model(graph)
    loss = F.binary_cross_entropy_with_logits(
        obs_logits.squeeze(-1),
        graph.y_obs.squeeze(-1),
    )

    pred_obs = torch.sigmoid(obs_logits).cpu().numpy()
    true_obs = graph.y_obs.cpu().numpy()
    ler = compute_ler(pred_obs, true_obs)

    return {"loss": loss.item(), "ler": ler}


def evaluate_mwpm_baseline(sampler: StimCircuitSampler, n_shots: int) -> float:
    """Compute MWPM baseline LER for comparison."""
    try:
        import pymatching
    except ImportError:
        return float("nan")

    samples = sampler.sample(n_shots)
    dem = sampler.dem
    matcher = pymatching.Matching.from_detector_error_model(dem)

    pred_obs = matcher.decode_batch(samples["dets"])
    true_obs = samples["obs"]

    return compute_ler(pred_obs, true_obs)


def _move_graph_to_device(graph: DEMGraph, device: torch.device) -> DEMGraph:
    """Move all tensors in DEMGraph to device."""
    return DEMGraph(
        x_det=graph.x_det.to(device),
        det_events=graph.det_events.to(device),
        edge_index=graph.edge_index.to(device),
        edge_weight=graph.edge_weight.to(device),
        obs_mask=graph.obs_mask.to(device),
        y_obs=graph.y_obs.to(device),
        num_detectors=graph.num_detectors,
        num_edges=graph.num_edges,
        num_observables=graph.num_observables,
    )


def main():
    parser = argparse.ArgumentParser(description="Circuit-level MGHD training")
    
    # Data parameters
    parser.add_argument("--distance", "-d", type=int, default=3, help="Code distance")
    parser.add_argument("--rounds", "-r", type=int, default=5, help="Syndrome rounds")
    parser.add_argument("--dep", type=float, default=0.001, help="Depolarizing probability")
    
    # Curriculum parameters
    parser.add_argument(
        "--distance-curriculum", type=str, default=None,
        help="Comma-separated distances to cycle (e.g., '3,5,7'). Overrides --distance."
    )
    parser.add_argument(
        "--rounds-curriculum", type=str, default=None,
        help="Comma-separated rounds to cycle (e.g., '3,5,7'). Overrides --rounds."
    )
    parser.add_argument(
        "--dep-curriculum", type=str, default=None,
        help="Comma-separated dep values to cycle (e.g., '0.001,0.003,0.005'). Overrides --dep."
    )
    parser.add_argument("--epochs-per-curriculum", type=int, default=1, help="Epochs per curriculum step")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--shots-per-batch", type=int, default=256, help="Shots per batch")
    parser.add_argument("--batches-per-epoch", type=int, default=10, help="Batches per epoch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    
    # Model parameters
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n-gnn-iters", type=int, default=6, help="GNN iterations")
    parser.add_argument("--n-match-iters", type=int, default=5, help="Soft matching iterations")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Infrastructure
    parser.add_argument(
        "--amp", type=str, default="bf16", choices=["off", "fp16", "bf16", "auto"],
        help="Mixed precision mode: off/fp16/bf16/auto"
    )
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--early-stop-patience", type=int, default=0, help="Early stop patience (0=disabled)")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0, help="Min improvement for early stop")
    
    # Output
    parser.add_argument("--save", type=str, default=None, help="Save directory")
    parser.add_argument("--save-root", type=str, default="data/results", help="Root for auto-named runs")
    parser.add_argument("--eval-shots", type=int, default=1000, help="Evaluation shots")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--progress-prints", type=int, default=1, help="Mid-epoch progress prints")
    
    args = parser.parse_args()

    # DDP setup
    rank = 0
    world_size = 1
    local_rank = 0
    is_distributed = False

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        is_distributed = True
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        if rank == 0:
            print(f"Initialized DDP: {world_size} GPUs")

    # Seeds
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank if is_distributed else 0)
    else:
        device = torch.device("cpu")
    if rank == 0:
        print(f"Using device: {device}")

    # Mixed precision setup
    amp_mode = args.amp.lower()
    use_amp = False
    amp_dtype = torch.float32
    if device.type == "cuda" and amp_mode != "off":
        if amp_mode == "auto":
            amp_mode = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        if amp_mode == "bf16" and torch.cuda.is_bf16_supported():
            use_amp = True
            amp_dtype = torch.bfloat16
        elif amp_mode == "fp16":
            use_amp = True
            amp_dtype = torch.float16
        if rank == 0 and use_amp:
            print(f"Using AMP with {amp_dtype}")
    
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    def autocast_ctx():
        if use_amp:
            return torch.amp.autocast("cuda", dtype=amp_dtype)
        return nullcontext()

    # Parse curriculum
    def parse_curriculum(spec, default, dtype=float):
        if spec is None:
            return [dtype(default)]
        return [dtype(x.strip()) for x in spec.split(",")]

    distance_curriculum = parse_curriculum(args.distance_curriculum, args.distance, int)
    rounds_curriculum = parse_curriculum(args.rounds_curriculum, args.rounds, int)
    dep_curriculum = parse_curriculum(args.dep_curriculum, args.dep, float)

    # Create model
    model = MGHDCircuit(
        d_model=args.d_model,
        n_gnn_iters=args.n_gnn_iters,
        n_match_iters=args.n_match_iters,
        dropout=args.dropout,
    ).to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank])

    if rank == 0:
        print(f"Created MGHDCircuit: d_model={args.d_model}, n_gnn_iters={args.n_gnn_iters}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Resume
    start_epoch = 1
    best_ler = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        state = ckpt.get("model", ckpt)
        if is_distributed:
            model.module.load_state_dict(state)
        else:
            model.load_state_dict(state)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_ler = ckpt.get("ler", float("inf"))
        if rank == 0:
            print(f"Resumed from {args.resume}, epoch {start_epoch}, best_ler={best_ler:.4f}")

    # Save directory
    if args.save:
        save_dir = Path(args.save)
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_dir = Path(args.save_root) / f"circuit_{timestamp}_d{args.distance}"
    if rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Initial sampler for baseline (use first curriculum values)
    sampler = StimCircuitSampler(
        distance=distance_curriculum[0],
        rounds=rounds_curriculum[0],
        dep=dep_curriculum[0],
    )

    # MWPM baseline
    if rank == 0:
        print("Computing MWPM baseline...")
        mwpm_ler = evaluate_mwpm_baseline(sampler, args.eval_shots)
        print(f"MWPM baseline LER: {mwpm_ler:.4f}")

        # Save config
        config = vars(args)
        config["mwpm_baseline_ler"] = mwpm_ler
        config["distance_curriculum"] = distance_curriculum
        config["rounds_curriculum"] = rounds_curriculum
        config["dep_curriculum"] = dep_curriculum
        (save_dir / "config.json").write_text(json.dumps(config, indent=2))
    else:
        mwpm_ler = 0.0

    if is_distributed:
        mwpm_tensor = torch.tensor([mwpm_ler], device=device)
        dist.broadcast(mwpm_tensor, src=0)
        mwpm_ler = mwpm_tensor.item()

    # Training loop
    history = []
    last_improve_epoch = start_epoch
    patience = args.early_stop_patience

    if rank == 0:
        print("\nStarting training...")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # Curriculum selection
        curriculum_idx = ((epoch - 1) // args.epochs_per_curriculum)
        d_cur = distance_curriculum[curriculum_idx % len(distance_curriculum)]
        r_cur = rounds_curriculum[curriculum_idx % len(rounds_curriculum)]
        p_cur = dep_curriculum[curriculum_idx % len(dep_curriculum)]

        # Recreate sampler if curriculum changed
        if (d_cur != sampler._distance or r_cur != sampler._rounds or 
            abs(p_cur - sampler._dep) > 1e-9):
            sampler = StimCircuitSampler(distance=d_cur, rounds=r_cur, dep=p_cur)
            if rank == 0:
                print(f"  [Curriculum] d={d_cur}, rounds={r_cur}, dep={p_cur}")

        # Train
        model.train()
        total_loss = 0.0
        total_ler = 0.0
        total_samples = 0

        for batch_idx in range(args.batches_per_epoch):
            graph = sampler.sample_with_dem_graph(args.shots_per_batch)
            graph = _move_graph_to_device(graph, device)

            with autocast_ctx():
                obs_logits, edge_probs = model(graph)
                loss = F.binary_cross_entropy_with_logits(
                    obs_logits.squeeze(-1),
                    graph.y_obs.squeeze(-1),
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                pred_obs = torch.sigmoid(obs_logits).cpu().numpy()
                true_obs = graph.y_obs.cpu().numpy()
                ler = compute_ler(pred_obs, true_obs)

            total_loss += loss.item() * args.shots_per_batch
            total_ler += ler * args.shots_per_batch
            total_samples += args.shots_per_batch

            # Mid-epoch progress
            if rank == 0 and args.progress_prints > 0:
                if (batch_idx + 1) % max(1, args.batches_per_epoch // args.progress_prints) == 0:
                    print(f"  batch {batch_idx+1}/{args.batches_per_epoch}, loss={loss.item():.4f}")

        train_loss = total_loss / total_samples
        train_ler = total_ler / total_samples

        # Evaluate
        eval_metrics = evaluate(model, sampler, n_shots=args.eval_shots, device=device)
        scheduler.step()
        dt = time.time() - t0

        # Sync metrics in DDP
        if is_distributed:
            metrics_tensor = torch.tensor([train_loss, train_ler, eval_metrics["loss"], eval_metrics["ler"]], device=device)
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.AVG)
            train_loss, train_ler, eval_loss, eval_ler = metrics_tensor.tolist()
        else:
            eval_loss = eval_metrics["loss"]
            eval_ler = eval_metrics["ler"]

        # Log
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ler": train_ler,
            "eval_loss": eval_loss,
            "eval_ler": eval_ler,
            "mwpm_ratio": eval_ler / mwpm_ler if mwpm_ler > 0 else float("inf"),
            "lr": scheduler.get_last_lr()[0],
            "distance": d_cur,
            "rounds": r_cur,
            "dep": p_cur,
            "time": dt,
        }
        history.append(entry)

        if rank == 0:
            print(
                f"Epoch {epoch:3d} | "
                f"loss={train_loss:.4f} | "
                f"LER={eval_ler:.4f} | "
                f"vs MWPM={entry['mwpm_ratio']:.2f}x | "
                f"d={d_cur} | "
                f"{dt:.1f}s"
            )

        # Checkpointing
        if rank == 0:
            model_state = model.module.state_dict() if is_distributed else model.state_dict()
            
            if eval_ler < best_ler - args.early_stop_min_delta:
                best_ler = eval_ler
                last_improve_epoch = epoch
                torch.save(
                    {"model": model_state, "epoch": epoch, "ler": best_ler},
                    save_dir / "best.pt",
                )

            torch.save(
                {"model": model_state, "epoch": epoch, "ler": eval_ler},
                save_dir / "last.pt",
            )
            (save_dir / "history.json").write_text(json.dumps(history, indent=2))

        # Early stopping
        if patience > 0 and (epoch - last_improve_epoch) >= patience:
            if rank == 0:
                print(f"Early stopping: no improvement for {patience} epochs")
            break

    # Final summary
    if rank == 0:
        print("\n" + "=" * 60)
        print(f"Training complete!")
        print(f"Best LER: {best_ler:.4f} (vs MWPM {mwpm_ler:.4f} = {best_ler/mwpm_ler:.2f}x)")
        print(f"Results saved to: {save_dir}")

    if is_distributed:
        dist.destroy_process_group()

    return str(save_dir / "best.pt")


if __name__ == "__main__":
    main()
