"""MGHD trainer: online surface-code training with Stim DEM-based sampling.

Generates circuit-level syndrome data via Stim detector error models,
routes teacher supervision (PyMatching from DEM), and packs 3D subgraphs
into MGHDv2 input tensors for training.
CUDA is initialized only inside ``main``.
"""

# NOTE: Initialize CUDA only in main(). This file defines dataset, model, loop.
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset

from mghd.core.core import MGHDv2, PackedCrop, CropMeta, pack_cluster
from mghd.decoders.lsd import clustered as cc

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def collate_packed(batch):
    """Stack crops into a single batched PackedCrop."""
    if not batch:
        return None

    # Stack 3D tensors
    x_nodes = torch.stack([c.x_nodes for c in batch])
    node_mask = torch.stack([c.node_mask for c in batch])
    node_type = torch.stack([c.node_type for c in batch])
    g_token = torch.stack([c.g_token for c in batch])
    y_bits = torch.stack([c.y_bits for c in batch])
    s_sub = torch.stack([c.s_sub for c in batch])

    # Concatenate and shift 2D/1D tensors
    edge_indices = []
    edge_attrs = []
    edge_masks = []
    seq_idxs = []
    seq_masks = []

    N_max = batch[0].x_nodes.shape[0]

    for i, c in enumerate(batch):
        shift = i * N_max
        edge_indices.append(c.edge_index + shift)
        edge_attrs.append(c.edge_attr)
        edge_masks.append(c.edge_mask)
        seq_idxs.append(c.seq_idx)
        seq_masks.append(c.seq_mask)

    edge_index = torch.cat(edge_indices, dim=1)
    edge_attr = torch.cat(edge_attrs, dim=0)
    edge_mask = torch.cat(edge_masks, dim=0)
    seq_idx = torch.stack(seq_idxs, dim=0)
    seq_mask = torch.stack(seq_masks, dim=0)

    # Keep H_sub as list for per-sample processing
    H_sub = [c.H_sub for c in batch]

    # Meta: use first one, but we might need per-sample meta for projection
    meta = batch[0].meta
    # We attach the list of metas to the batch meta for reference
    meta.batch_metas = [c.meta for c in batch]

    return PackedCrop(
        x_nodes=x_nodes,
        node_mask=node_mask,
        node_type=node_type,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_mask=edge_mask,
        seq_idx=seq_idx,
        seq_mask=seq_mask,
        g_token=g_token,
        y_bits=y_bits,
        s_sub=s_sub,
        meta=meta,
        H_sub=H_sub,  # List of H_subs
        idx_data_local=None,
        idx_check_local=None,
    )


def _load_hparams(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_packed_contract(packed: PackedCrop, node_feat_dim: int, edge_feat_dim: int) -> None:
    node_dim = int(packed.x_nodes.shape[-1])
    if node_dim != int(node_feat_dim):
        raise ValueError(
            f"Packed crop node feature dimension {node_dim} does not match expected "
            f"{node_feat_dim}. Check data_contract.node_feat_dim or erasure setting."
        )
    if packed.edge_attr is None:
        raise ValueError("Packed crop is missing edge_attr entries required by the model")
    edge_dim = int(packed.edge_attr.shape[-1])
    if edge_dim != int(edge_feat_dim):
        raise ValueError(
            f"Packed crop edge feature dimension {edge_dim} does not match expected "
            f"{edge_feat_dim}. Update data_contract.edge_feat_dim or regenerate crops."
        )


def bce_binary_head_loss(
    logits: torch.Tensor,
    node_mask: torch.Tensor,
    node_type: torch.Tensor,
    y_bits: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Binary cross‑entropy on data‑qubit nodes with optional smoothing/weights."""
    # Use only data-qubit nodes (type==0) and valid mask
    is_data = (node_type == 0) & node_mask
    if is_data.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    # logits: [N_max, 2]; target bit in {0,1}
    target = y_bits.clamp_min(0).clamp_max(1).long()
    smoothing = float(max(0.0, min(0.5, label_smoothing)))
    ce = nn.CrossEntropyLoss(reduction="none", label_smoothing=smoothing)
    loss_all = ce(logits[is_data], target[is_data])
    if sample_weight is not None:
        weight = sample_weight.to(logits.device).clamp_min(0.5).clamp_max(3.0)
        loss_all = loss_all * weight
    return loss_all.mean()


def focal_binary_head_loss(
    logits: torch.Tensor,
    node_mask: torch.Tensor,
    node_type: torch.Tensor,
    y_bits: torch.Tensor,
    *,
    alpha: float | None = 0.25,
    gamma: float = 2.0,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Focal loss on data‑qubit nodes (binary head) with optional weights."""
    is_data = (node_type == 0) & node_mask
    if is_data.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    targets = y_bits.clamp_min(0).clamp_max(1).long()
    logits_sel = logits[is_data]
    targets_sel = targets[is_data]

    ce = F.cross_entropy(logits_sel, targets_sel, reduction="none")
    probs = torch.softmax(logits_sel, dim=-1)
    p_t = probs[torch.arange(probs.size(0), device=probs.device), targets_sel]
    loss = (1.0 - p_t).pow(float(gamma)) * ce

    if alpha is not None:
        a = float(alpha)
        alpha_t = torch.where(
            targets_sel == 1,
            torch.tensor(a, device=probs.device),
            torch.tensor(1.0 - a, device=probs.device),
        )
        loss = alpha_t * loss
    if sample_weight is not None:
        weight = sample_weight.to(logits.device).clamp_min(0.5).clamp_max(3.0)
        loss = loss * weight
    return loss.mean()


def parity_auxiliary_loss(
    logits: torch.Tensor,
    node_mask: torch.Tensor,
    node_type: torch.Tensor,
    H_sub: np.ndarray | list[np.ndarray],
    s_sub: torch.Tensor,
) -> torch.Tensor:
    """Small regularizer encouraging parity consistency within the crop."""
    # logits: [B, N, 2] or [N, 2]
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
        node_mask = node_mask.unsqueeze(0)
        node_type = node_type.unsqueeze(0)
        if not isinstance(H_sub, list):
            H_sub = [H_sub]
        if s_sub.dim() == 1:
            s_sub = s_sub.unsqueeze(0)

    batch_size = logits.shape[0]
    total_loss = torch.tensor(0.0, device=logits.device)

    for i in range(batch_size):
        h = H_sub[i]
        if h is None or h.shape[0] == 0:
            continue

        l = logits[i]
        nm = node_mask[i]
        nt = node_type[i]

        # Differentiable XOR expectation ~= parity of Bernoulli probs
        with torch.no_grad():
            is_data = ((nt == 0) & nm).cpu().numpy()
            # map logits indices -> data-qubits used by H_sub
            data_idx = np.nonzero(is_data)[0]

        if len(data_idx) == 0:
            continue

        p = torch.sigmoid(l[:, 1] - l[:, 0])  # P(bit=1)
        p_data = p[data_idx]

        # Expected parity for each check row: E[⊕] ≈ 0.5*(1 - ∏(1-2p_i)) over participating data
        # H_sub columns are already local data-qubit order [0..nQ-1]
        H = torch.as_tensor(h, dtype=torch.float32, device=logits.device)

        # Map logits to local data-qubit region: we assume data-qubits occupy [0:nQ)
        eps = 1e-6
        prod_terms = []
        for r in range(H.size(0)):
            idx = torch.nonzero(H[r] > 0.5, as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                prod_terms.append(torch.tensor(1.0, device=logits.device))
            else:
                sel = p_data[idx]
                prod_terms.append(torch.clamp(1 - 2 * sel, -1 + eps, 1 - eps).prod())

        if not prod_terms:
            continue

        prod = torch.stack(prod_terms)
        E_par = 0.5 * (1 - prod)
        # Correct target: H·e ≡ s (mod 2)  => expected parity should match s_sub (0/1)
        s = s_sub[i][: h.shape[0]].to(E_par.device).float()
        total_loss = total_loss + (E_par - s).pow(2).mean()

    return total_loss / batch_size


def projection_aware_logits_to_bits(
    logits: torch.Tensor,
    projector_kwargs: dict[str, Any],
    *,
    data_mask: torch.Tensor,
) -> np.ndarray:
    """Project probabilities to ML bits via the exact GF(2) projector.

    Converts per‑node logits to probabilities, extracts data‑qubit region, and
    invokes the clustered projector (or thresholds as a fallback).
    """
    # Convert logits -> per-qubit probabilities -> run exact projector in GF(2) to ML correction.
    # restrict to data-qubits only to match H_sub columns
    probs1_full = torch.sigmoid(logits[:, 1] - logits[:, 0])
    probs1 = probs1_full[data_mask].detach().cpu().numpy()
    # Adapter to clustered projector (discover exact signature and adapt)
    if hasattr(cc, "ml_parity_project"):
        H_sub = projector_kwargs.get("H_sub")
        if H_sub is None:
            return (probs1 > 0.5).astype(np.uint8)
        s_sub = projector_kwargs.get("s_sub", None)
        if s_sub is None:
            raise ValueError(
                "projection_aware_logits_to_bits requires s_sub (true syndrome) but got None"
            )
        if torch.is_tensor(s_sub):
            s_sub = s_sub.detach().cpu().numpy()
        s_sub = np.asarray(s_sub, dtype=np.uint8).ravel()
        if H_sub is not None and hasattr(H_sub, "shape"):
            s_sub = s_sub[: int(H_sub.shape[0])]
        if H_sub is not None and isinstance(H_sub, np.ndarray):
            H_sub = sp.csr_matrix(H_sub)
        bits = cc.ml_parity_project(H_sub, s_sub, probs1)  # np.uint8
    else:
        # Fallback: threshold
        bits = (probs1 > 0.5).astype(np.uint8)
    return bits  # length == #data_qubits


def train_inprocess(ns) -> str:
    """Run the trainer in‑process (mirrors CLI ``main``) and return best.pt path."""
    args = ns
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=False)
    parser.add_argument("--online", action="store_true", help="Enable on-the-fly Stim DEM sampling")
    parser.add_argument("--family", type=str, default="surface")
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument(
        "--p", type=float, default=0.005, help="Physical error rate for online sampling"
    )
    parser.add_argument(
        "--p-curriculum",
        type=str,
        default=None,
        help="Comma-separated list of p values to cycle across epochs in online mode (e.g., '0.01,0.006,0.003')",
    )
    parser.add_argument(
        "--epochs-per-p",
        type=int,
        default=1,
        help="Epochs to spend on each p value in --p-curriculum",
    )
    parser.add_argument("--shots-per-epoch", type=int, default=256)
    parser.add_argument(
        "--sampler",
        type=str,
        default="stim",
        choices=["stim", "synthetic"],
        help=(
            "Sampler backend for online mode: "
            "'stim' uses DEM-based circuit-level sampling (default), "
            "'synthetic' uses fast code-capacity IID noise for testing."
        ),
    )
    parser.add_argument(
        "--noise-model",
        type=str,
        default="SI1000",
        help="Stim noise model name passed to stim.Circuit.generated() (e.g., SI1000, uniform).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=0,
        help="Number of syndrome extraction rounds (0 = use distance d).",
    )
    parser.add_argument(
        "--after-clifford-depolarization",
        type=float,
        default=0.0,
        help="Depolarization rate after Clifford gates for Stim circuit generation.",
    )
    parser.add_argument("--teacher-mix", type=str, default="pymatching=1.0")
    parser.add_argument("--profile", type=str, default="S")
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--lr", type=float, default=5.952925899948483e-05)
    parser.add_argument("--wd", type=float, default=6.65850238574699e-05)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--parity-lambda", type=float, default=0.0)
    parser.add_argument("--projection-aware", type=int, default=0)
    parser.add_argument(
        "--online-fast",
        action="store_true",
        help=(
            "Speed-oriented online settings. Disables expensive projection/parity auxiliaries "
            "and enables periodic progress heartbeats."
        ),
    )
    parser.add_argument(
        "--online-fast-keep-aux",
        action="store_true",
        help=(
            "Keep projection/parity auxiliaries enabled even when --online-fast is set. "
            "Useful when training quality is prioritized over maximum throughput."
        ),
    )
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--use-focal", action="store_true", help="Use focal loss for per-qubit labels"
    )
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--noise-injection", type=float, default=0.009883059279379016)
    parser.add_argument("--grad-clip", type=float, default=0.8545326095750816)
    parser.add_argument(
        "--hparams", type=str, default=None, help="Path to JSON hyperparameters file"
    )
    parser.add_argument(
        "--save",
        type=str,
        required=False,
        default=None,
        help="If omitted or contains '{auto}', an auto-named run dir is created under --save-root",
    )
    parser.add_argument(
        "--save-root", type=str, default="data/results", help="Root directory for auto-named runs"
    )
    parser.add_argument(
        "--save-auto", action="store_true", help="Force auto-named save directory under --save-root"
    )
    parser.add_argument("--seed", type=int, default=42)
    # Progress reporting (prints per epoch; 1 = only near end, 0 = disable mid-epoch prints)
    parser.add_argument("--progress-prints", type=int, default=1)
    parser.add_argument(
        "--progress-seconds",
        type=float,
        default=0.0,
        help=(
            "If >0, emit periodic progress heartbeats every N seconds "
            "(useful when per-epoch logging is sparse)."
        ),
    )
    parser.add_argument(
        "--teacher-workers",
        type=int,
        default=4,
        help="Thread pool workers used to prefetch teacher labels",
    )
    parser.add_argument(
        "--teacher-decode-batch-size",
        type=int,
        default=16,
        help="Shots per distance bucket to decode together in online teacher path",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of data loading workers for online training",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=8,
        help="DataLoader prefetch factor (batches per worker). Increase if GPU idle waiting for data.",
    )
    parser.add_argument(
        "--amp",
        type=str,
        default="bf16",
        choices=["off", "fp16", "bf16", "auto"],
        help="Mixed precision mode for speed: off/fp16/bf16/auto (default bf16 on H100)",
    )
    # Early stopping
    parser.add_argument(
        "--distance-curriculum",
        type=str,
        default=None,
        help="Comma-separated list of distances to sample from (e.g., '3,5,7'). Overrides --distance.",
    )
    parser.add_argument(
        "--cluster-halo",
        type=int,
        default=0,
        help=(
            "Component halo used during online crop extraction. "
            "0 keeps strict connected components, 1 adds one-hop halo."
        ),
    )
    parser.add_argument(
        "--component-scope",
        type=str,
        default="active",
        choices=["active", "full"],
        help=(
            "Crop extraction scope for online mode: "
            "'active' keeps connected active components (faster), "
            "'full' keeps one full side graph per shot-side (better global distance signal)."
        ),
    )
    parser.add_argument(
        "--edge-prune-thresh",
        type=float,
        default=1e-3,
        help=(
            "Probability threshold for edge pruning in DEM graph. "
            "Edges with merged probability below this value are dropped. "
            "Lower = more edges (larger graphs), higher = fewer edges (faster)."
        ),
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop if no improvement for this many epochs (0 disables)",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum loss improvement to reset patience",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (loads model weights and sets start epoch)",
    )
    if not hasattr(args, "data_root"):
        args = parser.parse_args()

    # DDP Setup
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
        print(f"Initialized DDP: rank {rank}/{world_size}, local_rank {local_rank}")

    # Handle sampler selection
    sampler_choice = str(getattr(args, "sampler", "stim")).lower()
    if rank == 0:
        print(f"Sampler: {sampler_choice}")

    # Auto-scale pad limits for large distances if not provided
    if not hasattr(args, "N_max"):
        d_param = int(getattr(args, "distance", 3))
        dc = str(getattr(args, "distance_curriculum", "") or "")
        if dc:
            dc_list = [int(x) for x in dc.split(",") if x.strip()]
            if dc_list:
                d_param = max(dc_list)
        n_rounds = int(getattr(args, "rounds", 0)) or d_param
        # Circuit-level: detectors = d*(d-1)/2 * n_rounds * 2 (Z+X), plus boundary
        args.N_max = max(512, int(4.0 * d_param**2 * n_rounds))
        args.E_max = max(4096, int(16.0 * d_param**2 * n_rounds))
        args.S_max = max(512, int(4.0 * d_param**2 * n_rounds))
        if rank == 0:
            print(f"Auto-scaled pad limits for d={d_param}, rounds={n_rounds}: N_max={args.N_max}, E_max={args.E_max}")

    # Default structural hyperparameters (can be overridden via JSON/CLI namespace)
    # NOTE: node_feat_dim=10 (added time coord), g_token_dim=14 (added n_rounds, noise_scale)
    defaults = {
        "d_model": 192,
        "d_state": 80,
        "n_iters": 8,
        "msg_net_dropout_p": 0.0,
        "gru_dropout_p": 0.0,
        "se_reduction": 4,
        "node_feat_dim": 10,
        "edge_feat_dim": 3,
        "g_token_dim": 14,
    }
    for attr, value in defaults.items():
        if not hasattr(args, attr):
            setattr(args, attr, value)
    if not hasattr(args, "msg_net_size"):
        args.msg_net_size = max(96, int(getattr(args, "d_model", 192)))

    hp = None
    if getattr(args, "hparams", None):
        try:
            hp = _load_hparams(args.hparams)
        except Exception:
            hp = None
    if isinstance(hp, dict):
        hp_model = hp.get("model_architecture", {}) or {}
        hp_mamba = hp.get("mamba_parameters", {}) or {}
        hp_attn = hp.get("attention_mechanism", {}) or {}
        hp_data = hp.get("data_contract", {}) or {}
        hp_train = hp.get("training_parameters", {}) or {}

        args.d_model = int(hp_mamba.get("d_model", args.d_model))
        args.d_state = int(hp_mamba.get("d_state", args.d_state))
        args.n_iters = int(hp_model.get("n_iters", args.n_iters))
        args.msg_net_size = int(hp_model.get("msg_net_size", max(96, args.d_model)))
        args.msg_net_dropout_p = float(hp_model.get("msg_net_dropout_p", args.msg_net_dropout_p))
        args.gru_dropout_p = float(hp_model.get("gru_dropout_p", args.gru_dropout_p))
        args.se_reduction = int(hp_attn.get("se_reduction", args.se_reduction))

        node_feat_dim = int(hp_data.get("node_feat_dim", args.node_feat_dim))
        if bool(hp_data.get("erasure_enabled", False)):
            node_feat_dim = 10
        edge_feat_dim = int(hp_data.get("edge_feat_dim", args.edge_feat_dim))
        g_token_dim = int(hp_data.get("g_token_dim", args.g_token_dim))
        args.node_feat_dim = node_feat_dim
        args.edge_feat_dim = edge_feat_dim
        args.g_token_dim = g_token_dim

        args.lr = float(hp_train.get("lr", args.lr))
        args.wd = float(hp_train.get("weight_decay", args.wd))
        args.epochs = int(hp_train.get("epochs", args.epochs))
        args.batch = int(hp_train.get("batch_size", args.batch))
        args.label_smoothing = float(hp_train.get("label_smoothing", args.label_smoothing))
        args.grad_clip = float(hp_train.get("gradient_clip", args.grad_clip))
        args.noise_injection = float(hp_train.get("noise_injection", args.noise_injection))

    args.msg_net_size = int(getattr(args, "msg_net_size", max(96, int(args.d_model))))
    if bool(getattr(args, "online", False)) and float(getattr(args, "erasure_frac", 0.0)) > 0.0:
        args.node_feat_dim = max(int(args.node_feat_dim), 10)

    expected_node_dim = int(args.node_feat_dim)
    expected_edge_dim = int(args.edge_feat_dim)

    # Resolve save directory (auto if requested or missing)
    def _auto_name(family: str, distance: int, qpu_profile: str | None) -> str:
        ts = time.strftime("%Y%m%d-%H%M%S")
        qpu = "none"
        if qpu_profile:
            import os as _os

            qpu = _os.path.splitext(_os.path.basename(qpu_profile))[0]
        return f"{ts}_{family}_d{int(distance)}_{qpu}"

    if (
        args.save_auto
        or args.save is None
        or (isinstance(args.save, str) and "{auto}" in args.save)
    ):
        auto = _auto_name(
            getattr(args, "family", "code"),
            int(getattr(args, "distance", 0)),
            getattr(args, "qpu_profile", None),
        )
        base = Path(getattr(args, "save_root", "data/results"))
        base.mkdir(parents=True, exist_ok=True)
        args.save = str(base / auto)
    else:
        # Support placeholder replacement in explicit paths
        if "{auto}" in args.save:
            auto = _auto_name(
                getattr(args, "family", "code"),
                int(getattr(args, "distance", 0)),
                getattr(args, "qpu_profile", None),
            )
            args.save = args.save.replace("{auto}", auto)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Prefer CUDA when available; optionally require it via MGHD_REQUIRE_CUDA=1.
    require_cuda = os.getenv("MGHD_REQUIRE_CUDA", "").lower() in {"1", "true", "yes"}
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    elif require_cuda:
        raise RuntimeError(
            "CUDA GPU is required but not available (MGHD_REQUIRE_CUDA=1). "
            "Install a CUDA-enabled PyTorch build and run on a GPU node."
        )
    else:
        device = torch.device("cpu")
    # Mixed precision + TF32 tuning for H100 throughput
    amp_mode = str(getattr(args, "amp", "bf16")).lower()
    use_amp = False
    amp_dtype = torch.float32
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if amp_mode != "off":
            if amp_mode == "fp16":
                amp_dtype = torch.float16
            elif amp_mode == "auto":
                bf16_ok = (
                    hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
                )
                amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
            else:
                amp_dtype = torch.bfloat16
            use_amp = amp_dtype != torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    def _autocast():
        if device.type == "cuda":
            return torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp)
        return nullcontext()

    # Build model with optional hyperparameters from JSON/data contract
    m_kwargs = {
        "d_model": int(getattr(args, "d_model", 192)),
        "d_state": int(getattr(args, "d_state", 80)),
        "n_iters": int(getattr(args, "n_iters", 8)),
        "gnn_msg_net_size": int(
            getattr(args, "msg_net_size", max(96, getattr(args, "d_model", 192)))
        ),
        "gnn_msg_dropout": float(getattr(args, "msg_net_dropout_p", 0.0)),
        "gnn_gru_dropout": float(getattr(args, "gru_dropout_p", 0.0)),
        "se_reduction": int(getattr(args, "se_reduction", 4)),
        "node_feat_dim": expected_node_dim,
        "edge_feat_dim": expected_edge_dim,
        "g_dim": int(getattr(args, "g_token_dim", 12)),
    }
    model = MGHDv2(profile=args.profile, **m_kwargs).to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def _backward_and_step(loss: torch.Tensor):
        """Handle AMP-aware backward/step with grad clipping."""
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            opt.step()
        opt.zero_grad(set_to_none=True)

    start_epoch = 1
    if getattr(args, "resume", None):
        ckpt_path = args.resume
        if os.path.isfile(ckpt_path):
            if rank == 0:
                print(f"Resuming from checkpoint: {ckpt_path}")
            # Map location to CPU first to avoid GPU OOM or device mismatch, then load_state_dict handles move
            checkpoint = torch.load(ckpt_path, map_location="cpu")

            # The saved state_dict is unwrapped (see save logic below).
            # If current model is DDP, we load into model.module.
            state_dict = checkpoint["model"]

            # Check if g_proj needs initialization
            if "g_proj.weight" in state_dict:
                g_dim = state_dict["g_proj.weight"].shape[1]
                if is_distributed:
                    model.module.ensure_g_proj(g_dim, device)
                else:
                    model.ensure_g_proj(g_dim, device)
                if rank == 0:
                    print(f"Initialized g_proj with input dim {g_dim} from checkpoint.")

            if is_distributed:
                load_info = model.module.load_state_dict(state_dict, strict=False)
            else:
                load_info = model.load_state_dict(state_dict, strict=False)
            if rank == 0:
                missing = list(getattr(load_info, "missing_keys", []))
                unexpected = list(getattr(load_info, "unexpected_keys", []))
                if missing:
                    print(
                        "Checkpoint load note: missing keys (showing up to 10): "
                        + ", ".join(missing[:10])
                    )
                if unexpected:
                    print(
                        "Checkpoint load note: unexpected keys (showing up to 10): "
                        + ", ".join(unexpected[:10])
                    )

            if "optimizer" in checkpoint:
                try:
                    opt.load_state_dict(checkpoint["optimizer"])
                    if rank == 0:
                        print("Loaded optimizer state.")
                except Exception as e:
                    if rank == 0:
                        print(f"Failed to load optimizer state: {e}")

            start_epoch = checkpoint.get("epoch", 0) + 1

            # Align scheduler without stepping before first optimizer step
            if start_epoch > 1:
                sched.last_epoch = start_epoch - 1

            if rank == 0:
                print(f"Resumed model at epoch {start_epoch - 1}. Next epoch: {start_epoch}")
        else:
            if rank == 0:
                print(f"Checkpoint not found at {ckpt_path}, starting from scratch.")

    loader = None
    use_online = bool(getattr(args, "online", False))
    if use_online and bool(getattr(args, "online_fast", False)):
        keep_aux = bool(getattr(args, "online_fast_keep_aux", False))
        if not keep_aux:
            if int(getattr(args, "projection_aware", 0)) != 0:
                args.projection_aware = 0
            if float(getattr(args, "parity_lambda", 0.0)) != 0.0:
                args.parity_lambda = 0.0
        if int(getattr(args, "progress_prints", 0)) <= 1:
            args.progress_prints = 20
        if float(getattr(args, "progress_seconds", 0.0)) <= 0.0:
            args.progress_seconds = 20.0
        if int(getattr(args, "prefetch_factor", 0)) < 4:
            args.prefetch_factor = 4
        if rank == 0:
            print(
                "[online-fast] Enabled: "
                f"projection_aware={int(getattr(args, 'projection_aware', 0))} "
                f"parity_lambda={float(getattr(args, 'parity_lambda', 0.0))} "
                f"keep_aux={keep_aux} progress_prints={args.progress_prints} "
                f"progress_seconds={args.progress_seconds}"
            )
    if not use_online:
        raise RuntimeError(
            "Offline (crop-shard) training has been removed. Use --online mode."
        )

    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)
    # Persist run metadata for reproducibility
    try:
        base_p = float(getattr(args, "p", 0.0))
        n_rounds = int(getattr(args, "rounds", 0)) or int(args.distance)
        p_curriculum_vals = (
            [float(x) for x in str(getattr(args, "p_curriculum", "")).split(",") if x.strip()]
            if getattr(args, "p_curriculum", None)
            else []
        )
        run_meta = {
            "family": args.family,
            "distance": int(args.distance),
            "distance_curriculum": (
                [int(x) for x in str(getattr(args, "distance_curriculum", "")).split(",") if x.strip()]
                or [int(args.distance)]
            ),
            "component_scope": str(getattr(args, "component_scope", "active")),
            "online": use_online,
            "p": base_p,
            "p_curriculum": p_curriculum_vals if p_curriculum_vals else None,
            "noise_model": str(getattr(args, "noise_model", "SI1000")),
            "rounds": n_rounds,
            "sampler": sampler_choice,
            "teacher_mix": getattr(args, "teacher_mix", None),
            "shots_per_epoch": int(getattr(args, "shots_per_epoch", 0)),
            "epochs": int(args.epochs),
            "batch": int(args.batch),
            "workers": int(getattr(args, "workers", 0)),
            "profile": str(getattr(args, "profile", "S")),
            "node_feat_dim": int(getattr(args, "node_feat_dim", 10)),
            "edge_feat_dim": int(getattr(args, "edge_feat_dim", 3)),
            "g_token_dim": int(getattr(args, "g_token_dim", 14)),
            "lr": float(getattr(args, "lr", 0.0)),
            "wd": float(getattr(args, "wd", 0.0)),
            "amp": str(getattr(args, "amp", "off")),
            "seed": int(args.seed),
        }
        (save_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))
    except Exception:
        pass

    best_loss = float("inf")
    history: list[dict[str, Any]] = []
    last_improve_epoch = 0
    if getattr(args, "resume", None):
        hist_jsonl = save_dir / "train_log.jsonl"
        if hist_jsonl.exists():
            try:
                with hist_jsonl.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        if not isinstance(obj, dict):
                            continue
                        if "epoch" not in obj or "loss" not in obj:
                            continue
                        history.append(obj)
            except Exception:
                history = []
        if history:
            best_loss = min(float(item.get("loss", float("inf"))) for item in history)
            best_epochs = [
                int(item.get("epoch", 0))
                for item in history
                if float(item.get("loss", float("inf"))) <= best_loss + 1e-12
            ]
            if best_epochs:
                last_improve_epoch = max(best_epochs)

    def _parse_teacher_mix(spec: str) -> dict[str, float]:
        """Parse a comma-separated teacher weight spec into a dict.

        Example: "pymatching=1.0" → {"pymatching": 1.0}
        """
        weights: dict[str, float] = {}
        if not spec:
            weights["pymatching"] = 1.0
            return weights
        for chunk in spec.split(","):
            if "=" not in chunk:
                continue
            name, value = chunk.split("=", 1)
            try:
                weights[name.strip().lower()] = float(value)
            except ValueError:
                continue
        if sum(weights.values()) <= 0.0:
            weights["pymatching"] = 1.0
        return weights

    # Optional curriculum over p for online mode
    p_list = None
    if use_online and getattr(args, "p_curriculum", None):
        try:
            p_list = [float(x) for x in str(args.p_curriculum).split(",") if x.strip()]
        except Exception:
            p_list = None
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        last_heartbeat = t0
        total_loss = 0.0
        n_items = 0
        distance_loss_sum: dict[int, float] = defaultdict(float)
        distance_item_count: dict[int, int] = defaultdict(int)
        if use_online:
            # Per-sample p mixing: pass the full p_list so each shot samples its own p.
            # Falls back to single fixed p if no curriculum.
            if p_list:
                p_epoch = p_list  # list signals per-sample mixing
            else:
                p_epoch = float(getattr(args, "p", 0.005))

            # =====================================================================
            # DEM-based circuit-level supervision (Stim + PyMatching teacher)
            # =====================================================================
            teacher_mix = _parse_teacher_mix(
                getattr(args, "teacher_mix", "pymatching=1.0")
            )

            # Setup DataLoader for parallel generation
            workers = max(0, int(getattr(args, "workers", 0)))
            shots_target = int(getattr(args, "shots_per_epoch", args.batch))

            # Divide shots among ranks
            if is_distributed:
                shots_target = shots_target // world_size

            # Pass per-rank shot budget; dataset will split this across workers to avoid duplication
            dataset = OnlineSurfaceDataset(
                args, p_epoch, epoch, shots_target, teacher_mix, rank=rank
            )

            # In DDP mode, divide batch size across GPUs
            effective_batch = args.batch // world_size if is_distributed else args.batch

            loader = DataLoader(
                dataset,
                batch_size=effective_batch,
                num_workers=workers,
                collate_fn=collate_packed,
                pin_memory=True if torch.cuda.is_available() else False,
                persistent_workers=bool(workers > 0),
                prefetch_factor=int(getattr(args, "prefetch_factor", 2)) if workers > 0 else None,
            )

            steps_done = 0

            prog_stride = (
                max(1, shots_target // int(getattr(args, "progress_prints", 1)))
                if int(getattr(args, "progress_prints", 1)) > 1
                else 0
            )

            # OnlineIterableDataset yields a variable number of crops per shot (and may yield
            # zero crops for empty syndromes). This can cause different ranks to see different
            # numbers of batches and deadlock in DDP allreduces. DDP's join() context handles
            # uneven inputs by inserting shadow collectives on ranks that run out early.
            join_ctx = model.join() if is_distributed and hasattr(model, "join") else nullcontext()
            with join_ctx:
                for batch in loader:
                    if not batch:
                        continue

                    # batch is a single PackedCrop (batched)
                    packed = batch
                    _validate_packed_contract(packed, expected_node_dim, expected_edge_dim)
                    packed = move_to(packed, device)

                    with _autocast():
                        logits, node_mask = model(packed=packed)
                        # logits: (B*N, 2), node_mask: (B*N,)

                        # Reshape for structured losses
                        B = packed.x_nodes.shape[0]
                        N = packed.x_nodes.shape[1]
                        logits_reshaped = logits.view(B, N, 2)
                        node_mask_reshaped = node_mask.view(B, N)
                        node_type_reshaped = packed.node_type  # (B, N)

                        hard = 1.0
                        sample_weight = torch.tensor(hard, dtype=torch.float32, device=device)

                        if bool(getattr(args, "use_focal", False)):
                            loss_bce = focal_binary_head_loss(
                                logits,
                                node_mask,
                                packed.node_type.view(-1),
                                packed.y_bits.view(-1),
                                alpha=float(getattr(args, "focal_alpha", 0.25)),
                                gamma=float(getattr(args, "focal_gamma", 2.0)),
                                sample_weight=sample_weight,
                            )
                        else:
                            loss_bce = bce_binary_head_loss(
                                logits,  # flattened
                                node_mask,  # flattened
                                packed.node_type.view(-1),
                                packed.y_bits.view(-1),
                                sample_weight=sample_weight,
                                label_smoothing=args.label_smoothing,
                            )

                        loss_par = args.parity_lambda * parity_auxiliary_loss(
                            logits_reshaped,
                            node_mask_reshaped,
                            node_type_reshaped,
                            H_sub=packed.H_sub,
                            s_sub=packed.s_sub,
                        )

                        loss_proj = torch.tensor(0.0, device=device)
                        if args.projection_aware:
                            proj_loss_sum = 0.0
                            for i in range(B):
                                l = logits_reshaped[i]
                                nm = node_mask_reshaped[i]
                                nt = node_type_reshaped[i]
                                yb = packed.y_bits[i]
                                h = (
                                    packed.H_sub[i] if isinstance(packed.H_sub, list) else packed.H_sub
                                )

                                data_mask = (nt == 0) & nm

                                side = "Z"
                                if hasattr(packed.meta, "batch_metas"):
                                    side = getattr(packed.meta.batch_metas[i], "side", "Z")
                                elif hasattr(packed.meta, "side"):
                                    side = packed.meta.side

                                proj_bits = projection_aware_logits_to_bits(
                                    l,
                                    projector_kwargs={
                                        "H_sub": h,
                                        "side": side,
                                        "s_sub": packed.s_sub[i],
                                    },
                                    data_mask=data_mask,
                                )
                                with torch.no_grad():
                                    mask_data = data_mask.detach().cpu().numpy()
                                    target_full = (
                                        yb.detach().cpu().numpy().clip(0, 1).astype(np.uint8)
                                    )
                                    target_data = target_full[mask_data]
                                proj_target = torch.from_numpy(target_data).to(device)
                                proj_pred = torch.from_numpy(proj_bits.astype(np.int64)).to(device)
                                raw_bits = (
                                    torch.sigmoid(l[:, 1] - l[:, 0])[data_mask] > 0.5
                                ).long()
                                p_loss = 0.5 * F.l1_loss(
                                    proj_pred.float(), proj_target.float()
                                ) + 0.2 * F.l1_loss(proj_pred.float(), raw_bits.float())
                                proj_loss_sum += p_loss
                            loss_proj = proj_loss_sum / B

                        sample_loss = loss_bce + loss_par + 0.5 * loss_proj

                        batch_loss = sample_loss

                    _backward_and_step(batch_loss)

                    batch_size = B
                    total_loss += batch_loss.detach().item() * batch_size
                    n_items += batch_size
                    with torch.no_grad():
                        batch_metas = getattr(packed.meta, "batch_metas", None)
                        if batch_metas:
                            dist_ids = torch.as_tensor(
                                [
                                    int(getattr(meta_i, "d", getattr(args, "distance", 0)))
                                    for meta_i in batch_metas
                                ],
                                device=logits_reshaped.device,
                                dtype=torch.long,
                            )
                            for dist_id in torch.unique(dist_ids):
                                sample_sel = dist_ids == dist_id
                                if not bool(torch.any(sample_sel)):
                                    continue
                                logits_sel = logits_reshaped[sample_sel].reshape(-1, 2)
                                mask_sel = node_mask_reshaped[sample_sel].reshape(-1)
                                type_sel = node_type_reshaped[sample_sel].reshape(-1)
                                y_sel = packed.y_bits[sample_sel].reshape(-1)
                                d_loss = bce_binary_head_loss(
                                    logits_sel,
                                    mask_sel,
                                    type_sel,
                                    y_sel,
                                    sample_weight=None,
                                    label_smoothing=args.label_smoothing,
                                ).detach()
                                d_key = int(dist_id.item())
                                d_count = int(sample_sel.sum().item())
                                distance_loss_sum[d_key] += float(d_loss.item()) * d_count
                                distance_item_count[d_key] += d_count
                        else:
                            d_key = int(getattr(args, "distance", 0))
                            distance_loss_sum[d_key] += float(batch_loss.detach().item()) * int(batch_size)
                            distance_item_count[d_key] += int(batch_size)
                    steps_done += 1

                    if prog_stride and (steps_done % prog_stride == 0) and rank == 0:
                        prog = {
                            "epoch": epoch,
                            "step": int(steps_done),
                            "avg": float(total_loss / max(n_items, 1)),
                            "secs": float(time.time() - t0),
                        }
                        if p_epoch is not None:
                            prog["p"] = "mixed" if isinstance(p_epoch, list) else float(p_epoch)
                        print(json.dumps(prog, separators=(",", ":")), flush=True)
                    heartbeat_s = float(getattr(args, "progress_seconds", 0.0))
                    now = time.time()
                    if (
                        rank == 0
                        and heartbeat_s > 0.0
                        and (now - last_heartbeat) >= heartbeat_s
                    ):
                        hb = {
                            "heartbeat": True,
                            "epoch": int(epoch),
                            "step": int(steps_done),
                            "avg": float(total_loss / max(n_items, 1)),
                            "secs": float(now - t0),
                            "items": int(n_items),
                        }
                        if distance_item_count:
                            hb["distance_counts"] = {
                                str(k): int(v) for k, v in sorted(distance_item_count.items())
                            }
                        if p_epoch is not None:
                            hb["p"] = "mixed" if isinstance(p_epoch, list) else float(p_epoch)
                        print(json.dumps(hb, separators=(",", ":")), flush=True)
                        last_heartbeat = now

        if not use_online:
            raise RuntimeError(
                "Offline (crop-shard) training has been removed. Use --online mode."
            )

        sched.step()
        dt = time.time() - t0
        avg = total_loss / max(n_items, 1)
        epoch_distance_loss = {
            str(k): float(distance_loss_sum[k] / max(distance_item_count.get(k, 1), 1))
            for k in sorted(distance_loss_sum.keys())
            if distance_item_count.get(k, 0) > 0
        }
        epoch_distance_counts = {
            str(k): int(distance_item_count[k])
            for k in sorted(distance_item_count.keys())
            if distance_item_count[k] > 0
        }
        history.append(
            {
                "epoch": epoch,
                "loss": avg,
                "count": n_items,
                "secs": dt,
                "distance_loss": epoch_distance_loss if epoch_distance_loss else None,
                "distance_counts": epoch_distance_counts if epoch_distance_counts else None,
            }
        )

        if rank == 0:
            _ckpt_config = {
                "profile": str(getattr(args, "profile", "S")),
                "d_model": int(getattr(args, "d_model", 192)),
                "d_state": int(getattr(args, "d_state", 80)),
                "n_iters": int(getattr(args, "n_iters", 8)),
                "node_feat_dim": int(getattr(args, "node_feat_dim", 10)),
                "edge_feat_dim": int(getattr(args, "edge_feat_dim", 3)),
                "g_token_dim": int(getattr(args, "g_token_dim", 14)),
                "component_scope": str(getattr(args, "component_scope", "active")),
                "noise_model": str(getattr(args, "noise_model", "SI1000")),
            }
            torch.save(
                {
                    "model": model.module.state_dict() if is_distributed else model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "epoch": epoch,
                    "loss": avg,
                    "model_config": _ckpt_config,
                },
                save_dir / "last.pt",
            )
            if avg < best_loss - float(getattr(args, "early_stop_min_delta", 0.0)):
                best_loss = avg
                torch.save(
                    {
                        "model": model.module.state_dict()
                        if is_distributed
                        else model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "epoch": epoch,
                        "loss": avg,
                        "model_config": _ckpt_config,
                    },
                    save_dir / "best.pt",
                )

            log_obj = {"epoch": epoch, "loss": avg, "secs": dt}
            if use_online:
                log_obj["p"] = "mixed" if isinstance(p_epoch, list) else float(p_epoch if "p_epoch" in locals() else getattr(args, "p", 0.0))
            if epoch_distance_loss:
                log_obj["distance_loss"] = epoch_distance_loss
            if epoch_distance_counts:
                log_obj["distance_counts"] = epoch_distance_counts
            # Print epoch summary and flush so users see it promptly
            print(json.dumps(log_obj, separators=(",", ":")), flush=True)

            # Persist logs incrementally each epoch (JSONL + JSON snapshot)
            try:
                with (save_dir / "train_log.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(log_obj, separators=(",", ":")) + "\n")
                (save_dir / "train_log.json").write_text(
                    json.dumps(history, indent=2), encoding="utf-8"
                )
            except Exception:
                pass

        # Early stopping check
        patience = int(getattr(args, "early_stop_patience", 0))
        min_delta = float(getattr(args, "early_stop_min_delta", 0.0))
        if patience > 0:
            # Broadcast best_loss to all ranks to ensure consistent stopping
            if is_distributed:
                best_loss_tensor = torch.tensor(best_loss, device=device)
                dist.all_reduce(best_loss_tensor, op=dist.ReduceOp.MIN)
                best_loss = best_loss_tensor.item()

            if avg <= best_loss + 1e-12:
                last_improve_epoch = epoch
            if (epoch - last_improve_epoch) >= patience:
                if rank == 0:
                    print(
                        json.dumps(
                            {"early_stop": True, "epoch": epoch, "best_loss": best_loss},
                            separators=(",", ":"),
                        )
                    )
                break

    # Final snapshot (in case of early termination without last write)
    if rank == 0:
        try:
            (save_dir / "train_log.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    if is_distributed:
        dist.destroy_process_group()

    return os.path.join(args.save, "best.pt")


def move_to(p: PackedCrop, device):
    """Move all tensors in a PackedCrop to a target device (in place)."""
    p.x_nodes = p.x_nodes.to(device)
    p.node_mask = p.node_mask.to(device)
    p.node_type = p.node_type.to(device)
    p.edge_index = p.edge_index.to(device)
    p.edge_attr = p.edge_attr.to(device)
    p.edge_mask = p.edge_mask.to(device)
    p.seq_idx = p.seq_idx.to(device)
    p.seq_mask = p.seq_mask.to(device)
    p.g_token = p.g_token.to(device)
    p.y_bits = p.y_bits.to(device)
    p.s_sub = p.s_sub.to(device)
    return p


# ─── DEM-based Stim sampling infrastructure ───────────────────────────────────
#
# Graph structure: "merged-edge" bipartite graph.
#
#   type-1 nodes = detectors  (carry the syndrome)
#   type-0 nodes = edges      (unique detector-pairs / boundary-det from the DEM)
#
# Each unique (detector_set, observable_set) in the DEM becomes one edge-node.
# Probabilities of multiple DEM instructions that map to the same edge are
# combined: p_merged = 1 - prod(1 - p_i).
#
# This keeps the graph orders of magnitude smaller than treating every decomposed
# DEM fault as a separate node, and it maps cleanly onto MGHD's bipartite
# (data, check) architecture.  The model predicts per-edge-node labels; the
# observable correction is the XOR of observable_flags for predicted-active edges.
#
# Active-component decomposition works on the detector adjacency graph (small
# and local), then we pull in the edge-nodes for each component.
# ---------------------------------------------------------------------------

# Global cache for workers to avoid rebuilding circuits/DEM/teachers every iteration
_WORKER_DEM_CACHE: dict[tuple, dict] = {}


def _build_stim_circuit(
    distance: int,
    rounds: int,
    p: float,
    noise_model: str = "SI1000",
    after_clifford_depolarization: float = 0.0,
) -> "stim.Circuit":
    """Build a Stim circuit for rotated surface code memory experiment."""
    import stim

    kwargs: dict = {
        "distance": distance,
        "rounds": rounds,
    }
    # SI1000 uses a single 'p' parameter for all noise
    if noise_model.upper() == "SI1000":
        kwargs["after_clifford_depolarization"] = p
        kwargs["after_reset_flip_probability"] = p
        kwargs["before_measure_flip_probability"] = p
        kwargs["before_round_data_depolarization"] = p
    elif after_clifford_depolarization > 0:
        kwargs["after_clifford_depolarization"] = after_clifford_depolarization
    else:
        kwargs["after_clifford_depolarization"] = p

    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        **kwargs,
    )


def _build_dem_info(
    distance: int,
    rounds: int,
    p: float,
    noise_model: str = "SI1000",
    edge_prune_thresh: float = 1e-3,
):
    """Build DEM info with merged-edge graph representation.

    Every unique (frozenset_of_detectors, frozenset_of_observables) becomes
    one edge-node.  Multiple DEM instructions mapping to the same key are
    combined by probability.  Edges with merged probability below
    ``edge_prune_thresh`` are dropped to keep graphs manageable.

    Returns dict with keys:
        circuit, dem, sampler, matching,
        H_edge    – np.ndarray (n_det, n_edges) uint8 incidence matrix,
        L_obs     – np.ndarray (n_obs, n_edges) uint8 observable flags,
        edge_probs – np.ndarray (n_edges,) merged probabilities,
        det_coords – np.ndarray (n_det, 3)  detector (x, y, t),
        edge_coords – np.ndarray (n_edges, 3) edge-node midpoint coords,
        det_adj   – np.ndarray (n_det, n_det) uint8 detector adjacency,
        n_det, n_edges, n_obs,
        edge_key_to_idx – dict mapping (frozenset, frozenset) → edge index,
        det_pair_to_edge – dict mapping frozenset({d0, d1}) → edge index
                           (for teacher label mapping),
    """
    import stim
    import pymatching

    circuit = _build_stim_circuit(distance, rounds, p, noise_model)
    dem = circuit.detector_error_model(decompose_errors=True)
    sampler = circuit.compile_detector_sampler()
    n_det = circuit.num_detectors
    n_obs = circuit.num_observables

    # ── Parse DEM and merge by unique (det_set, obs_set) ──────────────
    from collections import defaultdict
    edge_probs_raw: dict[tuple, list[float]] = defaultdict(list)
    for instruction in dem.flattened():
        if instruction.type == "error":
            prob = instruction.args_copy()[0]
            dets = frozenset(
                t.val for t in instruction.targets_copy() if t.is_relative_detector_id()
            )
            obs = frozenset(
                t.val for t in instruction.targets_copy() if t.is_logical_observable_id()
            )
            edge_probs_raw[(dets, obs)].append(prob)

    # Build deterministic ordering of edge-nodes, pruning low-prob edges
    all_keys_sorted = sorted(edge_probs_raw.keys(), key=lambda k: (sorted(k[0]), sorted(k[1])))
    edge_keys: list[tuple] = []
    edge_probs_list: list[float] = []
    for key in all_keys_sorted:
        survival = 1.0
        for pi in edge_probs_raw[key]:
            survival *= 1.0 - pi
        merged_p = 1.0 - survival
        if merged_p >= edge_prune_thresh:
            edge_keys.append(key)
            edge_probs_list.append(merged_p)

    edge_key_to_idx = {k: i for i, k in enumerate(edge_keys)}
    n_edges = len(edge_keys)
    edge_probs = np.array(edge_probs_list, dtype=np.float64)

    # Incidence matrix  H_edge[det, edge] = 1
    H_edge = np.zeros((n_det, n_edges), dtype=np.uint8)
    L_obs = np.zeros((n_obs, n_edges), dtype=np.uint8)
    for i, (dets, obs) in enumerate(edge_keys):
        for d_id in dets:
            if d_id < n_det:
                H_edge[d_id, i] = 1
        for o_id in obs:
            if o_id < n_obs:
                L_obs[o_id, i] = 1

    # Detector adjacency (for active-component decomposition)
    det_adj = np.zeros((n_det, n_det), dtype=np.uint8)
    det_pair_to_edge: dict[frozenset, int] = {}
    for i, (dets, obs) in enumerate(edge_keys):
        dets_list = sorted(dets)
        if len(dets_list) == 2:
            det_adj[dets_list[0], dets_list[1]] = 1
            det_adj[dets_list[1], dets_list[0]] = 1
            det_pair_to_edge[frozenset(dets_list)] = i
        elif len(dets_list) == 1:
            # boundary edge → store as {det, -1}
            det_pair_to_edge[frozenset([dets_list[0], -1])] = i

    # Detector coordinates
    det_coord_dict = dem.get_detector_coordinates()
    det_coords = np.zeros((n_det, 3), dtype=np.float32)
    for det_id, coords in det_coord_dict.items():
        if det_id < n_det:
            for j, c in enumerate(coords[:3]):
                det_coords[det_id, j] = float(c)

    # Edge-node coordinates = midpoint of connected detectors
    edge_coords = np.zeros((n_edges, 3), dtype=np.float32)
    for i in range(n_edges):
        connected = np.where(H_edge[:, i])[0]
        if len(connected) > 0:
            edge_coords[i] = det_coords[connected].mean(axis=0)

    # PyMatching teacher from DEM
    matching = pymatching.Matching.from_detector_error_model(dem)

    return {
        "circuit": circuit,
        "dem": dem,
        "sampler": sampler,
        "matching": matching,
        "H_edge": H_edge,
        "L_obs": L_obs,
        "edge_probs": edge_probs,
        "det_coords": det_coords,
        "edge_coords": edge_coords,
        "det_adj": det_adj,
        "n_det": n_det,
        "n_edges": n_edges,
        "n_obs": n_obs,
        "edge_key_to_idx": edge_key_to_idx,
        "det_pair_to_edge": det_pair_to_edge,
    }


def _get_dem_info(
    distance: int,
    rounds: int,
    p: float,
    noise_model: str = "SI1000",
    edge_prune_thresh: float = 1e-3,
):
    """Get or create cached DEM info."""
    global _WORKER_DEM_CACHE
    key = (distance, rounds, p, noise_model, edge_prune_thresh)
    if key not in _WORKER_DEM_CACHE:
        _WORKER_DEM_CACHE[key] = _build_dem_info(
            distance, rounds, p, noise_model, edge_prune_thresh
        )
    return _WORKER_DEM_CACHE[key]


def _teacher_labels_from_matching(
    det_bits: np.ndarray,
    matching,
    det_pair_to_edge: dict,
    n_edges: int,
) -> np.ndarray:
    """Convert PyMatching decoded result to per-edge-node labels.

    PyMatching's ``decode_to_matched_dets_array`` returns detector pairs.
    We map each pair to the corresponding merged edge-node via
    ``det_pair_to_edge``.
    """
    y_bits = np.zeros(n_edges, dtype=np.uint8)
    try:
        matched = matching.decode_to_matched_dets_array(det_bits)
    except Exception:
        return y_bits

    for pair in matched:
        d0, d1 = int(pair[0]), int(pair[1])
        # boundary match: PyMatching uses large sentinel or -1
        if d0 < 0 or d1 < 0:
            real_det = d0 if d1 < 0 else d1
            key = frozenset([real_det, -1])
        else:
            key = frozenset([d0, d1])
        eidx = det_pair_to_edge.get(key)
        if eidx is not None:
            y_bits[eidx] = 1
    return y_bits


def pack_dem_cluster(
    H_edge: np.ndarray,
    det_coords: np.ndarray,
    edge_coords: np.ndarray,
    det_bits: np.ndarray,
    y_bits_edge: np.ndarray,
    *,
    edge_indices: np.ndarray | None = None,
    det_indices: np.ndarray | None = None,
    d: int,
    rounds: int,
    p: float,
    N_max: int,
    E_max: int,
    S_max: int,
) -> PackedCrop | None:
    """Pack a merged-edge DEM subgraph into a PackedCrop for MGHDv2.

    Bipartite structure:
        type-0 nodes = edge-nodes (unique DEM edges; we predict their flip)
        type-1 nodes = detectors  (carry the syndrome)
        graph edges  = H_sub incidence (detector ↔ edge-node)

    Parameters
    ----------
    H_edge : (n_det, n_edges) incidence matrix (full or will be sliced)
    det_coords, edge_coords : (n, 3) coordinates
    det_bits : (n_det,) syndrome
    y_bits_edge : (n_edges,) teacher labels (which edge-nodes are active)
    edge_indices, det_indices : optional subsets for component extraction
    d, rounds, p : metadata for conditioning
    N_max, E_max, S_max : padding limits
    """
    # ── Extract subgraph ──────────────────────────────────────────────
    if edge_indices is not None and det_indices is not None:
        H_sub = H_edge[np.ix_(det_indices, edge_indices)].astype(np.uint8)
        xy_edge = edge_coords[edge_indices]
        xy_det = det_coords[det_indices]
        synd = det_bits[det_indices].astype(np.uint8)
        y_local = y_bits_edge[edge_indices].astype(np.uint8)
    else:
        H_sub = np.asarray(H_edge, dtype=np.uint8)
        xy_edge = edge_coords
        xy_det = det_coords
        synd = det_bits.astype(np.uint8)
        y_local = y_bits_edge.astype(np.uint8)

    nC, nQ = H_sub.shape  # nC = detectors, nQ = edge-nodes
    if nQ == 0 or nC == 0:
        return None
    N = nQ + nC
    if N > N_max:
        return None

    # ── Node features (10-dim) ────────────────────────────────────────
    deg_edge = H_sub.sum(axis=0).astype(np.float32)  # per edge-node
    deg_det = H_sub.sum(axis=1).astype(np.float32)   # per detector

    all_xyz = np.vstack([xy_edge, xy_det])
    coord_min = all_xyz.min(axis=0)
    coord_max = all_xyz.max(axis=0)
    coord_span = np.maximum(coord_max - coord_min, 1.0)

    xy_e_norm = (xy_edge - coord_min) / coord_span
    xy_d_norm = (xy_det - coord_min) / coord_span

    node_type = np.concatenate([np.zeros(nQ, dtype=np.int8),
                                np.ones(nC, dtype=np.int8)])
    degree = np.concatenate([deg_edge, deg_det])
    xy_all = np.vstack([xy_e_norm, xy_d_norm]).astype(np.float32)

    # (x, y, t, type, degree, k, r, d_norm, rounds_norm, syndrome) = 10
    parts = [
        xy_all,                                                     # 3
        node_type[:, None].astype(np.float32),                     # 1
        degree[:, None],                                            # 1
        np.full((N, 1), float(nC), dtype=np.float32),              # k
        np.full((N, 1), float(nQ), dtype=np.float32),              # r
        np.full((N, 1), float(d) / 20.0, dtype=np.float32),        # d_norm
        np.full((N, 1), float(rounds) / 20.0, dtype=np.float32),   # rounds_norm
    ]
    synd_clipped = synd[:nC] if synd.size >= nC else np.pad(synd, (0, nC - synd.size))
    synd_col = np.zeros((N, 1), dtype=np.float32)
    synd_col[nQ:, 0] = synd_clipped.astype(np.float32)
    parts.append(synd_col)                                         # 1  total=10

    base_nodes = np.concatenate(parts, axis=1)

    # ── Graph edges from H_sub ────────────────────────────────────────
    ci, qi = np.nonzero(H_sub)
    src = qi.astype(np.int64)
    dst = (nQ + ci).astype(np.int64)
    edge_index = np.stack([src, dst], axis=0)
    E_tot = edge_index.shape[1]
    if E_tot > E_max:
        return None

    edge_attr = np.zeros((E_tot, 3), dtype=np.float32)
    edge_attr[:, 0] = 1.0

    # ── Mamba sequence: (t, hilbert_2d) ordering on detectors ─────────
    from mghd.core.core import _quantize_xy01, _hilbert_index_2d
    det_xy01 = xy_d_norm[:, :2]
    qxy = _quantize_xy01(det_xy01, levels=64)
    h2d = _hilbert_index_2d(qxy, levels=64)
    t_q = (xy_d_norm[:, 2] * 1000).astype(np.int64)
    sort_key = t_q * 100000 + h2d
    check_order = np.argsort(sort_key, kind="stable")
    seq_idx = (nQ + check_order).astype(np.int64)
    S = nC
    if S > S_max:
        return None

    # ── g_token (14-dim) ──────────────────────────────────────────────
    p_safe = max(float(p), 1e-12)
    g_list = [
        float(nC), float(nQ), float(d), float(rounds),
        p_safe, float(np.log10(p_safe)),
        float(nC) / max(1.0, float(d ** 2)),
        float(nQ) / max(1.0, float(d ** 2)),
        float(coord_span[0]), float(coord_span[1]), float(coord_span[2]),
        float(nC) / max(1.0, float(N)),
        float(np.sum(synd_clipped)) / max(1.0, float(nC)),
        0.0,
    ]
    g_token = torch.tensor(g_list[:14], dtype=torch.float32)

    # ── Pad and tensorize ─────────────────────────────────────────────
    x_nodes = torch.zeros((N_max, base_nodes.shape[1]), dtype=torch.float32)
    node_mask = torch.zeros(N_max, dtype=torch.bool)
    node_type_t = torch.zeros(N_max, dtype=torch.int8)
    x_nodes[:N] = torch.from_numpy(base_nodes)
    node_mask[:N] = True
    node_type_t[:N] = torch.from_numpy(node_type)

    ei_t = torch.zeros((2, E_max), dtype=torch.long)
    ea_t = torch.zeros((E_max, 3), dtype=torch.float32)
    em_t = torch.zeros(E_max, dtype=torch.bool)
    ei_t[:, :E_tot] = torch.from_numpy(edge_index)
    ea_t[:E_tot] = torch.from_numpy(edge_attr)
    em_t[:E_tot] = True

    si_t = torch.zeros(S_max, dtype=torch.long)
    sm_t = torch.zeros(S_max, dtype=torch.bool)
    si_t[:S] = torch.from_numpy(seq_idx)
    sm_t[:S] = True

    y_t = torch.full((N_max,), -1, dtype=torch.int8)
    y_t[:nQ] = torch.from_numpy(y_local.astype(np.int8))

    s_t = torch.zeros(S_max, dtype=torch.int8)
    s_t[:S] = torch.from_numpy(synd_clipped.astype(np.int8))

    bbox_xywh = (
        int(coord_min[0]), int(coord_min[1]),
        int(coord_span[0]) + 1, int(coord_span[1]) + 1,
    )
    meta = CropMeta(
        k=nC, r=nQ, bbox_xywh=bbox_xywh, side="DEM",
        d=d, p=p, kappa=N, seed=0,
        bucket_id=-1, pad_nodes=N_max, pad_edges=E_max, pad_seq=S_max,
    )

    return PackedCrop(
        x_nodes=x_nodes, node_mask=node_mask, node_type=node_type_t,
        edge_index=ei_t, edge_attr=ea_t, edge_mask=em_t,
        seq_idx=si_t, seq_mask=sm_t,
        g_token=g_token, y_bits=y_t, s_sub=s_t,
        meta=meta, H_sub=H_sub,
    )


class OnlineSurfaceDataset(IterableDataset):
    """DEM-based iterable dataset for circuit-level surface code training.

    Uses Stim to generate detection events, PyMatching as teacher, and packs
    3-D (x, y, t) merged-edge bipartite subgraphs for MGHDv2.
    """

    def __init__(self, args, p_epoch, epoch, shots_total, teacher_mix, rank=0):
        self.args = args
        if isinstance(p_epoch, list):
            self.p_list = p_epoch
            self.p_epoch = p_epoch[0]
        else:
            self.p_list = None
            self.p_epoch = p_epoch
        self.epoch = epoch
        self.shots_to_do = shots_total
        self.teacher_mix = teacher_mix
        self.rank = rank
        self.noise_model = str(getattr(args, "noise_model", "SI1000"))

        # Distance curriculum
        self.distances = [self.args.distance]
        if getattr(self.args, "distance_curriculum", None):
            try:
                parsed = [int(x) for x in self.args.distance_curriculum.split(",") if x.strip()]
                if parsed:
                    self.distances = sorted(set(parsed))
            except ValueError:
                pass

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id = worker_info.id
            num_workers = max(1, worker_info.num_workers)

        base = self.shots_to_do // num_workers
        remainder = self.shots_to_do % num_workers
        shots_for_worker = base + (1 if worker_id < remainder else 0)
        if shots_for_worker <= 0:
            return

        seed = self.args.seed + self.epoch * 10000 + self.rank * 100 + worker_id
        rng = np.random.default_rng(seed)

        # Build balanced distance schedule
        distance_schedule: list[int] = []
        if len(self.distances) == 1:
            distance_schedule = [int(self.distances[0])] * shots_for_worker
        else:
            reps, rem = divmod(shots_for_worker, len(self.distances))
            for dv in self.distances:
                distance_schedule.extend([int(dv)] * reps)
            if rem > 0:
                tail = list(self.distances)
                rng.shuffle(tail)
                distance_schedule.extend(tail[:rem])
            rng.shuffle(distance_schedule)

        from scipy.sparse import csr_matrix as _csr
        from scipy.sparse.csgraph import connected_components as _cc

        edge_prune = float(getattr(self.args, "edge_prune_thresh", 1e-3))

        for dist in distance_schedule:
            p_shot = float(rng.choice(self.p_list)) if self.p_list else float(self.p_epoch)
            n_rounds = int(getattr(self.args, "rounds", 0)) or dist

            info = _get_dem_info(dist, n_rounds, p_shot, self.noise_model, edge_prune)

            # ── Sample one shot ───────────────────────────────────────
            det_bits_all, obs_all = info["sampler"].sample(
                shots=1, separate_observables=True
            )
            det_bits = det_bits_all[0].astype(np.uint8)

            # ── Teacher: PyMatching → per-edge-node labels ────────────
            y_bits = _teacher_labels_from_matching(
                det_bits, info["matching"],
                info["det_pair_to_edge"], info["n_edges"],
            )

            # ── Decompose into active components ──────────────────────
            component_scope = str(getattr(self.args, "component_scope", "active")).lower()

            if component_scope == "full":
                pack = pack_dem_cluster(
                    H_edge=info["H_edge"],
                    det_coords=info["det_coords"],
                    edge_coords=info["edge_coords"],
                    det_bits=det_bits,
                    y_bits_edge=y_bits,
                    d=dist, rounds=n_rounds, p=p_shot,
                    N_max=self.args.N_max,
                    E_max=self.args.E_max,
                    S_max=self.args.S_max,
                )
                if pack is not None:
                    yield pack
                continue

            # Active decomposition on the detector adjacency graph
            fired = np.where(det_bits > 0)[0]
            if len(fired) == 0:
                continue

            det_adj = info["det_adj"]
            H_edge_full = info["H_edge"]

            # 1-hop expansion on detector graph
            active_set = set(fired.tolist())
            for f in fired:
                active_set.update(np.where(det_adj[f] > 0)[0].tolist())
            active_dets = np.array(sorted(active_set), dtype=np.intp)

            # Connected components among active detectors
            adj_sub = det_adj[np.ix_(active_dets, active_dets)]
            n_comp, labels = _cc(_csr(adj_sub.astype(np.float32)), directed=False)

            for comp_id in range(n_comp):
                comp_det_local = np.where(labels == comp_id)[0]
                comp_dets = active_dets[comp_det_local]
                comp_det_set = set(comp_dets.tolist())

                # TIGHT: only edge-nodes where ALL connected detectors
                # are within this component (avoids pulling in distant edges)
                H_comp = H_edge_full[comp_dets, :]
                candidate_edges = np.where(H_comp.sum(axis=0) > 0)[0]
                tight_edges = []
                for eidx in candidate_edges:
                    all_dets_of_edge = set(np.where(H_edge_full[:, eidx] > 0)[0].tolist())
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
                    y_bits_edge=y_bits,
                    edge_indices=comp_edges,
                    det_indices=comp_dets,
                    d=dist, rounds=n_rounds, p=p_shot,
                    N_max=self.args.N_max,
                    E_max=self.args.E_max,
                    S_max=self.args.S_max,
                )
                if pack is not None:
                    yield pack


def main():
    """Unified CLI entrypoint supporting offline and online modes.

    Delegates to ``train_inprocess``, which accepts both offline (``--data-root``)
    and online (``--online``) training paths with TAD/RL/erasure options.
    """
    from types import SimpleNamespace

    # Passing a dummy namespace will make train_inprocess parse sys.argv
    train_inprocess(SimpleNamespace())


if __name__ == "__main__":
    main()
