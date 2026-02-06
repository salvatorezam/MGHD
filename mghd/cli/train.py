"""MGHD trainer: offline (crops) and online (CUDA‑Q) modes.

Defines a small offline dataset loader for crop shards and an online training
loop that samples syndromes on the fly, routes teacher supervision, and packs
subgraphs into MGHDv2 input tensors. Detector bits are handled in canonical
Z→X order throughout (matching Stim/DEM). CUDA is initialized only inside
``main``.
"""

# NOTE: Initialize CUDA only in main(). This file defines dataset, model, loop.
from __future__ import annotations

import argparse
import glob
import json
import os
import random
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
from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler

from mghd.codes.qpu_profile import load_qpu_profile
from mghd.core.core import MGHDv2, PackedCrop, pack_cluster
from mghd.decoders.lsd import clustered as cc  # uses projector functions
from mghd.decoders.lsd_teacher import LSDTeacher
from mghd.decoders.mwpf_teacher import MWPFTeacher
from mghd.decoders.mwpm_ctx import MWPMatchingContext
from mghd.decoders.nvqldpc_teacher import NvQldpcTeacher
from mghd.qpu.adapters.surface_sampler import sample_round, split_components_for_side
from mghd.tad import context as tad_context
from mghd.tad import weighting as tad_weighting

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

_TEACHER_POOL: ThreadPoolExecutor | None = None


class CropShardDataset(Dataset):
    """Index all ``.npz`` shards under ``root`` and expose packed crops."""

    def __init__(self, root: str, curriculum: str = "stage1", max_shards: int | None = None):
        self.files = sorted(glob.glob(os.path.join(root, "**/*.npz"), recursive=True))
        if max_shards is not None:
            self.files = self.files[:max_shards]
        # Load all items from all shards
        self.items = []
        for f in self.files:
            arr = np.load(f, allow_pickle=True)
            packed_array = arr["packed"]
            # Handle both single item and array of items
            if packed_array.shape == ():
                # Single item stored as scalar
                self.items.append((f, packed_array.item()))
            else:
                # Array of items
                for it in packed_array:
                    self.items.append((f, it))
        print(f"Loaded {len(self.items)} crop items from {len(self.files)} shards")

    def __len__(self):
        """Return number of packed items across all shards."""
        return len(self.items)

    def __getitem__(self, idx):
        """Load one PackedCrop (as torch tensors) and attach teacher metadata."""
        _, item = self.items[idx]

        def to_tensor(x, dtype=None):
            if isinstance(x, torch.Tensor):
                return torch.as_tensor(x, dtype=dtype)
            return torch.as_tensor(x, dtype=dtype)

        seq_idx_t = to_tensor(item["seq_idx"], torch.long)
        s_sub_raw = item.get("s_sub", None)
        if s_sub_raw is None:
            s_sub_t = torch.zeros_like(seq_idx_t, dtype=torch.int8)
        else:
            s_sub_t = to_tensor(s_sub_raw, torch.int8)

        packed = PackedCrop(
            x_nodes=to_tensor(item["x_nodes"], torch.float32),
            node_mask=to_tensor(item["node_mask"], torch.bool),
            node_type=to_tensor(item["node_type"], torch.int8),
            edge_index=to_tensor(item["edge_index"], torch.long),
            edge_attr=to_tensor(item["edge_attr"], torch.float32),
            edge_mask=to_tensor(item["edge_mask"], torch.bool),
            seq_idx=seq_idx_t,
            seq_mask=to_tensor(item["seq_mask"], torch.bool),
            g_token=to_tensor(item["g_token"], torch.float32),
            y_bits=to_tensor(item["y_bits"], torch.int8),
            s_sub=s_sub_t,
            meta=item["meta"],
            H_sub=item.get("H_sub", None),
            idx_data_local=item.get("idx_data_local", None),
            idx_check_local=item.get("idx_check_local", None),
        )
        # attach teacher meta for weighting/logging
        packed.teacher = item.get("teacher", "mwpf")
        packed.teacher_weight = int(item.get("teacher_weight", 0))
        packed.teacher_valid = bool(item.get("teacher_valid", True))
        packed.teacher_matched_local_ml = bool(item.get("teacher_matched_local_ml", False))
        return packed


def make_bucket_sampler(ds: CropShardDataset, *, stage="stage1", seed=0):
    """
    Bucket by (r, kappa=size) and coarse p to ensure hard cases are seen.
    """
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for i, (_, item) in enumerate(ds.items):
        meta = item["meta"]
        r = int(meta.get("r", 0))
        kappa = int(meta.get("kappa", meta.get("k", 0)))
        # coarse p bin
        p = float(meta.get("p", 0.005))
        pbin = 0 if p < 0.003 else (1 if p < 0.01 else 2)
        buckets[(min(r, 8), min(kappa, 32), pbin)].append(i)
    order = []
    # Interleave buckets to avoid collapse
    keys = sorted(buckets.keys())
    heads = dict.fromkeys(keys, 0)
    while True:
        progressed = False
        for k in keys:
            lst = buckets[k]
            h = heads[k]
            if h < len(lst):
                order.append(lst[h])
                heads[k] = h + 1
                progressed = True
        if not progressed:
            break
    rng.shuffle(order)

    class _S(Sampler):
        def __iter__(self):
            return iter(order)

        def __len__(self):
            return len(order)

    return _S()


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
    parser.add_argument("--online", action="store_true", help="Enable on-the-fly CUDA-Q sampling")
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
    parser.add_argument("--qpu-profile", type=str, default=None)
    parser.add_argument(
        "--context-source", type=str, default="none", choices=["none", "qiskit", "cirq", "cudaq"]
    )
    parser.add_argument("--shots-per-epoch", type=int, default=256)
    parser.add_argument(
        "--sampler",
        type=str,
        default="cudaq",
        choices=["cudaq", "stim", "synthetic"],
        help=(
            "Sampler backend for online mode: "
            "'cudaq' uses the CUDA-Q adapter (or falls back), "
            "'synthetic' uses a fast phenomenological sampler (parity-check syndromes), "
            "'stim' is circuit-level detector sampling and is NOT compatible with MGHDv2's per-qubit "
            "supervision pipeline (use a DEM/fault-space or observable-training path instead)."
        ),
    )
    parser.add_argument(
        "--noise-model",
        type=str,
        default="auto",
        choices=["auto", "garnet", "generic_cl"],
        help=(
            "Noise model family for sampler=cudaq. "
            "'auto' selects garnet when --qpu-profile is set, else generic_cl."
        ),
    )
    parser.add_argument(
        "--generic-p1q",
        type=float,
        default=0.0015,
        help="Base 1Q depolarizing probability for generic_cl noise.",
    )
    parser.add_argument(
        "--generic-p2q",
        type=float,
        default=0.01,
        help="Base 2Q depolarizing probability for generic_cl noise.",
    )
    parser.add_argument(
        "--generic-pidle",
        type=float,
        default=0.0008,
        help="Base idle error probability per idle_ref_ns for generic_cl noise.",
    )
    parser.add_argument(
        "--generic-pmeas0",
        type=float,
        default=0.02,
        help="Base readout assignment error P(meas=1|state=0) for generic_cl noise.",
    )
    parser.add_argument(
        "--generic-pmeas1",
        type=float,
        default=0.02,
        help="Base readout assignment error P(meas=0|state=1) for generic_cl noise.",
    )
    parser.add_argument(
        "--generic-phook",
        type=float,
        default=0.0,
        help="Optional correlated hook-error probability per CZ for generic_cl noise.",
    )
    parser.add_argument(
        "--generic-pcrosstalk",
        type=float,
        default=0.0,
        help="Optional spectator crosstalk probability per active layer for generic_cl noise.",
    )
    parser.add_argument(
        "--generic-idle-ref-ns",
        type=float,
        default=20.0,
        help="Reference layer duration for generic idle scaling.",
    )
    parser.add_argument(
        "--phenomenological",
        action="store_true",
        help="Shortcut for requesting the fast phenomenological sampler (sets sampler=synthetic)",
    )
    parser.add_argument(
        "--erasure-frac",
        type=float,
        default=0.0,
        help="Optional erasure injection fraction for online sampling",
    )
    parser.add_argument("--teacher-mix", type=str, default="mwpf=1.0,mwpm=0.0,lsd=0.0")
    parser.add_argument(
        "--online-rl",
        action="store_true",
        help="Enable LinTS scaling of TAD overrides in online mode",
    )
    parser.add_argument("--profile", type=str, default="S")
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--lr", type=float, default=5.952925899948483e-05)
    parser.add_argument("--wd", type=float, default=6.65850238574699e-05)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--parity-lambda", type=float, default=0.03)
    parser.add_argument("--projection-aware", type=int, default=1)
    parser.add_argument(
        "--online-fast",
        action="store_true",
        help=(
            "Speed-oriented online settings. Disables expensive projection/parity auxiliaries "
            "and enables periodic progress heartbeats."
        ),
    )
    parser.add_argument("--label-smoothing", type=float, default=0.13831652882929857)
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
    # Optional post-run teacher comparison report (writes teacher_eval.txt)
    parser.add_argument("--post-eval", action="store_true")
    parser.add_argument("--post-eval-sampler", type=str, default="stim")
    parser.add_argument("--post-eval-shots-per-batch", type=int, default=16)
    parser.add_argument("--post-eval-batches", type=int, default=2)
    parser.add_argument(
        "--teacher-workers",
        type=int,
        default=4,
        help="Thread pool workers used to prefetch teacher labels",
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

    # Handle sampler selection and environment variable
    sampler_choice = str(getattr(args, "sampler", "cudaq"))
    if getattr(args, "phenomenological", False):
        sampler_choice = "synthetic"
        args.sampler = "synthetic"

    generic_env_keys = (
        "MGHD_GENERIC_P1Q",
        "MGHD_GENERIC_P2Q",
        "MGHD_GENERIC_PIDLE",
        "MGHD_GENERIC_PMEAS0",
        "MGHD_GENERIC_PMEAS1",
        "MGHD_GENERIC_PHOOK",
        "MGHD_GENERIC_PCROSSTALK",
        "MGHD_GENERIC_IDLE_REF_NS",
    )

    # Set environment variables for sampling backend
    # - "stim": Use Stim circuit-level noise (MGHD_SAMPLER=stim)
    # - "synthetic": Use fast phenomenological noise (MGHD_SYNTHETIC=1)
    # - "cudaq": Use CUDA-Q backend (default)
    if sampler_choice == "stim":
        os.environ["MGHD_SAMPLER"] = "stim"
        os.environ.pop("MGHD_SYNTHETIC", None)
        os.environ.pop("MGHD_NOISE_MODEL", None)
        for key in generic_env_keys:
            os.environ.pop(key, None)
    elif sampler_choice == "synthetic":
        os.environ["MGHD_SYNTHETIC"] = "1"
        os.environ.pop("MGHD_SAMPLER", None)
        os.environ.pop("MGHD_NOISE_MODEL", None)
        for key in generic_env_keys:
            os.environ.pop(key, None)
    elif sampler_choice == "cudaq":
        os.environ.pop("MGHD_SYNTHETIC", None)
        os.environ.pop("MGHD_SAMPLER", None)
        noise_model = str(getattr(args, "noise_model", "auto")).lower()
        if noise_model == "auto":
            noise_model = "garnet" if getattr(args, "qpu_profile", None) else "generic_cl"
        os.environ["MGHD_NOISE_MODEL"] = noise_model
        if noise_model == "generic_cl":
            os.environ["MGHD_GENERIC_P1Q"] = str(float(getattr(args, "generic_p1q", 0.0015)))
            os.environ["MGHD_GENERIC_P2Q"] = str(float(getattr(args, "generic_p2q", 0.01)))
            os.environ["MGHD_GENERIC_PIDLE"] = str(float(getattr(args, "generic_pidle", 0.0008)))
            os.environ["MGHD_GENERIC_PMEAS0"] = str(float(getattr(args, "generic_pmeas0", 0.02)))
            os.environ["MGHD_GENERIC_PMEAS1"] = str(float(getattr(args, "generic_pmeas1", 0.02)))
            os.environ["MGHD_GENERIC_PHOOK"] = str(float(getattr(args, "generic_phook", 0.0)))
            os.environ["MGHD_GENERIC_PCROSSTALK"] = str(
                float(getattr(args, "generic_pcrosstalk", 0.0))
            )
            os.environ["MGHD_GENERIC_IDLE_REF_NS"] = str(
                float(getattr(args, "generic_idle_ref_ns", 20.0))
            )
        else:
            for key in generic_env_keys:
                os.environ.pop(key, None)
        if rank == 0:
            print(f"CUDA-Q noise model: {noise_model}")

    if bool(getattr(args, "online", False)) and sampler_choice == "stim":
        raise ValueError(
            "Online training with `--sampler stim` is not supported for MGHDv2 per-qubit training. "
            "Stim's circuit-level sampler produces space-time detector events; "
            "the current MGHDv2 training loop expects parity-check syndromes (synZ/synX) "
            "and per-qubit correction labels from MWPF/LSD/MWPM. "
            "Use `--sampler synthetic` for phenomenological training, or switch to a circuit-level "
            "DEM/fault-space or observable-based training pipeline."
        )

    # Auto-scale pad limits for large distances if not provided
    if not hasattr(args, "N_max"):
        d_param = int(getattr(args, "distance", 3))
        # Base 512 is good for d<=15. For d=31, we need ~2000.
        # We use a safe scaling: N_max ~ 3*d^2, E_max ~ 12*d^2
        args.N_max = max(512, int(3.0 * d_param**2))
        args.E_max = max(4096, int(12.0 * d_param**2))
        args.S_max = max(512, int(2.0 * d_param**2))
        if rank == 0:
            print(f"Auto-scaled pad limits for d={d_param}: N_max={args.N_max}, E_max={args.E_max}")

    # Default structural hyperparameters (can be overridden via JSON/CLI namespace)
    defaults = {
        "d_model": 192,
        "d_state": 80,
        "n_iters": 8,
        "msg_net_dropout_p": 0.0,
        "gru_dropout_p": 0.0,
        "se_reduction": 4,
        "node_feat_dim": 9,
        "edge_feat_dim": 3,
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
        args.node_feat_dim = node_feat_dim
        args.edge_feat_dim = edge_feat_dim

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
                model.module.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)

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
                "[online-fast] Enabled: projection_aware=0 parity_lambda=0 "
                f"progress_prints={args.progress_prints} progress_seconds={args.progress_seconds}"
            )
    if not use_online:
        ds = CropShardDataset(args.data_root)
        sampler = make_bucket_sampler(ds, stage="stage1", seed=args.seed)
        loader = DataLoader(
            ds,
            batch_size=args.batch,
            sampler=sampler,
            collate_fn=lambda batch: batch,
        )

    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)
    # Persist run metadata for reproducibility
    try:
        run_meta = {
            "family": args.family,
            "distance": int(args.distance),
            "online": use_online,
            "p": float(getattr(args, "p", 0.0)),
            "p_curriculum": [
                float(x) for x in str(getattr(args, "p_curriculum", "")).split(",") if x.strip()
            ]
            if getattr(args, "p_curriculum", None)
            else None,
            "epochs_per_p": int(getattr(args, "epochs_per_p", 1)),
            "teacher_mix": getattr(args, "teacher_mix", None),
            "noise_model": os.environ.get("MGHD_NOISE_MODEL", None),
            "generic_noise": {
                "p1q": os.environ.get("MGHD_GENERIC_P1Q", None),
                "p2q": os.environ.get("MGHD_GENERIC_P2Q", None),
                "pidle": os.environ.get("MGHD_GENERIC_PIDLE", None),
                "pmeas0": os.environ.get("MGHD_GENERIC_PMEAS0", None),
                "pmeas1": os.environ.get("MGHD_GENERIC_PMEAS1", None),
                "phook": os.environ.get("MGHD_GENERIC_PHOOK", None),
                "pcrosstalk": os.environ.get("MGHD_GENERIC_PCROSSTALK", None),
                "idle_ref_ns": os.environ.get("MGHD_GENERIC_IDLE_REF_NS", None),
            },
            "qpu_profile": getattr(args, "qpu_profile", None),
            "context_source": getattr(args, "context_source", None),
            "erasure_frac": float(getattr(args, "erasure_frac", 0.0)),
            "shots_per_epoch": int(getattr(args, "shots_per_epoch", 0)),
            "epochs": int(args.epochs),
            "batch": int(args.batch),
            "seed": int(args.seed),
            "online_fast": bool(getattr(args, "online_fast", False)),
            "progress_seconds": float(getattr(args, "progress_seconds", 0.0)),
        }
        (save_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))
    except Exception:
        pass

    best_loss = float("inf")
    history: list[dict[str, Any]] = []
    last_improve_epoch = 0

    def _build_tad_for_code(code_obj):
        """Build transpilation‑aware context and per‑qubit LLR overrides.

        Uses the provided QPU profile JSON (``--qpu-profile``) and the
        requested ``--context-source`` to extract a schedule IR, construct
        TAD weighting maps, and return:
        - qpu_prof: loaded profile object
        - ctx_vec: fixed global context vector for this epoch (or None)
        - llr_overrides: per‑qubit log‑likelihood adjustments (or None)
        """
        qpu_prof = None
        ctx_vec = None
        llr_overrides = None
        if args.qpu_profile and args.context_source != "none":
            try:
                qpu_prof = load_qpu_profile(args.qpu_profile)
            except Exception:
                qpu_prof = None
        if qpu_prof is not None:
            # Attempt schedule IR extraction via adapters if native circuit present
            try:
                native = None
                for attr in (
                    "native_circuit",
                    "reference_circuit",
                    "circuit",
                    "qc",
                    "quantum_circuit",
                ):
                    if hasattr(code_obj, attr):
                        native = getattr(code_obj, attr)
                        if native is not None:
                            break
                schedule_ir = []
                if native is not None:
                    import sys as _sys

                    if args.context_source == "qiskit" and "qiskit" in _sys.modules:
                        from mghd.qpu.adapters import qiskit_adapter

                        schedule_ir = qiskit_adapter.to_schedule_ir(native)
                    elif args.context_source == "cirq" and "cirq" in _sys.modules:
                        from mghd.qpu.adapters import cirq_adapter  # type: ignore

                        schedule_ir = cirq_adapter.to_schedule_ir(native)
                    elif args.context_source == "cudaq" and "cudaq" in _sys.modules:
                        from mghd.qpu.adapters import cudaq_adapter  # type: ignore

                        schedule_ir = cudaq_adapter.to_schedule_ir(native)
                n_qubits = getattr(code_obj, "n", None) or int(code_obj.Hx.shape[1])
                maps = tad_weighting.schedule_to_weight_maps(schedule_ir, qpu_prof, n_qubits)
                feats = tad_weighting.feature_vector(schedule_ir)
                gate_vocab = {g: i for i, g in enumerate(sorted(feats.get("gate_hist", {}).keys()))}
                ctx_vec = tad_context.context_vector(feats, gate_vocab)
                # Build per-qubit LLR override
                import numpy as _np

                llr = _np.zeros(int(n_qubits), dtype=_np.float32)
                for layer in (maps.get("w_qubit", {}) or {}).values():
                    for q, w in layer.items():
                        idx = int(q)
                        if 0 <= idx < llr.size:
                            llr[idx] += float(w)
                llr_overrides = llr if llr.size else None
            except Exception:
                ctx_vec = None
                llr_overrides = None
        return qpu_prof, ctx_vec, llr_overrides

    def _parse_teacher_mix(spec: str):
        """Parse a comma‑separated teacher weight spec into a dict.

        Example: "mwpf=1.0,mwpm=0.5,lsd=0.25" → {"mwpf":1.0,"mwpm":0.5,"lsd":0.25}
        Missing teachers default to 0.0; if all are zero, defaults to MWPF=1.0.
        """
        weights = {
            "mwpf": 0.0,
            "mwpm": 0.0,
            "lsd": 0.0,
            "nvqldpc": 0.0,
            "oracle": 0.0,
        }
        if not spec:
            weights["mwpf"] = 1.0
            return weights
        for chunk in spec.split(","):
            if "=" not in chunk:
                continue
            name, value = chunk.split("=", 1)
            try:
                weights[name.strip().lower()] = float(value)
            except ValueError:
                continue
        for k in weights:
            weights[k] = max(0.0, weights[k])
        if sum(weights.values()) == 0.0:
            weights["mwpf"] = 1.0
        return weights

    bandit = None
    prev_epoch_loss = None
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
        if use_online:
            # Select p for this epoch (curriculum if provided)
            if p_list:
                step = max(1, int(getattr(args, "epochs_per_p", 1)))
                idx = min((epoch - 1) // step, len(p_list) - 1)
                p_epoch = float(p_list[idx])
            else:
                p_epoch = float(getattr(args, "p", 0.005))

            # =====================================================================
            # Per-qubit supervision path (cudaq/synthetic samplers)
            # =====================================================================
            # Teacher setup per epoch
            family = args.family
            # Build code object
            from mghd.codes.registry import get_code

            code = get_code(family, distance=args.distance)
            # Initialize teachers once per epoch
            teacher_mix = _parse_teacher_mix(
                getattr(args, "teacher_mix", "mwpf=1.0,mwpm=0.0,lsd=0.0")
            )
            label_teacher_weight = sum(
                teacher_mix.get(k, 0.0) for k in ("mwpf", "mwpm", "lsd", "nvqldpc", "oracle")
            )
            if label_teacher_weight <= 0.0:
                raise ValueError(
                    "Invalid `--teacher-mix`: no label-producing teachers selected. "
                    "This MGHDv2 training loop requires per-qubit labels from one of "
                    "`mwpf`, `mwpm`, `lsd`, `nvqldpc`, or `oracle`."
                )

            # TAD context/overrides
            qpu_prof, ctx_vec, llr_overrides = _build_tad_for_code(code)
            # Initialize bandit once if requested and context available
            if args.online_rl and ctx_vec is not None and bandit is None:
                try:
                    from mghd.tad.rl.lin_ts import LinTSBandit

                    bandit = LinTSBandit(d=ctx_vec.size, prior_var=5.0, noise_var=0.5)
                except Exception:
                    bandit = None

            # Setup DataLoader for parallel generation
            workers = max(0, int(getattr(args, "workers", 0)))
            shots_target = int(getattr(args, "shots_per_epoch", args.batch))

            # Divide shots among ranks
            if is_distributed:
                shots_target = shots_target // world_size

            # Pass per-rank shot budget; dataset will split this across workers to avoid duplication
            dataset = OnlineSurfaceDataset(
                args, p_epoch, epoch, shots_target, teacher_mix, ctx_vec, llr_overrides, rank=rank
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
                            h = packed.H_sub[i] if isinstance(packed.H_sub, list) else packed.H_sub

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
                                target_full = yb.detach().cpu().numpy().clip(0, 1).astype(np.uint8)
                                target_data = target_full[mask_data]
                            proj_target = torch.from_numpy(target_data).to(device)
                            proj_pred = torch.from_numpy(proj_bits.astype(np.int64)).to(device)
                            raw_bits = (torch.sigmoid(l[:, 1] - l[:, 0])[data_mask] > 0.5).long()
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
                steps_done += 1

                if prog_stride and (steps_done % prog_stride == 0) and rank == 0:
                    prog = {
                        "epoch": epoch,
                        "step": int(steps_done),
                        "avg": float(total_loss / max(n_items, 1)),
                        "secs": float(time.time() - t0),
                    }
                    if p_epoch is not None:
                        prog["p"] = float(p_epoch)
                    print(json.dumps(prog, separators=(",", ":")), flush=True)
                heartbeat_s = float(getattr(args, "progress_seconds", 0.0))
                now = time.time()
                if rank == 0 and heartbeat_s > 0.0 and (now - last_heartbeat) >= heartbeat_s:
                    hb = {
                        "heartbeat": True,
                        "epoch": int(epoch),
                        "step": int(steps_done),
                        "avg": float(total_loss / max(n_items, 1)),
                        "secs": float(now - t0),
                        "items": int(n_items),
                    }
                    if p_epoch is not None:
                        hb["p"] = float(p_epoch)
                    print(json.dumps(hb, separators=(",", ":")), flush=True)
                    last_heartbeat = now
        else:
            steps_done = 0
            steps_total = len(loader) if hasattr(loader, "__len__") else 0
            prog_stride = (
                max(1, steps_total // int(getattr(args, "progress_prints", 1)))
                if steps_total and int(getattr(args, "progress_prints", 1)) > 1
                else 0
            )
            global _TEACHER_POOL
            if _TEACHER_POOL is None:
                workers = max(1, int(getattr(args, "teacher_workers", 4)))
                _TEACHER_POOL = ThreadPoolExecutor(max_workers=workers)

            def _teacher_prefetch(packed: PackedCrop) -> dict:
                return {
                    "valid": bool(getattr(packed, "teacher_valid", True)),
                    "matched_local_ml": bool(getattr(packed, "teacher_matched_local_ml", False)),
                }

            futures: list = []
            teacher_outs: list = []
            for batch in loader:
                if not batch:
                    continue
                futures.clear()
                moved = []
                for packed in batch:
                    futures.append(_TEACHER_POOL.submit(_teacher_prefetch, packed))
                    _validate_packed_contract(packed, expected_node_dim, expected_edge_dim)
                    moved.append(move_to(packed, device))
                teacher_outs = [f.result() for f in futures]
                if args.noise_injection > 0.0:
                    std = float(args.noise_injection)
                    for packed in moved:
                        packed.x_nodes = packed.x_nodes + torch.randn_like(packed.x_nodes) * std

                batch_loss = torch.zeros((), device=device)
                for packed, teach in zip(moved, teacher_outs):
                    with _autocast():
                        logits, node_mask = model(packed=packed)
                        logits = logits.squeeze(0)
                        node_mask = node_mask.squeeze(0)

                        hard = (0.5 if teach.get("valid", True) else 1.5) + (
                            0.5 if teach.get("matched_local_ml", False) else 1.0
                        )
                        sample_weight = torch.tensor(hard, dtype=torch.float32, device=device)

                        if bool(getattr(args, "use_focal", False)):
                            loss_bce = focal_binary_head_loss(
                                logits,
                                node_mask,
                                packed.node_type,
                                packed.y_bits,
                                alpha=float(getattr(args, "focal_alpha", 0.25)),
                                gamma=float(getattr(args, "focal_gamma", 2.0)),
                                sample_weight=sample_weight,
                            )
                        else:
                            loss_bce = bce_binary_head_loss(
                                logits,
                                node_mask,
                                packed.node_type,
                                packed.y_bits,
                                sample_weight=sample_weight,
                                label_smoothing=args.label_smoothing,
                            )

                        loss_par = args.parity_lambda * parity_auxiliary_loss(
                            logits,
                            node_mask,
                            packed.node_type,
                            H_sub=packed.H_sub,
                            s_sub=packed.s_sub,
                        )

                        loss_proj = torch.tensor(0.0, device=device)
                        if args.projection_aware:
                            data_mask = (packed.node_type == 0) & node_mask
                            proj_bits = projection_aware_logits_to_bits(
                                logits,
                                projector_kwargs={
                                    "H_sub": packed.H_sub,
                                    "side": getattr(packed.meta, "side", "Z"),
                                    "s_sub": packed.s_sub,
                                },
                                data_mask=data_mask,
                            )
                            with torch.no_grad():
                                mask_data = data_mask.detach().cpu().numpy()
                                target_full = (
                                    packed.y_bits.detach().cpu().numpy().clip(0, 1).astype(np.uint8)
                                )
                                target_data = target_full[mask_data]

                            proj_target = torch.from_numpy(target_data).to(device)
                            proj_pred = torch.from_numpy(proj_bits.astype(np.int64)).to(device)

                            raw_bits = (
                                torch.sigmoid(logits[:, 1] - logits[:, 0])[data_mask] > 0.5
                            ).long()
                            loss_proj = 0.5 * F.l1_loss(
                                proj_pred.float(), proj_target.float()
                            ) + 0.2 * F.l1_loss(proj_pred.float(), raw_bits.float())

                        sample_loss = loss_bce + loss_par + 0.5 * loss_proj
                        batch_loss = batch_loss + sample_loss

                batch_size = len(moved)
                batch_loss = batch_loss / batch_size
                _backward_and_step(batch_loss)

                total_loss += batch_loss.detach().item() * batch_size
                n_items += batch_size
                steps_done += 1
                if prog_stride and (steps_done % prog_stride == 0):
                    prog = {
                        "epoch": epoch,
                        "step": int(steps_done),
                        "steps": int(steps_total),
                        "avg": float(total_loss / max(n_items, 1)),
                        "secs": float(time.time() - t0),
                    }
                    print(json.dumps(prog, separators=(",", ":")), flush=True)
                heartbeat_s = float(getattr(args, "progress_seconds", 0.0))
                now = time.time()
                if rank == 0 and heartbeat_s > 0.0 and (now - last_heartbeat) >= heartbeat_s:
                    hb = {
                        "heartbeat": True,
                        "epoch": int(epoch),
                        "step": int(steps_done),
                        "steps": int(steps_total),
                        "avg": float(total_loss / max(n_items, 1)),
                        "secs": float(now - t0),
                        "items": int(n_items),
                    }
                    print(json.dumps(hb, separators=(",", ":")), flush=True)
                    last_heartbeat = now

        sched.step()
        dt = time.time() - t0
        avg = total_loss / max(n_items, 1)
        history.append({"epoch": epoch, "loss": avg, "count": n_items, "secs": dt})
        # Bandit posterior update with simple reward: 1.0 if loss decreased, else 0.0
        if bandit is not None and prev_epoch_loss is not None and ctx_vec is not None:
            reward = 1.0 if avg < prev_epoch_loss else 0.0
            bandit.update(ctx_vec, reward)
        prev_epoch_loss = avg

        if rank == 0:
            torch.save(
                {
                    "model": model.module.state_dict() if is_distributed else model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "epoch": epoch,
                    "loss": avg,
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
                    },
                    save_dir / "best.pt",
                )

            log_obj = {"epoch": epoch, "loss": avg, "secs": dt}
            if use_online:
                log_obj["p"] = float(p_epoch if "p_epoch" in locals() else getattr(args, "p", 0.0))
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

    # Optional teacher comparison report
    if hasattr(args, "post_eval") and bool(getattr(args, "post_eval", False)) and rank == 0:
        try:
            import subprocess, sys as _sys, os as _os

            shots = int(getattr(args, "post_eval_shots_per_batch", 16))
            batches = int(getattr(args, "post_eval_batches", 2))
            sampler = str(getattr(args, "post_eval_sampler", "stim"))
            cmd = [
                _sys.executable,
                "-m",
                "mghd.tools.teacher_eval",
                "--families",
                str(args.family),
                "--distances",
                str(args.distance),
                "--sampler",
                sampler,
                "--shots-per-batch",
                str(shots),
                "--batches",
                str(batches),
            ]
            env = _os.environ.copy()
            env.setdefault("PYTHONPATH", _os.getcwd())
            cp = subprocess.run(cmd, capture_output=True, text=True, env=env)
            (save_dir / "teacher_eval.txt").write_text(
                cp.stdout + ("\n--- STDERR ---\n" + cp.stderr if cp.stderr else "")
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


def sanity_train(
    crop_root: str,
    epochs: int = 2,
    batch_size: int = 2,
    lr: float = 1e-4,
    hparams: str | None = None,
):
    """Quick sanity training over a small offline crop dir (for tests/demos)."""
    import os
    import tempfile
    from types import SimpleNamespace

    ns = SimpleNamespace()
    ns.data_root = crop_root  # Changed from crop_root to data_root to match arg parser
    ns.epochs = epochs
    ns.batch = batch_size  # Changed from batch_size to batch to match arg parser
    ns.lr = lr
    ns.wd = 1e-5  # Changed from weight_decay to wd
    ns.profile = "S"  # Default profile
    ns.ema = 0.999
    ns.parity_lambda = 0.03
    ns.projection_aware = 1
    ns.seed = 42
    ns.label_smoothing = 0.1
    ns.use_focal = False
    ns.focal_alpha = 0.25
    ns.focal_gamma = 2.0
    ns.noise_injection = 0.0
    ns.grad_clip = 1.0
    ns.node_feat_dim = 9
    ns.edge_feat_dim = 3
    ns.hparams = hparams

    # Create temporary directory for save path
    temp_dir = tempfile.mkdtemp()
    ns.save = os.path.join(temp_dir, "sanity_test")

    print(f"Running sanity training with {epochs} epochs, batch_size={batch_size}, lr={lr}")
    print(f"Temporary save path: {ns.save}")

    try:
        result_path = train_inprocess(ns)
        print(f"Training completed, result: {result_path}")

        # Return the trained model for inspection
        from mghd.core.core import MGHDv2

        model = MGHDv2(profile=ns.profile)
        if torch.cuda.is_available():
            model = model.cuda()
        return model
    finally:
        # Clean up temp directory
        import shutil

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


# Global cache for workers to avoid rebuilding teachers every epoch/iteration
_WORKER_TEACHERS_CACHE = {}


class OnlineSurfaceDataset(IterableDataset):
    """
    Iterable dataset for online surface code simulation.
    Generates crops on-the-fly using multiple workers.
    """

    def __init__(
        self,
        args,
        p_epoch,
        epoch,
        shots_total,
        teacher_mix,
        ctx_vec=None,
        llr_overrides=None,
        rank=0,
    ):
        self.args = args
        self.p_epoch = p_epoch
        self.epoch = epoch
        # shots_total refers to the per-rank budget; workers will partition this in __iter__
        self.shots_to_do = shots_total
        self.teacher_mix = teacher_mix
        self.ctx_vec = ctx_vec
        self.llr_overrides = llr_overrides
        self.rank = rank

        # Parse distance curriculum
        self.distances = [self.args.distance]
        if getattr(self.args, "distance_curriculum", None):
            try:
                self.distances = [
                    int(x) for x in self.args.distance_curriculum.split(",") if x.strip()
                ]
            except ValueError:
                print(
                    f"Warning: Invalid distance curriculum '{self.args.distance_curriculum}', using default {self.args.distance}"
                )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = max(1, worker_info.num_workers)

        # Split the per-rank shot budget across workers to avoid duplicating work
        base = self.shots_to_do // num_workers
        remainder = self.shots_to_do % num_workers
        shots_for_this_worker = base + (1 if worker_id < remainder else 0)
        if shots_for_this_worker <= 0:
            return

        # Seed RNG for this worker
        # Ensure distinct seeds across ranks and workers
        seed = self.args.seed + self.epoch * 10000 + self.rank * 100 + worker_id
        rng = np.random.default_rng(seed)

        # Use global cache for teachers to persist across epochs in persistent workers
        global _WORKER_TEACHERS_CACHE

        shots_done = 0
        while shots_done < shots_for_this_worker:
            # Pick a distance for this shot
            d = int(rng.choice(self.distances))

            # Per-shot seed
            shot_seed = int(rng.integers(0, 2**31 - 1))

            sample = sample_round(
                d=d,
                p=self.p_epoch,
                seed=shot_seed,
                profile_path=self.args.qpu_profile if self.args.qpu_profile else None,
            )

            # Get or create teachers for this distance
            if d not in _WORKER_TEACHERS_CACHE:
                _WORKER_TEACHERS_CACHE[d] = self._init_teachers_for_d(d)

            mwpf_teacher, mwpm_ctx, lsd_teacher, nvqldpc_teacher = _WORKER_TEACHERS_CACHE[d]

            # Handle erasure
            erase_local = None
            if self.args.erasure_frac > 0:
                n_q = len(sample["coords_q"])
                # Generate erasure mask (1 if erased, 0 otherwise)
                erase_mask = (rng.random(n_q) < self.args.erasure_frac).astype(np.uint8)

                # Always set erase_local if erasure is enabled, to ensure consistent feature dim
                erase_local = erase_mask

                if np.any(erase_mask):
                    # Inject erasure noise into syndromes
                    # Erasure = random Pauli X/Y/Z/I.
                    # We simulate this as random X flip (50%) and random Z flip (50%)
                    # applied to the erased qubits.

                    # Random errors on erased qubits
                    erasure_err_x = (rng.random(n_q) < 0.5).astype(np.uint8) * erase_mask
                    erasure_err_z = (rng.random(n_q) < 0.5).astype(np.uint8) * erase_mask

                    # Update syndromes
                    # synZ checks X errors
                    # synX checks Z errors

                    Hz = sample["Hz"]
                    Hx = sample["Hx"]

                    # synZ += Hz @ erasure_err_x
                    additional_synZ = (Hz @ erasure_err_x) % 2
                    sample["synZ"] = (sample["synZ"] ^ additional_synZ).astype(np.uint8)

                    # synX += Hx @ erasure_err_z
                    additional_synX = (Hx @ erasure_err_z) % 2
                    sample["synX"] = (sample["synX"] ^ additional_synX).astype(np.uint8)

            yield from self._process_sample(
                sample,
                shot_seed,
                mwpf_teacher,
                mwpm_ctx,
                lsd_teacher,
                rng,
                d,
                erase_local,
                nvqldpc_teacher=nvqldpc_teacher,
            )

            shots_done += 1

    def _init_teachers_for_d(self, d):
        # Build Hx/Hz directly from a sample so that
        # - parity-check dimensions match the online sampler's syndromes, and
        # - nvqldpc (and other teachers) see a consistent PCM.
        base_p = float(getattr(self.args, "p", 0.005))
        sample = sample_round(
            d=d,
            p=base_p,
            seed=0,
            profile_path=self.args.qpu_profile if getattr(self.args, "qpu_profile", None) else None,
        )

        Hx = np.asarray(sample["Hx"], dtype=np.uint8)
        Hz = np.asarray(sample["Hz"], dtype=np.uint8)

        mx = Hx.shape[0]
        mz = Hz.shape[0]
        dets_per_fault = []
        for col in range(Hx.shape[1]):
            dets = []
            dets.extend(np.flatnonzero(Hx[:, col]).tolist())
            dets.extend((mx + np.flatnonzero(Hz[:, col])).tolist())
            dets_per_fault.append(dets)

        code = SimpleNamespace(
            Hx=Hx,
            Hz=Hz,
            detectors_per_fault=dets_per_fault,
            fault_weights=[1.0] * Hx.shape[1],
            num_detectors=mx + mz,
            n=Hx.shape[1],
        )

        commutes = not np.any((Hx % 2) @ (Hz.T % 2) % 2)
        use_mwpf = self.teacher_mix.get("mwpf", 0.0) > 0.0 and commutes
        use_mwpm = self.teacher_mix.get("mwpm", 0.0) > 0.0
        use_lsd = self.teacher_mix.get("lsd", 0.0) > 0.0
        use_nvqldpc = self.teacher_mix.get("nvqldpc", 0.0) > 0.0

        mwpf_teacher = None
        if use_mwpf:
            try:
                mwpf_teacher = MWPFTeacher(code)
            except Exception:
                mwpf_teacher = None

        mwpm_ctx = MWPMatchingContext() if use_mwpm else None

        lsd_teacher = None
        if use_lsd:
            try:
                lsd_teacher = LSDTeacher(code.Hx, code.Hz)
            except Exception:
                lsd_teacher = None

        nvqldpc_teacher = None
        if use_nvqldpc:
            # For nvqldpc we enforce strict behavior: construction errors
            # should propagate rather than silently disabling the teacher.
            nvqldpc_teacher = NvQldpcTeacher(code.Hx, code.Hz)

        return mwpf_teacher, mwpm_ctx, lsd_teacher, nvqldpc_teacher

    def _process_sample(
        self,
        sample,
        seed,
        mwpf_teacher,
        mwpm_ctx,
        lsd_teacher,
        rng,
        d,
        erase_local=None,
        nvqldpc_teacher=None,
    ):
        # Logic extracted from the main loop

        # Check if we have native Stim detectors (preferred path)
        use_native_dets = (
            "detectors" in sample and sample.get("dem_meta", {}).get("backend") == "stim_native"
        )

        if use_native_dets:
            # Use native Stim detector ordering - this is the correct path
            dets_global = sample["detectors"][np.newaxis, :].astype(np.uint8)
        else:
            # Legacy path: Pack detectors in canonical Z→X order for MWPFTeacher
            dets_global = np.concatenate(
                [
                    sample["synZ"][np.newaxis, :].astype(np.uint8),
                    sample["synX"][np.newaxis, :].astype(np.uint8),
                ],
                axis=1,
            )

        # Global per-fault scaling dict
        mwpf_scale = None
        if hasattr(self, "llr_overrides") and self.llr_overrides is not None:
            probs = 1.0 / (1.0 + np.exp(self.llr_overrides))
            scale_full = np.clip(probs / 0.5, 0.1, 10.0)
            mwpf_scale = {int(i): float(s) for i, s in enumerate(scale_full)}

        fault_ids_global = None
        if mwpf_teacher is not None:
            try:
                out_mwpf = mwpf_teacher.decode_batch(dets_global, mwpf_scale=mwpf_scale)
                fid_arr = np.asarray(out_mwpf.get("fault_ids"), dtype=np.int32)
                if fid_arr.ndim == 2 and fid_arr.shape[0] >= 1:
                    fault_ids_global = fid_arr[0]
            except Exception:
                fault_ids_global = None

        # Compute LSD once per sample
        ex_lsd = ez_lsd = None
        if lsd_teacher is not None:
            try:
                ex_arr, ez_arr = lsd_teacher.decode_batch_xz(
                    syndromes_x=sample["synX"][None, :],
                    syndromes_z=sample["synZ"][None, :],
                    llr_overrides=self.llr_overrides if hasattr(self, "llr_overrides") else None,
                )
                ex_lsd, ez_lsd = ex_arr[0], ez_arr[0]
            except Exception:
                ex_lsd = ez_lsd = None

        # Compute NvQldpc once per sample (GPU BP+OSD teacher, strict)
        ex_nq = ez_nq = None
        if nvqldpc_teacher is not None:
            ex_arr, ez_arr = nvqldpc_teacher.decode_batch_xz(
                syndromes_x=sample["synX"][None, :],
                syndromes_z=sample["synZ"][None, :],
            )
            ex_nq, ez_nq = ex_arr[0], ez_arr[0]

        oracle_enabled = self.teacher_mix.get("oracle", 0.0) > 0.0
        oracle_ex = sample.get("ex_glob", None)
        oracle_ez = sample.get("ez_glob", None)
        if oracle_enabled and (oracle_ex is None or oracle_ez is None):
            raise RuntimeError(
                "Oracle supervision requested but sampler did not return ex_glob/ez_glob."
            )

        for side in ("Z", "X"):
            comps = split_components_for_side(
                side=side,
                Hx=sample["Hx"],
                Hz=sample["Hz"],
                synZ=sample["synZ"],
                synX=sample["synX"],
                coords_q=sample["coords_q"],
                coords_c=sample["coords_c"],
            )
            for comp in comps:
                H_sub = comp["H_sub"]
                synd_bits = comp["synd_bits"]
                qubit_indices = comp["qubit_indices"]

                outputs = {}
                # MWPF
                if fault_ids_global is not None:
                    local_bits = np.zeros(H_sub.shape[1], dtype=np.uint8)
                    valid_ids = fault_ids_global[fault_ids_global >= 0]
                    if valid_ids.size:
                        mask = np.isin(qubit_indices, valid_ids)
                        local_bits[mask] = 1
                    outputs["mwpf"] = (local_bits, int(local_bits.sum()))

                # MWPM
                if mwpm_ctx is not None:
                    bits_pm, w_pm = mwpm_ctx.decode(H_sub, synd_bits, side)
                    outputs["mwpm"] = (bits_pm.astype(np.uint8), int(w_pm))

                # LSD
                if lsd_teacher is not None and (ex_lsd is not None and ez_lsd is not None):
                    bits_global = ex_lsd if side == "Z" else ez_lsd
                    if qubit_indices.size and bits_global.size > qubit_indices.max():
                        bits_local = bits_global[qubit_indices].astype(np.uint8)
                        outputs["lsd"] = (bits_local, int(bits_local.sum()))

                # Oracle labels (true sampled errors), if provided by sampler
                if oracle_enabled:
                    bits_global = oracle_ex if side == "Z" else oracle_ez
                    if (
                        bits_global is not None
                        and qubit_indices.size
                        and bits_global.size > qubit_indices.max()
                    ):
                        bits_local = bits_global[qubit_indices].astype(np.uint8)
                        outputs["oracle"] = (bits_local, int(bits_local.sum()))

                # NvQldpc (GPU BP+OSD)
                if nvqldpc_teacher is not None and (ex_nq is not None and ez_nq is not None):
                    bits_global = ex_nq if side == "Z" else ez_nq
                    if qubit_indices.size and bits_global.size > qubit_indices.max():
                        bits_local = bits_global[qubit_indices].astype(np.uint8)
                        outputs["nvqldpc"] = (bits_local, int(bits_local.sum()))

                # Choose teacher
                weighted = []
                for name, (bits, w) in outputs.items():
                    if self.teacher_mix.get(name, 0.0) > 0:
                        weighted.append((name, bits, w, self.teacher_mix[name]))

                chosen_bits = None
                if weighted:
                    total_w = sum(w for *_, w in weighted)
                    r = float(rng.random() * max(total_w, 1e-9))
                    acc = 0.0
                    for name, bits, w, tw in weighted:
                        acc += tw
                        if r <= acc:
                            chosen_bits = bits
                            break

                if chosen_bits is None:
                    continue

                # Extract local erasure if present
                local_erasure = None
                if erase_local is not None:
                    # erase_local is full array of size n_qubits
                    # we need to extract the subset for this component
                    if qubit_indices.size > 0 and qubit_indices.max() < erase_local.size:
                        local_erasure = erase_local[qubit_indices]
                    else:
                        local_erasure = np.zeros(len(qubit_indices), dtype=np.uint8)
                elif self.args.erasure_frac > 0:
                    # Force zero erasure if not provided but expected
                    local_erasure = np.zeros(len(qubit_indices), dtype=np.uint8)

                pack = pack_cluster(
                    H_sub=H_sub,
                    xy_qubit=comp["xy_qubit"],
                    xy_check=comp["xy_check"],
                    synd_Z_then_X_bits=synd_bits,
                    k=int(comp["k"]),
                    r=int(comp["r"]),
                    bbox_xywh=tuple(int(v) for v in comp["bbox_xywh"]),
                    kappa_stats=comp.get("kappa_stats", {}),
                    y_bits_local=chosen_bits,
                    side=side,
                    d=d,
                    p=self.p_epoch,
                    seed=seed,
                    N_max=self.args.N_max if hasattr(self.args, "N_max") else 512,
                    E_max=self.args.E_max if hasattr(self.args, "E_max") else 4096,
                    S_max=self.args.S_max if hasattr(self.args, "S_max") else 512,
                    g_extra=self.ctx_vec if hasattr(self, "ctx_vec") else None,
                    erase_local=local_erasure,
                )
                # We don't validate contract here to save time, or we can.
                # _validate_packed_contract(pack, self.args.node_feat_dim, self.args.edge_feat_dim)
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
