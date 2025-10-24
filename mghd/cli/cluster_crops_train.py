# NOTE: Initialize CUDA only in main(). This file defines dataset, model, loop.
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time
from collections import defaultdict
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mghd_clustered import cluster_core as cc  # uses projector functions
from mghd_public.features_v2 import PackedCrop
from mghd_public.model_v2 import MGHDv2
from torch.utils.data import Dataset, Sampler


class CropShardDataset(Dataset):
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
        return len(self.items)
        
    def __getitem__(self, idx):
        _, item = self.items[idx]

        def to_tensor(x, dtype=None):
            if isinstance(x, torch.Tensor):
                return torch.as_tensor(x, dtype=dtype)
            return torch.as_tensor(x, dtype=dtype)

        packed = PackedCrop(
            x_nodes=to_tensor(item["x_nodes"], torch.float32),
            node_mask=to_tensor(item["node_mask"], torch.bool),
            node_type=to_tensor(item["node_type"], torch.int8),
            edge_index=to_tensor(item["edge_index"], torch.long),
            edge_attr=to_tensor(item["edge_attr"], torch.float32),
            edge_mask=to_tensor(item["edge_mask"], torch.bool),
            seq_idx=to_tensor(item["seq_idx"], torch.long),
            seq_mask=to_tensor(item["seq_mask"], torch.bool),
            g_token=to_tensor(item["g_token"], torch.float32),
            y_bits=to_tensor(item["y_bits"], torch.int8),
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
        buckets[(min(r,8), min(kappa,32), pbin)].append(i)
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
    # Single-sample per item; batch via stacking on a new dim for tensors that allow it.
    # Keep compatibility with existing microbatching by treating batch_size as micro-accumulation.
    return batch

def bce_binary_head_loss(
    logits: torch.Tensor,
    node_mask: torch.Tensor,
    node_type: torch.Tensor,
    y_bits: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    # Use only data-qubit nodes (type==0) and valid mask
    is_data = (node_type == 0) & node_mask
    if is_data.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    # logits: [N_max, 2]; target bit in {0,1}
    target = y_bits.clamp_min(0).clamp_max(1).long()
    ce = nn.CrossEntropyLoss(reduction="none")  # on 2-logits
    loss_all = ce(logits[is_data], target[is_data])
    if sample_weight is not None:
        weight = sample_weight.to(logits.device).clamp_min(0.5).clamp_max(3.0)
        loss_all = loss_all * weight
    return loss_all.mean()

def parity_auxiliary_loss(
    logits: torch.Tensor,
    node_mask: torch.Tensor,
    node_type: torch.Tensor,
    H_sub: np.ndarray,
) -> torch.Tensor:
    # Differentiable XOR expectation ~= parity of Bernoulli probs
    with torch.no_grad():
        is_data = ((node_type == 0) & node_mask).cpu().numpy()
        # map logits indices -> data-qubits used by H_sub
        data_idx = np.nonzero(is_data)[0]
    if len(data_idx) == 0 or H_sub.shape[0] == 0:
        return torch.tensor(0.0, device=logits.device)
    p = torch.sigmoid(logits[:, 1] - logits[:, 0])  # P(bit=1)
    p_data = p[data_idx]
    # Expected parity for each check row: E[⊕] ≈ 0.5*(1 - ∏(1-2p_i)) over participating data
    # H_sub columns are already local data-qubit order [0..nQ-1]
    H = torch.as_tensor(H_sub, dtype=torch.float32, device=logits.device)
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
    prod = torch.stack(prod_terms)
    E_par = 0.5 * (1 - prod)
    # Encourage small E_par (close to 0 or target) — here we push toward 0.5 neutrality lightly
    return (E_par - 0.5).pow(2).mean()

def projection_aware_logits_to_bits(
    logits: torch.Tensor,
    projector_kwargs: dict[str, Any],
    *,
    data_mask: torch.Tensor,
) -> np.ndarray:
    # Convert logits -> per-qubit probabilities -> run exact projector in GF(2) to ML correction.
    # restrict to data-qubits only to match H_sub columns
    probs1_full = torch.sigmoid(logits[:, 1] - logits[:, 0])
    probs1 = probs1_full[data_mask].detach().cpu().numpy()
    # Adapter to cluster_core projector (discover exact signature and adapt)
    if hasattr(cc, "ml_parity_project"):
        H_sub = projector_kwargs.get("H_sub")
        s_sub_default = (
            np.zeros(H_sub.shape[0], dtype=np.uint8)
            if H_sub is not None
            else np.array([], dtype=np.uint8)
        )
        s_sub = projector_kwargs.get("s_sub", s_sub_default)
        if H_sub is not None and isinstance(H_sub, np.ndarray):
            H_sub = sp.csr_matrix(H_sub)
        bits = cc.ml_parity_project(H_sub, s_sub, probs1)  # np.uint8
    else:
        # Fallback: threshold
        bits = (probs1 > 0.5).astype(np.uint8)
    return bits  # length == #data_qubits

def train_inprocess(ns) -> str:
    """
    In-process entry that mirrors CLI `main()`.
    Returns path to best checkpoint.
    """
    # ns: argparse.Namespace with fields matching CLI flags
    args = ns
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=False)
    parser.add_argument("--profile", type=str, default="S")
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--wd", type=float, default=7e-5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--parity-lambda", type=float, default=0.03)
    parser.add_argument("--projection-aware", type=int, default=1)
    parser.add_argument("--save", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    
    # If called in-process, args already populated. If missing fields, parse CLI.
    if not hasattr(args, "data_root"):
        args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MGHDv2(profile=args.profile).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    ds = CropShardDataset(args.data_root)
    sampler = make_bucket_sampler(ds, stage="stage1", seed=args.seed)
    # iterate items one by one to keep padding/masks simple; sampler controls curriculum
    os.makedirs(args.save, exist_ok=True)

    best_loss = float("inf")
    for epoch in range(1, args.epochs+1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_items = 0
        for idx in sampler:
            packed = ds[idx]
            logits, node_mask = model(packed=move_to(packed, device))
            # BCE on data-qubits
            # hard-case upweighting: when MWPF!=MWPM or !teacher_valid -> upweight
            hard = (0.5 if packed.teacher_valid else 1.5) + (
                0.5 if packed.teacher_matched_local_ml else 1.0
            )
            sample_weight = torch.tensor(hard, dtype=torch.float32, device=device)
            loss_bce = bce_binary_head_loss(
                logits,
                packed.node_mask.to(device),
                packed.node_type.to(device),
                packed.y_bits.to(device),
                sample_weight=sample_weight,
            )
            # Parity aux
            loss_par = args.parity_lambda * parity_auxiliary_loss(
                logits,
                packed.node_mask.to(device),
                packed.node_type.to(device),
                H_sub=packed.H_sub,
            )
            # Projection-aware term
            loss_proj = torch.tensor(0.0, device=device)
            if args.projection_aware:
                data_mask = (packed.node_type.to(device) == 0) & packed.node_mask.to(device)
                proj_bits = projection_aware_logits_to_bits(
                    logits,
                    projector_kwargs={
                        "H_sub": packed.H_sub,
                        "side": getattr(packed.meta, "side", "Z"),
                    },
                    data_mask=data_mask,
                )
                with torch.no_grad():
                    # Extract target bits for data qubits only
                    mask_data = (packed.node_type.to(device) == 0) & packed.node_mask.to(device)
                    target_full = (
                        packed.y_bits.cpu().numpy().clip(0, 1)
                    ).astype(np.uint8)
                    target_data = target_full[mask_data.cpu().numpy()]  # Extract data qubit targets

                # proj_bits and target_data both have length = number of data qubits
                proj_target = torch.from_numpy(target_data).to(device)
                proj_pred = torch.from_numpy(proj_bits.astype(np.int64)).to(device)

                # Penalize projection flips against the raw threshold on data-qubits
                raw_bits = (torch.sigmoid(logits[:, 1] - logits[:, 0])[data_mask] > 0.5).long()
                loss_proj = 0.5 * F.l1_loss(
                    proj_pred.float(),
                    proj_target.float(),
                ) + 0.2 * F.l1_loss(proj_pred.float(), raw_bits.float())

            loss = loss_bce + loss_par + 0.5*loss_proj
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total_loss += loss.item()
            n_items += 1
        sched.step()
        dt = (time.time()-t0)
        avg = total_loss/max(n_items,1)
        ck = f"{args.save}/epoch{epoch:03d}.pt"
        torch.save({"model":model.state_dict(),"epoch":epoch,"loss":avg}, ck)
        if avg < best_loss:
            best_loss = avg
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "loss": avg},
                os.path.join(args.save, "best.pt"),
            )
        # Telemetry (teacher fractions, simple LER proxy vs teacher after projection)
        print(json.dumps({"epoch":epoch,"loss":avg,"secs":dt}, separators=(",",":")))
    return os.path.join(args.save,"best.pt")

def move_to(p: PackedCrop, device):
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
    return p

def sanity_train(crop_root: str, epochs: int = 2, batch_size: int = 2, lr: float = 1e-4):
    """Quick sanity training to validate the v2 system."""
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
    
    # Create temporary directory for save path
    temp_dir = tempfile.mkdtemp()
    ns.save = os.path.join(temp_dir, "sanity_test")
    
    print(f"Running sanity training with {epochs} epochs, batch_size={batch_size}, lr={lr}")
    print(f"Temporary save path: {ns.save}")
    
    try:
        result_path = train_inprocess(ns)
        print(f"Training completed, result: {result_path}")
        
        # Return the trained model for inspection
        from mghd_public.model_v2 import MGHDv2
        model = MGHDv2(profile=ns.profile)
        if torch.cuda.is_available():
            model = model.cuda()
        return model
    finally:
        # Clean up temp directory
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def main():
    # CLI wrapper that forwards to train_inprocess
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--profile", type=str, default="S")
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--wd", type=float, default=7e-5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--parity-lambda", type=float, default=0.03)
    parser.add_argument("--projection-aware", type=int, default=1)
    parser.add_argument("--save", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # Delegate
    train_inprocess(args)

if __name__ == "__main__":
    main()
