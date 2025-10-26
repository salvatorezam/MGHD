# NOTE: Initialize CUDA only in main(). This file defines dataset, model, loop.
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, Sampler, DataLoader

from mghd.decoders.lsd import clustered as cc  # uses projector functions
from mghd.core.core import MGHDv2, PackedCrop


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
    label_smoothing: float = 0.0,
) -> torch.Tensor:
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
    # Adapter to clustered projector (discover exact signature and adapt)
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
    args = ns
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=False)
    parser.add_argument("--profile", type=str, default="S")
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--lr", type=float, default=5.952925899948483e-05)
    parser.add_argument("--wd", type=float, default=6.65850238574699e-05)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--parity-lambda", type=float, default=0.03)
    parser.add_argument("--projection-aware", type=int, default=1)
    parser.add_argument("--label-smoothing", type=float, default=0.13831652882929857)
    parser.add_argument("--noise-injection", type=float, default=0.009883059279379016)
    parser.add_argument("--grad-clip", type=float, default=0.8545326095750816)
    parser.add_argument("--save", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    if not hasattr(args, "data_root"):
        args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Build model with optional hyperparameters from JSON
    hp_model = getattr(args, "_hp_model", {})
    hp_mamba = getattr(args, "_hp_mamba", {})
    hp_attn = getattr(args, "_hp_attn", {})
    m_kwargs = {}
    if hp_mamba:
        if "d_model" in hp_mamba:
            m_kwargs["d_model"] = int(hp_mamba["d_model"])  # type: ignore[assignment]
        if "d_state" in hp_mamba:
            m_kwargs["d_state"] = int(hp_mamba["d_state"])  # type: ignore[assignment]
    if hp_model:
        if "n_iters" in hp_model:
            m_kwargs["n_iters"] = int(hp_model["n_iters"])  # type: ignore[assignment]
        if "msg_net_size" in hp_model:
            m_kwargs["gnn_msg_net_size"] = int(hp_model["msg_net_size"])  # type: ignore[assignment]
        if "msg_net_dropout_p" in hp_model:
            m_kwargs["gnn_msg_dropout"] = float(hp_model["msg_net_dropout_p"])  # type: ignore[assignment]
        if "gru_dropout_p" in hp_model:
            m_kwargs["gnn_gru_dropout"] = float(hp_model["gru_dropout_p"])  # type: ignore[assignment]
    if hp_attn and "se_reduction" in hp_attn:
        m_kwargs["se_reduction"] = int(hp_attn["se_reduction"])  # type: ignore[assignment]
    model = MGHDv2(profile=args.profile, **m_kwargs).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

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

    best_loss = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_items = 0
        for batch in loader:
            if not batch:
                continue
            moved = [move_to(packed, device) for packed in batch]
            if args.noise_injection > 0.0:
                std = float(args.noise_injection)
                for packed in moved:
                    packed.x_nodes = packed.x_nodes + torch.randn_like(packed.x_nodes) * std
            opt.zero_grad(set_to_none=True)

            batch_loss = torch.zeros((), device=device)
            for packed in moved:
                logits, node_mask = model(packed=packed)
                logits = logits.squeeze(0)
                node_mask = node_mask.squeeze(0)

                hard = (0.5 if packed.teacher_valid else 1.5) + (0.5 if packed.teacher_matched_local_ml else 1.0)
                sample_weight = torch.tensor(hard, dtype=torch.float32, device=device)

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
                )

                loss_proj = torch.tensor(0.0, device=device)
                if args.projection_aware:
                    data_mask = (packed.node_type == 0) & node_mask
                    proj_bits = projection_aware_logits_to_bits(
                        logits,
                        projector_kwargs={"H_sub": packed.H_sub, "side": getattr(packed.meta, "side", "Z")},
                        data_mask=data_mask,
                    )
                    with torch.no_grad():
                        mask_data = data_mask.detach().cpu().numpy()
                        target_full = packed.y_bits.detach().cpu().numpy().clip(0, 1).astype(np.uint8)
                        target_data = target_full[mask_data]

                    proj_target = torch.from_numpy(target_data).to(device)
                    proj_pred = torch.from_numpy(proj_bits.astype(np.int64)).to(device)

                    raw_bits = (torch.sigmoid(logits[:, 1] - logits[:, 0])[data_mask] > 0.5).long()
                    loss_proj = 0.5 * F.l1_loss(proj_pred.float(), proj_target.float()) + 0.2 * F.l1_loss(proj_pred.float(), raw_bits.float())

                sample_loss = loss_bce + loss_par + 0.5 * loss_proj
                batch_loss = batch_loss + sample_loss

            batch_size = len(moved)
            batch_loss = batch_loss / batch_size
            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            opt.step()

            total_loss += batch_loss.item() * batch_size
            n_items += batch_size

        sched.step()
        dt = time.time() - t0
        avg = total_loss / max(n_items, 1)
        history.append({"epoch": epoch, "loss": avg, "count": n_items, "secs": dt})

        torch.save({"model": model.state_dict(), "epoch": epoch, "loss": avg}, save_dir / "last.pt")
        if avg < best_loss:
            best_loss = avg
            torch.save({"model": model.state_dict(), "epoch": epoch, "loss": avg}, save_dir / "best.pt")

        print(json.dumps({"epoch": epoch, "loss": avg, "secs": dt}, separators=(",", ":")))

    (save_dir / "train_log.json").write_text(json.dumps(history, indent=2))

    return os.path.join(args.save, "best.pt")
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
    ns.label_smoothing = 0.13831652882929857
    ns.noise_injection = 0.009883059279379016
    ns.grad_clip = 0.8545326095750816
    
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

def main():
    # CLI wrapper that forwards to train_inprocess
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--profile", type=str, default="S")
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--lr", type=float, default=5.952925899948483e-05)
    parser.add_argument("--wd", type=float, default=6.65850238574699e-05)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--parity-lambda", type=float, default=0.03)
    parser.add_argument("--projection-aware", type=int, default=1)
    parser.add_argument("--label-smoothing", type=float, default=0.13831652882929857)
    parser.add_argument("--noise-injection", type=float, default=0.009883059279379016)
    parser.add_argument("--grad-clip", type=float, default=0.8545326095750816)
    parser.add_argument("--save", type=str, required=True)
    parser.add_argument("--hparams", type=str, default=None, help="Path to JSON hyperparameters file")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # Optionally load hyperparameters and apply to args
    if args.hparams:
        try:
            with open(args.hparams, "r", encoding="utf-8") as fh:
                hp = json.load(fh)
        except Exception:
            hp = None
        if isinstance(hp, dict):
            tp = hp.get("training_parameters", {}) or {}
            args.lr = float(tp.get("lr", args.lr))
            args.wd = float(tp.get("weight_decay", args.wd))
            args.label_smoothing = float(tp.get("label_smoothing", args.label_smoothing))
            args.grad_clip = float(tp.get("gradient_clip", args.grad_clip))
            args.noise_injection = float(tp.get("noise_injection", args.noise_injection))
            args.epochs = int(tp.get("epochs", args.epochs))
            args.batch = int(tp.get("batch_size", args.batch))
            setattr(args, "_hp_model", hp.get("model_architecture", {}) or {})
            setattr(args, "_hp_mamba", hp.get("mamba_parameters", {}) or {})
            setattr(args, "_hp_attn", hp.get("attention_mechanism", {}) or {})
    # Delegate
    train_inprocess(args)

if __name__ == "__main__":
    main()
