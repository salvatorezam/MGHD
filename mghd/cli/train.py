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
from mghd.core.core import MGHDv2, PackedCrop, pack_cluster
from mghd.decoders.lsd_teacher import LSDTeacher
from mghd.decoders.mwpf_teacher import MWPFTeacher
from mghd.decoders.mwpm_ctx import MWPMatchingContext
from mghd.qpu.adapters.garnet_adapter import sample_round, split_components_for_side
from mghd.tad import weighting as tad_weighting
from mghd.tad import context as tad_context
from mghd.codes.qpu_profile import load_qpu_profile


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
    """Identity collate: crops are processed one by one in the training loop.

    Retains compatibility with micro‑accumulation styles where the trainer
    iterates over items and aggregates gradients manually.
    """
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

def parity_auxiliary_loss(
    logits: torch.Tensor,
    node_mask: torch.Tensor,
    node_type: torch.Tensor,
    H_sub: np.ndarray,
) -> torch.Tensor:
    """Small regularizer encouraging parity consistency within the crop."""
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
    """Run the trainer in‑process (mirrors CLI ``main``) and return best.pt path."""
    args = ns
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=False)
    parser.add_argument("--online", action="store_true", help="Enable on-the-fly CUDA-Q sampling")
    parser.add_argument("--family", type=str, default="surface")
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--p", type=float, default=0.005, help="Physical error rate for online sampling")
    parser.add_argument("--p-curriculum", type=str, default=None, help="Comma-separated list of p values to cycle across epochs in online mode (e.g., '0.01,0.006,0.003')")
    parser.add_argument("--epochs-per-p", type=int, default=1, help="Epochs to spend on each p value in --p-curriculum")
    parser.add_argument("--qpu-profile", type=str, default=None)
    parser.add_argument("--context-source", type=str, default="none", choices=["none","qiskit","cirq","cudaq"]) 
    parser.add_argument("--shots-per-epoch", type=int, default=256)
    parser.add_argument("--erasure-frac", type=float, default=0.0, help="Optional erasure injection fraction for online sampling")
    parser.add_argument("--teacher-mix", type=str, default="mwpf=1.0,mwpm=0.0,lsd=0.0")
    parser.add_argument("--online-rl", action="store_true", help="Enable LinTS scaling of TAD overrides in online mode")
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
    parser.add_argument("--hparams", type=str, default=None, help="Path to JSON hyperparameters file")
    parser.add_argument("--save", type=str, required=False, default=None, help="If omitted or contains '{auto}', an auto-named run dir is created under --save-root")
    parser.add_argument("--save-root", type=str, default="data/results", help="Root directory for auto-named runs")
    parser.add_argument("--save-auto", action="store_true", help="Force auto-named save directory under --save-root")
    parser.add_argument("--seed", type=int, default=42)
    # Progress reporting (prints per epoch; 1 = only near end, 0 = disable mid-epoch prints)
    parser.add_argument("--progress-prints", type=int, default=1)
    # Optional post-run teacher comparison report (writes teacher_eval.txt)
    parser.add_argument("--post-eval", action="store_true")
    parser.add_argument("--post-eval-sampler", type=str, default="stim")
    parser.add_argument("--post-eval-shots-per-batch", type=int, default=16)
    parser.add_argument("--post-eval-batches", type=int, default=2)
    if not hasattr(args, "data_root"):
        args = parser.parse_args()
        # Optionally load hyperparameters and apply to args
        if getattr(args, "hparams", None):
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

    # Resolve save directory (auto if requested or missing)
    def _auto_name(family: str, distance: int, qpu_profile: str | None) -> str:
        ts = time.strftime("%Y%m%d-%H%M%S")
        qpu = "none"
        if qpu_profile:
            import os as _os
            qpu = _os.path.splitext(_os.path.basename(qpu_profile))[0]
        return f"{ts}_{family}_d{int(distance)}_{qpu}"

    if args.save_auto or args.save is None or (isinstance(args.save, str) and "{auto}" in args.save):
        auto = _auto_name(getattr(args, "family", "code"), int(getattr(args, "distance", 0)), getattr(args, "qpu_profile", None))
        base = Path(getattr(args, "save_root", "data/results"))
        base.mkdir(parents=True, exist_ok=True)
        args.save = str(base / auto)
    else:
        # Support placeholder replacement in explicit paths
        if "{auto}" in args.save:
            auto = _auto_name(getattr(args, "family", "code"), int(getattr(args, "distance", 0)), getattr(args, "qpu_profile", None))
            args.save = args.save.replace("{auto}", auto)

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
    # If online with erasures, expand node features by 1 (erasure flag)
    if bool(getattr(args, "online", False)) and float(getattr(args, "erasure_frac", 0.0)) > 0.0:
        m_kwargs["node_feat_dim"] = 9
    model = MGHDv2(profile=args.profile, **m_kwargs).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    loader = None
    use_online = bool(getattr(args, "online", False))
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
            "p_curriculum": [float(x) for x in str(getattr(args, "p_curriculum", "")).split(",") if x.strip()] if getattr(args, "p_curriculum", None) else None,
            "epochs_per_p": int(getattr(args, "epochs_per_p", 1)),
            "teacher_mix": getattr(args, "teacher_mix", None),
            "qpu_profile": getattr(args, "qpu_profile", None),
            "context_source": getattr(args, "context_source", None),
            "erasure_frac": float(getattr(args, "erasure_frac", 0.0)),
            "shots_per_epoch": int(getattr(args, "shots_per_epoch", 0)),
            "epochs": int(args.epochs),
            "batch": int(args.batch),
            "seed": int(args.seed),
        }
        (save_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))
    except Exception:
        pass

    best_loss = float("inf")
    history: list[dict[str, Any]] = []
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
                for attr in ("native_circuit","reference_circuit","circuit","qc","quantum_circuit"):
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
                gate_vocab = {g:i for i,g in enumerate(sorted(feats.get("gate_hist",{}).keys()))}
                ctx_vec = tad_context.context_vector(feats, gate_vocab)
                # Build per-qubit LLR override
                import numpy as _np
                llr = _np.zeros(int(n_qubits), dtype=_np.float32)
                for layer in (maps.get("w_qubit",{}) or {}).values():
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
        weights = {"mwpf": 1.0, "mwpm": 0.0, "lsd": 0.0}
        if not spec:
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
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
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
            # Teacher setup per epoch
            family = args.family
            if family != "surface":
                raise NotImplementedError("Online training currently supports family='surface'")
            # Build code object
            from mghd.codes.registry import get_code
            code = get_code(family, distance=args.distance)
            # Initialize MWPFTeacher once per epoch
            mwpf_teacher = None
            try:
                mwpf_teacher = MWPFTeacher(code)
            except Exception:
                mwpf_teacher = None
            mwpm_ctx = MWPMatchingContext()
            lsd_teacher = None
            try:
                lsd_teacher = LSDTeacher(code.Hx, code.Hz)
            except Exception:
                lsd_teacher = None
            teacher_mix = _parse_teacher_mix(getattr(args, "teacher_mix", "mwpf=1.0,mwpm=0.0,lsd=0.0"))
            # TAD context/overrides
            qpu_prof, ctx_vec, llr_overrides = _build_tad_for_code(code)
            # Initialize bandit once if requested and context available
            if args.online_rl and ctx_vec is not None and bandit is None:
                try:
                    from mghd.tad.rl.lin_ts import LinTSBandit
                    bandit = LinTSBandit(d=ctx_vec.size, prior_var=5.0, noise_var=0.5)
                except Exception:
                    bandit = None
            import numpy as _np
            rng = _np.random.default_rng(args.seed + epoch)
            batch_buf: list[PackedCrop] = []
            shots_target = int(getattr(args, "shots_per_epoch", args.batch))
            shots_done = 0
            erase_data_mask = None  # optional per-batch erasure mask; None by default
            # Mid-epoch progress prints (optional)
            prog_stride = (max(1, shots_target // int(getattr(args, "progress_prints", 1)))
                           if int(getattr(args, "progress_prints", 1)) > 1 else 0)
            while shots_done < shots_target:
                seed = int(rng.integers(0, 2**31 - 1))
                sample = sample_round(d=args.distance, p=p_epoch, seed=seed, profile_path=args.qpu_profile if args.qpu_profile else None)
                # Pack detectors in canonical Z→X order for MWPFTeacher
                dets_global = np.concatenate([
                    sample["synZ"][np.newaxis, :].astype(np.uint8),
                    sample["synX"][np.newaxis, :].astype(np.uint8),
                ], axis=1)
                # Global per-fault scaling dict
                mwpf_scale = None
                if llr_overrides is not None:
                    probs = 1.0/(1.0+_np.exp(llr_overrides))
                    scale_full = _np.clip(probs/0.5, 0.1, 10.0)
                    mwpf_scale = {int(i): float(s) for i, s in enumerate(scale_full)}
                fault_ids_global = None
                if mwpf_teacher is not None:
                    try:
                        out_mwpf = mwpf_teacher.decode_batch(dets_global, mwpf_scale=mwpf_scale)
                        fid_arr = _np.asarray(out_mwpf.get("fault_ids"), dtype=_np.int32)
                        if fid_arr.ndim == 2 and fid_arr.shape[0] >= 1:
                            fault_ids_global = fid_arr[0]
                    except Exception:
                        fault_ids_global = None
                for side in ("Z","X"):
                    comps = split_components_for_side(
                        side=side,
                        Hx=sample["Hx"], Hz=sample["Hz"],
                        synZ=sample["synZ"], synX=sample["synX"],
                        coords_q=sample["coords_q"], coords_c=sample["coords_c"],
                    )
                    for comp in comps:
                        H_sub = comp["H_sub"]
                        synd_bits = comp["synd_bits"]
                        qubit_indices = comp["qubit_indices"]
                        # Candidate outputs
                        outputs = {}
                        # MWPFTeacher fault_ids mapped to local bits
                        if fault_ids_global is not None:
                            local_bits = _np.zeros(H_sub.shape[1], dtype=_np.uint8)
                            valid_ids = fault_ids_global[fault_ids_global >= 0]
                            if valid_ids.size:
                                mask = _np.isin(qubit_indices, valid_ids)
                                local_bits[mask] = 1
                            outputs["mwpf"] = (local_bits, int(local_bits.sum()))
                        # MWPM fallback
                        bits_pm, w_pm = mwpm_ctx.decode(H_sub, synd_bits, side)
                        outputs["mwpm"] = (bits_pm.astype(_np.uint8), int(w_pm))
                        # LSD global converted to local
                        if lsd_teacher is not None:
                            try:
                                ex_glob, ez_glob = lsd_teacher.decode_batch_xz(
                                    syndromes_x=sample["synX"][None,:],
                                    syndromes_z=sample["synZ"][None,:],
                                    llr_overrides=llr_overrides,
                                )
                                bits_global = ex_glob[0] if side=="Z" else ez_glob[0]
                                if qubit_indices.size and bits_global.size > qubit_indices.max():
                                    bits_local = bits_global[qubit_indices].astype(_np.uint8)
                                    outputs["lsd"] = (bits_local, int(bits_local.sum()))
                            except Exception:
                                pass
                        # choose teacher by weighted random among valid
                        weighted = []
                        for name,(bits, w) in outputs.items():
                            if teacher_mix.get(name,0.0) > 0:
                                weighted.append((name, bits, w, teacher_mix[name]))
                        chosen_bits = None
                        if weighted:
                            total_w = sum(w for *_, w in weighted)
                            r = float(rng.random() * max(total_w,1e-9))
                            acc=0.0
                            for name,bits,w,tw in weighted:
                                acc += tw
                                if r <= acc:
                                    chosen_bits = bits
                                    break
                        if chosen_bits is None:
                            continue
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
                            d=args.distance,
                            p=args.p,
                            seed=seed,
                            N_max=args.N_max if hasattr(args,"N_max") else 512,
                            E_max=args.E_max if hasattr(args,"E_max") else 4096,
                            S_max=args.S_max if hasattr(args,"S_max") else 512,
                            g_extra=ctx_vec,
                            erase_local=(erase_data_mask[qubit_indices] if erase_data_mask is not None and qubit_indices.size else None),
                        )
                        batch_buf.append(move_to(pack, device))
                        if len(batch_buf) >= args.batch:
                            # run a step
                            moved = batch_buf
                            batch_buf = []
                            batch_loss = torch.zeros((), device=device)
                            for packed in moved:
                                logits, node_mask = model(packed=packed)
                                logits = logits.squeeze(0)
                                node_mask = node_mask.squeeze(0)
                                hard = 1.0
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
                                        target_full = packed.y_bits.detach().cpu().numpy().clip(0,1).astype(_np.uint8)
                                        target_data = target_full[mask_data]
                                    proj_target = torch.from_numpy(target_data).to(device)
                                    proj_pred = torch.from_numpy(proj_bits.astype(_np.int64)).to(device)
                                    raw_bits = (torch.sigmoid(logits[:,1]-logits[:,0])[data_mask] > 0.5).long()
                                    loss_proj = 0.5*F.l1_loss(proj_pred.float(), proj_target.float()) + 0.2*F.l1_loss(proj_pred.float(), raw_bits.float())
                                sample_loss = loss_bce + loss_par + 0.5*loss_proj
                                batch_loss = batch_loss + sample_loss
                            batch_size = len(moved)
                            batch_loss = batch_loss / batch_size
                            batch_loss.backward()
                            nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
                            opt.step(); opt.zero_grad(set_to_none=True)
                            total_loss += batch_loss.item() * batch_size
                            n_items += batch_size
                            shots_done += 1
                            # Periodic progress print within epoch
                            if prog_stride and (shots_done % prog_stride == 0):
                                prog = {"epoch": epoch,
                                        "step": int(shots_done),
                                        "steps": int(shots_target),
                                        "avg": float(total_loss / max(n_items, 1)),
                                        "secs": float(time.time() - t0)}
                                if p_epoch is not None:
                                    prog["p"] = float(p_epoch)
                                print(json.dumps(prog, separators=(",", ":")), flush=True)
                            if shots_done >= shots_target:
                                break
                if shots_done >= shots_target:
                    break
        else:
            steps_done = 0
            steps_total = len(loader) if hasattr(loader, "__len__") else 0
            prog_stride = (max(1, steps_total // int(getattr(args, "progress_prints", 1)))
                           if steps_total and int(getattr(args, "progress_prints", 1)) > 1 else 0)
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
            steps_done += 1
            if prog_stride and (steps_done % prog_stride == 0):
                prog = {"epoch": epoch,
                        "step": int(steps_done),
                        "steps": int(steps_total),
                        "avg": float(total_loss / max(n_items, 1)),
                        "secs": float(time.time() - t0)}
                print(json.dumps(prog, separators=(",", ":")), flush=True)

        sched.step()
        dt = time.time() - t0
        avg = total_loss / max(n_items, 1)
        history.append({"epoch": epoch, "loss": avg, "count": n_items, "secs": dt})
        # Bandit posterior update with simple reward: 1.0 if loss decreased, else 0.0
        if bandit is not None and prev_epoch_loss is not None and ctx_vec is not None:
            reward = 1.0 if avg < prev_epoch_loss else 0.0
            bandit.update(ctx_vec, reward)
        prev_epoch_loss = avg

        torch.save({"model": model.state_dict(), "epoch": epoch, "loss": avg}, save_dir / "last.pt")
        if avg < best_loss:
            best_loss = avg
            torch.save({"model": model.state_dict(), "epoch": epoch, "loss": avg}, save_dir / "best.pt")

        log_obj = {"epoch": epoch, "loss": avg, "secs": dt}
        if use_online:
            log_obj["p"] = float(p_epoch if 'p_epoch' in locals() else getattr(args, "p", 0.0))
        print(json.dumps(log_obj, separators=(",", ":")))

    (save_dir / "train_log.json").write_text(json.dumps(history, indent=2))

    # Optional teacher comparison report
    if hasattr(args, "post_eval") and bool(getattr(args, "post_eval", False)):
        try:
            import subprocess, sys as _sys, os as _os
            shots = int(getattr(args, "post_eval_shots_per_batch", 16))
            batches = int(getattr(args, "post_eval_batches", 2))
            sampler = str(getattr(args, "post_eval_sampler", "stim"))
            cmd = [
                _sys.executable, "-m", "mghd.tools.teacher_eval",
                "--families", str(args.family),
                "--distances", str(args.distance),
                "--sampler", sampler,
                "--shots-per-batch", str(shots),
                "--batches", str(batches),
            ]
            env = _os.environ.copy()
            env.setdefault("PYTHONPATH", _os.getcwd())
            cp = subprocess.run(cmd, capture_output=True, text=True, env=env)
            (save_dir / "teacher_eval.txt").write_text(cp.stdout + ("\n--- STDERR ---\n" + cp.stderr if cp.stderr else ""))
        except Exception:
            pass

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
    return p

def sanity_train(crop_root: str, epochs: int = 2, batch_size: int = 2, lr: float = 1e-4):
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
    """Unified CLI entrypoint supporting offline and online modes.

    Delegates to ``train_inprocess``, which accepts both offline (``--data-root``)
    and online (``--online``) training paths with TAD/RL/erasure options.
    """
    from types import SimpleNamespace
    # Passing a dummy namespace will make train_inprocess parse sys.argv
    train_inprocess(SimpleNamespace())

if __name__ == "__main__":
    main()
