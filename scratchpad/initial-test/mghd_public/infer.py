"""Public MGHD inference helper for rotated d=3."""

from __future__ import annotations

import inspect
import math
import time
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, Sequence

import numpy as np
import scipy.sparse as sp
import torch

from .config import MGHDConfig
from .model import load_mghd_checkpoint
from .features import features_rotated_d3, features_from_subgraph
from .features_v2 import pack_cluster, PackedCrop, infer_bucket_id
from mghd_clustered.microbatcher import CrossShotBatcher, stack_crops_pinned
from mghd_clustered.cluster_core import gf2_nullspace


__all__ = ['warmup_and_capture', 'MGHDDecoderPublic', 'load_mghd_auto']


def _detect_model_version(ckpt_path: str) -> str:
    """Detect if checkpoint contains v1 or v2 model by examining state dict structure."""
    try:
        import torch
        state = torch.load(ckpt_path, map_location="cpu")
        
        # Handle different checkpoint formats
        state_dict = state
        if "state_dict" in state:
            state_dict = state["state_dict"]
        elif "model" in state:
            state_dict = state["model"]  # Training script format
        
        # v2 indicators: check for v2-specific layer patterns
        v2_indicators = [
            "seq_encoder",  # MaskedMamba component
            "g_proj",       # Global token projection (key v2 indicator)
            "gnn.node_in",  # v2 GNN structure
        ]
        
        # v1 indicators: check for v1-specific patterns
        v1_indicators = [
            "pe_enc",       # Position encoding in v1
            "pe_mlp",       # Position MLP in v1
            "static_",      # Static layout patterns
            "_Hx_static",   # v1 static matrices
            "_Hz_static",   # v1 static matrices
        ]
        
        # Count evidence for each version
        v2_score = sum(1 for key in state_dict.keys() if any(ind in key for ind in v2_indicators))
        v1_score = sum(1 for key in state_dict.keys() if any(ind in key for ind in v1_indicators))
        
        # Also check for class name in metadata if available
        model_class = state.get("hyper_parameters", {}).get("_target_", "")
        if "MGHDv2" in model_class or "model_v2" in model_class:
            return "v2"
        if "MGHD" in model_class and "v2" not in model_class:
            return "v1"
            
        # Decide based on scores
        if v2_score > v1_score:
            return "v2"
        elif v1_score > v2_score:
            return "v1"
        else:
            # Default to v1 for backward compatibility
            return "v1"
            
    except Exception:
        # Default to v1 if detection fails
        return "v1"


def load_mghd_auto(ckpt_path: str, cfg: MGHDConfig = None, device: str = "cpu", **kwargs):
    """Auto-detect and load v1 or v2 model based on checkpoint structure.
    
    Returns:
        For v1: (model, load_info) tuple compatible with MGHDDecoderPublic
        For v2: (model, load_info) with model being MGHDv2 instance
    """
    version = _detect_model_version(ckpt_path)
    
    if version == "v2":
        # Load v2 model
        try:
            from .model_v2 import MGHDv2
            import torch
            
            # Load checkpoint
            state = torch.load(ckpt_path, map_location="cpu")
            
            # Handle different checkpoint formats
            state_dict = state
            if "state_dict" in state:
                state_dict = state["state_dict"]
            elif "model" in state:
                state_dict = state["model"]  # Training script format
            
            # Create model instance with default config if needed
            if cfg is None:
                # v2 models don't need the complex MGHDConfig, create a simple one
                cfg = type('SimpleConfig', (), {
                    'n_checks': 72,
                    'n_qubits': 72, 
                    'n_node_inputs': 4,
                    'profile': 'v2'
                })()
            
            # Create model
            model = MGHDv2(d_model=192, d_state=80)  # v2 defaults
            model.load_state_dict(state_dict, strict=False)
            model.to(device).eval()
            
            # Attach config to model for compatibility
            model.cfg = cfg
            
            load_info = {
                "version": "v2",
                "checkpoint_path": ckpt_path,
                "device": device,
                "model_class": "MGHDv2",
            }
            
            return model, load_info
            
        except Exception as e:
            # Fallback to v1 if v2 loading fails
            print(f"Warning: v2 loading failed ({e}), falling back to v1")
            version = "v1"
    
    if version == "v1":
        # Load v1 model using existing mechanism
        if cfg is None:
            # Use default d=3 config
            cfg = MGHDConfig(n_checks=72, n_qubits=72, n_node_inputs=4)
        
        model, load_info = load_mghd_checkpoint(ckpt_path, cfg, device=device)
        load_info["version"] = "v1"
        return model, load_info
        
    else:
        raise ValueError(f"Unknown model version: {version}")


def warmup_and_capture(model, device, side: str, *, use_fixed_d3=True) -> dict:
    """
    Runs a single synthetic MGHD forward designed to be non-empty so CUDA graph capture happens.
    Returns dict with {'warmup_us': float, 'graph_used': bool, 'path': 'fast'|'fixed_d3'}.
    """
    if not hasattr(model, '_bound') or not model._bound:
        raise RuntimeError("Model must be bound to code before warmup")
    
    device = torch.device(device)
    side = side.upper()
    if side not in {"X", "Z"}:
        raise ValueError("side must be 'X' or 'Z'")
    
    # Synchronize before timing
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    
    start_time = time.time()
    
    # Try to build a synthetic non-empty batch
    path_used = "fixed_d3"  # Default fallback
    graph_used = False
    
    try:
        # Attempt fast path: create a 1-cluster subgraph with >= 2 checks and >= 2 qubits
        # For d=3 surface code, create a minimal subgraph
        if not use_fixed_d3:
            # Build a synthetic small subgraph (2x2 checks, 4 qubits)
            m_sub, n_sub = 4, 4  # 4 checks, 4 qubits
            # Simple connectivity: each check connects to 2 qubits
            row_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
            col_ids = np.array([0, 1, 1, 2, 2, 3, 3, 0], dtype=np.int32)  # Cycle
            data = np.ones(len(row_ids), dtype=np.int32)
            H_sub = sp.csr_matrix((data, (row_ids, col_ids)), shape=(m_sub, n_sub))
            
            # Non-zero syndrome to ensure model does work
            s_sub = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
            
            # Create subgraph features
            from .features import features_from_subgraph
            feats = features_from_subgraph(H_sub, s_sub, n_node_inputs=model.cfg.n_node_inputs)
            
            # Put tensors on device
            for key in feats:
                if torch.is_tensor(feats[key]):
                    feats[key] = feats[key].to(device)
            
            # Call model
            with torch.no_grad():
                _ = model._call_model(feats)
            
            path_used = "fast"
            
        else:
            # Fall back to fixed d=3 pack
            # Create a non-empty syndrome for the appropriate side
            if side == "Z":
                syndrome = np.zeros(model.cfg.n_checks // 2, dtype=np.float32)
                syndrome[0] = 1.0  # Ensure non-zero
            else:  # X
                syndrome = np.zeros(model.cfg.n_checks // 2, dtype=np.float32)
                syndrome[0] = 1.0  # Ensure non-zero
            
            # Use the model's d3 batched interface with CUDA graph capture
            with torch.no_grad():
                _, report = model.priors_from_d3_fullgraph_batched(
                    [syndrome], 
                    temp=model.temp if hasattr(model, 'temp') else 1.0,
                    bucket=side
                )
                graph_used = report.get('graph_used', False)
            
            path_used = "fixed_d3"
    
    except Exception:
        # Fallback: just call the most basic forward
        try:
            with torch.no_grad():
                # Use the simplest possible call with minimal synthetic data
                syndrome = np.zeros(model.cfg.n_checks // 2, dtype=np.float32)
                syndrome[0] = 1.0  # Non-zero to ensure work
                _ = model.priors_from_syndrome(syndrome, side=side)
            path_used = "fixed_d3"
        except Exception:
            # Last resort: just time an empty forward
            path_used = "minimal"
    
    # Synchronize after timing
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    
    end_time = time.time()
    warmup_us = (end_time - start_time) * 1e6  # Convert to microseconds
    
    return {
        'warmup_us': float(warmup_us),
        'graph_used': bool(graph_used),
        'path': path_used
    }


def _collate_flat_batches(batches: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor | List[Tuple[int, int, int]]]:
    """Concatenate variable-size subgraph feature dicts into one flat batch."""
    if not batches:
        raise ValueError("_collate_flat_batches requires at least one batch entry")

    node_inputs_list: List[torch.Tensor] = []
    src_list: List[torch.Tensor] = []
    dst_list: List[torch.Tensor] = []
    meta: List[Tuple[int, int, int]] = []
    node_base = 0

    for b in batches:
        node_inputs = b["node_inputs"]
        if node_inputs.ndim == 2:
            node_inputs = node_inputs.unsqueeze(0)
        node_inputs_list.append(node_inputs)

        m_sub = int(b["n_checks"])
        n_sub = int(b["n_qubits"])

        src = b["src_ids"].long() + node_base
        dst = b["dst_ids"].long() + node_base
        src_list.append(src)
        dst_list.append(dst)

        meta.append((m_sub, n_sub, node_base))
        node_base += (m_sub + n_sub)

    node_inputs_cat = torch.cat(node_inputs_list, dim=1)
    node_inputs_flat = node_inputs_cat.view(-1, node_inputs_cat.shape[-1])
    src_ids = torch.cat(src_list, dim=0)
    dst_ids = torch.cat(dst_list, dim=0)
    edge_index = torch.stack([src_ids, dst_ids], dim=0)

    return {
        "node_inputs": node_inputs_cat,
        "node_inputs_flat": node_inputs_flat,
        "src_ids": src_ids,
        "dst_ids": dst_ids,
        "edge_index": edge_index,
        "edge_attr": None,
        "meta": meta,
    }


def _pad_to_multiple(value: int, base: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if base <= 0:
        raise ValueError("base must be positive")
    padded = int(math.ceil(max(value, 1) / base) * base)
    if minimum is not None:
        padded = max(padded, minimum)
    if maximum is not None:
        padded = min(padded, maximum)
    return padded


def _v2_bucket_key(H_sub: sp.csr_matrix, *, max_nodes: int = 512, max_edges: int = 4096, max_seq: int = 512) -> Tuple[int, int, int]:
    n_checks, n_qubits = H_sub.shape
    nodes = n_checks + n_qubits
    edges = int(H_sub.nnz)
    seq = n_checks
    node_pad = _pad_to_multiple(nodes, 8, minimum=16, maximum=max_nodes)
    edge_pad = _pad_to_multiple(max(edges, 1), 16, minimum=32, maximum=max_edges)
    seq_pad = _pad_to_multiple(seq, 8, minimum=8, maximum=max_seq)
    return (node_pad, edge_pad, seq_pad)


def _to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if tensor.device == device:
        return tensor
    return tensor.to(device, non_blocking=True)


def _collate_v2_batch(
    entries: List[Tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray | None]],
    extras: List[Dict[str, Any]],
    bucket: Tuple[int, int, int],
) -> Tuple[SimpleNamespace, List[Dict[str, Any]]]:
    node_pad, edge_pad, seq_pad = bucket
    packed_crops: List[PackedCrop] = []
    cluster_infos: List[Dict[str, Any]] = []

    for (H_sub, s_sub, _q_l2g, _c_l2g), extra in zip(entries, extras):
        pack = pack_cluster(
            H_sub=H_sub.toarray(),
            xy_qubit=extra["xy_qubit"],
            xy_check=extra["xy_check"],
            synd_Z_then_X_bits=np.asarray(s_sub, dtype=np.uint8),
            k=extra["k"],
            r=extra["r"],
            bbox_xywh=extra["bbox"],
            kappa_stats=extra["kappa_stats"],
            y_bits_local=np.zeros(extra["k"], dtype=np.uint8),
            side=extra["side"],
            d=extra["d"],
            p=extra["p"],
            seed=0,
            N_max=node_pad,
            E_max=edge_pad,
            S_max=seq_pad,
            add_jump_edges=False,
        )
        packed_crops.append(pack)

    B = len(packed_crops)
    if B == 0:
        raise ValueError("_collate_v2_batch requires at least one crop")

    feat_dim = packed_crops[0].x_nodes.shape[1]
    edge_feat_dim = packed_crops[0].edge_attr.shape[1]
    g_dim = int(packed_crops[0].g_token.numel())

    x_nodes = torch.zeros((B, node_pad, feat_dim), dtype=torch.float32)
    node_mask = torch.zeros((B, node_pad), dtype=torch.bool)
    node_type = torch.zeros((B, node_pad), dtype=torch.int8)
    g_tokens = torch.zeros((B, g_dim), dtype=torch.float32)

    edge_index = torch.zeros((2, B * edge_pad), dtype=torch.long)
    edge_attr = torch.zeros((B * edge_pad, edge_feat_dim), dtype=torch.float32)
    edge_mask = torch.zeros((B * edge_pad,), dtype=torch.bool)

    seq_idx = torch.zeros((B * seq_pad,), dtype=torch.long)
    seq_mask = torch.zeros((B * seq_pad,), dtype=torch.bool)

    node_offset = 0

    for i, (pack, extra, entry) in enumerate(zip(packed_crops, extras, entries)):
        H_sub, _s_sub, q_l2g, _c_l2g = entry

        x_nodes[i, :, :] = pack.x_nodes
        node_mask[i, :] = pack.node_mask
        node_type[i, :] = pack.node_type
        g_tokens[i, :] = pack.g_token.view(-1).to(torch.float32)

        edge_base = i * edge_pad
        seq_base = i * seq_pad

        edge_index[:, edge_base : edge_base + edge_pad] = pack.edge_index + node_offset
        edge_attr[edge_base : edge_base + edge_pad, :] = pack.edge_attr
        edge_mask[edge_base : edge_base + edge_pad] = pack.edge_mask

        seq_idx[seq_base : seq_base + seq_pad] = pack.seq_idx + node_offset
        seq_mask[seq_base : seq_base + seq_pad] = pack.seq_mask

        data_positions = torch.nonzero((pack.node_type == 0) & pack.node_mask, as_tuple=False).squeeze(-1)

        cluster_infos.append(
            dict(
                data_idx=(data_positions + node_offset),
                q_l2g=q_l2g,
                nullity=int(extra["r"]),
                data_count=int(extra["k"]),
                bucket=bucket,
            )
        )

        node_offset += node_pad

    batch = SimpleNamespace(
        x_nodes=x_nodes,
        node_mask=node_mask,
        node_type=node_type,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_mask=edge_mask,
        seq_idx=seq_idx,
        seq_mask=seq_mask,
        g_token=g_tokens,
        batch_size=B,
        nodes_pad=node_pad,
        edges_pad=edge_pad,
        seq_pad=seq_pad,
        bucket=bucket,
    )

    return batch, cluster_infos


def _get_v2_graph_state(
    cache: Dict[Tuple[int, int, int, int], Dict[str, Any]],
    key: Tuple[int, int, int],
    batch_size: int,
    packed: SimpleNamespace,
    model,
) -> Dict[str, Any]:
    full_key = (key[0], key[1], key[2], batch_size)
    state = cache.get(full_key)
    if state is not None:
        return state

    tensors: Dict[str, torch.Tensor] = {}
    for attr in ("x_nodes", "node_mask", "node_type", "edge_index", "edge_attr", "edge_mask", "seq_idx", "seq_mask", "g_token"):
        value = getattr(packed, attr)
        tensors[attr] = value.clone()

    static_ns = SimpleNamespace(**tensors)
    if tensors["x_nodes"].is_cuda:
        # Warm up once outside of the capture so modules that lazily allocate
        # device state (e.g. v2's adaptive global projection) are materialised
        # before we enter CUDA graph capture. Otherwise CUDA forbids those ops
        # during capture and we observe "operation not permitted" failures.
        with torch.cuda.device(tensors["x_nodes"].device):
            with torch.no_grad():
                model(static_ns)
            torch.cuda.synchronize(tensors["x_nodes"].device)
    else:
        pass
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logits, _ = model(static_ns)
    state = dict(graph=graph, tensors=tensors, output=logits)
    cache[full_key] = state
    return state


def _run_v2_batch(
    decoder,
    bucket: Tuple[int, int, int],
    batch: SimpleNamespace,
) -> torch.Tensor:
    device = decoder.device
    pack_device = SimpleNamespace(
        x_nodes=_to_device(batch.x_nodes, device),
        node_mask=_to_device(batch.node_mask, device),
        node_type=_to_device(batch.node_type, device),
        edge_index=_to_device(batch.edge_index, device),
        edge_attr=_to_device(batch.edge_attr, device),
        edge_mask=_to_device(batch.edge_mask, device),
        seq_idx=_to_device(batch.seq_idx, device),
        seq_mask=_to_device(batch.seq_mask, device),
        g_token=_to_device(batch.g_token, device),
        batch_size=batch.batch_size,
        nodes_pad=batch.nodes_pad,
    )

    if device.type == "cuda" and decoder._graph_capture_enabled:
        try:
            state = _get_v2_graph_state(decoder._v2_graph_states, bucket, batch.batch_size, pack_device, decoder.model)
            for name, tensor in state["tensors"].items():
                tensor.copy_(getattr(pack_device, name))
            state["graph"].replay()
            logits = state["output"]
            decoder._last_graph_used = True
            return logits
        except RuntimeError:
            # CUDA graph capture may fail for certain dynamic control-flow
            # patterns. Disable capture for the remainder of the session and
            # fall back to eager execution so validation can proceed.
            decoder._graph_capture_enabled = False
            decoder._v2_graph_states.clear()
            decoder._last_graph_used = False

    logits, _ = decoder.model(pack_device)
    return logits


class MGHDDecoderPublic:
    """Thin wrapper around the training MGHD module for inference."""

    def __init__(
        self,
        ckpt_path: str,
        cfg: MGHDConfig = None,
        device: str = "cpu",
        *,
        expert: str = "auto",
        graph_capture: bool = False,
        max_batch_d3: int = 512,
    ):
        self.expert = expert
        
        # Auto-detect model version if expert='auto'
        if expert == "auto":
            self.model, self.load_info = load_mghd_auto(ckpt_path, cfg, device=device)
            self.model_version = self.load_info["version"]
        elif expert == "v2":
            self.model, self.load_info = load_mghd_auto(ckpt_path, cfg, device=device)
            if self.load_info.get("version") != "v2":
                raise ValueError("Requested expert='v2' but checkpoint is not a v2 model")
            self.model_version = "v2"
        else:
            # Legacy path: assume v1 if not auto/v2
            if cfg is None:
                cfg = MGHDConfig(n_checks=72, n_qubits=72, n_node_inputs=4)
            self.model, self.load_info = load_mghd_checkpoint(ckpt_path, cfg, device=device)
            self.model_version = "v1"
            self.load_info["version"] = "v1"
        
        self.model.to(device).eval()
        self.device = torch.device(device)
        
        # Use config from load_info if available, otherwise use provided/default
        if cfg is None:
            # Try to get config from model or use defaults
            if hasattr(self.model, 'cfg'):
                self.cfg = self.model.cfg
            else:
                self.cfg = MGHDConfig(n_checks=72, n_qubits=72, n_node_inputs=4)
        else:
            self.cfg = cfg
            
        self._bound = False
        self._Hz = None
        self._Hx = None
        self._H_combined = None
        self._n_z = None
        self._n_x = None
        self._graph_capture_enabled = bool(graph_capture) and self.device.type == "cuda"
        self._graph_max_batch = int(max(1, max_batch_d3))
        self._graph_state = None  # lazily created for fixed d=3 batching
        self._v2_graph_states: Dict[Tuple[int, int, int, int], Dict[str, Any]] = {}
        self._last_graph_used = False

    def bind_code(self, Hx: sp.csr_matrix, Hz: sp.csr_matrix) -> None:
        self._Hx = Hx.tocsr()
        self._Hz = Hz.tocsr()
        self._n_x = int(self._Hx.shape[0])
        self._n_z = int(self._Hz.shape[0])
        self._H_combined = sp.vstack([self._Hz, self._Hx]).tocsr()
        
        # Handle v1/v2 differences in code binding
        if self.model_version == "v1":
            # v1 models need authoritative matrices and static indices
            self.model.set_authoritative_mats(self._Hx.toarray(), self._Hz.toarray(), device=self.device)
            self.model._ensure_static_indices(self.device)
        else:  # v2
            # v2 models have compatibility shims but don't require these operations
            if hasattr(self.model, 'set_authoritative_mats'):
                self.model.set_authoritative_mats(self._Hx.toarray(), self._Hz.toarray(), device=self.device)
        
        self._bound = True

    def _device_info(self) -> Dict[str, object]:
        info: Dict[str, object] = {
            "device": str(self.device),
            "model_version": self.model_version,
            "expert_mode": self.expert,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        }
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            info.update(
                name=props.name,
                multi_processor_count=int(props.multi_processor_count),
                sm=f"{props.major}.{props.minor}",
            )
        else:
            info.setdefault("name", str(self.device))
        return info

    def _call_model(self, feats: Dict[str, torch.Tensor] | PackedCrop) -> torch.Tensor:
        if isinstance(feats, PackedCrop):
            logits, _ = self.model(feats)
            return logits

        if self.model_version == "v2":
            raise RuntimeError("PackedCrop features required for MGHD v2 model")

        # v1 model calling (original logic)
        m = self.model
        sig = inspect.signature(m.forward)
        params = list(sig.parameters.keys())

        node_inputs = feats["node_inputs"].to(self.device)
        if node_inputs.ndim == 3:
            node_inputs = node_inputs.view(-1, node_inputs.shape[-1])

        node_inputs_flat = feats["node_inputs_flat"].to(self.device)

        bundle = {
            "node_inputs": node_inputs,
            "node_inputs_flat": node_inputs_flat,
            "src_ids": feats["src_ids"].to(self.device),
            "dst_ids": feats["dst_ids"].to(self.device),
        }
        if feats.get("edge_index") is not None:
            bundle["edge_index"] = feats["edge_index"].to(self.device)
        if feats.get("edge_attr") is not None:
            bundle["edge_attr"] = feats["edge_attr"].to(self.device)

        inter = {k: bundle[k] for k in bundle if k in params}
        try:
            if inter:
                return m(**inter)
        except TypeError:
            pass

        try:
            return m({k: bundle[k] for k in bundle})
        except TypeError:
            pass

        for order in ("node_inputs_flat", "node_inputs"):
            if all(k in bundle for k in (order, "src_ids", "dst_ids")):
                try:
                    return m(bundle[order], bundle["src_ids"], bundle["dst_ids"])
                except TypeError:
                    continue

        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            try:
                return m(**bundle)
            except TypeError:
                pass

        raise RuntimeError(
            "Cannot call RawMGHD.forward with provided features. "
            f"Forward params = {params}; provided keys = {list(bundle.keys())}"
        )

    def _get_d3_base(self) -> Dict[str, torch.Tensor]:
        if not getattr(self, "_bound", False):
            raise RuntimeError("MGHDDecoderPublic must be bound before building d=3 features")
        if getattr(self, "_d3_base", None) is None:
            zeros = np.zeros(self.cfg.n_checks, dtype=np.float32)
            self._d3_base = features_rotated_d3(
                self._H_combined,
                zeros,
                n_checks=self.cfg.n_checks,
                n_qubits=self.cfg.n_qubits,
                n_node_inputs=self.cfg.n_node_inputs,
            )
        return self._d3_base

    def _init_mb_report(self) -> Dict[str, Any]:
        return {
            "fast_path_batches": 0,
            "fixed_d3_batches": 0,
            "fallback_loops": 0,
            "batch_sizes": [],
            "graph_used": False,
            "graph_used_shots": 0,
            "device": self._device_info(),
        }

    def _fill_d3_checks(
        self,
        node_inputs: torch.Tensor,
        syndromes: List[np.ndarray],
        *,
        bucket: str | None,
    ) -> None:
        node_inputs.zero_()
        if not syndromes:
            return
        checks = self.cfg.n_checks
        half = checks // 2
        bucket_norm = bucket.upper() if isinstance(bucket, str) else None
        device = node_inputs.device
        for idx, syn in enumerate(syndromes):
            s_arr = np.asarray(syn, dtype=np.float32).ravel()
            if s_arr.size == checks:
                node_inputs[idx, :checks, 0] = torch.from_numpy(s_arr).to(device)
            elif s_arr.size == half and bucket_norm in {"X", "Z"}:
                if bucket_norm == "Z":
                    node_inputs[idx, :half, 0] = torch.from_numpy(s_arr).to(device)
                else:
                    node_inputs[idx, half : half + s_arr.size, 0] = torch.from_numpy(s_arr).to(device)
            else:
                raise ValueError(
                    "Syndrome shape mismatch for d=3 fallback: "
                    f"expected {checks} or {half}, got {s_arr.size}"
                )

    def _prepare_d3_batch(
        self,
        syndromes: List[np.ndarray],
        *,
        bucket: str | None,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        base = self._get_d3_base()
        nodes = int(self.cfg.n_checks + self.cfg.n_qubits)
        feats = int(self.cfg.n_node_inputs)
        batch = len(syndromes)
        node_inputs = torch.zeros((batch, nodes, feats), dtype=torch.float32, device=device)
        self._fill_d3_checks(node_inputs, syndromes, bucket=bucket)
        node_inputs_flat = node_inputs.view(-1, feats)
        base_src = base["src_ids"].to(device)
        base_dst = base["dst_ids"].to(device)
        if batch == 0:
            src_ids = torch.empty(0, dtype=torch.long, device=device)
            dst_ids = torch.empty(0, dtype=torch.long, device=device)
        else:
            offsets = torch.arange(batch, device=device, dtype=torch.long) * nodes
            src_ids = torch.cat([base_src + off for off in offsets], dim=0)
            dst_ids = torch.cat([base_dst + off for off in offsets], dim=0)
        return node_inputs, node_inputs_flat, src_ids, dst_ids

    def _extract_probs(self, raw: torch.Tensor, batch: int, temp: float) -> List[np.ndarray]:
        nodes = int(self.cfg.n_checks + self.cfg.n_qubits)
        probs_list: List[np.ndarray] = []
        for i in range(batch):
            offset = i * nodes
            logits = self._normalize_logits(
                raw,
                int(self.cfg.n_checks),
                int(self.cfg.n_qubits),
                offset=offset,
            )
            logit_diff = (logits[:, 1] - logits[:, 0]) / float(temp)
            p = torch.sigmoid(logit_diff).clamp(1e-6, 1 - 1e-6)
            probs_list.append(p.detach().cpu().numpy().astype(np.float64))
        return probs_list

    def _build_graph_state(self, capacity: int) -> None:
        nodes = int(self.cfg.n_checks + self.cfg.n_qubits)
        feats = int(self.cfg.n_node_inputs)
        device = self.device
        node_inputs = torch.zeros((capacity, nodes, feats), dtype=torch.float32, device=device)
        node_inputs_flat = node_inputs.view(-1, feats)
        base = self._get_d3_base()
        base_src = base["src_ids"].to(device)
        base_dst = base["dst_ids"].to(device)
        offsets = torch.arange(capacity, device=device, dtype=torch.long) * nodes
        src_ids = torch.cat([base_src + off for off in offsets], dim=0)
        dst_ids = torch.cat([base_dst + off for off in offsets], dim=0)
        batch = {
            "node_inputs": node_inputs,
            "node_inputs_flat": node_inputs_flat,
            "src_ids": src_ids,
            "dst_ids": dst_ids,
        }
        torch.cuda.synchronize(device)
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            _ = self._call_model(batch)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = self._call_model(batch)
        self._graph_state = {
            "capacity": capacity,
            "node_inputs": node_inputs,
            "node_inputs_flat": node_inputs_flat,
            "src_ids": src_ids,
            "dst_ids": dst_ids,
            "graph": graph,
            "graph_output": graph_output,
        }

    def _priors_from_d3_eager(
        self,
        syndromes: List[np.ndarray],
        *,
        temp: float,
        bucket: str | None,
    ) -> Tuple[List[np.ndarray], bool]:
        batch = len(syndromes)
        if batch == 0:
            return [], False
        node_inputs, node_inputs_flat, src_ids, dst_ids = self._prepare_d3_batch(
            syndromes, bucket=bucket, device=self.device
        )
        raw = self._call_model(
            {
                "node_inputs": node_inputs,
                "node_inputs_flat": node_inputs_flat,
                "src_ids": src_ids,
                "dst_ids": dst_ids,
            }
        )
        return self._extract_probs(raw, batch, temp), False

    def _priors_from_d3_graph(
        self,
        syndromes: List[np.ndarray],
        *,
        temp: float,
        bucket: str | None,
    ) -> Tuple[List[np.ndarray], bool]:
        batch = len(syndromes)
        if batch == 0 or not self._graph_capture_enabled or self.device.type != "cuda":
            return self._priors_from_d3_eager(syndromes, temp=temp, bucket=bucket)
        capacity = batch
        if self._graph_state is None or capacity > self._graph_state["capacity"]:
            if capacity > self._graph_max_batch:
                return self._priors_from_d3_eager(syndromes, temp=temp, bucket=bucket)
            self._build_graph_state(capacity)
        state = self._graph_state
        node_inputs = state["node_inputs"]
        self._fill_d3_checks(node_inputs, syndromes, bucket=bucket)
        if batch < state["capacity"]:
            node_inputs[batch:, :, :].zero_()
        torch.cuda.synchronize(self.device)
        state["graph"].replay()
        torch.cuda.synchronize(self.device)
        raw = state["graph_output"]
        return self._extract_probs(raw, batch, temp), True

    def _normalize_logits(self, raw: torch.Tensor, n_checks: int, n_qubits: int, *, offset: int = 0) -> torch.Tensor:
        out = raw[-1] if isinstance(raw, (list, tuple)) else raw

        # Handle different tensor shapes
        if out.ndim == 3:
            if out.shape[0] == 1:
                out = out.squeeze(0)  # Remove batch dimension if present
            else:
                # For (iters, nodes, channels), take the last iteration
                out = out[-1]
        
        if out.ndim != 2:
            raise AssertionError(f"Expected [nodes, C] after normalization, got {tuple(out.shape)}")
        
        nodes, C = out.shape
        start = int(offset)
        stop = start + n_checks + n_qubits
        if stop > nodes:
            raise AssertionError(f"Requested slice [{start}:{stop}] exceeds tensor with {nodes} nodes")
        segment = out[start:stop, :]
        qubit_logits = segment[n_checks:n_checks + n_qubits, :]
        
        if C == 2:
            return qubit_logits
        if C == 1:
            return torch.stack([-qubit_logits[:, 0], qubit_logits[:, 0]], dim=1)
        if C == 4:
            flip = qubit_logits[:, 1]
            noflip = torch.logsumexp(torch.stack([qubit_logits[:, 0], qubit_logits[:, 2], qubit_logits[:, 3]], dim=1), dim=1)
            return torch.stack([noflip, flip], dim=1)
        raise AssertionError(f"Unexpected channel dim {C}")

    @torch.no_grad()
    def priors_from_syndrome(self, s: np.ndarray, *, side: str) -> np.ndarray:
        if not self._bound:
            raise RuntimeError("Call bind_code(Hx, Hz) before requesting priors.")
        side = side.upper()
        if side not in {"X", "Z"}:
            raise ValueError("side must be 'X' or 'Z'")

        if self.model_version == "v2":
            raise NotImplementedError("priors_from_syndrome is not supported for MGHD v2 models")

        vec = np.zeros(self._n_z + self._n_x, dtype=np.float32)
        if side == "Z":
            vec[: self._n_z] = np.asarray(s, dtype=np.float32).ravel()
        else:
            vec[self._n_z :] = np.asarray(s, dtype=np.float32).ravel()

        feats = features_rotated_d3(
            self._H_combined,
            vec,
            n_checks=self.cfg.n_checks,
            n_qubits=self.cfg.n_qubits,
            n_node_inputs=self.cfg.n_node_inputs,
        )
        raw = self._call_model(feats)
        logits = self._normalize_logits(raw, self.cfg.n_checks, self.cfg.n_qubits)
        logit_diff = logits[:, 1] - logits[:, 0]
        probs = torch.sigmoid(logit_diff).clamp(1e-6, 1 - 1e-6).detach().cpu().numpy()
        assert probs.shape[0] == self.cfg.n_qubits, f"probs len {probs.shape[0]} != n_qubits {self.cfg.n_qubits}"
        return probs.astype(np.float64)

    @torch.no_grad()
    def priors_from_subgraph(self, H_sub: sp.csr_matrix, s_sub: np.ndarray, *, temp: float = 1.0) -> np.ndarray:
        if self.model_version == "v2":
            raise NotImplementedError("Use priors_from_subgraphs_batched with metadata for MGHD v2 models")
        feats = features_from_subgraph(H_sub, s_sub, n_node_inputs=self.cfg.n_node_inputs)
        raw = self._call_model(feats)
        m_sub = int(feats["n_checks"]); n_sub = int(feats["n_qubits"])
        logits = self._normalize_logits(raw, m_sub, n_sub)
        logit_diff = (logits[:,1] - logits[:,0]) / float(temp)
        probs = torch.sigmoid(logit_diff).clamp(1e-6, 1-1e-6).detach().cpu().numpy()
        return probs.astype(np.float64)

    @torch.no_grad()
    def priors_from_d3_fullgraph_batched(
        self,
        syndromes: List[np.ndarray],
        *,
        temp: float = 1.0,
        bucket: str | None = None,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        report = self._init_mb_report()
        batch = len(syndromes)
        if batch == 0:
            report["fixed_d3_batches"] = 0
            report["batch_sizes"] = []
            return [], report

        probs_list, graph_used = self._priors_from_d3_graph(
            syndromes,
            temp=temp,
            bucket=bucket,
        )
        report["fixed_d3_batches"] = 1
        report["batch_sizes"].append(batch)
        report["graph_used"] = graph_used
        return probs_list, report

    @torch.no_grad()
    def priors_from_subgraphs_batched(
        self,
        items: List[
            Tuple[sp.csr_matrix, np.ndarray, np.ndarray]
            | Tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray]
        ],
        *,
        temp: float = 1.0,
        bucket: str | None = None,
        use_masked_fullgraph_fallback: bool = True,
        bucket_spec: Sequence[Tuple[int, int, int]] | None = None,
        microbatch: int = 64,
        flush_ms: float = 1.0,
        use_graphs: bool = True,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Return per-cluster priors using a single MGHD forward when possible."""

        if not items:
            return [], self._init_mb_report()

        processed: List[Tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray | None]] = []
        extras: List[Dict[str, Any] | None] = []
        feats: List[Dict[str, torch.Tensor]] = []

        for entry in items:
            if len(entry) == 3:
                H_sub, s_sub, q_l2g = entry  # type: ignore[misc]
                c_l2g = None
                extra = None
            elif len(entry) >= 5:
                H_sub, s_sub, q_l2g, c_l2g, extra = entry[:5]  # type: ignore[misc]
            elif len(entry) >= 4:
                H_sub, s_sub, q_l2g, c_l2g = entry[:4]  # type: ignore[misc]
                extra = None
            else:
                raise ValueError("Each item must provide at least (H_sub, s_sub, q_l2g)")

            if not sp.issparse(H_sub):
                raise TypeError("H_sub must be a scipy sparse matrix")

            processed.append(
                (
                    H_sub,
                    s_sub,
                    np.asarray(q_l2g, dtype=np.int64),
                    None if c_l2g is None else np.asarray(c_l2g, dtype=np.int64),
                )
            )
            extras.append(extra)
            if self.model_version != "v2":
                feats.append(features_from_subgraph(H_sub, s_sub, n_node_inputs=self.cfg.n_node_inputs))

        report = self._init_mb_report()
        batch_size = len(processed)
        report["batch_sizes"].append(batch_size)
        if self.model_version == "v2":
            probs_list, v2_report = self._priors_from_subgraphs_v2(
                processed,
                extras,
                temp=temp,
                bucket_spec=bucket_spec,
                microbatch=microbatch,
                flush_ms=flush_ms,
                use_graphs=use_graphs,
            )
            report.update({k: v2_report.get(k, report.get(k)) for k in v2_report})
            return probs_list, report
        try:
            batch = _collate_flat_batches(feats)
            meta = batch.pop("meta")
            raw = self._call_model(batch)
            probs_list: List[np.ndarray] = []
            for (m_sub, n_sub, node_base), (_H_sub, _s_sub, _q_l2g, _c_l2g) in zip(meta, processed):
                logits = self._normalize_logits(raw, m_sub, n_sub, offset=node_base)
                logit_diff = (logits[:, 1] - logits[:, 0]) / float(temp)
                p = torch.sigmoid(logit_diff).clamp(1e-6, 1 - 1e-6)
                probs_list.append(p.detach().cpu().numpy().astype(np.float64))
            report["fast_path_batches"] = 1
            return probs_list, report
        except Exception:
            if not use_masked_fullgraph_fallback:
                raise

            total_checks = int(self.cfg.n_checks)
            half = total_checks // 2
            syndromes_full: List[np.ndarray] = []
            for _H_sub, s_sub, _q_l2g, c_l2g in processed:
                s_full = np.zeros(total_checks, dtype=np.float32)
                s_vals = np.asarray(s_sub, dtype=np.float32)
                c_idx = np.asarray(c_l2g, dtype=np.int64)
                if bucket is None:
                    s_full[c_idx] = s_vals
                else:
                    side = bucket.upper()
                    if side == "Z":
                        s_full[c_idx] = s_vals
                    elif side == "X":
                        s_full[c_idx + half] = s_vals
                    else:
                        raise ValueError(f"Unknown bucket '{bucket}' for masked fallback")
                syndromes_full.append(s_full)

            probs_list, fallback_report = self.priors_from_d3_fullgraph_batched(
                syndromes_full,
                temp=temp,
                bucket=None,
            )
            final_probs: List[np.ndarray] = []
            for probs_full, (_H_sub, _s_sub, q_l2g, _c_l2g) in zip(probs_list, processed):
                final_probs.append(np.asarray(probs_full, dtype=np.float64)[q_l2g])
            fallback_report["batch_sizes"] = [batch_size]
            return final_probs, fallback_report

        raise RuntimeError("priors_from_subgraphs_batched failed to produce probabilities")

    def _move_packed_crop(self, packed: PackedCrop, device: torch.device) -> PackedCrop:
        tensor_fields = (
            "x_nodes",
            "node_mask",
            "node_type",
            "edge_index",
            "edge_attr",
            "edge_mask",
            "seq_idx",
            "seq_mask",
            "g_token",
            "y_bits",
        )
        for field in tensor_fields:
            value = getattr(packed, field, None)
            if torch.is_tensor(value):
                setattr(packed, field, value.to(device))
        return packed

    def _priors_from_subgraphs_v2(
        self,
        processed: List[Tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray | None]],
        extras: List[Dict[str, Any] | None],
        *,
        temp: float,
        bucket_spec: Sequence[Tuple[int, int, int]] | None,
        microbatch: int,
        flush_ms: float,
        use_graphs: bool,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        if any(extra is None for extra in extras):
            raise ValueError("V2 model requires geometry metadata for every subgraph")

        if not processed:
            return [], self._init_mb_report()

        device = self.device
        use_graphs = bool(use_graphs and self._graph_capture_enabled and device.type == "cuda")

        if not bucket_spec:
            buckets = sorted({_v2_bucket_key(entry[0]) for entry in processed})
            bucket_spec = buckets
        bucket_spec = tuple(bucket_spec)

        batcher = CrossShotBatcher(
            bucket_spec=bucket_spec,
            microbatch=microbatch,
            flush_ms=flush_ms,
            pin_host=True,
        )

        report = self._init_mb_report()
        report.setdefault("bucket_histogram", {})
        report.setdefault("graph_used_shots", 0)

        final_probs: List[np.ndarray | None] = [None] * len(processed)

        class _BucketContext:
            def __init__(self, decoder: "MGHDDecoderPublic", meta: Dict[str, int], enabled: bool) -> None:
                self.decoder = decoder
                self.model = decoder.model
                self.device = decoder.device
                self.meta = meta
                self.enabled = enabled
                self.static_inputs: SimpleNamespace | None = None
                self.static_output: Tuple[torch.Tensor, torch.Tensor] | None = None
                self.graph: torch.cuda.CUDAGraph | None = None
                self.graph_used = 0

            def ensure(self) -> None:
                if not self.enabled or self.graph is not None:
                    return
                try:
                    self.static_inputs = self.model.allocate_static_batch(
                        batch_size=self.meta["batch_size"],
                        nodes_pad=self.meta["nodes_pad"],
                        edges_pad=self.meta["edges_pad"],
                        seq_pad=self.meta["seq_pad"],
                        feat_dim=self.meta["feat_dim"],
                        edge_feat_dim=self.meta["edge_feat_dim"],
                        g_dim=self.meta["g_dim"],
                        device=self.device,
                    )
                    warm_logits, warm_mask = self.model(self.static_inputs)
                    torch.cuda.synchronize(self.device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        logits_cap, mask_cap = self.model(self.static_inputs)
                    self.static_output = (logits_cap, mask_cap)
                    self.graph = graph
                except Exception:
                    self.enabled = False
                    self.static_inputs = None
                    self.static_output = None
                    self.graph = None

            def run(self, host_batch: SimpleNamespace) -> Tuple[Tuple[torch.Tensor, torch.Tensor], bool]:
                if self.enabled and self.graph is not None and self.static_inputs is not None and self.static_output is not None:
                    self.model.copy_into_static(self.static_inputs, host_batch, non_blocking=True)
                    torch.cuda.synchronize(self.device)
                    self.graph.replay()
                    torch.cuda.synchronize(self.device)
                    self.graph_used += host_batch.batch_size
                    return self.model.gather_from_static(self.static_output), True
                dev_batch = self.model.move_packed_to_device(host_batch, self.device)
                return self.model(dev_batch), False

        bucket_contexts: Dict[Tuple[int, int], _BucketContext] = {}
        report_graph_used = False

        def process_ready(ready: Dict[int, List[PackedCrop]]) -> None:
            nonlocal report_graph_used
            for bucket_id, crops in ready.items():
                if not crops:
                    continue
                host_batch, cluster_infos, meta = stack_crops_pinned(crops, pin=True)
                ctx_key = (bucket_id, meta["batch_size"])
                ctx = bucket_contexts.get(ctx_key)
                if ctx is None:
                    ctx = _BucketContext(self, meta, use_graphs)
                    if use_graphs:
                        ctx.ensure()
                    bucket_contexts[ctx_key] = ctx
                outputs, used_graph = ctx.run(host_batch)
                logits, node_mask = outputs
                self._last_graph_used = used_graph
                cluster_probs_t = self.model.scatter_outputs(logits, cluster_infos, temp=temp)
                for info, probs_tensor in zip(cluster_infos, cluster_probs_t):
                    idx = info.get("record_index")
                    if idx is None:
                        raise RuntimeError("Packed crop is missing record index for scattering")
                    final_probs[idx] = probs_tensor.detach().cpu().numpy().astype(np.float64)

                report["fast_path_batches"] += 1
                report["batch_sizes"].append(len(crops))
                bucket_key = f"{meta['nodes_pad']}-{meta['edges_pad']}-{meta['seq_pad']}"
                hist = report.setdefault("bucket_histogram", {})
                hist[bucket_key] = hist.get(bucket_key, 0) + len(crops)
                if used_graph:
                    report_graph_used = True
                    report["graph_used_shots"] = report.get("graph_used_shots", 0) + len(crops)

        now_ms = lambda: time.monotonic() * 1e3

        for idx, (entry, extra) in enumerate(zip(processed, extras)):
            assert extra is not None
            H_sub, s_sub, q_l2g, c_l2g = entry
            pack = pack_cluster(
                H_sub=H_sub.toarray(),
                xy_qubit=extra["xy_qubit"],
                xy_check=extra["xy_check"],
                synd_Z_then_X_bits=np.asarray(s_sub, dtype=np.uint8),
                k=extra["k"],
                r=extra["r"],
                bbox_xywh=extra["bbox"],
                kappa_stats=extra["kappa_stats"],
                y_bits_local=np.zeros(extra["k"], dtype=np.uint8),
                side=extra["side"],
                d=extra["d"],
                p=extra["p"],
                seed=0,
                N_max=None,
                E_max=None,
                S_max=None,
                bucket_spec=bucket_spec,
                add_jump_edges=False,
            )
            setattr(pack, "_record_index", idx)
            batcher.add(pack)
            ready = batcher.pop_ready_batches()
            if ready:
                process_ready(ready)
            if batcher.flush_due(now_ms()):
                ready = batcher.pop_ready_batches()
                if ready:
                    process_ready(ready)

        for bucket_id, crops in batcher.finalize():
            process_ready({bucket_id: crops})

        if any(prob is None for prob in final_probs):
            raise RuntimeError("priors_from_subgraphs_batched failed to produce probabilities")

        report["graph_used"] = report_graph_used
        return [prob for prob in final_probs if prob is not None], report
