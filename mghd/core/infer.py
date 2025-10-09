"""Minimal MGHD v2 inference helpers."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from .features_v2 import PackedCrop, pack_cluster
from .model_v2 import MGHDv2


TensorFields: Tuple[str, ...] = (
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


def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        if "state_dict" in state:
            return state["state_dict"]
        if "model" in state:
            return state["model"]
    if not isinstance(state, dict):
        raise RuntimeError("Unexpected checkpoint format: expected a state-dict mapping")
    return state


def _ensure_array(array: np.ndarray | sp.csr_matrix) -> np.ndarray:
    if sp.issparse(array):
        return np.asarray(array.toarray(), dtype=np.uint8)
    return np.asarray(array, dtype=np.uint8)


class MGHDDecoderPublic:
    """Thin wrapper that keeps only the MGHD v2 inference surface."""

    def __init__(self, ckpt_path: str, device: str = "cpu", *, profile: str = "S") -> None:
        self.device = torch.device(device)
        state_dict = _load_state_dict(ckpt_path)
        self.model = MGHDv2(profile=profile)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device).eval()

        self.model_version = "v2"
        self._graph_capture_enabled = False
        self._bound = False
        self._Hx: sp.csr_matrix | None = None
        self._Hz: sp.csr_matrix | None = None

    def bind_code(self, Hx: sp.csr_matrix, Hz: sp.csr_matrix) -> None:
        self._Hx = Hx.tocsr()
        self._Hz = Hz.tocsr()
        if hasattr(self.model, "set_authoritative_mats"):
            self.model.set_authoritative_mats(self._Hx.toarray(), self._Hz.toarray(), device=self.device)
        self._bound = True

    def set_message_iters(self, n_iters: int | None) -> None:
        self.model.set_message_iters(n_iters)

    # ----------------------------------------------------------------------
    # Batched cluster inference
    # ----------------------------------------------------------------------
    def priors_from_subgraphs_batched(
        self,
        items: Sequence[Tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]],
        *,
        temp: float = 1.0,
        bucket: str | None = None,
        bucket_spec: Sequence[Tuple[int, int, int]] | None = None,
        microbatch: int = 64,
        flush_ms: float = 1.0,
        use_graphs: bool = False,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        if not self._bound:
            raise RuntimeError("Call bind_code(Hx, Hz) before requesting priors.")
        if not items:
            return [], self._init_report()

        probs: List[np.ndarray] = []
        for entry in items:
            H_sub, s_sub, q_l2g, c_l2g, extra = self._normalize_entry(entry)
            pack = self._build_pack(H_sub, s_sub, extra, bucket_spec)
            pack = self._move_packed_crop(pack, self.device)
            logits, node_mask = self.model(pack)
            data_mask = node_mask & (pack.node_type == 0)
            logits_data = logits[data_mask]
            logit_diff = (logits_data[:, 1] - logits_data[:, 0]) / float(temp)
            local_probs = torch.sigmoid(logit_diff).clamp(1e-6, 1 - 1e-6)
            values = local_probs.detach().cpu().numpy()
            if values.shape[0] != q_l2g.size:
                raise RuntimeError("Probability vector length mismatch for subgraph")
            probs.append(values)

        report = self._init_report()
        report["fast_path_batches"] = 1 if items else 0
        report["batch_sizes"].append(len(items))
        report["bucket_histogram"]["default"] = len(items)
        return probs, report

    def priors_from_syndrome(self, *_args, **_kwargs) -> np.ndarray:  # pragma: no cover - legacy guard
        raise RuntimeError("MGHD v2 decoder only supports packed crop inference")

    def _init_mb_report(self) -> Dict[str, Any]:
        return self._init_report()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _normalize_entry(
        self,
        entry: Tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]],
    ) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        if len(entry) < 5:
            raise ValueError("Each entry must include (H_sub, s_sub, q_l2g, c_l2g, meta)")
        H_sub, s_sub, q_l2g, c_l2g, meta = entry[:5]
        if meta is None:
            raise ValueError("Geometry metadata is required for MGHD v2 crops")
        q_l2g_arr = np.asarray(q_l2g, dtype=np.int64)
        c_l2g_arr = np.asarray(c_l2g, dtype=np.int64)
        return H_sub, np.asarray(s_sub, dtype=np.uint8), q_l2g_arr, c_l2g_arr, meta

    def _build_pack(
        self,
        H_sub: sp.csr_matrix,
        s_sub: np.ndarray,
        meta: Dict[str, Any],
        bucket_spec: Sequence[Tuple[int, int, int]] | None,
    ) -> PackedCrop:
        H_dense = _ensure_array(H_sub)
        n_checks, n_qubits = H_dense.shape
        edges = int(H_dense.sum())
        kappa_stats = dict(meta.get("kappa_stats", {}))
        if "size" not in kappa_stats:
            kappa_stats["size"] = float(n_checks + n_qubits)

        pack = pack_cluster(
            H_sub=H_dense,
            xy_qubit=np.asarray(meta["xy_qubit"], dtype=np.int32),
            xy_check=np.asarray(meta.get("xy_check", np.zeros((0, 2), dtype=np.int32))),
            synd_Z_then_X_bits=s_sub,
            k=int(meta.get("k", n_qubits)),
            r=int(meta.get("r", 0)),
            bbox_xywh=tuple(int(v) for v in meta.get("bbox", (0, 0, 1, 1))),
            kappa_stats=kappa_stats,
            y_bits_local=np.zeros(n_qubits, dtype=np.uint8),
            side=str(meta.get("side", "Z")),
            d=int(meta.get("d", 3)),
            p=float(meta.get("p", 0.0)),
            seed=0,
            N_max=n_qubits + n_checks,
            E_max=max(edges, 1),
            S_max=max(n_checks, 1),
            bucket_spec=bucket_spec,
            add_jump_edges=False,
        )
        return pack

    def _move_packed_crop(self, pack: PackedCrop, device: torch.device) -> PackedCrop:
        for name in TensorFields:
            value = getattr(pack, name, None)
            if torch.is_tensor(value):
                setattr(pack, name, value.to(device, non_blocking=True))
        return pack

    def _device_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {"device": str(self.device)}
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            info.update(name=props.name, sm=f"{props.major}.{props.minor}")
        return info

    def _init_report(self) -> Dict[str, Any]:
        return {
            "fast_path_batches": 0,
            "fixed_d3_batches": 0,
            "fallback_loops": 0,
            "batch_sizes": [],
            "graph_used": False,
            "bucket_histogram": {},
            "device": self._device_info(),
        }


def warmup_and_capture(*_args, **_kwargs) -> Dict[str, Any]:
    """Compatibility stub retained for legacy call sites."""
    return {"warmup_us": 0.0, "graph_used": False, "path": "vanilla"}


__all__ = ["MGHDDecoderPublic", "warmup_and_capture"]
