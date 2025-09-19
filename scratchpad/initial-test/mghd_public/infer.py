"""Public MGHD inference helper for rotated d=3."""

from __future__ import annotations

import inspect
import numpy as np
import scipy.sparse as sp
import torch
from typing import Dict

from .config import MGHDConfig
from .model import load_mghd_checkpoint
from .features import features_rotated_d3, features_from_subgraph


class MGHDDecoderPublic:
    """Thin wrapper around the training MGHD module for inference."""

    def __init__(self, ckpt_path: str, cfg: MGHDConfig, device: str = "cpu"):
        self.model, self.load_info = load_mghd_checkpoint(ckpt_path, cfg, device=device)
        self.model.to(device).eval()
        self.device = torch.device(device)
        self.cfg = cfg
        self._bound = False
        self._Hz = None
        self._Hx = None
        self._H_combined = None
        self._n_z = None
        self._n_x = None

    def bind_code(self, Hx: sp.csr_matrix, Hz: sp.csr_matrix) -> None:
        self._Hx = Hx.tocsr()
        self._Hz = Hz.tocsr()
        self._n_x = int(self._Hx.shape[0])
        self._n_z = int(self._Hz.shape[0])
        self._H_combined = sp.vstack([self._Hz, self._Hx]).tocsr()
        self.model.set_authoritative_mats(self._Hx.toarray(), self._Hz.toarray(), device=self.device)
        self.model._ensure_static_indices(self.device)
        self._bound = True

    def _call_model(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
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

    def _normalize_logits(self, raw: torch.Tensor, n_checks: int, n_qubits: int) -> torch.Tensor:
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
        assert nodes >= n_checks + n_qubits, f"nodes={nodes} < {n_checks + n_qubits}"
        qubit_logits = out[n_checks:n_checks + n_qubits, :]
        
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
        feats = features_from_subgraph(H_sub, s_sub, n_node_inputs=self.cfg.n_node_inputs)
        raw = self._call_model(feats)
        m_sub = int(feats["n_checks"]); n_sub = int(feats["n_qubits"])
        logits = self._normalize_logits(raw, m_sub, n_sub)
        logit_diff = (logits[:,1] - logits[:,0]) / float(temp)
        probs = torch.sigmoid(logit_diff).clamp(1e-6, 1-1e-6).detach().cpu().numpy()
        return probs.astype(np.float64)
