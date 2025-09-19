"""Public feature pipeline for rotated d=3 MGHD inference."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from typing import Dict, Tuple


def tanner_from_H(H: sp.csr_matrix, n_checks: int, n_qubits: int) -> Tuple[torch.Tensor, torch.Tensor]:
    H = H.tocoo()
    m, n = H.shape
    assert m == n_checks and n == n_qubits

    src_f = H.row
    dst_f = H.col + m
    src_b = H.col + m
    dst_b = H.row

    src = torch.from_numpy(np.concatenate([src_f, src_b]).astype(np.int64))
    dst = torch.from_numpy(np.concatenate([dst_f, dst_b]).astype(np.int64))
    return src, dst


def features_rotated_d3(
    H_side: sp.csr_matrix,
    s: np.ndarray,
    *,
    n_checks: int | None = None,
    n_qubits: int | None = None,
    n_node_inputs: int = 9,
) -> Dict[str, torch.Tensor]:
    n_checks = int(H_side.shape[0]) if n_checks is None else n_checks
    n_qubits = int(H_side.shape[1]) if n_qubits is None else n_qubits

    s = np.asarray(s, dtype=np.float32).ravel()
    assert s.size == n_checks, f"syndrome length {s.size} != n_checks {n_checks}"
    nodes = n_checks + n_qubits

    node_inputs = torch.zeros((1, nodes, n_node_inputs), dtype=torch.float32)
    node_inputs[0, :n_checks, 0] = torch.from_numpy(s)

    node_inputs_flat = node_inputs.view(-1, n_node_inputs)

    src_ids, dst_ids = tanner_from_H(H_side, n_checks, n_qubits)
    edge_index = torch.stack([src_ids, dst_ids], dim=0)

    return {
        "node_inputs": node_inputs,
        "node_inputs_flat": node_inputs_flat,
        "src_ids": src_ids,
        "dst_ids": dst_ids,
        "edge_index": edge_index,
        "edge_attr": None,
        "n_checks": torch.tensor(n_checks),
        "n_qubits": torch.tensor(n_qubits),
    }


def features_from_subgraph(H_sub: sp.csr_matrix, s_sub: np.ndarray, *, n_node_inputs: int = 9) -> Dict[str, torch.Tensor]:
    """
    Generic (m_sub checks, n_sub qubits):
      - node order: checks first, then qubits
      - channel 0 on check nodes carries the sub-syndrome bits
      - build C<->Q and Q<->C edges from H_sub
    """
    m_sub, n_sub = H_sub.shape
    s = np.asarray(s_sub, dtype=np.float32).ravel()
    assert s.size == m_sub
    nodes = m_sub + n_sub
    x = torch.zeros((1, nodes, n_node_inputs), dtype=torch.float32)
    x[0, :m_sub, 0] = torch.from_numpy(s)
    x_flat = x.view(-1, n_node_inputs)
    # edges
    Hc = H_sub.tocoo()
    src_f = Hc.row
    dst_f = Hc.col + m_sub
    src_b = Hc.col + m_sub
    dst_b = Hc.row
    src = torch.from_numpy(np.concatenate([src_f, src_b]).astype(np.int64))
    dst = torch.from_numpy(np.concatenate([dst_f, dst_b]).astype(np.int64))
    edge_index = torch.stack([src, dst], dim=0)
    return {
        "node_inputs": x,
        "node_inputs_flat": x_flat,
        "src_ids": src,
        "dst_ids": dst,
        "edge_index": edge_index,
        "edge_attr": None,
        "n_checks": torch.tensor(m_sub),
        "n_qubits": torch.tensor(n_sub),
    }
