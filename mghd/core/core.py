"""Consolidated MGHD v2 runtime stack (features, model, decoder)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Lightweight config dataclass
# ---------------------------------------------------------------------------


@dataclass
class MGHDConfig:
    """Minimal configuration for reconstructing an MGHDv2 instance.

    Fields
    - gnn: keyword args for the graph message-passing core
    - mamba: keyword args for the sequence encoder (state-space model)
    - profile: model size/profile tag (e.g., "S")
    - dist: nominal code distance used for defaults (metadata only)
    - n_qubits/n_checks: reference sizes (metadata only)
    - n_node_inputs/n_node_outputs: feature and head sizes
    """

    gnn: Dict[str, Any]
    mamba: Dict[str, Any]
    profile: str = "S"
    dist: int = 3
    n_qubits: int = 9
    n_checks: int = 8
    n_node_inputs: int = 9
    n_node_outputs: int = 2  # binary head for rotated d=3

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def to_dict(cfg: MGHDConfig) -> Dict[str, Any]:
    """Backward compatible helper returning the dataclass fields as a dict."""
    return cfg.to_dict()


# ---------------------------------------------------------------------------
# Minimal CSS helpers (moved from codes.pcm_real for consolidation)
# ---------------------------------------------------------------------------


def rotated_surface_pcm(d: int, side: str) -> sp.csr_matrix:
    """Return rotated surface code parity-check matrix for odd distance d.

    Constructs the proper rotated planar surface code where stabilizers are
    centered on plaquettes (at half-integer lattice coordinates), not on data qubits.

    Interior stabilizers have weight 4, boundary stabilizers have weight 2.
    The stabilizer type alternates in a checkerboard pattern on the dual lattice.

    For X-memory configuration:
    - X stabilizers (detect Z errors): on rough boundaries (left/right) + interior
    - Z stabilizers (detect X errors): on smooth boundaries (top/bottom) + interior
    """
    if d % 2 == 0 or d < 1:
        raise ValueError("rotated_surface_pcm requires odd d >= 1")

    side = side.upper()
    if side not in {"X", "Z"}:
        raise ValueError("side must be 'X' or 'Z'")

    n_qubits = d * d
    # For rotated surface code: (d-1)^2/2 interior + boundary stabilizers
    # Total stabilizers = d^2 - 1 (split evenly between X and Z)
    n_checks = (d * d - 1) // 2

    rows: List[int] = []
    cols: List[int] = []
    row_idx = 0

    def q_index(r: int, c: int) -> int:
        return r * d + c

    # Interior plaquettes: weight-4 stabilizers
    for r in range(d - 1):
        for c in range(d - 1):
            parity = (r + c) % 2
            # X stabilizers at even parity, Z at odd
            is_x_type = parity == 0
            if (side == "X" and is_x_type) or (side == "Z" and not is_x_type):
                qubits = [
                    q_index(r, c),
                    q_index(r + 1, c),
                    q_index(r, c + 1),
                    q_index(r + 1, c + 1),
                ]
                for q in qubits:
                    rows.append(row_idx)
                    cols.append(q)
                row_idx += 1

    # Boundary stabilizers: weight-2
    if side == "Z":
        # Z stabilizers on smooth boundaries (top and bottom)
        # Top: where parity at virtual row -1 would be odd
        for c in range(d - 1):
            if ((-1) + c) % 2 != 0:
                for q in [q_index(0, c), q_index(0, c + 1)]:
                    rows.append(row_idx)
                    cols.append(q)
                row_idx += 1
        # Bottom
        for c in range(d - 1):
            if ((d - 1) + c) % 2 != 0:
                for q in [q_index(d - 1, c), q_index(d - 1, c + 1)]:
                    rows.append(row_idx)
                    cols.append(q)
                row_idx += 1
    else:  # side == "X"
        # X stabilizers on rough boundaries (left and right)
        # Left: where parity at virtual col -1 would be even
        for r in range(d - 1):
            if (r + (-1)) % 2 == 0:
                for q in [q_index(r, 0), q_index(r + 1, 0)]:
                    rows.append(row_idx)
                    cols.append(q)
                row_idx += 1
        # Right
        for r in range(d - 1):
            if (r + (d - 1)) % 2 == 0:
                for q in [q_index(r, d - 1), q_index(r + 1, d - 1)]:
                    rows.append(row_idx)
                    cols.append(q)
                row_idx += 1

    if row_idx != n_checks:
        raise AssertionError(f"Constructed {row_idx} checks, expected {n_checks}")

    data = np.ones(len(rows), dtype=np.uint8)
    H = sp.coo_matrix((data, (rows, cols)), shape=(n_checks, n_qubits)).tocsr()
    return H


# ---------------------------------------------------------------------------
# Packed crop data contracts and helpers
# ---------------------------------------------------------------------------


@dataclass
class CropMeta:
    """Lightweight metadata describing a packed crop and pad/bucket sizes."""

    k: int
    r: int
    bbox_xywh: Tuple[int, int, int, int]
    side: str
    d: int
    p: float
    kappa: int
    seed: int
    sha_short: str = ""
    bucket_id: int = -1
    pad_nodes: int = 0
    pad_edges: int = 0
    pad_seq: int = 0


@dataclass
class PackedCrop:
    """All tensors required for a single crop forward pass and supervision."""

    x_nodes: torch.Tensor
    node_mask: torch.Tensor
    node_type: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    edge_mask: torch.Tensor
    seq_idx: torch.Tensor
    seq_mask: torch.Tensor
    g_token: torch.Tensor
    y_bits: torch.Tensor
    s_sub: torch.Tensor  # (S_max,) padded syndrome bits for checks (row order of H_sub)
    meta: CropMeta
    H_sub: np.ndarray | None = None
    idx_data_local: np.ndarray | None = None
    idx_check_local: np.ndarray | None = None


SYND_FEAT_IDX = 8  # after xy(2), type(1), degree(1), k/r/bw/bh(4) => 8 dims


def _degree_from_Hsub(H_sub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (data_degree, check_degree) from a 0/1 submatrix H_sub."""
    deg_data = H_sub.sum(axis=0).astype(np.int32)
    deg_check = H_sub.sum(axis=1).astype(np.int32)
    return deg_data, deg_check


def _normalize_xy(xy: np.ndarray, bbox_xywh: Tuple[int, int, int, int]) -> np.ndarray:
    """Translate/scale absolute coordinates into [0,1] within the bbox."""
    x0, y0, w, h = bbox_xywh
    w = max(w, 1)
    h = max(h, 1)
    out = xy.copy().astype(np.float32)
    out[:, 0] = (out[:, 0] - x0) / float(w)
    out[:, 1] = (out[:, 1] - y0) / float(h)
    return out


def _quantize_xy01(xy01: np.ndarray, levels: int = 64) -> np.ndarray:
    """Uniformly quantize points in [0,1]^2 onto a levels×levels grid."""
    return np.clip((xy01 * (levels - 1) + 0.5).astype(np.int32), 0, levels - 1)


def _rot(n: int, x: int, y: int, rx: int, ry: int) -> Tuple[int, int]:
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = y, x
    return x, y


def _hilbert_xy_to_d(n: int, x: int, y: int) -> int:
    s = 1 << (max(1, int(math.ceil(math.log2(max(n, 1))))))
    d = 0
    xx, yy = x, y
    t = s // 2
    while t > 0:
        rx = 1 if (xx & t) else 0
        ry = 1 if (yy & t) else 0
        d += t * t * ((3 * rx) ^ ry)
        xx, yy = _rot(t, xx, yy, rx, ry)
        t //= 2
    return d


def _hilbert_index_2d(qxy: np.ndarray, levels: int = 64) -> np.ndarray:
    """Return Hilbert indices for integer grid points qxy in [0,levels)."""
    idx = np.zeros(qxy.shape[0], dtype=np.int64)
    for i, (qx, qy) in enumerate(qxy):
        idx[i] = _hilbert_xy_to_d(levels, int(qx), int(qy))
    return idx


def hilbert_order_within_bbox(
    check_xy: np.ndarray, bbox_xywh: Tuple[int, int, int, int]
) -> np.ndarray:
    """Stable argsort of check nodes by 2D Hilbert order inside bbox."""
    xy01 = _normalize_xy(check_xy, bbox_xywh)
    qxy = _quantize_xy01(xy01, levels=64)
    hilbert = _hilbert_index_2d(qxy, levels=64)
    return np.argsort(hilbert, kind="stable")


def infer_bucket_id(
    n_nodes: int,
    n_edges: int,
    n_seq: int,
    bucket_spec: Sequence[Tuple[int, int, int]],
) -> int:
    """Pick first (node_cap, edge_cap, seq_cap) triple that fits sizes."""
    for idx, (node_cap, edge_cap, seq_cap) in enumerate(bucket_spec):
        if n_nodes <= node_cap and n_edges <= edge_cap and n_seq <= seq_cap:
            return idx
    raise ValueError(f"No bucket accommodates nodes={n_nodes}, edges={n_edges}, seq={n_seq}")


def pack_cluster(
    H_sub: np.ndarray,
    xy_qubit: np.ndarray,
    xy_check: np.ndarray,
    synd_Z_then_X_bits: np.ndarray,
    *,
    k: int,
    r: int,
    bbox_xywh: Tuple[int, int, int, int],
    kappa_stats: Dict[str, float],
    y_bits_local: np.ndarray,
    side: str,
    d: int,
    p: float,
    seed: int,
    N_max: Optional[int],
    E_max: Optional[int],
    S_max: Optional[int],
    bucket_spec: Optional[Sequence[Tuple[int, int, int]]] = None,
    add_jump_edges: bool = True,
    jump_k: int = 2,
    g_extra: Optional[np.ndarray] = None,
    erase_local: Optional[np.ndarray] = None,
) -> PackedCrop:
    """Pack a local subgraph into fixed-pad tensors for MGHDv2.

    Produces node/edge/sequence tensors, a global conditioning token, and the
    supervision vector y_bits (data-qubit region) with optional erasure feature.
    """
    assert side in ("Z", "X")
    nC, nQ = H_sub.shape
    deg_data, deg_check = _degree_from_Hsub(H_sub)
    xy_q01 = _normalize_xy(xy_qubit, bbox_xywh)
    xy_c01 = _normalize_xy(xy_check, bbox_xywh)

    node_type = np.concatenate(
        [
            np.zeros(nQ, dtype=np.int8),
            np.ones(nC, dtype=np.int8),
        ]
    )
    degree = np.concatenate([deg_data, deg_check]).astype(np.float32)
    xy01 = np.vstack([xy_q01, xy_c01]).astype(np.float32)

    _, _, bw, bh = bbox_xywh
    g_list: List[float] = [float(k), float(r), float(bw), float(bh)]
    for key in ("size", "radius", "ecc", "bdensity"):
        if key in kappa_stats:
            g_list.append(float(kappa_stats[key]))
    # Optional global context features (e.g., TAD schedule/context vector)
    if g_extra is not None:
        try:
            extra = np.asarray(g_extra, dtype=np.float32).ravel().tolist()
            g_list.extend(extra)
        except Exception:
            pass
    g_token = torch.tensor(g_list, dtype=torch.float32)

    # Optional per-node erasure feature (data-qubits only)
    if erase_local is not None:
        try:
            er = np.asarray(erase_local, dtype=np.float32).ravel()
            if er.size != nQ:
                er = None
        except Exception:
            er = None
    else:
        er = None

    # Base per-node features (8 dims): xy(2), type(1), degree(1), k/r/bw/bh(4)
    # Append syndrome as feature index 8 (on check nodes only). If erasure is
    # enabled, it becomes the final feature dim.
    parts = [
        xy01,
        node_type[:, None].astype(np.float32),
        degree[:, None],
        np.full((nQ + nC, 1), float(k), dtype=np.float32),
        np.full((nQ + nC, 1), float(r), dtype=np.float32),
        np.full((nQ + nC, 1), float(bw), dtype=np.float32),
        np.full((nQ + nC, 1), float(bh), dtype=np.float32),
    ]
    synd = np.asarray(synd_Z_then_X_bits, dtype=np.uint8).ravel() & 1
    if synd.size < nC:
        synd = np.pad(synd, (0, nC - synd.size), mode="constant")
    elif synd.size > nC:
        synd = synd[:nC]
    synd_col = np.zeros((nQ + nC, 1), dtype=np.float32)
    synd_col[nQ:, 0] = synd.astype(np.float32)
    parts.append(synd_col)
    # Optional per-node erasure flag (only when provided): adds +1 feature dim
    if er is not None:
        er_col = np.concatenate([er.astype(np.float32), np.zeros(nC, dtype=np.float32)])[:, None]
        parts.append(er_col)
    base_nodes = np.concatenate(parts, axis=1)

    ci, qi = np.nonzero(H_sub)
    src = qi.astype(np.int64)
    dst = (nQ + ci).astype(np.int64)
    edge_index = np.stack([src, dst], axis=0)
    E = edge_index.shape[1]
    edge_attr = np.zeros((E, 3), dtype=np.float32)
    edge_attr[:, 0] = 1.0

    if add_jump_edges and nQ > 0 and jump_k >= 2:
        adj_data = np.zeros((nQ, nQ), dtype=np.uint8)
        for cc, qq in zip(ci, qi):
            neighbors = np.nonzero(H_sub[cc])[0]
            adj_data[qq, neighbors] = 1
            adj_data[neighbors, qq] = 1
        pow_adj = adj_data.copy()
        for _ in range(2, jump_k + 1):
            pow_adj = (pow_adj @ adj_data) > 0
        jj = np.transpose(np.nonzero(pow_adj))
        if jj.size > 0:
            js = jj[:, 0].astype(np.int64)
            jt = jj[:, 1].astype(np.int64)
            j_index = np.stack([js, jt], axis=0)
            j_attr = np.zeros((js.shape[0], 3), dtype=np.float32)
            j_attr[:, 1] = 1.0
            j_attr[:, 2] = float(jump_k)
            edge_index = np.concatenate([edge_index, j_index], axis=1)
            edge_attr = np.concatenate([edge_attr, j_attr], axis=0)

    check_order = hilbert_order_within_bbox(xy_check, bbox_xywh)
    # Sequence encoder runs over check nodes (Hilbert order), which are offset by nQ
    # because nodes are stored as [data..., checks...].
    seq_idx = (nQ + check_order).astype(np.int64)

    N = nQ + nC
    E_tot = edge_index.shape[1]
    S = nC
    if bucket_spec:
        bucket_idx = infer_bucket_id(N, E_tot, S, bucket_spec)
        N_max, E_max, S_max = bucket_spec[bucket_idx]
    else:
        bucket_idx = -1
        if N_max is None or E_max is None or S_max is None:
            raise ValueError("N_max, E_max, S_max must be provided when bucket_spec is None")

    assert N <= N_max and E_tot <= E_max and S <= S_max, "Increase pad limits."

    x_nodes = torch.zeros((N_max, base_nodes.shape[1]), dtype=torch.float32)
    node_mask = torch.zeros((N_max,), dtype=torch.bool)
    node_type_t = torch.zeros((N_max,), dtype=torch.int8)
    x_nodes[:N] = torch.from_numpy(base_nodes)
    node_mask[:N] = True
    node_type_t[:N] = torch.from_numpy(node_type)

    edge_index_t = torch.zeros((2, E_max), dtype=torch.long)
    edge_attr_t = torch.zeros((E_max, edge_attr.shape[1]), dtype=torch.float32)
    edge_mask = torch.zeros((E_max,), dtype=torch.bool)
    edge_index_t[:, :E_tot] = torch.from_numpy(edge_index)
    edge_attr_t[:E_tot] = torch.from_numpy(edge_attr)
    edge_mask[:E_tot] = True

    seq_idx_t = torch.zeros((S_max,), dtype=torch.long)
    seq_mask = torch.zeros((S_max,), dtype=torch.bool)
    seq_idx_t[:S] = torch.from_numpy(seq_idx)
    seq_mask[:S] = True

    y_bits_t = torch.full((N_max,), -1, dtype=torch.int8)
    y_bits_t[:nQ] = torch.from_numpy(y_bits_local.astype(np.int8))

    s_sub_t = torch.zeros((S_max,), dtype=torch.int8)
    s_sub_t[:S] = torch.from_numpy(synd.astype(np.int8))

    meta = CropMeta(
        k=k,
        r=r,
        bbox_xywh=bbox_xywh,
        side=side,
        d=d,
        p=p,
        kappa=int(kappa_stats.get("size", N)),
        seed=seed,
        bucket_id=bucket_idx,
        pad_nodes=int(N_max),
        pad_edges=int(E_max),
        pad_seq=int(S_max),
    )
    return PackedCrop(
        x_nodes=x_nodes,
        node_mask=node_mask,
        node_type=node_type_t,
        edge_index=edge_index_t,
        edge_attr=edge_attr_t,
        edge_mask=edge_mask,
        seq_idx=seq_idx_t,
        seq_mask=seq_mask,
        g_token=g_token,
        y_bits=y_bits_t,
        s_sub=s_sub_t,
        meta=meta,
        H_sub=H_sub.astype(np.uint8),
        idx_data_local=np.arange(nQ, dtype=np.int32),
        idx_check_local=np.arange(nC, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# Core neural blocks
# ---------------------------------------------------------------------------


class GraphDecoderCore(nn.Module):
    """Iterative message-passing core used by MGHDv2's graph head.

    Runs a small MLP for edge messages and a GRU to update node states for a
    fixed number of iterations; exposes a 2‑logit per‑node head at each step
    (the caller typically takes the last step).
    """

    def __init__(
        self,
        *,
        n_iters: int = 7,
        n_node_features: int = 10,
        n_node_inputs: int = 9,
        n_edge_features: int = 11,
        n_node_outputs: int = 2,
        msg_net_size: int = 96,
        msg_net_dropout_p: float = 0.0,
        gru_dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_iters = n_iters
        self.n_node_features = n_node_features
        self.n_node_inputs = n_node_inputs
        self.n_edge_features = n_edge_features
        self.n_node_outputs = n_node_outputs

        self.final_digits = nn.Linear(self.n_node_features, self.n_node_outputs)
        self.msg_net = nn.Sequential(
            nn.Linear(2 * n_node_features + n_edge_features, msg_net_size),
            nn.ReLU(),
            nn.Dropout(msg_net_dropout_p),
            nn.Linear(msg_net_size, msg_net_size),
            nn.ReLU(),
            nn.Dropout(msg_net_dropout_p),
            nn.Linear(msg_net_size, msg_net_size),
            nn.ReLU(),
            nn.Dropout(msg_net_dropout_p),
            nn.Linear(msg_net_size, n_edge_features),
        )
        self.gru = nn.GRU(input_size=n_edge_features + n_node_inputs, hidden_size=n_node_features)
        self.gru_drop = nn.Dropout(gru_dropout_p)

    def forward(
        self,
        node_inputs: torch.Tensor,
        src_ids: torch.Tensor,
        dst_ids: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per‑node logits for one graph.

        Parameters
        - node_inputs: [N, F] node features (already projected to hidden dim)
        - src_ids/dst_ids: [E] edge endpoints (0‑based node indices)

        Returns
        - [T, N, 2] logits (T iterations)
        """
        device = node_inputs.device
        node_states = torch.zeros(node_inputs.shape[0], self.n_node_features, device=device)
        outputs_tensor = torch.zeros(
            self.n_iters,
            node_inputs.shape[0],
            self.n_node_outputs,
            device=device,
        )

        for i in range(self.n_iters):
            if edge_attr is None:
                edge_feat = torch.zeros(
                    src_ids.shape[0],
                    self.n_edge_features,
                    device=node_inputs.device,
                    dtype=node_inputs.dtype,
                )
            else:
                edge_feat = edge_attr.to(node_inputs.device, dtype=node_inputs.dtype)
            msg_in = torch.cat((node_states[src_ids], node_states[dst_ids], edge_feat), dim=1)
            messages = self.msg_net(msg_in)
            agg_msg = torch.zeros(
                node_inputs.shape[0], self.n_edge_features, device=device, dtype=messages.dtype
            )
            agg_msg.index_add_(dim=0, index=dst_ids, source=messages)
            gru_inputs = torch.cat((agg_msg, node_inputs), dim=1)

            _, node_states = self.gru(
                gru_inputs.view(1, node_inputs.shape[0], -1),
                node_states.view(1, node_inputs.shape[0], -1),
            )
            node_states = node_states.squeeze(0)
            outputs_tensor[i] = self.final_digits(node_states)
            node_states = self.gru_drop(node_states)

        return outputs_tensor


class ChannelSE(nn.Module):
    """Squeeze-and-Excitation for channel attention."""

    def __init__(self, channels: int, reduction: int = 4, use_hsigmoid: bool = False) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.act = nn.SiLU()
        if use_hsigmoid:
            self.gate = lambda t: torch.clamp((t + 3.0) / 6.0, 0.0, 1.0)
        else:
            self.gate = torch.sigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.mean(dim=1)
        excite = self.fc2(self.act(self.fc1(squeeze)))
        weights = self.gate(excite).unsqueeze(1)
        return x * weights


class GraphDecoder(nn.Module):
    """Wrapper that normalizes kwargs and instantiates GraphDecoderCore."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        normalized = dict(kwargs)
        if "n_node_features" not in normalized and "n_node_inputs" in normalized:
            normalized["n_node_features"] = normalized["n_node_inputs"]
        if "msg_net_size" not in normalized and "n_node_inputs" in normalized:
            normalized["msg_net_size"] = max(96, normalized["n_node_inputs"])
        if "n_node_outputs" not in normalized:
            normalized["n_node_outputs"] = 2
        self.core = GraphDecoderCore(**normalized)

    def forward(
        self,
        node_inputs: torch.Tensor,
        src_ids: torch.Tensor,
        dst_ids: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pass through to the underlying core; see GraphDecoderCore.forward."""
        return self.core(node_inputs, src_ids, dst_ids, edge_attr=edge_attr)


def _mamba_constructors() -> List[Any]:
    """Return available Mamba encoder constructors, or an empty list if none.

    In production we expect at least one of these symbols to be importable. In
    CPU-only or minimal test environments, we degrade gracefully by returning
    an empty list and letting SequenceEncoder fall back to an identity module.
    """
    candidates: List[Any] = []
    for module, attr in (
        ("poc_my_models", "MambaEncoder"),
        ("poc_my_models", "MambaStack"),
        ("poc_my_models", "Mamba"),
        ("mamba_ssm", "Mamba"),
    ):
        try:
            mod = __import__(module, fromlist=[attr])
            candidates.append(getattr(mod, attr))
        except Exception:
            continue
    return candidates


class SequenceEncoder(nn.Module):
    """Mask‑aware adapter over Mamba encoder implementations.

    Selects a working constructor from several common entry points and wraps the
    resulting module with a layer norm and simple scatter‑add back to node space.
    """

    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        candidates = _mamba_constructors()
        self.core = None
        for ctor in candidates:
            for factory in (
                lambda: ctor(d_model=d_model, d_state=d_state),
                lambda: ctor(d_model=d_model),
                lambda: ctor(d_state=d_state),
                lambda: ctor(),
            ):
                try:
                    self.core = factory()
                    break
                except TypeError:
                    continue
                except Exception:
                    continue
            if self.core is not None:
                break
        if self.core is None:
            # Fallback: identity encoder (no sequence modeling) for CPU-only CI.
            self.core = nn.Identity()
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        seq_idx: torch.Tensor,
        seq_mask: torch.Tensor,
        *,
        node_type: torch.Tensor,
    ) -> torch.Tensor:
        """Encode check‑node slice and scatter back to node features.

        Uses ``seq_idx``/``seq_mask`` to pick and order checks in the sequence
        encoder and then fuses the output additively into ``x``.
        """
        if seq_idx.numel() == 0 or seq_mask.sum() == 0:
            return x

        # Batched path: x=[B,N,C], seq_idx=[B,S], seq_mask=[B,S]
        if x.dim() == 3 and seq_idx.dim() == 2 and seq_mask.dim() == 2:
            B, N, C = x.shape
            idx = seq_idx.long().clamp(min=0, max=max(0, N - 1))
            mask = seq_mask.bool()

            x_chk = x.gather(1, idx.unsqueeze(-1).expand(-1, -1, C))
            try:
                y_chk = self.core(x_chk)
            except Exception:
                y_chk = x_chk
            y_chk = self.out_norm(y_chk)

            # Ensure dtype alignment for index_copy in mixed precision
            if y_chk.dtype != x.dtype:
                y_chk = y_chk.to(x.dtype)
                x_chk = x_chk.to(x.dtype)

            x_scatter = x.clone()
            x_flat = x_scatter.view(B * N, C)
            idx_flat = idx + (torch.arange(B, device=x.device).unsqueeze(1) * N)
            idx_flat_valid = idx_flat[mask]
            vals = (x_chk + y_chk)[mask]
            x_flat.index_copy_(0, idx_flat_valid, vals)
            return x_flat.view(B, N, C)

        if x.dim() == 3:
            if seq_idx.dim() != 1 or seq_mask.dim() != 1:
                raise ValueError(
                    "For batched x_nodes, expected seq_idx/seq_mask to have shape [B,S] "
                    f"(got seq_idx {tuple(seq_idx.shape)}, seq_mask {tuple(seq_mask.shape)})."
                )
            # Backwards-compatible path: treat batched nodes as a single disjoint
            # union indexed by flattened node IDs.
            B, N, C = x.shape
            x = x.view(B * N, C)
            node_type = node_type.view(B * N)
        elif seq_idx.dim() != 1 or seq_mask.dim() != 1:
            raise ValueError(
                "Expected seq_idx/seq_mask to be 1D for unbatched x_nodes "
                f"(got seq_idx {tuple(seq_idx.shape)}, seq_mask {tuple(seq_mask.shape)})."
            )
        valid = seq_mask.nonzero(as_tuple=False).squeeze(-1)
        idx = seq_idx[valid].long()
        x_chk = x.index_select(0, idx)
        x_chk_3d = x_chk.unsqueeze(0)
        try:
            y_chk = self.core(x_chk_3d).squeeze(0)
        except Exception:
            y_chk = x_chk
        y_chk = self.out_norm(y_chk)
        # Ensure dtype alignment for index_copy in mixed precision
        if y_chk.dtype != x.dtype:
            y_chk = y_chk.to(x.dtype)
            x_chk = x_chk.to(x.dtype)
        x_scatter = x.clone()
        x_scatter.index_copy_(0, idx, x_chk + y_chk)
        if "B" in locals():
            return x_scatter.view(B, N, C)
        return x_scatter


class GraphDecoderAdapter(nn.Module):
    """Thin adapter over GraphDecoder for v2 crops (sets shapes/heads)."""

    def __init__(
        self,
        hidden_dim: int,
        edge_feat_dim: int,
        n_iters: int,
        n_node_outputs: int = 2,
        *,
        msg_net_size: Optional[int] = None,
        msg_net_dropout_p: float = 0.0,
        gru_dropout_p: float = 0.0,
    ):
        super().__init__()
        self.n_node_outputs = int(n_node_outputs)
        self.core = GraphDecoder(
            n_iters=n_iters,
            n_node_inputs=hidden_dim,
            n_node_outputs=self.n_node_outputs,
            n_edge_features=edge_feat_dim,
            msg_net_size=max(96, hidden_dim) if msg_net_size is None else int(msg_net_size),
            msg_net_dropout_p=float(msg_net_dropout_p),
            gru_dropout_p=float(gru_dropout_p),
        )
        self.iter_override: Optional[int] = None

    def forward(
        self,
        x_nodes: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute last‑iteration logits [N,2] with masked edges."""
        src_ids = edge_index[0]
        dst_ids = edge_index[1]
        attr = edge_attr
        if edge_mask is not None:
            valid_edges = edge_mask
            src_ids = src_ids[valid_edges]
            dst_ids = dst_ids[valid_edges]
            if attr is not None:
                attr = attr[valid_edges]
        output = self.core(x_nodes, src_ids, dst_ids, edge_attr=attr)
        idx = output.shape[0] - 1
        if self.iter_override is not None:
            idx = max(0, min(self.iter_override - 1, output.shape[0] - 1))
        return output[idx]

    def set_iteration_override(self, n_iters: Optional[int]) -> None:
        self.iter_override = None if not n_iters or n_iters <= 0 else int(n_iters)


# ---------------------------------------------------------------------------
# MGHD v2 model
# ---------------------------------------------------------------------------


class MGHDv2(nn.Module):
    """Distance-agnostic MGHD v2 with Mamba, channel attention, and graph message passing."""

    def __init__(
        self,
        profile: str = "S",
        *,
        d_model: int = 192,
        d_state: int = 80,
        n_iters: int = 8,
        node_feat_dim: int = 9,
        edge_feat_dim: int = 3,
        g_dim: Optional[int] = None,
        se_reduction: int = 4,
        gnn_msg_net_size: Optional[int] = None,
        gnn_msg_dropout: float = 0.0,
        gnn_gru_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if g_dim is None:
            g_dim = max(8, node_feat_dim)
        self.d_model = d_model
        self.seq_encoder = SequenceEncoder(d_model=d_model, d_state=d_state)
        self.se = ChannelSE(channels=d_model, reduction=int(se_reduction))
        self.gnn = GraphDecoderAdapter(
            hidden_dim=d_model,
            edge_feat_dim=d_model,
            n_iters=n_iters,
            msg_net_size=gnn_msg_net_size,
            msg_net_dropout_p=gnn_msg_dropout,
            gru_dropout_p=gnn_gru_dropout,
        )
        self.node_in = nn.Linear(node_feat_dim, d_model)
        self.edge_in = nn.Linear(edge_feat_dim, d_model)
        self.g_proj: Optional[nn.Linear] = None

    def forward(self, packed: PackedCrop) -> Tuple[torch.Tensor, torch.Tensor]:
        x = packed.x_nodes.float()
        eidx = packed.edge_index.long()
        eatt = packed.edge_attr.float()
        emask = packed.edge_mask.bool()
        gtok = packed.g_token.float()

        # Guard against feature-dim mismatches (e.g., erasure channel optional)
        if self.node_in.in_features != x.shape[-1]:
            self.ensure_node_in(int(x.shape[-1]), x.device)
        if self.edge_in.in_features != eatt.shape[-1]:
            self.ensure_edge_in(int(eatt.shape[-1]), eatt.device)

        if x.dim() == 3:
            batch_size, nodes_pad, _ = x.shape
            nmask = packed.node_mask.bool()
            node_type = packed.node_type.long()

            if gtok.dim() == 2:
                g_dim = gtok.size(-1)
                if self.g_proj is None or self.g_proj.in_features != g_dim:
                    self.ensure_g_proj(g_dim, gtok.device)
                g_bias = self.g_proj(gtok).unsqueeze(1).expand(batch_size, nodes_pad, -1)
            else:
                g_dim = gtok.numel()
                if self.g_proj is None or self.g_proj.in_features != g_dim:
                    self.ensure_g_proj(g_dim, gtok.device)
                g_bias = self.g_proj(gtok.view(-1)).view(1, 1, -1).expand(batch_size, nodes_pad, -1)

            x = self.node_in(x) + g_bias
            x = self.seq_encoder(
                x,
                packed.seq_idx.long(),
                packed.seq_mask.bool(),
                node_type=node_type,
            )
            x = self.se(x)

            x_flat = x.view(batch_size * nodes_pad, -1)
            nmask_flat = nmask.view(batch_size * nodes_pad)
            e = self.edge_in(eatt)
            logits = self.gnn(x_flat, eidx, e, nmask_flat, emask)
            return logits, nmask_flat

        nodes_pad = x.shape[0]
        nmask = packed.node_mask.bool()
        node_type = packed.node_type.long()

        if gtok.dim() == 2:
            g_dim = gtok.size(-1)
            if self.g_proj is None or self.g_proj.in_features != g_dim:
                self.ensure_g_proj(g_dim, gtok.device)
            g_bias = self.g_proj(gtok).unsqueeze(1).expand(1, nodes_pad, -1).reshape(nodes_pad, -1)
        else:
            g_dim = gtok.numel()
            if self.g_proj is None or self.g_proj.in_features != g_dim:
                self.ensure_g_proj(g_dim, gtok.device)
            g_bias = self.g_proj(gtok.view(-1)).unsqueeze(0).expand(nodes_pad, -1)

        x = self.node_in(x) + g_bias
        e = self.edge_in(eatt)
        x = self.seq_encoder(
            x,
            packed.seq_idx.long(),
            packed.seq_mask.bool(),
            node_type=node_type,
        )
        x = self.se(x.unsqueeze(0)).squeeze(0)
        logits = self.gnn(x, eidx, e, nmask, emask)
        return logits, nmask

    def allocate_static_batch(
        self,
        *,
        batch_size: int,
        nodes_pad: int,
        edges_pad: int,
        seq_pad: int,
        feat_dim: int,
        edge_feat_dim: int,
        g_dim: int,
        device: torch.device,
    ) -> SimpleNamespace:
        def zeros(shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=device)

        return SimpleNamespace(
            x_nodes=zeros((batch_size, nodes_pad, feat_dim), torch.float32),
            node_mask=zeros((batch_size, nodes_pad), torch.bool),
            node_type=zeros((batch_size, nodes_pad), torch.int8),
            edge_index=zeros((2, batch_size * edges_pad), torch.long),
            edge_attr=zeros((batch_size * edges_pad, edge_feat_dim), torch.float32),
            edge_mask=zeros((batch_size * edges_pad,), torch.bool),
            seq_idx=zeros((batch_size, seq_pad), torch.long),
            seq_mask=zeros((batch_size, seq_pad), torch.bool),
            g_token=zeros((batch_size, g_dim), torch.float32),
            batch_size=batch_size,
            nodes_pad=nodes_pad,
        )

    def copy_into_static(
        self, static_ns: SimpleNamespace, host_ns: SimpleNamespace, *, non_blocking: bool = True
    ) -> None:
        for name in (
            "x_nodes",
            "node_mask",
            "node_type",
            "edge_index",
            "edge_attr",
            "edge_mask",
            "seq_idx",
            "seq_mask",
            "g_token",
        ):
            getattr(static_ns, name).copy_(getattr(host_ns, name), non_blocking=non_blocking)

    def move_packed_to_device(
        self, host_ns: SimpleNamespace, device: torch.device
    ) -> SimpleNamespace:
        tensors: Dict[str, torch.Tensor] = {}
        for name in (
            "x_nodes",
            "node_mask",
            "node_type",
            "edge_index",
            "edge_attr",
            "edge_mask",
            "seq_idx",
            "seq_mask",
            "g_token",
        ):
            tensors[name] = getattr(host_ns, name).to(device, non_blocking=True)
        tensors["batch_size"] = host_ns.batch_size
        tensors["nodes_pad"] = host_ns.nodes_pad
        return SimpleNamespace(**tensors)

    def gather_from_static(
        self, static_output: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, node_mask = static_output
        return logits, node_mask

    def scatter_outputs(
        self,
        logits: torch.Tensor,
        cluster_infos: Sequence[Dict[str, torch.Tensor]],
        *,
        temp: float = 1.0,
    ) -> List[torch.Tensor]:
        probs_all = torch.sigmoid((logits[:, 1] - logits[:, 0]) / float(temp)).clamp(1e-6, 1 - 1e-6)
        out: List[torch.Tensor] = []
        for info in cluster_infos:
            data_idx = info["data_idx"].to(probs_all.device)
            out.append(probs_all.index_select(0, data_idx))
        return out

    def set_message_iters(self, n_iters: Optional[int]) -> None:
        self.gnn.set_iteration_override(n_iters)

    def ensure_g_proj(self, g_dim: int, device: torch.device) -> None:
        if self.g_proj is None or self.g_proj.in_features != g_dim:
            layer = nn.Linear(g_dim, self.node_in.out_features)
            layer.to(device)
            self.g_proj = layer

    def ensure_node_in(self, in_features: int, device: torch.device) -> None:
        if self.node_in.in_features != in_features:
            layer = nn.Linear(in_features, self.node_in.out_features)
            layer.to(device)
            self.node_in = layer

    def ensure_edge_in(self, in_features: int, device: torch.device) -> None:
        if self.edge_in.in_features != in_features:
            layer = nn.Linear(in_features, self.edge_in.out_features)
            layer.to(device)
            self.edge_in = layer

    def set_authoritative_mats(self, *_args, **_kwargs) -> None:
        return None

    def set_rotated_layout(self, *_args, **_kwargs) -> None:
        return None


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------


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
    "s_sub",
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

    def __init__(
        self, ckpt_path: str, device: str = "cpu", *, profile: str = "S", node_feat_dim: int = 9
    ) -> None:
        self.device = torch.device(device)
        state_dict = _load_state_dict(ckpt_path)
        self.model = MGHDv2(profile=profile, node_feat_dim=node_feat_dim)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device).eval()

        self.model_version = "v2"
        self._graph_capture_enabled = False
        self._bound = False
        self._Hx: Optional[sp.csr_matrix] = None
        self._Hz: Optional[sp.csr_matrix] = None
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_in: Optional[PackedCrop] = None
        self._static_logits: Optional[torch.Tensor] = None
        self._static_mask: Optional[torch.Tensor] = None

    def bind_code(self, Hx: sp.csr_matrix, Hz: sp.csr_matrix) -> None:
        self._Hx = Hx.tocsr()
        self._Hz = Hz.tocsr()
        if hasattr(self.model, "set_authoritative_mats"):
            self.model.set_authoritative_mats(
                self._Hx.toarray(), self._Hz.toarray(), device=self.device
            )
        self._bound = True

    def set_message_iters(self, n_iters: Optional[int]) -> None:
        self.model.set_message_iters(n_iters)

    def priors_from_subgraphs_batched(
        self,
        items: Sequence[Tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]],
        *,
        temp: float = 1.0,
        bucket: Optional[str] = None,
        bucket_spec: Optional[Sequence[Tuple[int, int, int]]] = None,
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
            H_sub, s_sub, q_l2g, c_l2g, meta = self._normalize_entry(entry)
            pack = self._build_pack(H_sub, s_sub, meta, bucket_spec)
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

    def priors_from_syndrome(self, *_args, **_kwargs) -> np.ndarray:  # pragma: no cover
        raise RuntimeError("MGHD v2 decoder only supports packed crop inference")

    def warmup_and_capture(self, example_packed: PackedCrop, iters: int = 3) -> Optional[bool]:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return None
        pack = self._move_packed_crop(example_packed, self.device)
        stream = torch.cuda.Stream(device=self.device)
        torch.cuda.synchronize(self.device)
        with torch.cuda.stream(stream):
            for _ in range(max(1, int(iters))):
                _ = self.model(pack)
        torch.cuda.synchronize(self.device)
        self._graph = torch.cuda.CUDAGraph()
        self._static_in = pack
        with torch.cuda.graph(self._graph):  # type: ignore[attr-defined]
            logits, node_mask = self.model(self._static_in)
            self._static_logits = logits.clone()
            self._static_mask = node_mask.clone()
        self._graph_capture_enabled = True
        return True

    @torch.inference_mode()
    def fast_infer(self, packed: PackedCrop):
        if self._graph is None or self._static_in is None:
            pack = self._move_packed_crop(packed, self.device)
            return self.model(pack)
        pack = self._move_packed_crop(packed, self.device)
        self._copy_into_static_pack(self._static_in, pack)
        self._graph.replay()
        return self._static_logits, self._static_mask

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
        bucket_spec: Optional[Sequence[Tuple[int, int, int]]],
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

    def _copy_into_static_pack(self, dst: PackedCrop, src: PackedCrop) -> None:
        for name in TensorFields:
            src_val = getattr(src, name, None)
            dst_val = getattr(dst, name, None)
            if torch.is_tensor(src_val) and torch.is_tensor(dst_val):
                if src_val.shape != dst_val.shape:
                    raise ValueError(
                        f"Tensor field '{name}' shape mismatch for CUDA graph replay: "
                        f"{tuple(src_val.shape)} vs {tuple(dst_val.shape)}"
                    )
                dst_val.copy_(src_val)

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


__all__ = [
    "MGHDConfig",
    "to_dict",
    "CropMeta",
    "PackedCrop",
    "pack_cluster",
    "hilbert_order_within_bbox",
    "infer_bucket_id",
    "GraphDecoderCore",
    "ChannelSE",
    "GraphDecoder",
    "MGHDv2",
    "MGHDDecoderPublic",
    "warmup_and_capture",
]
