# NOTE: No CUDA/CUDA-Q initialization at import.
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, Sequence
import math
import numpy as np
import torch

# ---- Data contracts ---------------------------------------------------------

@dataclass
class CropMeta:
    k: int                   # |C| data qubits in crop
    r: int                   # nullity (kernel dimension)
    bbox_xywh: Tuple[int,int,int,int]   # (x0,y0,w,h) in lattice coords
    side: str                # 'Z' or 'X'
    d: int
    p: float
    kappa: int               # cluster size (nodes)
    seed: int
    sha_short: str = ""
    bucket_id: int = -1
    pad_nodes: int = 0
    pad_edges: int = 0
    pad_seq: int = 0

@dataclass
class PackedCrop:
    # Node features [N_max, F_n], mask [N_max]
    x_nodes: torch.Tensor
    node_mask: torch.Tensor  # bool
    node_type: torch.Tensor  # int8 {0: data-qubit, 1: check}
    # Edge indices [2, E_max], features [E_max, F_e], mask [E_max]
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    edge_mask: torch.Tensor  # bool
    # Mamba sequence over check nodes [S_max], mask [S_max]
    seq_idx: torch.Tensor
    seq_mask: torch.Tensor   # bool
    # Global token [F_g]
    g_token: torch.Tensor
    # Labels per data-qubit (local indexing) [N_max], mask is node_mask & (node_type==0)
    y_bits: torch.Tensor
    meta: CropMeta
    # Extra projector data (kept CPU by default)
    H_sub: np.ndarray | None = None          # checks x data_qubits (uint8)
    idx_data_local: np.ndarray | None = None # local data-qubit indices (0..nQ-1)
    idx_check_local: np.ndarray | None = None# local check indices (0..nC-1)

# ---- Utilities --------------------------------------------------------------

def _degree_from_Hsub(H_sub: np.ndarray) -> np.ndarray:
    # H_sub: Tanner adj matrix in bipartite (checks x data-qubits) OR COO edges.
    # Expect edges provided separately; degree is computed from adjacency.
    # Here we accept a binary 2D array (checks x data).
    deg_data = H_sub.sum(axis=0).astype(np.int32)
    deg_check = H_sub.sum(axis=1).astype(np.int32)
    return deg_data, deg_check

def _normalize_xy(xy: np.ndarray, bbox_xywh: Tuple[int,int,int,int]) -> np.ndarray:
    x0, y0, w, h = bbox_xywh
    # map to [0,1] via Δx/w, Δy/h; avoid division by zero
    w = max(w, 1)
    h = max(h, 1)
    out = xy.copy().astype(np.float32)
    out[:, 0] = (out[:, 0] - x0) / float(w)
    out[:, 1] = (out[:, 1] - y0) / float(h)
    return out

def _quantize_xy01(xy01: np.ndarray, levels: int = 64) -> np.ndarray:
    # quantize to integer grid for Hilbert indexing
    q = np.clip((xy01 * (levels - 1) + 0.5).astype(np.int32), 0, levels - 1)
    return q

def _hilbert_index_2d(qxy: np.ndarray, levels: int = 64) -> np.ndarray:
    # Compute Hilbert index for 2D points on levels x levels grid.
    # Implementation adapted from standard bitwise algorithm.
    n = qxy.shape[0]
    idx = np.zeros(n, dtype=np.int64)
    for i in range(n):
        x = int(qxy[i, 0]); y = int(qxy[i, 1])
        idx[i] = _hilbert_xy_to_d(levels, x, y)
    return idx

def _hilbert_xy_to_d(n: int, x: int, y: int) -> int:
    # n is grid size (power of 2 recommended but we clamp)
    # Convert (x,y) to Hilbert distance d. We coerce n to next power of 2.
    s = 1 << (max(1, int(math.ceil(math.log2(max(n,1))))) )
    d = 0
    rx = ry = 0
    t = s // 2
    xx, yy = x, y
    while t > 0:
        rx = 1 if (xx & t) else 0
        ry = 1 if (yy & t) else 0
        d += t * t * ((3 * rx) ^ ry)
        xx, yy = _rot(t, xx, yy, rx, ry)
        t //= 2
    return d

def _rot(n, x, y, rx, ry):
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        # Swap x and y
        x, y = y, x
    return x, y

def hilbert_order_within_bbox(check_xy: np.ndarray, bbox_xywh: Tuple[int,int,int,int]) -> np.ndarray:
    xy01 = _normalize_xy(check_xy, bbox_xywh)
    qxy = _quantize_xy01(xy01, levels=64)
    h = _hilbert_index_2d(qxy, levels=64)
    return np.argsort(h, kind="stable")

# ---- Public packer ----------------------------------------------------------

def infer_bucket_id(n_nodes: int, n_edges: int, n_seq: int, bucket_spec: Sequence[Tuple[int, int, int]]) -> int:
    for idx, (node_cap, edge_cap, seq_cap) in enumerate(bucket_spec):
        if n_nodes <= node_cap and n_edges <= edge_cap and n_seq <= seq_cap:
            return idx
    raise ValueError(
        f"No bucket accommodates nodes={n_nodes}, edges={n_edges}, seq={n_seq}"
    )


def pack_cluster(
    H_sub: np.ndarray,
    xy_qubit: np.ndarray,
    xy_check: np.ndarray,
    synd_Z_then_X_bits: np.ndarray,
    *,
    k: int,
    r: int,
    bbox_xywh: Tuple[int,int,int,int],
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
) -> PackedCrop:
    """
    Build distance-agnostic, padded tensors for a cluster crop.
    - H_sub: binary matrix [n_checks, n_qubits_in_crop]
    - xy_qubit: int array [n_qubits, 2] absolute coords
    - xy_check: int array [n_checks, 2] absolute coords
    - synd_Z_then_X_bits: uint8 syndrome of this crop in Z-then-X order for checks
    - y_bits_local: uint8 labels per local data-qubit (teacher ensemble output)
    - k, r, bbox_xywh, kappa_stats: metadata
    """
    assert side in ("Z","X")
    nC, nQ = H_sub.shape
    # Node features (data-qubits + checks)
    deg_data, deg_check = _degree_from_Hsub(H_sub)
    xy_q01 = _normalize_xy(xy_qubit, bbox_xywh)
    xy_c01 = _normalize_xy(xy_check, bbox_xywh)

    # Assemble nodes: data-qubits then checks for stable local indexing
    node_type = np.concatenate([np.zeros(nQ, dtype=np.int8), np.ones(nC, dtype=np.int8)], axis=0)
    degree    = np.concatenate([deg_data, deg_check], axis=0).astype(np.float32)
    xy01      = np.vstack([xy_q01, xy_c01]).astype(np.float32)

    # Global token: [k, r, w, h, kappa_stats...]
    _,_,bw,bh = bbox_xywh
    g_list = [float(k), float(r), float(bw), float(bh)]
    # add a few stable κ stats if provided
    for key in ("size","radius","ecc","bdensity"):
        if key in kappa_stats:
            g_list.append(float(kappa_stats[key]))
    g_token = torch.tensor(g_list, dtype=torch.float32)

    # Node features: [Δx, Δy, node_type, degree, k, r, w, h]
    base_nodes = np.concatenate([
        xy01, node_type[:,None].astype(np.float32), degree[:,None],
        np.full((nQ+nC,1), float(k), dtype=np.float32),
        np.full((nQ+nC,1), float(r), dtype=np.float32),
        np.full((nQ+nC,1), float(bw), dtype=np.float32),
        np.full((nQ+nC,1), float(bh), dtype=np.float32),
    ], axis=1)

    # Edge indices (Tanner edges): checks are offset by nQ in node indexing
    ci, qi = np.nonzero(H_sub)  # ci in [0..nC), qi in [0..nQ)
    src = (qi).astype(np.int64)         # data-qubit node id
    dst = (nQ + ci).astype(np.int64)    # check node id
    edge_index = np.stack([src, dst], axis=0)  # shape [2, E]
    # Edge features: [is_tanner=1, is_jump=0, hop_len=0]
    E = edge_index.shape[1]
    edge_attr = np.zeros((E, 3), dtype=np.float32)
    edge_attr[:,0] = 1.0

    # Optional: add k-hop jump edges among data-qubits as long-range hints
    if add_jump_edges and nQ > 0 and jump_k >= 2:
        # simple BFS radius-k over data-qubit adjacency induced by checks
        # build data-data graph via shared checks
        adj_data = np.zeros((nQ, nQ), dtype=np.uint8)
        for cc, qq in zip(ci, qi):
            # mark neighbors sharing check cc
            neighbors = np.nonzero(H_sub[cc])[0]
            adj_data[qq, neighbors] = 1
            adj_data[neighbors, qq] = 1
        # compute hops by repeated squaring up to jump_k
        pow_adj = adj_data.copy()
        for kstep in range(2, jump_k+1):
            pow_adj = (pow_adj @ adj_data) > 0
        jj = np.transpose(np.nonzero(pow_adj))
        if jj.size > 0:
            js = jj[:,0].astype(np.int64)
            jt = jj[:,1].astype(np.int64)
            j_index = np.stack([js, jt], axis=0)  # data->data
            j_attr  = np.zeros((js.shape[0], 3), dtype=np.float32)
            j_attr[:,1] = 1.0  # is_jump
            j_attr[:,2] = float(jump_k)
            edge_index = np.concatenate([edge_index, j_index], axis=1)
            edge_attr  = np.concatenate([edge_attr, j_attr], axis=0)

    # Sequence for Mamba over checks: Hilbert order within bbox; keep Z-then-X outside
    check_order = hilbert_order_within_bbox(xy_check, bbox_xywh)
    seq_idx = check_order.astype(np.int64)  # local indices 0..nC-1

    # ---- Padding/masks ----
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
    edge_attr_t  = torch.zeros((E_max, edge_attr.shape[1]), dtype=torch.float32)
    edge_mask    = torch.zeros((E_max,), dtype=torch.bool)
    edge_index_t[:, :E_tot] = torch.from_numpy(edge_index)
    edge_attr_t[:E_tot]     = torch.from_numpy(edge_attr)
    edge_mask[:E_tot]       = True

    seq_idx_t = torch.zeros((S_max,), dtype=torch.long)
    seq_mask  = torch.zeros((S_max,), dtype=torch.bool)
    seq_idx_t[:S] = torch.from_numpy(seq_idx)
    seq_mask[:S]  = True

    y_bits_t = torch.full((N_max,), -1, dtype=torch.int8)
    # Place labels only on data-qubit section [0:nQ)
    y_bits_t[:nQ] = torch.from_numpy(y_bits_local.astype(np.int8))

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
        x_nodes=x_nodes, node_mask=node_mask, node_type=node_type_t,
        edge_index=edge_index_t, edge_attr=edge_attr_t, edge_mask=edge_mask,
        seq_idx=seq_idx_t, seq_mask=seq_mask,
        g_token=g_token, y_bits=y_bits_t, meta=meta,
        H_sub=H_sub.astype(np.uint8),
        idx_data_local=np.arange(nQ, dtype=np.int32),
        idx_check_local=np.arange(nC, dtype=np.int32),
    )

__all__ = ["CropMeta","PackedCrop","pack_cluster","hilbert_order_within_bbox","infer_bucket_id"]
