"""Erasure-aware peeling + cluster decoder for CSS codes."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from mghd.decoders.lsd.clustered import solve_on_erasure


def _peeling(
    H: np.ndarray,
    s: np.ndarray,
    erase_cols: np.ndarray,
    erase_rows: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform classical peeling on the erased subgraph."""

    E = erase_cols.astype(bool)
    rows_keep = np.ones(H.shape[0], dtype=bool) if erase_rows is None else ~erase_rows.astype(bool)
    Hs = (H[rows_keep][:, E] & 1).astype(np.uint8)
    ss = (s[rows_keep] & 1).astype(np.uint8)

    col_ids = np.flatnonzero(E)
    x_partial = np.zeros(H.shape[1], dtype=np.uint8)

    deg = Hs.sum(axis=1)
    changed = True
    while changed:
        idx = np.where(deg == 1)[0]
        if idx.size == 0:
            break
        changed = False
        for r in idx:
            cols = np.flatnonzero(Hs[r])
            if cols.size != 1:
                continue
            c = int(cols[0])
            x_partial[col_ids[c]] ^= ss[r]
            rows_with_c = np.where(Hs[:, c])[0]
            ss[rows_with_c] ^= ss[r]
            Hs[rows_with_c, c] ^= 1
            deg[rows_with_c] = Hs[rows_with_c].sum(axis=1)
            changed = True

    col_mask = Hs.sum(axis=0) > 0
    row_mask = Hs.sum(axis=1) > 0
    H_res = Hs[row_mask][:, col_mask]
    s_res = ss[row_mask]
    col_ids_res = col_ids[col_mask]
    return x_partial, H_res, s_res, col_ids_res


def _solve_clusters(
    H_res: np.ndarray,
    s_res: np.ndarray,
    col_ids_res: np.ndarray,
    max_cluster: int = 256,
) -> np.ndarray:
    """Solve residual stopping-set clusters exactly via GF(2) helper."""

    ncols = H_res.shape[1]
    if ncols == 0:
        return np.zeros(0, dtype=np.uint8)

    adj: List[set] = [set() for _ in range(ncols)]
    for r in range(H_res.shape[0]):
        cols = np.flatnonzero(H_res[r])
        for i in range(len(cols) - 1):
            a = cols[i]
            for j in range(i + 1, len(cols)):
                b = cols[j]
                adj[a].add(b)
                adj[b].add(a)

    seen = np.zeros(ncols, dtype=bool)
    x_local = np.zeros(ncols, dtype=np.uint8)
    for c0 in range(ncols):
        if seen[c0]:
            continue
        comp: List[int] = []
        queue = [c0]
        seen[c0] = True
        rows_touch = np.zeros(H_res.shape[0], dtype=bool)
        while queue:
            u = queue.pop()
            comp.append(u)
            rows_touch |= H_res[:, u].astype(bool)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    queue.append(v)
        C = np.array(comp, dtype=int)
        if C.size == 0:
            continue
        if C.size <= max_cluster:
            Hc = H_res[rows_touch][:, C]
            sc = s_res[rows_touch]
            xc = solve_on_erasure(Hc, sc, np.ones(Hc.shape[1], dtype=np.uint8))
            x_local[C] ^= xc
    return x_local


class ErasureQLDPCPeelingTeacher:
    """Erasure-aware teacher for generic CSS (qLDPC) codes."""

    def __init__(self, Hx: np.ndarray, Hz: np.ndarray, max_cluster: int = 256) -> None:
        self.Hx = (Hx.astype(np.uint8) & 1)
        self.Hz = (Hz.astype(np.uint8) & 1)
        self.max_cluster = int(max_cluster)

    def _decode_one(
        self,
        H: np.ndarray,
        s: np.ndarray,
        erase_cols: np.ndarray,
        erase_rows: np.ndarray | None,
    ) -> np.ndarray:
        x_part, H_res, s_res, col_ids_res = _peeling(H, s, erase_cols, erase_rows)
        if H_res.size == 0:
            return x_part
        x_local = _solve_clusters(H_res, s_res, col_ids_res, self.max_cluster)
        if x_local.size:
            x_part = x_part.copy()
            x_part[col_ids_res] ^= x_local
        return x_part

    def decode_batch(
        self,
        syndromes_x: np.ndarray,
        syndromes_z: np.ndarray,
        erase_data_mask: np.ndarray,
        erase_det_mask: np.ndarray | None = None,
    ) -> Dict[str, Any]:
        syndromes_x = np.asarray(syndromes_x, dtype=np.uint8)
        syndromes_z = np.asarray(syndromes_z, dtype=np.uint8)
        erase_data_mask = np.asarray(erase_data_mask, dtype=np.uint8)
        if syndromes_x.ndim != 2 or syndromes_z.ndim != 2:
            raise ValueError("Syndromes must be batched")
        if syndromes_x.shape[0] != syndromes_z.shape[0]:
            raise ValueError("Batch sizes for X and Z must match")
        if erase_data_mask.shape[0] != syndromes_x.shape[0]:
            raise ValueError("erase_data_mask batch mismatch")
        B = syndromes_x.shape[0]
        n = self.Hx.shape[1]
        ex = np.zeros((B, n), dtype=np.uint8)
        ez = np.zeros((B, n), dtype=np.uint8)
        for b in range(B):
            erows_x = None
            erows_z = None
            if erase_det_mask is not None:
                rows_mask = erase_det_mask[b]
                erows_x = rows_mask[: self.Hx.shape[0]]
                erows_z = rows_mask[self.Hx.shape[0] : self.Hx.shape[0] + self.Hz.shape[0]]
            ex[b] = self._decode_one(self.Hx, syndromes_x[b], erase_data_mask[b], erows_x)
            ez[b] = self._decode_one(self.Hz, syndromes_z[b], erase_data_mask[b], erows_z)
        return {"which": "erasure_peeling", "ex": ex, "ez": ez}
