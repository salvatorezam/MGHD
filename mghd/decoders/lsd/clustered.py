from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable, Optional, Any, Sequence


@dataclass
class Cluster:
    """Simple cluster container for legacy API compatibility."""
    check_indices: np.ndarray


def _as_dense_uint8(M) -> np.ndarray:
    """Accept scipy sparse or numpy arrays; return uint8 {0,1}."""
    if hasattr(M, "toarray"):
        A = M.toarray()
    else:
        A = np.asarray(M)
    return (A.astype(np.uint8) & 1)


def gf2_row_echelon(A: np.ndarray):
    A = (A & 1).astype(np.uint8).copy()
    m, n = A.shape
    pivots = []
    r = 0
    for c in range(n):
        idx = None
        for i in range(r, m):
            if A[i, c]:
                idx = i
                break
        if idx is None:
            continue
        if idx != r:
            A[[r, idx]] = A[[idx, r]]
        for i in range(r + 1, m):
            if A[i, c]:
                A[i, :] ^= A[r, :]
        pivots.append((r, c))
        r += 1
        if r == m:
            break
    return A, pivots


def gf2_solve_particular(H: sp.csr_matrix, s: np.ndarray) -> np.ndarray:
    """Return one e0 with H e0 = s (mod2). Raises if inconsistent."""
    Hn = _as_dense_uint8(H)
    b = np.asarray(s, dtype=np.uint8).ravel().copy()
    A = np.concatenate([Hn, b[:, None]], axis=1)
    R, piv = gf2_row_echelon(A)
    m, n1 = Hn.shape
    e = np.zeros(n1, dtype=np.uint8)
    for r, c in reversed(piv):
        rhs = R[r, n1]
        ssum = 0
        for j in range(c + 1, n1):
            if R[r, j] and e[j]:
                ssum ^= 1
        e[c] = rhs ^ ssum
    for i in range(m):
        if not R[i, :n1].any() and R[i, n1]:
            raise ValueError("Inconsistent system H e = s over GF(2)")
    return e


def gf2_nullspace(H: sp.csr_matrix):
    """Return basis vectors (columns) of nullspace of H over GF(2) as dense uint8 matrix N (n×r)."""
    Hn = _as_dense_uint8(H)
    m, n = Hn.shape
    R, piv = gf2_row_echelon(Hn)
    pivot_cols = {c for _, c in piv}
    free_cols = [c for c in range(n) if c not in pivot_cols]
    basis = []
    for f in free_cols:
        v = np.zeros(n, dtype=np.uint8)
        v[f] = 1
        for r, c in reversed(piv):
            ssum = 0
            for j in range(c + 1, n):
                if R[r, j] and v[j]:
                    ssum ^= 1
            v[c] = ssum
        basis.append(v)
    if not basis:
        return np.zeros((n, 0), dtype=np.uint8)
    return np.stack(basis, axis=1)


# Tier-0 defaults for small-cluster solver
TIER0_K_MAX = 3
TIER0_R_MAX = 6


def solve_small_cluster_channel_ml(
    H_sub: sp.csr_matrix,
    s_sub: np.ndarray,
    *,
    p_channel: float,
    k_max: int = TIER0_K_MAX,
    r_cap: int = TIER0_R_MAX,
) -> Optional[np.ndarray]:
    """Channel-only exact ML for tiny clusters; returns None if cluster too large."""
    m_sub, n_sub = H_sub.shape
    e0 = gf2_solve_particular(H_sub, s_sub)
    N = gf2_nullspace(H_sub)
    r = N.shape[1]
    if not (n_sub <= k_max or r <= r_cap):
        return None

    eps = 1e-9
    p = np.clip(float(p_channel), eps, 1 - eps)
    w = np.full(n_sub, np.log((1 - p) / p), dtype=np.float64)

    best_e = e0.copy()
    best_cost = float(np.dot(w, best_e))

    if r == 0:
        pass
    else:
        for z_int in range(1 << r):
            e = e0.copy()
            bits = z_int
            k = 0
            while bits:
                if bits & 1:
                    e ^= N[:, k]
                bits >>= 1
                k += 1
            cost = float(np.dot(w, e))
            if cost < best_cost:
                best_cost = cost
                best_e = e

    parity = (H_sub @ best_e) % 2
    if not np.array_equal(parity.astype(np.uint8) % 2, s_sub.astype(np.uint8) % 2):
        raise AssertionError("Tier-0 solver produced invalid correction")
    return best_e.astype(np.uint8)


def _delta_cost(e: np.ndarray, column: np.ndarray, w: np.ndarray) -> float:
    mask = column.astype(bool)
    if not mask.any():
        return 0.0
    bits = e[mask].astype(np.float64)
    return float(np.dot(w[mask], 1.0 - 2.0 * bits))


def greedy_parity_project(H_sub: sp.csr_matrix, s_sub: np.ndarray, p_flip: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    m_sub, n_sub = H_sub.shape
    e = (p_flip > thresh).astype(np.uint8)
    r = (H_sub @ e) % 2
    r = (r.astype(np.uint8) ^ (s_sub.astype(np.uint8))).astype(np.uint8)
    if r.sum() == 0:
        return e
    eps = 1e-6
    logit = lambda x: np.log(np.clip(x, eps, 1 - eps)) - np.log(np.clip(1 - x, eps, 1 - eps))
    g = np.abs(logit(1 - p_flip) - logit(p_flip))
    Hc = H_sub.tocsr()
    safety = 4 * (m_sub + n_sub)
    it = 0
    while r.sum() > 0 and it < safety:
        it += 1
        i = int(np.flatnonzero(r)[0])
        lo, hi = Hc.indptr[i], Hc.indptr[i + 1]
        cols = Hc.indices[lo:hi]
        if cols.size == 0:
            break
        j = cols[np.argmax(g[cols])]
        e[j] ^= 1
        r = (r ^ (Hc[:, j].toarray().ravel().astype(np.uint8))).astype(np.uint8)
    return e


def ml_parity_project(
    H_sub: sp.csr_matrix | np.ndarray,
    s_sub: np.ndarray,
    p_flip: np.ndarray | None = None,
    r_cap: int = 20,
    stats_out: Dict[str, int] | None = None,
    probs_local: np.ndarray | None = None,
) -> np.ndarray:
    """Exact ML projection under independent bit model; fall back to greedy when r is large."""
    eps = 1e-6
    if probs_local is not None and p_flip is None:
        p_flip = probs_local
    if p_flip is None:
        p_flip = np.full(_as_dense_uint8(H_sub).shape[1], 0.5, dtype=np.float64)
    p = np.clip(np.asarray(p_flip, dtype=np.float64), eps, 1 - eps)
    w = np.log((1 - p) / p)

    e0 = gf2_solve_particular(sp.csr_matrix(H_sub), s_sub)  # particular
    N = gf2_nullspace(sp.csr_matrix(H_sub))                  # n_sub × r
    r = N.shape[1]

    if r == 0:
        if stats_out is not None:
            stats_out.update(states_visited=1, states_pruned=0)
        return e0
    if r > r_cap:
        if stats_out is not None:
            stats_out.update(states_visited=0, states_pruned=0)
        return greedy_parity_project(sp.csr_matrix(H_sub), s_sub, p_flip)

    N_bool = (N != 0)
    columns = [N[:, idx].astype(np.uint8) for idx in range(r)]
    suffix_cover = np.zeros((r + 1, N.shape[0]), dtype=bool)
    for idx in range(r - 1, -1, -1):
        suffix_cover[idx] = suffix_cover[idx + 1] | N_bool[:, idx]

    e = e0.copy()
    cost = float(np.dot(w, e))
    best_cost = cost
    best_e = e.copy()
    visited = 0
    pruned = 0

    def lower_bound(idx: int) -> float:
        cover = suffix_cover[idx]
        fixed = ~cover
        bound = 0.0
        if fixed.any():
            bound += float(np.dot(w[fixed], e[fixed].astype(np.float64)))
        if cover.any():
            bound += float(np.sum(np.minimum(0.0, w[cover])))
        return bound

    def search(idx: int) -> None:
        nonlocal cost, best_cost, best_e, visited, pruned, e
        lb = lower_bound(idx)
        if lb >= best_cost - 1e-12:
            pruned += 1
            return
        if idx == r:
            visited += 1
            if cost < best_cost - 1e-12:
                best_cost = cost
                best_e = e.copy()
            return

        column = columns[idx]
        delta = _delta_cost(e, column, w)
        if delta < 0:
            e ^= column
            cost += delta
            search(idx + 1)
            cost -= delta
            e ^= column
            search(idx + 1)
        else:
            search(idx + 1)
            e ^= column
            cost += delta
            search(idx + 1)
            cost -= delta
            e ^= column

    search(0)
    if stats_out is not None:
        stats_out.update(states_visited=visited, states_pruned=pruned)
    return best_e.astype(np.uint8)


def active_components(H: sp.csr_matrix, s: np.ndarray, *, halo: int = 0) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Build clusters from active checks (rows where s==1).
    Returns (list_of_check_idx_arrays, list_of_qubit_idx_arrays) in GLOBAL indices.
    """
    s = np.asarray(s, dtype=np.uint8).ravel()
    rows = np.flatnonzero(s)
    if rows.size == 0:
        return [], []

    H_act = H[rows, :]
    A = (H_act.T @ H_act).tocsr()
    A.data[:] = 1
    A.setdiag(0)
    A.eliminate_zeros()

    touched_q = np.flatnonzero((H_act != 0).sum(axis=0).A.ravel() > 0)
    seen = np.zeros(H.shape[1], dtype=bool)
    qubit_comps: List[np.ndarray] = []

    for q in touched_q:
        if seen[q]:
            continue
        comp = []
        dq = deque([q])
        seen[q] = True
        while dq:
            u = dq.popleft()
            comp.append(u)
            lo, hi = A.indptr[u], A.indptr[u + 1]
            for v in A.indices[lo:hi]:
                if not seen[v]:
                    seen[v] = True
                    dq.append(v)
        qubit_comps.append(np.array(comp, dtype=np.int64))

    Hc = H.tocsr()
    check_comps: List[np.ndarray] = []
    for i, comp in enumerate(qubit_comps):
        sub = Hc[:, comp]
        checks = np.flatnonzero((sub != 0).sum(axis=1).A.ravel() > 0)
        if halo > 0:
            sub2 = Hc[checks, :]
            halo_q = np.flatnonzero((sub2 != 0).sum(axis=0).A.ravel() > 0)
            comp = np.unique(np.concatenate([comp, halo_q]))
            qubit_comps[i] = comp
        check_comps.append(checks)

    return check_comps, qubit_comps


def extract_subproblem(
    H: sp.csr_matrix, s: np.ndarray, checks_idx: np.ndarray, qubits_idx: np.ndarray
) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    checks_idx = np.asarray(checks_idx, dtype=np.int64)
    qubits_idx = np.asarray(qubits_idx, dtype=np.int64)
    H_sub = H[checks_idx, :][:, qubits_idx].tocsr()
    s_sub = np.asarray(s, dtype=np.uint8).ravel()[checks_idx]
    q_l2g = qubits_idx.copy()
    c_l2g = checks_idx.copy()
    return H_sub, s_sub, q_l2g, c_l2g


def solve_on_erasure(
    H: np.ndarray, s: np.ndarray, mask_cols: np.ndarray, mask_rows: np.ndarray | None = None
) -> np.ndarray:
    H = np.asarray(H, dtype=np.uint8)
    s = np.asarray(s, dtype=np.uint8)
    mask_cols = np.asarray(mask_cols, dtype=np.uint8).astype(bool)
    if mask_rows is None:
        rows_keep = np.ones(H.shape[0], dtype=bool)
    else:
        mask_rows = np.asarray(mask_rows, dtype=np.uint8).astype(bool)
        rows_keep = ~mask_rows
    H_sub = H[rows_keep][:, mask_cols]
    s_sub = s[rows_keep]
    if not mask_cols.any() or not rows_keep.any():
        return np.zeros(H.shape[1], dtype=np.uint8)
    try:
        x_erased = gf2_solve_particular(sp.csr_matrix(H_sub), s_sub)
    except (ValueError, AssertionError):
        x_erased = np.zeros(mask_cols.sum(), dtype=np.uint8)
    x = np.zeros(H.shape[1], dtype=np.uint8)
    x[mask_cols] = x_erased
    return x


def infer_clusters_batched(Hx: np.ndarray, Hz: np.ndarray, SX: np.ndarray, SZ: np.ndarray):
    B = SX.shape[0]
    ex = np.zeros((B, Hx.shape[1]), dtype=np.uint8)
    ez = np.zeros((B, Hz.shape[1]), dtype=np.uint8)
    for b in range(B):
        ex[b] = ml_parity_project(Hx, SX[b], p_flip=None)
        ez[b] = ml_parity_project(Hz, SZ[b], p_flip=None)
    return ex, ez


class MGHDPrimaryClustered:
    """
    Primary clustered decoder that works with MGHD v2 models via MGHDDecoderPublic.
    Splits syndrome into clusters, queries MGHD for priors, and projects via ML parity.
    """

    def __init__(
        self,
        H: sp.csr_matrix,
        mghd: Any,
        *,
        halo: int = 0,
        thresh: float = 0.5,
        temp: float = 1.0,
        r_cap: int = 20,
        batched: bool = True,
        tier0_enable: bool = True,
        tier0_k_max: int | None = None,
        tier0_r_max: int | None = None,
        tier0_mode: str = "mixed",
        p_channel: float | None = None,
        default_p: float | None = None,
        bucket_spec: Sequence[Tuple[int, int, int]] | None = None,
        microbatch: int = 64,
        flush_ms: float = 1.0,
    ):
        import time  # local import to avoid test-time deps unless used
        self._time = time

        self.H = H.tocsr()
        self.mghd = mghd
        self.halo = int(halo)
        self.thresh = float(thresh)
        self.temp = float(temp)
        self.r_cap = int(r_cap)
        self.mb_mode = "batched" if batched else "unbatched"
        self.side = self._infer_side()
        self.tier0_enable = bool(tier0_enable)
        self.model_version = getattr(self.mghd, "model_version", "v2")
        self.distance = self._infer_distance()
        self.coords_qubit = self._compute_qubit_coords(self.distance)
        self.coords_check = self._compute_check_coords()

        if tier0_k_max is None and tier0_r_max is None:
            if tier0_mode in ("mixed", "mixed_tight", "aggressive"):
                self.tier0_k_max, self.tier0_r_max = 2, 1
            elif tier0_mode == "off":
                self.tier0_k_max, self.tier0_r_max = 0, 0
            else:
                self.tier0_k_max, self.tier0_r_max = TIER0_K_MAX, TIER0_R_MAX
        else:
            self.tier0_k_max = int(tier0_k_max or TIER0_K_MAX)
            self.tier0_r_max = int(tier0_r_max or TIER0_R_MAX)

        self.p_channel = p_channel
        self.default_p = default_p
        self.bucket_spec = tuple(bucket_spec) if bucket_spec else None
        self.microbatch = max(1, int(microbatch))
        self.flush_ms = max(0.0, float(flush_ms))

    def _infer_distance(self) -> int:
        n_qubits = int(self.H.shape[1])
        d = int(round(float(np.sqrt(n_qubits))))
        if d * d != n_qubits:
            d = n_qubits
        return max(1, d)

    def _compute_qubit_coords(self, d: int) -> np.ndarray:
        coords = []
        if d * d != self.H.shape[1]:
            for q in range(self.H.shape[1]):
                coords.append([float(q), 0.0])
        else:
            for r in range(d):
                for c in range(d):
                    coords.append([float(r + c), float(r - c)])
        return np.asarray(coords, dtype=np.float32)

    def _compute_check_coords(self) -> np.ndarray:
        m = self.H.shape[0]
        coords = np.zeros((m, 2), dtype=np.float32)
        for i in range(m):
            row = self.H.getrow(i)
            idx = row.indices
            if idx.size == 0:
                continue
            coords[i] = self.coords_qubit[idx].mean(axis=0)
        return coords

    @staticmethod
    def _histogram_from_sizes(sizes: Sequence[int]) -> Dict[str, int]:
        hist: Dict[str, int] = {}
        for size in sizes:
            bucket = 1
            while size > bucket:
                bucket *= 2
            key = str(bucket)
            hist[key] = hist.get(key, 0) + 1
        return hist

    def _sync_cuda(self) -> None:
        try:
            import torch
            if torch.cuda.is_available() and getattr(self.mghd, "device", None) is not None:
                if getattr(self.mghd.device, "type", "cpu") == "cuda":
                    torch.cuda.synchronize(self.mghd.device)
        except Exception:
            pass

    def _infer_side(self) -> str:
        return "Z"

    def decode(self, s: np.ndarray, perf_only: bool = False) -> Dict[str, Any]:
        s = np.asarray(s, dtype=np.uint8).ravel()
        H = self.H

        check_comps, qubit_comps = active_components(H, s, halo=self.halo)
        subproblems: List[Dict[str, Any]] = []
        for checks, qubits in zip(check_comps, qubit_comps):
            H_sub, s_sub, q_l2g, c_l2g = extract_subproblem(H, s, checks, qubits)
            extra_meta = {
                "xy_qubit": self.coords_qubit[q_l2g],
                "xy_check": self.coords_check[c_l2g],
                "k": int(H_sub.shape[1]),
                "r": int(H_sub.shape[1] - np.linalg.matrix_rank(_as_dense_uint8(H_sub))),
                "bbox": (0, 0, 1, 1),
                "kappa_stats": {"size": float(H_sub.shape[0] + H_sub.shape[1])},
                "side": self.side,
                "d": self.distance,
                "p": float(self.default_p or 0.01),
                "seed": 0,
            }
            subproblems.append(
                {"H_sub": H_sub, "s_sub": s_sub, "q_l2g": q_l2g, "c_l2g": c_l2g, "extra": extra_meta}
            )

        e = np.zeros(H.shape[1], dtype=np.uint8)
        sizes = [len(x["q_l2g"]) for x in subproblems]
        sizes_hist = self._histogram_from_sizes(sizes)

        mb_report: Dict[str, Any]
        probs_list: List[np.ndarray]
        if self.mb_mode == "batched" and subproblems:
            self._sync_cuda()
            t1 = self._time.perf_counter()
            items = [
                (entry["H_sub"], entry["s_sub"], entry["q_l2g"], entry["c_l2g"], entry["extra"]) for entry in subproblems
            ]
            probs_list, mb_report = self.mghd.priors_from_subgraphs_batched(
                items,
                temp=self.temp,
                bucket=self.side,
                bucket_spec=self.bucket_spec,
                microbatch=self.microbatch,
                flush_ms=self.flush_ms,
                use_graphs=getattr(self.mghd, "_graph_capture_enabled", False),
            )
            self._sync_cuda()
            t_mghd = (self._time.perf_counter() - t1) * 1e6
        else:
            probs_list = []
            t_mghd = 0.0
            mb_report = {"fast_path_batches": 0, "fixed_d3_batches": 0, "fallback_loops": 0, "batch_sizes": [], "graph_used": False, "device": {"device": str(getattr(self.mghd, "device", "cpu"))}}
            for entry in subproblems:
                H_sub = entry["H_sub"]
                s_sub = entry["s_sub"]
                q_l2g = entry["q_l2g"]
                c_l2g = entry["c_l2g"]
                extra = entry.get("extra")
                self._sync_cuda()
                t1 = self._time.perf_counter()
                probs_local, sub_report = self.mghd.priors_from_subgraphs_batched(
                    [(H_sub, s_sub, q_l2g, c_l2g, extra)],
                    temp=self.temp,
                    bucket=self.side,
                    bucket_spec=self.bucket_spec,
                    microbatch=self.microbatch,
                    flush_ms=self.flush_ms,
                    use_graphs=getattr(self.mghd, "_graph_capture_enabled", False),
                )
                probs = np.asarray(probs_local[0], dtype=np.float64)
                for key, value in sub_report.items():
                    if key == "batch_sizes":
                        mb_report["batch_sizes"].extend(value)
                    elif key in {"fast_path_batches", "fixed_d3_batches", "fallback_loops"}:
                        mb_report[key] = mb_report.get(key, 0) + value
                mb_report["graph_used"] = mb_report.get("graph_used", False) or sub_report.get("graph_used", False)
                self._sync_cuda()
                t_mghd += (self._time.perf_counter() - t1) * 1e6
                probs_list.append(probs)

        t_proj = 0.0
        for entry, probs in zip(subproblems, probs_list):
            H_sub = entry["H_sub"]
            s_sub = entry["s_sub"]
            q_l2g = entry["q_l2g"]
            if probs.shape[0] != H_sub.shape[1]:
                raise AssertionError("Probability vector length mismatch for subgraph")
            t_p0 = self._time.perf_counter()
            e_sub = ml_parity_project(H_sub, s_sub, probs, r_cap=self.r_cap)
            t_proj += (self._time.perf_counter() - t_p0) * 1e6
            parity = (H_sub @ e_sub) % 2
            parity = np.asarray(parity).ravel().astype(np.uint8) % 2
            if not np.array_equal(parity, s_sub % 2):
                raise AssertionError("ML parity projection failed to satisfy local checks")
            e[q_l2g] ^= e_sub.astype(np.uint8)

        return {
            "e": e,
            "sizes_hist": sizes_hist,
            "mghd_invoked": t_mghd,
            "proj_us": t_proj,
        }


__all__ = [
    "Cluster",
    "gf2_row_echelon",
    "gf2_solve_particular",
    "gf2_nullspace",
    "ml_parity_project",
    "greedy_parity_project",
    "active_components",
    "extract_subproblem",
    "solve_on_erasure",
    "infer_clusters_batched",
    "MGHDPrimaryClustered",
]

