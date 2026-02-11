from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable, Optional, Any, Sequence

try:  # Optional GPU acceleration
    import torch  # type: ignore

    _TORCH_OK = True
except Exception:  # pragma: no cover
    _TORCH_OK = False


@dataclass
class Cluster:
    """Simple cluster container for legacy API compatibility."""

    check_indices: np.ndarray
    qubit_indices: Optional[np.ndarray] = None
    side: Optional[str] = None


def _as_dense_uint8(M) -> np.ndarray:
    """Accept scipy sparse or numpy arrays; return uint8 {0,1}."""
    if hasattr(M, "toarray"):
        A = M.toarray()
    else:
        A = np.asarray(M)
    return A.astype(np.uint8) & 1


def gf2_row_echelon(A: np.ndarray, pivot_cols: int | None = None):
    """Gaussian elimination over GF(2) returning row-echelon form and pivots.

    Parameters
    - A: binary m×n ndarray (uint8/0-1)

    Returns
    - R: row-echelon form over GF(2)
    - pivots: list of (row, col) pivot positions
    """
    A = (A & 1).astype(np.uint8).copy()
    m, n = A.shape
    n_pivot = n if pivot_cols is None else max(0, min(int(pivot_cols), n))
    pivots = []
    r = 0
    for c in range(n_pivot):
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
    # Only pivot on the original H columns (not the augmented RHS column).
    R, piv = gf2_row_echelon(A, pivot_cols=Hn.shape[1])
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


def gf2_nullspace_basis(H: sp.csr_matrix) -> np.ndarray:
    """Return a dense uint8 matrix whose columns form a GF(2) nullspace basis of H.

    Alias of ``gf2_nullspace`` exposed for clarity in calling sites where the
    intent is to request a basis rather than an algorithm-specific structure.
    Shape is ``(n, r)`` where ``n`` is the number of columns of H and ``r`` is
    the nullity.
    """
    return gf2_nullspace(H)


def gf2_project_to_coset(
    H: sp.csr_matrix,
    s: np.ndarray,
    e_hint: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return one correction e in the coset {e : H e = s}.

    Computes any particular solution ``e0`` such that ``H e0 = s`` and, when a
    hint ``e_hint`` is supplied, nudges toward it within the coset by solving
    ``N t = (e_hint ⊕ e0)`` for a nullspace basis ``N`` of ``H``. If the system
    has empty nullspace or the hint is absent, simply returns ``e0``.

    This is intended for projection-aware postprocessing of model outputs in
    training/eval, keeping parity constraints exactly satisfied.
    """
    Hs = H.tocsr() if not isinstance(H, sp.csr_matrix) else H
    e0 = gf2_solve_particular(Hs, np.asarray(s, dtype=np.uint8).ravel())
    N = gf2_nullspace(Hs)  # n x r
    if e_hint is None or N.shape[1] == 0:
        return e0.astype(np.uint8)
    d = np.asarray(e_hint, dtype=np.uint8).ravel() ^ e0  # desired delta within coset
    try:
        # Solve N t = d (mod 2); returns a length-r vector t
        t = gf2_solve_particular(sp.csr_matrix(N), d)
    except Exception:
        return e0.astype(np.uint8)
    adj = (N @ (t & 1)) & 1
    return (e0 ^ adj).astype(np.uint8)


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
    """Cost change if we XOR `column` into current state `e` under weights `w`."""
    mask = column.astype(bool)
    if not mask.any():
        return 0.0
    bits = e[mask].astype(np.float64)
    return float(np.dot(w[mask], 1.0 - 2.0 * bits))


def greedy_parity_project(
    H_sub: sp.csr_matrix, s_sub: np.ndarray, p_flip: np.ndarray, thresh: float = 0.5
) -> np.ndarray:
    """Greedy correction that reduces parity residual using weighted heuristic.

    Not exact ML, but cheap and effective when exact search is too large.
    """
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

    try:
        e0 = gf2_solve_particular(sp.csr_matrix(H_sub), s_sub)  # particular
    except Exception:
        if stats_out is not None:
            stats_out.update(states_visited=0, states_pruned=0)
        return greedy_parity_project(sp.csr_matrix(H_sub), s_sub, p_flip)
    N = gf2_nullspace(sp.csr_matrix(H_sub))  # n_sub × r
    r = N.shape[1]

    if r == 0:
        if stats_out is not None:
            stats_out.update(states_visited=1, states_pruned=0)
        return e0
    if r > r_cap:
        if stats_out is not None:
            stats_out.update(states_visited=0, states_pruned=0)
        return greedy_parity_project(sp.csr_matrix(H_sub), s_sub, p_flip)

    N_bool = N != 0
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


def ml_parity_project_torch(
    H_sub: sp.csr_matrix | np.ndarray,
    s_sub: np.ndarray,
    p_flip: np.ndarray | None = None,
    r_cap: int = 24,
    stats_out: Dict[str, int] | None = None,
    probs_local: np.ndarray | None = None,
    device: str = "cuda",
) -> np.ndarray:
    """Torch-accelerated ML parity projection for small nullspaces (r≤24).

    Falls back to ml_parity_project when torch is unavailable or r is large.
    """
    if not _TORCH_OK:
        return ml_parity_project(H_sub, s_sub, p_flip, r_cap, stats_out, probs_local)
    eps = 1e-6
    if probs_local is not None and p_flip is None:
        p_flip = probs_local
    if p_flip is None:
        p_flip = np.full(_as_dense_uint8(H_sub).shape[1], 0.5, dtype=np.float64)
    p = np.clip(np.asarray(p_flip, dtype=np.float64), eps, 1 - eps)
    w = np.log((1 - p) / p).astype(np.float32)

    Hs = sp.csr_matrix(H_sub)
    try:
        e0 = gf2_solve_particular(Hs, s_sub)
    except Exception:
        if stats_out is not None:
            stats_out.update(states_visited=0, states_pruned=0)
        return greedy_parity_project(Hs, s_sub, p_flip)
    N = gf2_nullspace(Hs)
    r = int(N.shape[1])
    if r == 0 or r > r_cap:
        return ml_parity_project(H_sub, s_sub, p_flip, r_cap, stats_out, probs_local)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    w_t = torch.as_tensor(w, device=dev)
    e = torch.as_tensor(e0.astype(np.uint8), device=dev)
    cols = [torch.as_tensor(N[:, i].astype(np.uint8), device=dev) for i in range(r)]

    best_e = e.clone()
    best_cost = (w_t * e.float()).sum()
    visited = 0

    stack: list[tuple[int, torch.Tensor, torch.Tensor]] = [(0, e, best_cost)]
    while stack:
        idx, state, cost = stack.pop()
        if idx == r:
            visited += 1
            if cost < best_cost:
                best_cost = cost
                best_e = state.clone()
            continue
        col = cols[idx]
        # no flip
        stack.append((idx + 1, state, cost))
        # flip
        s2 = state ^ col
        delta = (w_t * (s2.float() - state.float())).sum()
        stack.append((idx + 1, s2, cost + delta))

    if stats_out is not None:
        stats_out.update(states_visited=int(visited), states_pruned=0)
    return best_e.detach().to("cpu").numpy().astype(np.uint8)


def active_components(
    H: sp.csr_matrix | np.ndarray,
    s: np.ndarray,
    *,
    halo: int = 0,
    basis: str | None = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]] | List[Cluster]:
    """
    Build clusters from active checks (rows where s==1).
    Returns (list_of_check_idx_arrays, list_of_qubit_idx_arrays) in GLOBAL indices.
    """
    s = np.asarray(s, dtype=np.uint8).ravel()
    rows = np.flatnonzero(s)
    if rows.size == 0:
        return [], []

    Hc = H if sp.issparse(H) else sp.csr_matrix(np.asarray(H, dtype=np.uint8))
    H_act = Hc[rows, :]
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

    Hc = Hc.tocsr()
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

    # Legacy compatibility: when a 'basis' kwarg is supplied (as in older tests),
    # return a list of Cluster objects with only check_indices populated using
    # adjacency among active checks via shared qubits.
    if basis is not None:
        # Build check-level adjacency among active rows
        # share[i,j] = number of shared data qubits between checks i and j
        share = (Hc @ Hc.T).tocsr()
        share.data[:] = 1
        share.setdiag(0)
        share.eliminate_zeros()
        active = set(map(int, rows.tolist()))
        seen: set[int] = set()
        clusters: list[Cluster] = []
        for r0 in rows:
            r0i = int(r0)
            if r0i in seen:
                continue
            comp = []
            dq = deque([r0i])
            seen.add(r0i)
            while dq:
                u = dq.popleft()
                comp.append(u)
                lo, hi = share.indptr[u], share.indptr[u + 1]
                for v in share.indices[lo:hi]:
                    vi = int(v)
                    if vi in active and vi not in seen:
                        seen.add(vi)
                        dq.append(vi)
            clusters.append(Cluster(check_indices=np.array(sorted(comp), dtype=np.int64)))
        return clusters
    return check_comps, qubit_comps


def extract_subproblem(
    H: sp.csr_matrix, s: np.ndarray, checks_idx: np.ndarray, qubits_idx: np.ndarray
) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """Extract a local subproblem H_sub, s_sub with index maps back to globals."""
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
    """Solve H e = s with columns (and optionally rows) erased.

    Columns masked true are treated as variables to solve; rows masked true are
    dropped. Returns a full-length solution vector (zeros else).
    """
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
    """Convenience projector for a batch of independent CSS subproblems."""
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
        side: str | None = None,
        halo: int = 0,
        component_scope: str = "active",
        thresh: float = 0.5,
        temp: float = 1.0,
        r_cap: int = 20,
        projection_mode: str = "if_needed",
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
        scope = str(component_scope).strip().lower()
        if scope not in {"active", "full"}:
            scope = "active"
        self.component_scope = scope
        self.thresh = float(thresh)
        self.temp = float(temp)
        self.r_cap = int(r_cap)
        mode = str(projection_mode).strip().lower()
        if mode not in {"always", "if_needed", "none"}:
            mode = "if_needed"
        self.projection_mode = mode
        self.mb_mode = "batched" if batched else "unbatched"
        self.side = self._infer_side(side)
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

    def _infer_side(self, side: str | None) -> str:
        if side is None:
            return "Z"
        value = str(side).strip().upper()
        if value not in {"X", "Z"}:
            raise ValueError(f"Invalid side='{side}'. Expected 'X' or 'Z'.")
        return value

    def decode(self, s: np.ndarray, perf_only: bool = False) -> Dict[str, Any]:
        s = np.asarray(s, dtype=np.uint8).ravel()
        H = self.H

        subproblems: List[Dict[str, Any]] = []
        if self.component_scope == "full":
            q_l2g = np.arange(H.shape[1], dtype=np.int64)
            c_l2g = np.arange(H.shape[0], dtype=np.int64)
            H_sub = H
            s_sub = s
            xy_qubit = self.coords_qubit[q_l2g]
            xy_check = self.coords_check[c_l2g]
            all_coords = np.vstack([xy_qubit, xy_check]) if xy_check.size else xy_qubit
            mins = all_coords.min(axis=0) if all_coords.size else np.array([0.0, 0.0], dtype=np.float32)
            maxs = all_coords.max(axis=0) if all_coords.size else np.array([0.0, 0.0], dtype=np.float32)
            bbox_xywh = (
                int(mins[0]),
                int(mins[1]),
                int(maxs[0] - mins[0] + 1),
                int(maxs[1] - mins[1] + 1),
            )
            k_local = int(H_sub.shape[0])
            r_local = int(H_sub.shape[1])
            subproblems.append(
                {
                    "H_sub": H_sub,
                    "s_sub": s_sub,
                    "q_l2g": q_l2g,
                    "c_l2g": c_l2g,
                    "extra": {
                        "xy_qubit": xy_qubit,
                        "xy_check": xy_check,
                        "k": k_local,
                        "r": r_local,
                        "bbox_xywh": bbox_xywh,
                        "kappa_stats": {
                            "k": k_local,
                            "r": r_local,
                            "density": float(k_local / max(1, r_local)),
                            "syndrome_weight": int(np.asarray(s_sub, dtype=np.uint8).sum()),
                            "scope": "full",
                        },
                        "side": self.side,
                        "d": self.distance,
                        "p": float(self.default_p or 0.01),
                        "seed": 0,
                        "add_jump_edges": False,
                        "jump_k": 1,
                    },
                }
            )
        else:
            check_comps, qubit_comps = active_components(H, s, halo=self.halo)
            for checks, qubits in zip(check_comps, qubit_comps):
                H_sub, s_sub, q_l2g, c_l2g = extract_subproblem(H, s, checks, qubits)
                xy_qubit = self.coords_qubit[q_l2g]
                xy_check = self.coords_check[c_l2g]
                all_coords = np.vstack([xy_qubit, xy_check]) if xy_check.size else xy_qubit
                mins = (
                    all_coords.min(axis=0)
                    if all_coords.size
                    else np.array([0.0, 0.0], dtype=np.float32)
                )
                maxs = (
                    all_coords.max(axis=0)
                    if all_coords.size
                    else np.array([0.0, 0.0], dtype=np.float32)
                )
                bbox_xywh = (
                    int(mins[0]),
                    int(mins[1]),
                    int(maxs[0] - mins[0] + 1),
                    int(maxs[1] - mins[1] + 1),
                )
                k_local = int(H_sub.shape[0])  # local checks
                r_local = int(H_sub.shape[1])  # local data qubits
                extra_meta = {
                    "xy_qubit": xy_qubit,
                    "xy_check": xy_check,
                    "k": k_local,
                    "r": r_local,
                    "bbox_xywh": bbox_xywh,
                    "kappa_stats": {
                        "k": k_local,
                        "r": r_local,
                        "density": float(k_local / max(1, r_local)),
                        "syndrome_weight": int(np.asarray(s_sub, dtype=np.uint8).sum()),
                    },
                    "side": self.side,
                    "d": self.distance,
                    "p": float(self.default_p or 0.01),
                    "seed": 0,
                    "add_jump_edges": False,
                    "jump_k": 1,
                }
                subproblems.append(
                    {
                        "H_sub": H_sub,
                        "s_sub": s_sub,
                        "q_l2g": q_l2g,
                        "c_l2g": c_l2g,
                        "extra": extra_meta,
                    }
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
                (entry["H_sub"], entry["s_sub"], entry["q_l2g"], entry["c_l2g"], entry["extra"])
                for entry in subproblems
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
            mb_report = {
                "fast_path_batches": 0,
                "fixed_d3_batches": 0,
                "fallback_loops": 0,
                "batch_sizes": [],
                "graph_used": False,
                "device": {"device": str(getattr(self.mghd, "device", "cpu"))},
            }
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
                mb_report["graph_used"] = mb_report.get("graph_used", False) or sub_report.get(
                    "graph_used", False
                )
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
            target = s_sub.astype(np.uint8) % 2
            raw_bits = (np.asarray(probs, dtype=np.float64) > float(self.thresh)).astype(np.uint8)
            if self.projection_mode == "none":
                e_sub = raw_bits
            else:
                use_projection = True
                if self.projection_mode == "if_needed":
                    raw_parity = np.asarray((H_sub @ raw_bits) % 2).ravel().astype(np.uint8) % 2
                    if np.array_equal(raw_parity, target):
                        use_projection = False
                if use_projection:
                    if _TORCH_OK:
                        try:
                            e_sub = ml_parity_project_torch(H_sub, s_sub, probs, r_cap=self.r_cap)
                        except Exception:
                            e_sub = ml_parity_project(H_sub, s_sub, probs, r_cap=self.r_cap)
                    else:
                        e_sub = ml_parity_project(H_sub, s_sub, probs, r_cap=self.r_cap)
                    parity = np.asarray((H_sub @ e_sub) % 2).ravel().astype(np.uint8) % 2
                    if not np.array_equal(parity, target):
                        e_sub = greedy_parity_project(H_sub, s_sub, probs)
                else:
                    e_sub = raw_bits
            t_proj += (self._time.perf_counter() - t_p0) * 1e6
            e[q_l2g] ^= e_sub.astype(np.uint8)

        return {
            "e": e,
            "sizes_hist": sizes_hist,
            "mghd_invoked": t_mghd,
            "proj_us": t_proj,
            "mghd_clusters": int(len(subproblems)),
        }


__all__ = [
    "Cluster",
    "gf2_row_echelon",
    "gf2_solve_particular",
    "gf2_nullspace_basis",
    "gf2_project_to_coset",
    "gf2_nullspace",
    "ml_parity_project",
    "ml_parity_project_torch",
    "greedy_parity_project",
    "active_components",
    "extract_subproblem",
    "solve_on_erasure",
    "infer_clusters_batched",
    "MGHDPrimaryClustered",
]
