from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable, Optional


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
        # find a row with 1 in column c at or below r
        idx = None
        for i in range(r, m):
            if A[i, c]:
                idx = i; break
        if idx is None:
            continue
        if idx != r:
            A[[r, idx]] = A[[idx, r]]
        # eliminate below
        for i in range(r+1, m):
            if A[i, c]:
                A[i, :] ^= A[r, :]
        pivots.append((r, c))
        r += 1
        if r == m: break
    return A, pivots


def gf2_solve_particular(H: sp.csr_matrix, s: np.ndarray) -> np.ndarray:
    """Return one e0 with H e0 = s (mod2). Raises if inconsistent (shouldn't for valid syndromes)."""
    Hn = _as_dense_uint8(H)
    b = np.asarray(s, dtype=np.uint8).ravel().copy()
    A = np.concatenate([Hn, b[:, None]], axis=1)
    R, piv = gf2_row_echelon(A)
    m, n1 = Hn.shape
    # back-substitution (reduced row echelon not required)
    e = np.zeros(n1, dtype=np.uint8)
    for r, c in reversed(piv):
        # sum of knowns in row r excluding col c
        rhs = R[r, n1]
        ssum = 0
        for j in range(c+1, n1):
            if R[r, j] and e[j]:
                ssum ^= 1
        e[c] = rhs ^ ssum
    # (optional) detect inconsistency: a zero row with RHS 1
    for i in range(m):
        if not R[i, :n1].any() and R[i, n1]:
            raise ValueError("Inconsistent system H e = s over GF(2)")
    return e


def gf2_nullspace(H: sp.csr_matrix):
    """Return basis vectors (columns) of the nullspace of H over GF(2) as a dense uint8 matrix N (n×r)."""
    Hn = _as_dense_uint8(H)
    m, n = Hn.shape
    R, piv = gf2_row_echelon(Hn)
    pivot_cols = {c for _, c in piv}
    free_cols = [c for c in range(n) if c not in pivot_cols]
    basis = []
    for f in free_cols:
        v = np.zeros(n, dtype=np.uint8); v[f] = 1
        # set pivot variables by back-substitution
        for r, c in reversed(piv):
            ssum = 0
            for j in range(c+1, n):
                if R[r, j] and v[j]:
                    ssum ^= 1
            v[c] = ssum  # since row r has 1 at c and zeros above
        basis.append(v)
    if not basis:
        return np.zeros((n, 0), dtype=np.uint8)
    return np.stack(basis, axis=1)  # n × r


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


def ml_parity_project(H_sub: sp.csr_matrix | np.ndarray,
                      s_sub: np.ndarray,
                      p_flip: np.ndarray | None = None,
                      r_cap: int = 20,
                      stats_out: Dict[str, int] | None = None,
                      probs_local: np.ndarray | None = None) -> np.ndarray:
    """
    Exact ML projection under independent bit model:
      minimize w·e subject to H_sub e = s_sub (mod2), w_j = log((1-p)/p).
    If nullity r > r_cap, fall back to greedy.
    
    Args:
        H_sub: Parity check matrix (sparse or dense)
        s_sub: Syndrome vector
        p_flip: Flip probabilities (None = uniform 0.5). Also accepts probs_local for legacy compatibility.
        r_cap: Max nullity before fallback to greedy
        stats_out: Optional dict to store search stats
        probs_local: Legacy parameter name for p_flip
    """
    eps = 1e-6
    # Handle legacy probs_local parameter
    if probs_local is not None and p_flip is None:
        p_flip = probs_local
    if p_flip is None:
        # Back-compat: treat as uniform 0.5 (uninformative) with tiny tilt
        p_flip = np.full(_as_dense_uint8(H_sub).shape[1], 0.5, dtype=np.float64)
    p = np.clip(np.asarray(p_flip, dtype=np.float64), eps, 1 - eps)
    w = np.log((1 - p) / p)  # positive if p<0.5

    e0 = gf2_solve_particular(H_sub, s_sub)  # particular solution
    N = gf2_nullspace(H_sub)                 # n_sub × r
    r = N.shape[1]

    if r == 0:
        if stats_out is not None:
            stats_out.update(states_visited=1, states_pruned=0)
        return e0

    if r > r_cap:
        # fallback
        if stats_out is not None:
            stats_out.update(states_visited=0, states_pruned=0)
        return greedy_parity_project(H_sub, s_sub, p_flip)

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

        # Prefer beneficial flip branch first to tighten best_cost sooner
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


def _delta_cost(e: np.ndarray, column: np.ndarray, w: np.ndarray) -> float:
    mask = column.astype(bool)
    if not mask.any():
        return 0.0
    bits = e[mask].astype(np.float64)
    return float(np.dot(w[mask], 1.0 - 2.0 * bits))


def active_components(H: sp.csr_matrix, s: np.ndarray, *, halo: int = 0) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Build clusters from active checks (rows where s==1).
    - Qubit adjacency A = (H_act.T @ H_act) > 0  (two qubits adjacent if they share an active check)
    - Find connected components among qubits that are touched by any active check.
    - For each qubit component, collect the set of checks incident to those qubits.
    - Optional 'halo'=1: expand qubits by one hop via all checks (not only active).
    Returns (list_of_check_idx_arrays, list_of_qubit_idx_arrays) in GLOBAL indices.
    If s has no 1s, returns empty lists.
    """
    s = np.asarray(s, dtype=np.uint8).ravel()
    rows = np.flatnonzero(s)
    if rows.size == 0:
        return [], []

    H_act = H[rows, :]
    A = (H_act.T @ H_act).tocsr()
    A.data[:] = 1
    A.setdiag(0); A.eliminate_zeros()

    touched_q = np.flatnonzero((H_act != 0).sum(axis=0).A.ravel() > 0)
    seen = np.zeros(H.shape[1], dtype=bool)
    qubit_comps: List[np.ndarray] = []

    for q in touched_q:
        if seen[q]: continue
        comp = []
        dq = deque([q]); seen[q] = True
        while dq:
            u = dq.popleft(); comp.append(u)
            lo, hi = A.indptr[u], A.indptr[u+1]
            for v in A.indices[lo:hi]:
                if not seen[v]:
                    seen[v] = True; dq.append(v)
        qubit_comps.append(np.array(comp, dtype=np.int64))

    # collect checks incident to each qubit component
    Hc = H.tocsr()
    check_comps: List[np.ndarray] = []
    for i, comp in enumerate(qubit_comps):
        # checks incident to comp (any H[i, comp] != 0)
        sub = Hc[:, comp]
        checks = np.flatnonzero((sub != 0).sum(axis=1).A.ravel() > 0)

        if halo > 0:
            # 1-hop halo: add qubits incident to these checks
            sub2 = Hc[checks, :]
            halo_q = np.flatnonzero((sub2 != 0).sum(axis=0).A.ravel() > 0)
            comp = np.unique(np.concatenate([comp, halo_q]))
            # update qubit comp after halo
            qubit_comps[i] = comp

        check_comps.append(checks)

    return check_comps, qubit_comps


def extract_subproblem(H: sp.csr_matrix,
                       s: np.ndarray,
                       checks_idx: np.ndarray,
                       qubits_idx: np.ndarray) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """
    Slice H and s for one component; return:
      H_sub (csr), s_sub (uint8), maps: q_local2global, c_local2global
    Local ordering: first all checks (0..m_sub-1), then all qubits (m_sub..)
    """
    checks_idx = np.asarray(checks_idx, dtype=np.int64)
    qubits_idx = np.asarray(qubits_idx, dtype=np.int64)
    H_sub = H[checks_idx, :][:, qubits_idx].tocsr()
    s_sub = np.asarray(s, dtype=np.uint8).ravel()[checks_idx]
    q_l2g = qubits_idx.copy()
    c_l2g = checks_idx.copy()
    return H_sub, s_sub, q_l2g, c_l2g


def greedy_parity_project(H_sub: sp.csr_matrix,
                          s_sub: np.ndarray,
                          p_flip: np.ndarray,
                          thresh: float = 0.5) -> np.ndarray:
    """
    Fast local ML-ish repair on the subgraph:
      1) Start with e_hat = (p_flip > thresh)
      2) Residual r = s_sub ⊕ (H_sub @ e_hat)
      3) While r != 0: for each 1 in r, toggle the incident qubit with maximum gain
         (gain = |logit(1-p) - logit(p)|), update r, continue until r==0 or no progress.
    Returns e_hat ∈ {0,1}^{n_sub}.
    """
    m_sub, n_sub = H_sub.shape
    e = (p_flip > thresh).astype(np.uint8)
    # residual
    r = (H_sub @ e) % 2
    r = (r.astype(np.uint8) ^ (s_sub.astype(np.uint8))).astype(np.uint8)

    if r.sum() == 0:
        return e

    # precompute gains
    eps = 1e-6
    logit = lambda x: np.log(np.clip(x, eps, 1-eps)) - np.log(np.clip(1-x, eps, 1-eps))
    g = np.abs(logit(1 - p_flip) - logit(p_flip))  # larger = more confident

    Hc = H_sub.tocsr()
    # Greedy repairs
    safety = 4 * (m_sub + n_sub)  # guard against loops
    it = 0
    while r.sum() > 0 and it < safety:
        it += 1
        # pick a check with residual 1
        i = int(np.flatnonzero(r)[0])
        # qubits incident to i
        lo, hi = Hc.indptr[i], Hc.indptr[i+1]
        cols = Hc.indices[lo:hi]
        if cols.size == 0:
            # nothing to toggle (shouldn't happen in valid H); break to avoid infinite loop
            break
        # pick qubit with max gain (ties arbitrary)
        j = cols[np.argmax(g[cols])]
        # toggle and update residual for neighboring checks
        e[j] ^= 1
        # r ^= H_sub[:, j]
        r = (r ^ (Hc[:, j].toarray().ravel().astype(np.uint8))).astype(np.uint8)

    return e


def solve_on_erasure(
    H: np.ndarray,
    s: np.ndarray,
    mask_cols: np.ndarray,
    mask_rows: np.ndarray | None = None,
) -> np.ndarray:
    """
    Solve for erased qubit values that reproduce the syndrome.
    
    Args:
        H: Parity check matrix (m × n)
        s: Syndrome vector (m,)
        mask_cols: Binary mask indicating erased qubits (n,). 1 = erased.
        mask_rows: Optional binary mask indicating erased checks (m,). 1 = erased (don't use).
    
    Returns:
        Correction vector x (n,) such that H[valid_rows] @ x = s[valid_rows] (mod 2)
        and x[i] = 0 for non-erased qubits.
    """
    H = np.asarray(H, dtype=np.uint8)
    s = np.asarray(s, dtype=np.uint8)
    mask_cols = np.asarray(mask_cols, dtype=np.uint8).astype(bool)
    
    # Identify valid (non-erased) rows
    if mask_rows is None:
        rows_keep = np.ones(H.shape[0], dtype=bool)
    else:
        mask_rows = np.asarray(mask_rows, dtype=np.uint8).astype(bool)
        rows_keep = ~mask_rows
    
    # Extract submatrix for erased columns and valid rows
    H_sub = H[rows_keep][:, mask_cols]
    s_sub = s[rows_keep]
    
    # If no erased qubits or no valid constraints, return zeros
    if not mask_cols.any() or not rows_keep.any():
        return np.zeros(H.shape[1], dtype=np.uint8)
    
    # Solve using GF(2) particular solution (find any solution)
    try:
        x_erased = gf2_solve_particular(H_sub, s_sub)
    except (ValueError, AssertionError):
        # No solution exists; return zeros
        x_erased = np.zeros(mask_cols.sum(), dtype=np.uint8)
    
    # Build full correction vector
    x = np.zeros(H.shape[1], dtype=np.uint8)
    x[mask_cols] = x_erased
    return x


# -------- Legacy compatibility wrappers (kept slim) --------
def active_components(H: np.ndarray, s: np.ndarray, basis: str | None = None):
    """Legacy API: ignore basis and return connected components of the active subgraph."""
    A = _as_dense_uint8(H)
    rows = np.where(np.asarray(s, dtype=np.uint8) & 1)[0]
    # build check-node graph via shared data qubits
    # adjacency between checks if they share any 1 in a column
    if A.ndim != 2:
        raise ValueError("H must be 2D")
    # Compute which checks share data qubits (any overlap, not mod 2)
    share = (A @ A.T)  # Count of shared qubits (over integers, not GF2)
    visited = set(); comps=[]
    for r in rows:
        if r in visited: continue
        stack=[r]; comp=[]
        while stack:
            u=stack.pop()
            if u in visited: continue
            visited.add(u); comp.append(u)
            # Find neighbors: other active checks that share at least one qubit
            nbrs = np.where(share[u] > 0)[0].tolist()
            for v in nbrs:
                if int(v) in rows and v not in visited:
                    stack.append(int(v))
        comps.append(Cluster(check_indices=np.array(sorted(comp), dtype=int)))
    return comps

def infer_clusters_batched(Hx: np.ndarray, Hz: np.ndarray,
                           SX: np.ndarray, SZ: np.ndarray):
    """Legacy API expected by tests: run ml_parity_project per batch element."""
    B = SX.shape[0]
    ex = np.zeros((B, Hx.shape[1]), dtype=np.uint8)
    ez = np.zeros((B, Hz.shape[1]), dtype=np.uint8)
    for b in range(B):
        ex[b] = ml_parity_project(Hx, SX[b], p_flip=None)
        ez[b] = ml_parity_project(Hz, SZ[b], p_flip=None)
    return ex, ez
