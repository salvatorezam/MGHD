"""Compatibility wrapper for legacy `core` imports plus local decoding utils."""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_lazy_mod: ModuleType | None = None


def _load() -> ModuleType:
    global _lazy_mod
    if _lazy_mod is None:
        _lazy_mod = importlib.import_module("mghd_main.core")
    return _lazy_mod


def __getattr__(name: str) -> Any:  # pragma: no cover - passthrough
    return getattr(_load(), name)


def __dir__() -> list[str]:  # pragma: no cover - patched at end of file
    return sorted(set(dir(_load())))


# ---------------------------------------------------------------------------
# Local decoding helpers (distance-agnostic clustering + ML projection)
# ---------------------------------------------------------------------------

@dataclass
class Cluster:
    """A connected component of active checks (syndrome=1) in one basis."""

    check_indices: np.ndarray  # shape [C_r], dtype=int
    data_indices: np.ndarray   # unique data-qubit indices touched by these checks
    basis: str                 # "X" or "Z"


@dataclass
class Subproblem:
    """Local decoding problem sliced to a cluster."""

    H_local: np.ndarray                # shape [C_r, C_c], dtype=uint8 (GF(2))
    s_local: np.ndarray                # shape [C_r], dtype=uint8
    p_local: Optional[np.ndarray]      # shape [C_c], dtype=float or None
    map_local_to_global: np.ndarray    # shape [C_c], dtype=int
    basis: str


def _connected_components_active(H: np.ndarray, s: np.ndarray) -> List[np.ndarray]:
    """Return connected components of active checks linked via shared data qubits."""

    if H.ndim != 2:
        raise ValueError("Parity-check matrix must be 2-D")
    m, _ = H.shape
    s = (s.astype(np.uint8) & 1).reshape(-1)
    if s.shape[0] != m:
        raise ValueError("Syndrome length must match number of checks")
    active = np.flatnonzero(s)
    if active.size == 0:
        return []

    row_cols = [np.flatnonzero(H[i]) for i in range(m)]
    col_rows: Dict[int, List[int]] = {}
    for i in active:
        for c in row_cols[i]:
            col_rows.setdefault(int(c), []).append(int(i))

    seen = np.zeros(m, dtype=bool)
    comps: List[np.ndarray] = []
    for r0 in active:
        if seen[r0]:
            continue
        queue = [int(r0)]
        seen[r0] = True
        comp = [int(r0)]
        while queue:
            r = queue.pop()
            for c in row_cols[r]:
                for r2 in col_rows.get(int(c), []):
                    if not seen[r2]:
                        seen[r2] = True
                        queue.append(r2)
                        comp.append(r2)
        comps.append(np.array(sorted(comp), dtype=int))
    return comps


def active_components(
    Hx_or_Hz: np.ndarray,
    syndrome: np.ndarray,
    *,
    basis: str,
    halo_steps: int = 0,
) -> List[Cluster]:
    """Compute clusters of active checks with optional halo growth."""

    comps = _connected_components_active(Hx_or_Hz, syndrome)
    clusters: List[Cluster] = []
    for rows in comps:
        grow_rows = set(int(r) for r in rows)
        if halo_steps > 0:
            for _ in range(halo_steps):
                covered_cols = np.flatnonzero(Hx_or_Hz[list(grow_rows)].any(axis=0))
                touching_rows = np.flatnonzero(Hx_or_Hz[:, covered_cols].any(axis=1))
                for r in touching_rows:
                    grow_rows.add(int(r))
            rows = np.array(sorted(grow_rows), dtype=int)
        cols = np.flatnonzero(Hx_or_Hz[rows].any(axis=0)) if rows.size else np.array([], dtype=int)
        clusters.append(
            Cluster(
                check_indices=rows.astype(int),
                data_indices=cols.astype(int),
                basis=basis,
            )
        )
    return clusters


def extract_subproblem(
    Hx_or_Hz: np.ndarray,
    syndrome: np.ndarray,
    cluster: Cluster,
    probs_local: Optional[np.ndarray] = None,
) -> Subproblem:
    """Slice H and s to the cluster and return local maps."""

    rows = cluster.check_indices
    cols = cluster.data_indices
    H_local = (Hx_or_Hz[rows][:, cols]).astype(np.uint8, copy=False)
    s_local = (syndrome[rows]).astype(np.uint8, copy=False)
    p_local = None
    if probs_local is not None:
        probs_local = np.asarray(probs_local, dtype=float)
        if probs_local.shape[0] != cols.shape[0]:
            raise ValueError("Length of probs_local must match number of local columns")
        p_local = probs_local
    return Subproblem(
        H_local=H_local,
        s_local=s_local,
        p_local=p_local,
        map_local_to_global=cols.astype(int, copy=False),
        basis=cluster.basis,
    )


def _gf2_rref(
    A: np.ndarray,
    b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[int], Optional[List[int]]]:
    """
    Row-reduced echelon form over GF(2) for Ax=b.

    Returns transformed (A, b), list of pivot columns, and list of free columns.
    If the system is inconsistent, free columns is ``None``.
    """

    A = (np.asarray(A, dtype=np.uint8) & 1).copy()
    b = (np.asarray(b, dtype=np.uint8) & 1).copy().reshape(-1)
    m, n = A.shape
    if b.shape[0] != m:
        raise ValueError("Right-hand side length must match number of rows")

    pivots: List[int] = []
    row = 0
    for col in range(n):
        candidates = np.flatnonzero(A[row:, col])
        if candidates.size == 0:
            continue
        pivot = int(candidates[0] + row)
        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]
            b[[row, pivot]] = b[[pivot, row]]
        for r in range(m):
            if r != row and A[r, col]:
                A[r] ^= A[row]
                b[r] ^= b[row]
        pivots.append(col)
        row += 1
        if row == m:
            break

    for r in range(row, m):
        if not A[r].any() and b[r]:
            return A, b, pivots, None  # inconsistent

    free_cols = [c for c in range(n) if c not in pivots]
    return A, b, pivots, free_cols


def _solve_with_free(
    A: np.ndarray,
    b: np.ndarray,
    pivots: List[int],
    free_cols: List[int],
    free_values: np.ndarray,
) -> np.ndarray:
    """Construct a solution vector given assignments to free variables."""

    n = A.shape[1]
    x = np.zeros(n, dtype=np.uint8)
    if free_cols:
        x[np.array(free_cols, dtype=int)] = free_values & 1

    for idx in range(len(pivots) - 1, -1, -1):
        row = idx
        col = pivots[idx]
        row_data = A[row]
        parity = int(np.sum(row_data & x) % 2)
        x[col] = parity ^ int(b[row])
    return x


def _enumerate_min_weight_solution(
    A: np.ndarray,
    b: np.ndarray,
    w: Optional[np.ndarray],
    *,
    r_cap: int = 18,
) -> np.ndarray:
    """Solve min-weight GF(2) system using enumeration when nullity is small."""

    A_rref, b_rref, pivots, free_cols = _gf2_rref(A, b)
    if free_cols is None:
        return np.zeros(A.shape[1], dtype=np.uint8)

    free_cols_arr = np.array(free_cols, dtype=int)
    x0 = _solve_with_free(A_rref, b_rref, pivots, free_cols, np.zeros(len(free_cols), dtype=np.uint8))

    def cost(vec: np.ndarray) -> float:
        if w is None:
            return float(vec.sum())
        return float(np.dot(w, vec))

    r = len(free_cols)
    if r <= r_cap:
        basis = []
        for i in range(r):
            mask = np.zeros(r, dtype=np.uint8)
            mask[i] = 1
            sol = _solve_with_free(A_rref, b_rref, pivots, free_cols, mask)
            basis.append(sol ^ x0)
        best = x0.copy()
        best_cost = cost(best)
        for mask in range(1, 1 << r):
            candidate = x0.copy()
            for j in range(r):
                if (mask >> j) & 1:
                    candidate ^= basis[j]
            cand_cost = cost(candidate)
            if cand_cost < best_cost:
                best = candidate
                best_cost = cand_cost
        return best

    # Nullity large: fallback to particular solution (can be upgraded later).
    return x0


def ml_parity_project(
    H_local: np.ndarray,
    s_local: np.ndarray,
    probs_local: Optional[np.ndarray] = None,
    *,
    r_cap: int = 18,
) -> np.ndarray:
    """
    Minimum-weight solution x over GF(2): H_local x = s_local.

    When ``probs_local`` is given, minimize the weighted cost using
    log-likelihood ratios derived from the probabilities.
    """

    H_local = np.asarray(H_local, dtype=np.uint8) & 1
    s_local = (np.asarray(s_local, dtype=np.uint8) & 1).reshape(-1)
    if H_local.shape[0] != s_local.shape[0]:
        raise ValueError("Row count of H_local must match length of s_local")

    weights = None
    if probs_local is not None:
        p = np.clip(np.asarray(probs_local, dtype=float).reshape(-1), 1e-6, 1 - 1e-6)
        if p.shape[0] != H_local.shape[1]:
            raise ValueError("probs_local must match number of columns in H_local")
        weights = np.log((1.0 - p) / p)

    return _enumerate_min_weight_solution(H_local, s_local, weights, r_cap=r_cap)


def infer_clusters_once(
    H: np.ndarray,
    s: np.ndarray,
    *,
    basis: str,
    priors: Optional[np.ndarray] = None,
    r_cap: int = 18,
) -> Tuple[np.ndarray, List[Cluster]]:
    """Decode a single-basis syndrome via clustering and local ML projection."""

    H = np.asarray(H, dtype=np.uint8) & 1
    n = H.shape[1]
    e = np.zeros(n, dtype=np.uint8)
    clusters = active_components(H, s, basis=basis)
    for cluster in clusters:
        local_priors = None
        if priors is not None:
            local_priors = np.asarray(priors, dtype=float)[cluster.data_indices]
        sub = extract_subproblem(H, s, cluster, local_priors)
        correction = ml_parity_project(sub.H_local, sub.s_local, sub.p_local, r_cap=r_cap)
        e[sub.map_local_to_global] ^= correction
    return e, clusters


def infer_clusters_batched(
    Hx: np.ndarray,
    Hz: np.ndarray,
    syndromes_x: np.ndarray,
    syndromes_z: np.ndarray,
    *,
    priors_x: Optional[np.ndarray] = None,
    priors_z: Optional[np.ndarray] = None,
    r_cap: int = 18,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batched cluster decoding for CSS codes (X and Z separately)."""

    syndromes_x = np.asarray(syndromes_x, dtype=np.uint8) & 1
    syndromes_z = np.asarray(syndromes_z, dtype=np.uint8) & 1
    if syndromes_x.ndim != 2 or syndromes_z.ndim != 2:
        raise ValueError("Syndromes must be 2-D (batch, checks)")
    if syndromes_x.shape[0] != syndromes_z.shape[0]:
        raise ValueError("Batch sizes must match for X and Z syndromes")

    B = syndromes_x.shape[0]
    n = Hx.shape[1]
    ex = np.zeros((B, n), dtype=np.uint8)
    ez = np.zeros((B, n), dtype=np.uint8)

    def _slice_priors(priors: Optional[np.ndarray], idx: int) -> Optional[np.ndarray]:
        if priors is None:
            return None
        arr = np.asarray(priors, dtype=float)
        if arr.ndim == 1:
            return arr
        return arr[idx]

    for b_idx in range(B):
        e_x, _ = infer_clusters_once(
            Hx,
            syndromes_x[b_idx],
            basis="X",
            priors=_slice_priors(priors_x, b_idx),
            r_cap=r_cap,
        )
        e_z, _ = infer_clusters_once(
            Hz,
            syndromes_z[b_idx],
            basis="Z",
            priors=_slice_priors(priors_z, b_idx),
            r_cap=r_cap,
        )
        ex[b_idx] = e_x
        ez[b_idx] = e_z

    return ex, ez


__all__ = [
    "Cluster",
    "Subproblem",
    "active_components",
    "extract_subproblem",
    "ml_parity_project",
    "infer_clusters_once",
    "infer_clusters_batched",
]


def __dir__() -> list[str]:  # pragma: no cover - updated listing
    return sorted(set(__all__) | set(dir(_load())))
