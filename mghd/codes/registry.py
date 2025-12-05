"""Registry of CSS code families with deterministic parity-check matrices.

This module exposes small, reproducible builders for common CSS families
(surface, BB, etc.) together with helpers to validate commutation, derive
logical operators, and construct lightweight CSSCode carriers used by
samplers/teachers.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np


def _ensure_binary(mat: np.ndarray) -> np.ndarray:
    """Cast to uint8 and reduce mod 2 (defensive normalization)."""
    arr = np.asarray(mat, dtype=np.uint8)
    return arr % 2


def check_css_commutation(hx: np.ndarray, hz: np.ndarray) -> None:
    """Raise if Hx Hz^T has any 1s (violates CSS commutation)."""
    hx_bin = _ensure_binary(hx)
    hz_bin = _ensure_binary(hz)
    comm = (hx_bin @ hz_bin.T) % 2
    if np.any(comm):
        raise ValueError("CSS commutation violated: Hx Hz^T has non-zero entries")


@dataclass(frozen=True)
class CodeSpec:
    """Immutable spec describing a concrete CSS code instance."""

    name: str
    n: int
    hx: np.ndarray
    hz: np.ndarray
    k: int | None = None
    d: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "hx", _ensure_binary(self.hx))
        object.__setattr__(self, "hz", _ensure_binary(self.hz))
        check_css_commutation(self.hx, self.hz)


@dataclass
class CSSCode:
    """Lightweight carrier for CSS codes with teacher‑friendly metadata."""

    name: str
    distance: int
    n: int
    k: int
    Hx: np.ndarray
    Hz: np.ndarray
    layout: dict[str, Any]
    detectors_per_fault: list[list[int]] | None = None
    fault_weights: list[float] | None = None
    num_detectors: int | None = None
    num_observables: int | None = None
    stim_circuit: object | None = None
    Lx: np.ndarray | None = None
    Lz: np.ndarray | None = None

    def detectors_to_syndromes(self, dets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Best-effort mapping assuming detectors align with checks (X block then Z)."""

        mx = int(self.Hx.shape[0])
        mz = int(self.Hz.shape[0])
        dets = np.asarray(dets)
        if dets.ndim != 2:
            dets = np.reshape(dets, (1, -1))
        if self.num_detectors is None or self.num_detectors < mx + mz:
            B = dets.shape[0]
            return (
                np.zeros((B, mx), dtype=np.uint8),
                np.zeros((B, mz), dtype=np.uint8),
            )
        sx = dets[:, :mx].astype(np.uint8)
        sz = dets[:, mx : mx + mz].astype(np.uint8)
        return sx, sz

    def data_to_observables(
        self, ex: np.ndarray | None, ez: np.ndarray | None
    ) -> np.ndarray | None:
        """Map data-space corrections to logical observable flips."""

        if self.Lx is None or self.Lz is None or self.num_observables is None:
            return None

        ex_arr = None if ex is None else np.asarray(ex, dtype=np.uint8)
        ez_arr = None if ez is None else np.asarray(ez, dtype=np.uint8)
        if ex_arr is None and ez_arr is None:
            return None

        if ex_arr is not None and ex_arr.ndim == 1:
            ex_arr = ex_arr[np.newaxis, :]
        if ez_arr is not None and ez_arr.ndim == 1:
            ez_arr = ez_arr[np.newaxis, :]

        B = 0
        if ex_arr is not None:
            B = ex_arr.shape[0]
        if ez_arr is not None:
            B = max(B, ez_arr.shape[0])
        if B == 0:
            return None

        if ex_arr is not None and ex_arr.shape[1] != self.Hx.shape[1]:
            return None
        if ez_arr is not None and ez_arr.shape[1] != self.Hz.shape[1]:
            return None

        z_cols = self.Lz.shape[0] if self.Lz is not None else 0
        x_cols = self.Lx.shape[0] if self.Lx is not None else 0
        z_obs = np.zeros((B, z_cols), dtype=np.uint8)
        x_obs = np.zeros((B, x_cols), dtype=np.uint8)

        if ex_arr is not None and self.Lz is not None and self.Lz.size:
            if ex_arr.shape[0] != B or ex_arr.shape[1] != self.Lz.shape[1]:
                return None
            z_obs = (ex_arr @ (self.Lz.T % 2)) % 2
        if ez_arr is not None and self.Lx is not None and self.Lx.size:
            if ez_arr.shape[0] != B or ez_arr.shape[1] != self.Lx.shape[1]:
                return None
            x_obs = (ez_arr @ (self.Lx.T % 2)) % 2

        return np.concatenate([z_obs, x_obs], axis=1)


def _assert_css(Hx: np.ndarray, Hz: np.ndarray) -> None:
    """Validate that (Hx, Hz) define a valid CSS pair (commutation holds)."""
    check_css_commutation(Hx, Hz)


def _gf2_rank(mat: np.ndarray) -> int:
    """Rank over GF(2) via Gaussian elimination (dense, small matrices)."""
    A = (np.asarray(mat, dtype=np.uint8) & 1).copy()
    rows, cols = A.shape
    rank = 0
    col = 0
    for r in range(rows):
        while col < cols and not A[r:, col].any():
            col += 1
        if col >= cols:
            break
        pivot = r + int(np.flatnonzero(A[r:, col])[0])
        if pivot != r:
            A[[r, pivot]] = A[[pivot, r]]
        for rr in range(rows):
            if rr != r and A[rr, col]:
                A[rr] ^= A[r]
        rank += 1
        col += 1
    return rank


def _gf2_rref(A: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Reduced row‑echelon form over GF(2) and pivot list (dense)."""
    M = (np.asarray(A, dtype=np.uint8) & 1).copy()
    m, n = M.shape
    pivots: list[tuple[int, int]] = []
    row = 0
    for col in range(n):
        pivot = None
        for r in range(row, m):
            if M[r, col]:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != row:
            M[[row, pivot]] = M[[pivot, row]]
        for r in range(m):
            if r != row and M[r, col]:
                M[r] ^= M[row]
        pivots.append((row, col))
        row += 1
        if row == m:
            break
    return M, pivots


def _gf2_nullspace_dense(A: np.ndarray) -> np.ndarray:
    """Dense nullspace basis over GF(2) (columns form a basis)."""
    M, pivots = _gf2_rref(A)
    n = M.shape[1]
    pivot_cols = {c for _, c in pivots}
    free_cols = [c for c in range(n) if c not in pivot_cols]
    basis: list[np.ndarray] = []
    for free in free_cols:
        v = np.zeros(n, dtype=np.uint8)
        v[free] = 1
        for r, c in pivots:
            if M[r, free]:
                v[c] ^= 1
        basis.append(v)
    if not basis:
        return np.zeros((n, 0), dtype=np.uint8)
    return np.stack(basis, axis=1)


def _reduce_mod_rowspace(
    vec: np.ndarray, R: np.ndarray, pivots: list[tuple[int, int]]
) -> np.ndarray:
    """Reduce a vector mod rowspace represented by (R, pivots) over GF(2)."""
    res = (np.asarray(vec, dtype=np.uint8) & 1).copy()
    for r, c in pivots:
        if res[c]:
            res ^= R[r]
    return res


def _select_css_logicals(stab: np.ndarray, dual: np.ndarray, target: int) -> np.ndarray:
    """Pick `target` independent logicals by reducing dual reps mod stab rowspace."""
    if target <= 0:
        return np.zeros((0, stab.shape[1]), dtype=np.uint8)
    R, pivots = _gf2_rref(stab)
    null_dual = _gf2_nullspace_dense(dual)
    if null_dual.shape[1] == 0:
        return np.zeros((0, stab.shape[1]), dtype=np.uint8)

    reps: list[np.ndarray] = []
    seen: set[bytes] = set()
    for j in range(null_dual.shape[1]):
        rep = _reduce_mod_rowspace(null_dual[:, j], R, pivots)
        if not rep.any():
            continue
        key = rep.tobytes()
        if key in seen:
            continue
        seen.add(key)
        reps.append(rep)

    reps.sort(key=lambda v: (int(v.sum()), v.tobytes()))
    basis: list[np.ndarray] = []
    for rep in reps:
        if not basis:
            basis.append(rep)
        else:
            M = np.stack(basis + [rep], axis=0)
            _, new_pivots = _gf2_rref(M)
            if len(new_pivots) > len(basis):
                basis.append(rep)
        if len(basis) == target:
            break
    if not basis:
        return np.zeros((0, stab.shape[1]), dtype=np.uint8)
    return np.stack(basis, axis=0)


def _compute_css_logicals(
    Hx: np.ndarray, Hz: np.ndarray, target: int
) -> tuple[np.ndarray, np.ndarray]:
    """Derive (Lx, Lz) bases for a CSS pair by symmetric reduction."""
    Lx_auto = _select_css_logicals(Hx, Hz, target)
    Lz_auto = _select_css_logicals(Hz, Hx, target)
    return Lx_auto, Lz_auto


def _fault_map(Hx: np.ndarray, Hz: np.ndarray) -> list[list[int]]:
    """Map each data qubit (column) to detector indices (Z block followed by X)."""
    mx, n = Hx.shape
    mz = Hz.shape[0]
    mapping: list[list[int]] = []
    for j in range(n):
        dets: list[int] = []
        if mx:
            dets.extend(np.flatnonzero(Hx[:, j]).tolist())
        if mz:
            dets.extend((mx + np.flatnonzero(Hz[:, j])).tolist())
        mapping.append(dets)
    return mapping


def _commutes_mod2(A: np.ndarray, B: np.ndarray) -> bool:
    """Return True if AB == BA over GF(2)."""

    A = _ensure_binary(A)
    B = _ensure_binary(B)
    return np.array_equal((A @ B) % 2, (B @ A) % 2)


def _make_css(
    *,
    name: str,
    distance: int,
    Hx: np.ndarray,
    Hz: np.ndarray,
    layout: dict[str, Any],
    detectors_per_fault: list[list[int]] | None = None,
    fault_weights: list[float] | None = None,
    num_observables: int | None = None,
    Lx: np.ndarray | None = None,
    Lz: np.ndarray | None = None,
) -> CSSCode:
    """Assemble a CSSCode from parity checks and optional logicals/weights."""
    Hx = _ensure_binary(Hx)
    Hz = _ensure_binary(Hz)
    _assert_css(Hx, Hz)
    n = int(Hx.shape[1])
    rank_x = _gf2_rank(Hx)
    rank_z = _gf2_rank(Hz)
    k_raw = n - rank_x - rank_z
    k = int(k_raw) if k_raw >= 0 else 0
    if detectors_per_fault is None:
        detectors_per_fault = _fault_map(Hx, Hz)
    if fault_weights is None:
        fault_weights = [1.0] * n
    num_detectors = int(Hx.shape[0] + Hz.shape[0])
    if num_observables is None:
        num_observables = k
    target_logicals = int(num_observables if num_observables is not None else k)
    if target_logicals > 0 and (Lx is None or Lz is None):
        auto_Lx, auto_Lz = _compute_css_logicals(Hx, Hz, target_logicals)
        if Lx is None:
            Lx = auto_Lx
        if Lz is None:
            Lz = auto_Lz
    return CSSCode(
        name=name,
        distance=distance,
        n=n,
        k=k,
        Hx=Hx,
        Hz=Hz,
        layout=layout,
        detectors_per_fault=detectors_per_fault,
        fault_weights=fault_weights,
        num_detectors=num_detectors,
        num_observables=num_observables,
        Lx=Lx,
        Lz=Lz,
    )


# ---------------------------------------------------------------------------
# Circulant helpers for generalized/bivariate bicycle codes
# ---------------------------------------------------------------------------


def circulant_from_taps(n: int, taps: Iterable[int]) -> np.ndarray:
    """n x n binary circulant with ones at "taps" in the first row."""

    row0 = np.zeros(n, dtype=np.uint8)
    for t in taps:
        row0[int(t) % n] ^= 1
    mat = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        mat[i] = np.roll(row0, i)
    return mat


def _perm_matrix(n: int) -> np.ndarray:
    P = np.zeros((n, n), dtype=np.uint8)
    P[np.arange(n), (np.arange(n) + 1) % n] = 1
    return P


def bccb_from_taps(n1: int, n2: int, taps_2d: Iterable[tuple[int, int]]) -> np.ndarray:
    """Block-circulant-with-circulant-blocks (BCCB) matrix over GF(2)."""

    Px = _perm_matrix(n1)
    Py = _perm_matrix(n2)
    mat = np.zeros((n1 * n2, n1 * n2), dtype=np.uint8)
    for ax, ay in taps_2d:
        Ax = np.linalg.matrix_power(Px, int(ax) % n1)
        Ay = np.linalg.matrix_power(Py, int(ay) % n2)
        mat ^= np.kron(Ax, Ay).astype(np.uint8)
    return mat


def save_npz(
    hx: np.ndarray, hz: np.ndarray, path: str | np.ndarray, meta: dict[str, Any], **extras: Any
) -> None:
    data: dict[str, Any] = {
        "hx": _ensure_binary(hx),
        "hz": _ensure_binary(hz),
    }
    data.update(meta)
    data.update(extras)
    np.savez_compressed(path, **data)


# ---------------------------------------------------------------------------
# Rotated surface code family
# ---------------------------------------------------------------------------


@cache
def build_surface_rotated_H(d: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Construct rotated planar surface code Hx/Hz with metadata for odd distance d.

    The rotated surface code places data qubits on a d x d grid.
    Stabilizers are centered on plaquettes (half-integer coordinates):
    - Interior plaquettes have weight 4 (touching 4 data qubits)
    - Boundary plaquettes have weight 2 (touching 2 data qubits)

    The stabilizer type (X or Z) alternates in a checkerboard pattern.
    For X-memory configuration:
    - Smooth boundaries (top/bottom) have Z stabilizers only
    - Rough boundaries (left/right) have X stabilizers only

    This ensures CSS commutativity: Hx @ Hz.T = 0 (mod 2).
    """
    if d < 1 or d % 2 == 0:
        raise ValueError("Rotated surface code requires odd distance >= 1")

    n_data = d * d
    hz_rows: list[np.ndarray] = []
    hx_rows: list[np.ndarray] = []

    def q_index(r: int, c: int) -> int:
        return r * d + c

    # Interior plaquettes: weight-4 stabilizers at (r+0.5, c+0.5)
    # touching qubits at (r,c), (r+1,c), (r,c+1), (r+1,c+1)
    for r in range(d - 1):
        for c in range(d - 1):
            qubits = [q_index(r, c), q_index(r + 1, c),
                      q_index(r, c + 1), q_index(r + 1, c + 1)]
            row = np.zeros(n_data, dtype=np.uint8)
            row[qubits] = 1
            # Checkerboard: (r + c) even => X stabilizer, odd => Z stabilizer
            if (r + c) % 2 == 0:
                hx_rows.append(row)
            else:
                hz_rows.append(row)

    # Boundary stabilizers: weight-2
    # Top boundary (smooth): Z stabilizers where interior parity would be odd
    for c in range(d - 1):
        if ((-1) + c) % 2 != 0:  # parity at virtual row -1
            row = np.zeros(n_data, dtype=np.uint8)
            row[[q_index(0, c), q_index(0, c + 1)]] = 1
            hz_rows.append(row)

    # Bottom boundary (smooth): Z stabilizers
    for c in range(d - 1):
        if ((d - 1) + c) % 2 != 0:  # parity at row d-1
            row = np.zeros(n_data, dtype=np.uint8)
            row[[q_index(d - 1, c), q_index(d - 1, c + 1)]] = 1
            hz_rows.append(row)

    # Left boundary (rough): X stabilizers where interior parity would be even
    for r in range(d - 1):
        if (r + (-1)) % 2 == 0:  # parity at virtual col -1
            row = np.zeros(n_data, dtype=np.uint8)
            row[[q_index(r, 0), q_index(r + 1, 0)]] = 1
            hx_rows.append(row)

    # Right boundary (rough): X stabilizers
    for r in range(d - 1):
        if (r + (d - 1)) % 2 == 0:  # parity at col d-1
            row = np.zeros(n_data, dtype=np.uint8)
            row[[q_index(r, d - 1), q_index(r + 1, d - 1)]] = 1
            hx_rows.append(row)

    hz = np.vstack(hz_rows) if hz_rows else np.zeros((0, n_data), dtype=np.uint8)
    hx = np.vstack(hx_rows) if hx_rows else np.zeros((0, n_data), dtype=np.uint8)

    meta = {
        "code": "surface_rotated",
        "distance": d,
        "N_bits": n_data,
        "N_syn": hz.shape[0] + hx.shape[0],
        "syndrome_order": "Z_first_then_X",
        "n_z": hz.shape[0],
        "n_x": hx.shape[0],
        "Hz_order": [f"Z{i}" for i in range(hz.shape[0])],
        "Hx_order": [f"X{i}" for i in range(hx.shape[0])],
        "data_qubit_order": list(range(n_data)),
    }
    return hx, hz, meta


@cache
def default_surface_rotated_layout(d: int) -> dict[str, Any]:
    """Reference planar layout description for the rotated surface code."""
    hx, hz, meta = build_surface_rotated_H(d)
    n_data = meta["N_bits"]
    ancilla_z = list(range(n_data, n_data + hz.shape[0]))
    ancilla_x = list(range(n_data + hz.shape[0], n_data + hz.shape[0] + hx.shape[0]))

    def rows_to_pairs(rows: np.ndarray, ancillas: list[int]) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        for anc, row in zip(ancillas, rows):
            for q in np.nonzero(row)[0]:
                pairs.append((int(anc), int(q)))
        return pairs

    return {
        "code": meta["code"],
        "distance": d,
        "syndrome_order": meta["syndrome_order"],
        "data": list(range(n_data)),
        "ancilla_z": ancilla_z,
        "ancilla_x": ancilla_x,
        "cz_layers": [rows_to_pairs(hz, ancilla_z), rows_to_pairs(hx, ancilla_x)],
        "prx_layers": [[(anc, "z") for anc in ancilla_z], [(anc, "x") for anc in ancilla_x]],
        "total_qubits": n_data + len(ancilla_z) + len(ancilla_x),
        "syndrome_schedule": "alternating",
    }


def logical_surface_rotated(d: int) -> dict[str, np.ndarray]:
    """Return simple weight‑d logicals crossing the center row/column (Lx/Lz)."""
    if d < 1 or d % 2 == 0:
        raise ValueError("Rotated surface code requires odd distance >= 1")
    n = d * d
    center = d // 2
    lx = np.zeros(n, dtype=np.uint8)
    lz = np.zeros(n, dtype=np.uint8)
    for r in range(d):
        lx[r * d + center] = 1
    for c in range(d):
        lz[center * d + c] = 1
    return {"Lx": lx, "Lz": lz}


def surface_rotated_spec(d: int) -> CodeSpec:
    """Package the rotated surface Hx/Hz into an immutable CodeSpec."""
    hx, hz, meta = build_surface_rotated_H(d)
    meta_copy = dict(meta)
    return CodeSpec(name=f"surface_d{d}", n=hx.shape[1], hx=hx, hz=hz, d=d, meta=meta_copy)


# ---------------------------------------------------------------------------
# CSSCode builders for training / teacher integration
# ---------------------------------------------------------------------------


def build_surface(distance: int, *, rotated: bool = True) -> CSSCode:
    """Build a CSSCode object for the rotated planar surface code."""
    d = int(distance)
    if d < 3 or d % 2 == 0:
        raise ValueError("distance must be odd and >= 3 for rotated surface")
    hx, hz, meta = build_surface_rotated_H(d)
    layout = {
        "meta": dict(meta),
        "layout": default_surface_rotated_layout(d),
        "rotated": rotated,
    }
    n = hx.shape[1]
    detectors_per_fault = _fault_map(hx, hz)
    return _make_css(
        name="surface",
        distance=d,
        Hx=hx,
        Hz=hz,
        layout=layout,
        detectors_per_fault=detectors_per_fault,
        fault_weights=[1.0] * n,
        num_observables=1,
    )


# ---------------------------------------------------------------------------
# BB (Bravyi-Bacon) families
# ---------------------------------------------------------------------------


def _bb_indices(size_x: int, size_y: int):
    def wrap_x(x: int) -> int:
        return x % size_x

    def wrap_y(y: int) -> int:
        return y % size_y

    def idx_h(x: int, y: int) -> int:
        return wrap_y(y) * size_x + wrap_x(x)

    def idx_v(x: int, y: int) -> int:
        return size_x * size_y + wrap_y(y) * size_x + wrap_x(x)

    return idx_h, idx_v, wrap_x, wrap_y


def bb_from_shifts(
    size_x: int,
    size_y: int,
    a: tuple[int, int],
    b: tuple[int, int],
) -> CodeSpec:
    if size_x <= 0 or size_y <= 0:
        raise ValueError("l and m must be positive")
    ax, ay = a
    bx, by = b
    n_qubits = 2 * size_x * size_y
    n_checks = size_x * size_y
    hx = np.zeros((n_checks, n_qubits), dtype=np.uint8)
    hz = np.zeros_like(hx)
    idx_h, idx_v, wrap_x, wrap_y = _bb_indices(size_x, size_y)

    for x in range(size_x):
        for y in range(size_y):
            row = y * size_x + x
            hx_edges = {
                idx_h(x, y),
                idx_h(x + 1, y),
                idx_h(x - 1, y),
                idx_h(x, y - 1),
                idx_h(x + ax, y + ay),
                idx_h(x - ax, y - ay),
            }
            hz_edges = {
                idx_v(x, y),
                idx_v(x + 1, y),
                idx_v(x - 1, y),
                idx_v(x, y - 1),
                idx_v(x + bx, y + by),
                idx_v(x - bx, y - by),
            }
            for qubit in hx_edges:
                hx[row, qubit] = 1
            for qubit in hz_edges:
                hz[row, qubit] = 1

    if not np.all(hx.sum(axis=1) == 6) or not np.all(hz.sum(axis=1) == 6):
        raise ValueError("BB checks must have weight 6")

    return CodeSpec(
        name=f"bb_l{size_x}_m{size_y}_a{a}_b{b}",
        n=n_qubits,
        hx=hx,
        hz=hz,
        meta={"l": size_x, "m": size_y, "a": a, "b": b, "syndrome_order": "Z_first_then_X"},
    )


def bb_gross() -> CodeSpec:
    return bb_from_shifts(size_x=12, size_y=6, a=(3, -1), b=(-1, -3))

def bb_double_gross() -> CodeSpec:
    """[[288, 12, 18]] bivariate-bicycle 'two-gross' code (ℓ=12, m=12)."""
    return bb_from_shifts(size_x=12, size_y=12, a=(3, -1), b=(-1, -3))

def build_gb_two_block(
    n: int,
    taps_a: Iterable[int],
    taps_b: Iterable[int],
    *,
    name: str = "gb",
) -> CSSCode:
    """Two-block generalized-bicycle code: Hx=[A|B], Hz=[B^T|A^T]."""

    A = circulant_from_taps(n, taps_a)
    B = circulant_from_taps(n, taps_b)
    if not _commutes_mod2(A, B):
        raise ValueError("Two-block GB construction requires commuting circulant matrices A and B")
    Hx = np.concatenate([A, B], axis=1)
    Hz = np.concatenate([B.T, A.T], axis=1)
    _assert_css(Hx, Hz)
    N = Hx.shape[1]
    dets_per_fault = []
    mx, mz = Hx.shape[0], Hz.shape[0]
    for j in range(N):
        dets = []
        dets.extend(np.flatnonzero(Hx[:, j]).tolist())
        dets.extend((mx + np.flatnonzero(Hz[:, j])).tolist())
        dets_per_fault.append(dets)
    return CSSCode(
        name=name,
        distance=-1,
        n=N,
        k=-1,
        Hx=Hx,
        Hz=Hz,
        layout={"n": n, "two_block": True},
        detectors_per_fault=dets_per_fault,
        fault_weights=[1.0] * N,
        num_detectors=mx + mz,
        num_observables=1,
    )


def build_bb_bivariate(
    n1: int,
    n2: int,
    taps_a_2d: Iterable[tuple[int, int]],
    taps_b_2d: Iterable[tuple[int, int]],
) -> CSSCode:
    """Bivariate bicycle via block-circulant matrices."""

    A = bccb_from_taps(n1, n2, taps_a_2d)
    B = bccb_from_taps(n1, n2, taps_b_2d)
    if not _commutes_mod2(A, B):
        raise ValueError("BCCB matrices for BB code must commute over GF(2)")
    Hx = np.concatenate([A, B], axis=1)
    Hz = np.concatenate([B.T, A.T], axis=1)
    _assert_css(Hx, Hz)
    N = Hx.shape[1]
    dets_per_fault = []
    mx, mz = Hx.shape[0], Hz.shape[0]
    for j in range(N):
        dets = []
        dets.extend(np.flatnonzero(Hx[:, j]).tolist())
        dets.extend((mx + np.flatnonzero(Hz[:, j])).tolist())
        dets_per_fault.append(dets)
    return CSSCode(
        name="bb",
        distance=-1,
        n=N,
        k=-1,
        Hx=Hx,
        Hz=Hz,
        layout={"n1": n1, "n2": n2, "two_block": True, "bivariate": True},
        detectors_per_fault=dets_per_fault,
        fault_weights=[1.0] * N,
        num_detectors=mx + mz,
        num_observables=1,
    )


def build_bb(
    n1: int = 17,
    n2: int = 17,
    *,
    taps_a_2d: Iterable[tuple[int, int]] = ((0, 0), (1, 0), (0, 1)),
    taps_b_2d: Iterable[tuple[int, int]] = ((0, 0), (2, 0), (0, 2)),
) -> CSSCode:
    return build_bb_bivariate(n1, n2, taps_a_2d, taps_b_2d)


def build_gross(distance: int | None = None, **kw: Any) -> CSSCode:
    """CSSCode wrapper for the [[144, 12, 12]] gross BB code."""
    if distance not in (None, 12):
        raise ValueError("Gross code has fixed distance 12")
    spec = bb_gross()
    Hx, Hz = spec.hx, spec.hz
    dets_per_fault = _fault_map(Hx, Hz)
    layout = {"family": "bb", **(spec.meta or {})}
    return _make_css(
        name="gross",
        distance=12,
        Hx=Hx,
        Hz=Hz,
        layout=layout,
        detectors_per_fault=dets_per_fault,
        fault_weights=[1.0] * spec.n,
        num_observables=12,
    )


def build_double_gross(distance: int | None = None, **kw: Any) -> CSSCode:
    """CSSCode wrapper for the [[288, 12, 18]] two-gross BB code."""
    if distance not in (None, 18):
        raise ValueError("Double-gross code has fixed distance 18")
    spec = bb_double_gross()
    Hx, Hz = spec.hx, spec.hz
    dets_per_fault = _fault_map(Hx, Hz)
    layout = {"family": "bb", **(spec.meta or {})}
    return _make_css(
        name="double_gross",
        distance=18,
        Hx=Hx,
        Hz=Hz,
        layout=layout,
        detectors_per_fault=dets_per_fault,
        fault_weights=[1.0] * spec.n,
        num_observables=12,
    )


# ---------------------------------------------------------------------------
# HGP (Tillich–Zémor) construction
# ---------------------------------------------------------------------------


def hgp_from_classical(H1: np.ndarray, H2: np.ndarray) -> CodeSpec:
    H1 = _ensure_binary(H1)
    H2 = _ensure_binary(H2)
    m1, n1 = H1.shape
    m2, n2 = H2.shape
    block1 = np.kron(H1, np.eye(n2, dtype=np.uint8))
    block2 = np.kron(np.eye(m1, dtype=np.uint8), H2.T)
    hx = np.concatenate((block1, block2), axis=1)
    block3 = np.kron(np.eye(n1, dtype=np.uint8), H2)
    block4 = np.kron(H1.T, np.eye(m2, dtype=np.uint8))
    hz = np.concatenate((block3, block4), axis=1)
    n = n1 * n2 + m1 * m2
    return CodeSpec(
        name="hgp",
        n=n,
        hx=hx,
        hz=hz,
        meta={"H1_shape": H1.shape, "H2_shape": H2.shape, "syndrome_order": "Z_first_then_X"},
    )


def build_hgp(H1: np.ndarray, H2: np.ndarray, *, name: str = "hgp") -> CSSCode:
    if H1 is None or H2 is None:
        raise ValueError("H1 and H2 must be provided for the HGP builder")
    H1 = _ensure_binary(np.asarray(H1, dtype=np.uint8))
    H2 = _ensure_binary(np.asarray(H2, dtype=np.uint8))
    spec = hgp_from_classical(H1, H2)
    Hx = spec.hx
    Hz = spec.hz
    layout = {"H1_shape": H1.shape, "H2_shape": H2.shape}
    return _make_css(
        name=name,
        distance=-1,
        Hx=Hx,
        Hz=Hz,
        layout=layout,
        fault_weights=[1.0] * Hx.shape[1],
    )


# ---------------------------------------------------------------------------
# QRM / Hamming families
# ---------------------------------------------------------------------------


def qrm_steane() -> CodeSpec:
    H = np.array(
        [
            [1, 1, 1, 0, 1, 0, 0],
            [1, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 1, 0, 0, 1],
        ],
        dtype=np.uint8,
    )
    return CodeSpec(
        name="steane", n=7, hx=H, hz=H, k=1, d=3, meta={"syndrome_order": "Z_first_then_X"}
    )


def build_steane() -> CSSCode:
    H = np.array(
        [
            [1, 0, 0, 1, 0, 1, 1],
            [0, 1, 0, 1, 1, 0, 1],
            [0, 0, 1, 0, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    layout = {"family": "steane"}
    Lz = np.array([[1, 0, 0, 0, 0, 0, 1]], dtype=np.uint8)
    Lx = Lz.copy()
    return _make_css(
        name="steane",
        distance=3,
        Hx=H,
        Hz=H,
        layout=layout,
        fault_weights=[1.0] * 7,
        num_observables=1,
        Lx=Lx,
        Lz=Lz,
    )


def _hamming_parity_matrix(m: int) -> np.ndarray:
    n = (1 << m) - 1
    cols = np.arange(1, n + 1, dtype=np.uint32)
    rows = []
    for bit in range(m):
        rows.append(((cols >> bit) & 1).astype(np.uint8))
    return np.vstack(rows)


def qrm_hamming(m: int) -> CodeSpec:
    if m < 2:
        raise ValueError("m must be >= 2 for Hamming family")
    H = _hamming_parity_matrix(m)
    n = H.shape[1]
    k = n - 2 * m
    return CodeSpec(
        name=f"qrm_hamming_m{m}",
        n=n,
        hx=H,
        hz=H,
        k=k,
        d=3,
        meta={"m": m, "syndrome_order": "Z_first_then_X"},
    )


DATA_DIR = Path(__file__).resolve().parent.parent / "color_cache"


def _load_cached_color(
    kind: str, distance: int
) -> tuple[np.ndarray, np.ndarray, int, dict[str, Any]] | None:
    path = DATA_DIR / f"color_{kind}_d{distance}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    Hx = data["Hx"].astype(np.uint8)
    Hz = data["Hz"].astype(np.uint8)
    n = int(data["n"])
    meta_raw = data.get("meta")
    if meta_raw is not None:
        if isinstance(meta_raw, np.ndarray):
            meta_raw = meta_raw.item()
        if isinstance(meta_raw, bytes):
            meta_raw = meta_raw.decode("utf-8")
        try:
            layout = json.loads(meta_raw)
        except Exception:
            layout = {"meta": meta_raw}
    else:
        layout = {}
    return Hx, Hz, n, layout


def _write_color_cache(
    kind: str, distance: int, Hx: np.ndarray, Hz: np.ndarray, n: int, layout: dict[str, Any]
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"color_{kind}_d{distance}.npz"
    payload = dict(layout)
    np.savez_compressed(
        path,
        Hx=Hx.astype(np.uint8),
        Hz=Hz.astype(np.uint8),
        n=int(n),
        meta=json.dumps(payload),
    )


def _build_color_via_external(
    kind: str, distance: int
) -> tuple[np.ndarray, np.ndarray, int, dict[str, Any]]:
    if kind == "666":
        try:
            from mghd.core import codes_external as cx
        except ImportError:
            import importlib

            cx = importlib.import_module("codes_external")
        builder = getattr(cx, "build_color_666_qecsim", None)
        if builder is None:
            raise RuntimeError(
                "codes_external missing 'build_color_666_qecsim'. Install optional deps or update cache."
            )
        return builder(distance)
    if kind == "488":
        try:
            from mghd.core import codes_external_488 as cx488
        except ImportError as exc:
            raise RuntimeError(
                "codes_external_488 unavailable; install panqec or quantum-pecos."
            ) from exc
        return cx488.build_color_488(distance)
    raise ValueError(f"Unsupported color code kind '{kind}'")


def _build_color(kind: str, distance: int) -> CSSCode:
    cached = _load_cached_color(kind, distance)
    if cached is not None:
        Hx, Hz, n, layout = cached
    else:
        try:
            Hx, Hz, n, layout = _build_color_via_external(kind, distance)
        except Exception as exc:
            raise RuntimeError(
                f"color_{kind} d={distance} requires cached matrices. "
                "Run `python -m tools.precompute_color_codes` after installing optional deps."
            ) from exc
        else:
            _write_color_cache(kind, distance, Hx, Hz, n, layout)
    layout = dict(layout)
    layout.setdefault("tiling", "6.6.6" if kind == "666" else "4.8.8")
    layout.setdefault("distance", distance)
    return _make_css(
        name=f"color_{kind}",
        distance=distance,
        Hx=Hx,
        Hz=Hz,
        layout=layout,
        fault_weights=[1.0] * Hx.shape[1],
        num_observables=1,
    )


def build_color_666_triangle(distance: int) -> CSSCode:
    return _build_color("666", distance)


def build_color_488_triangle(distance: int) -> CSSCode:
    return _build_color("488", distance)


def build_color(distance: int, *, tiling: str = "6.6.6") -> CSSCode:
    tiling = tiling.strip().lower()
    if tiling in {"6.6.6", "666", "honeycomb"}:
        return build_color_666_triangle(distance)
    if tiling in {"4.8.8", "488", "square-octagon"}:
        return build_color_488_triangle(distance)
    raise ValueError("tiling must be '6.6.6' or '4.8.8'")


REGISTRY = {
    "surface": lambda distance, **kw: build_surface(distance, **kw),
    "repetition": lambda distance, **kw: build_repetition(distance, **kw),
    "steane": lambda distance=None, **kw: build_steane(),
    "color": lambda distance, tiling="6.6.6", **kw: build_color(distance, tiling=tiling, **kw),
    "color_666": lambda distance, **kw: build_color_666_triangle(distance),
    "color_488": lambda distance, **kw: build_color_488_triangle(distance),
    "gb": lambda distance=None, n=47, taps_a=(0, 1, 3), taps_b=(0, 2, 5), **kw: build_gb_two_block(
        n, taps_a, taps_b, name="gb"
    ),
    "bb": lambda distance=None,
           n1=17,
           n2=17,
           taps_a_2d=((0, 0), (1, 0), (0, 1)),
           taps_b_2d=((0, 0), (2, 0), (0, 2)),
           **kw: build_bb_bivariate(n1, n2, taps_a_2d, taps_b_2d),
    "gross": lambda distance=None, **kw: build_gross(distance=distance, **kw),
    "double_gross": lambda distance=None, **kw: build_double_gross(distance=distance, **kw),
    "hgp": lambda distance=None, H1=None, H2=None, name="hgp", **kw: build_hgp(H1, H2, name=name),
}

_OPTIONAL_DISTANCE = {"steane", "gb", "bb", "gross", "double_gross", "hgp"}


def get_code(family: str, distance: int | None = None, **kw) -> CSSCode:
    if family not in REGISTRY:
        raise KeyError(f"Unknown family '{family}'. Available: {list(REGISTRY)}")
    if distance is None and family not in _OPTIONAL_DISTANCE:
        raise ValueError(f"Family '{family}' requires a distance parameter")
    builder = REGISTRY[family]
    return builder(distance=distance, **kw)


# ---------------------------------------------------------------------------
# Repetition family
# ---------------------------------------------------------------------------


def repetition(n_data: int) -> CodeSpec:
    if n_data < 2:
        raise ValueError("repetition code requires n_data >= 2")
    rows = n_data - 1
    hx = np.zeros((rows, n_data), dtype=np.uint8)
    for i in range(rows):
        hx[i, i] = 1
        hx[i, i + 1] = 1
    hz = np.ones((1, n_data), dtype=np.uint8)
    return CodeSpec(
        name=f"repetition_{n_data}",
        n=n_data,
        hx=hx,
        hz=hz,
        k=1,
        d=n_data,
        meta={"syndrome_order": "Z_first_then_X"},
    )


def build_repetition(distance: int, *, basis: str = "Z") -> CSSCode:
    L = int(distance)
    if L < 2:
        raise ValueError("distance must be >= 2 for repetition")
    basis_upper = basis.upper()
    if basis_upper not in {"X", "Z"}:
        raise ValueError("basis must be 'X' or 'Z'")
    n = L
    if basis_upper == "Z":
        Hz = np.zeros((L - 1, n), dtype=np.uint8)
        for i in range(L - 1):
            Hz[i, [i, i + 1]] = 1
        Hx = np.zeros((0, n), dtype=np.uint8)
        Lz = np.ones((1, n), dtype=np.uint8)
        Lx = np.zeros((0, n), dtype=np.uint8)
    else:
        Hx = np.zeros((L - 1, n), dtype=np.uint8)
        for i in range(L - 1):
            Hx[i, [i, i + 1]] = 1
        Hz = np.zeros((0, n), dtype=np.uint8)
        Lx = np.ones((1, n), dtype=np.uint8)
        Lz = np.zeros((0, n), dtype=np.uint8)
    layout = {"L": L, "basis": basis_upper}
    return _make_css(
        name="repetition",
        distance=L,
        Hx=Hx,
        Hz=Hz,
        layout=layout,
        fault_weights=[1.0] * n,
        Lx=Lx,
        Lz=Lz,
    )


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def load_matrix(path: str) -> np.ndarray:
    path = str(path)
    if path.endswith(".npz"):
        data = np.load(path)
        if len(data.files) == 1:
            return _ensure_binary(data[data.files[0]])
        if "arr_0" in data:
            return _ensure_binary(data["arr_0"])
        raise ValueError("NPZ must contain a single array or arr_0")
    if path.endswith(".npy"):
        return _ensure_binary(np.load(path))
    with open(path, encoding="utf-8") as fh:
        obj = json.load(fh)
    return _ensure_binary(np.asarray(obj, dtype=np.uint8))


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def _random_css_check(spec: CodeSpec, samples: int = 5) -> None:
    rng = np.random.default_rng(0)
    for _ in range(samples):
        e = rng.integers(0, 2, size=spec.n, dtype=np.uint8)
        syn_x = (spec.hx @ e) % 2
        syn_z = (spec.hz @ e) % 2
        if syn_x.shape[0] != spec.hx.shape[0] or syn_z.shape[0] != spec.hz.shape[0]:
            raise ValueError("Syndrome dimension mismatch")


def _self_test() -> None:
    specs: Iterable[CodeSpec] = [
        bb_gross(),
        bb_double_gross(),
        surface_rotated_spec(3),
        qrm_steane(),
        qrm_hamming(3),
        repetition(5),
    ]
    H1 = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    H2 = np.array([[1, 0, 1], [1, 1, 0]], dtype=np.uint8)
    specs = list(specs) + [hgp_from_classical(H1, H2)]
    for spec in specs:
        check_css_commutation(spec.hx, spec.hz)
        _random_css_check(spec)
        if spec.name.startswith("bb_l"):
            if spec.hx.shape != spec.hz.shape:
                raise ValueError("BB shapes mismatch")
            if not np.all(spec.hx.sum(axis=1) == 6):
                raise ValueError("BB X check weight not 6")
            if not np.all(spec.hz.sum(axis=1) == 6):
                raise ValueError("BB Z check weight not 6")
    print("codes_registry self-test OK")


sys.modules.setdefault("codes_registry", sys.modules[__name__])


if __name__ == "__main__":
    _self_test()
