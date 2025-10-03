"""Registry of CSS code families with deterministic parity-check matrices."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Any, Optional, Iterable, List, Tuple
import json
import numpy as np
import sys


def _ensure_binary(mat: np.ndarray) -> np.ndarray:
    arr = np.asarray(mat, dtype=np.uint8)
    return arr % 2


def check_css_commutation(hx: np.ndarray, hz: np.ndarray) -> None:
    hx_bin = _ensure_binary(hx)
    hz_bin = _ensure_binary(hz)
    comm = (hx_bin @ hz_bin.T) % 2
    if np.any(comm):
        raise ValueError("CSS commutation violated: Hx Hz^T has non-zero entries")


@dataclass(frozen=True)
class CodeSpec:
    name: str
    n: int
    hx: np.ndarray
    hz: np.ndarray
    k: Optional[int] = None
    d: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "hx", _ensure_binary(self.hx))
        object.__setattr__(self, "hz", _ensure_binary(self.hz))
        check_css_commutation(self.hx, self.hz)


@dataclass
class CSSCode:
    """Lightweight carrier for CSS codes with teacher-friendly metadata."""

    name: str
    distance: int
    n: int
    k: int
    Hx: np.ndarray
    Hz: np.ndarray
    layout: Dict[str, Any]
    detectors_per_fault: Optional[List[List[int]]] = None
    fault_weights: Optional[List[float]] = None
    num_detectors: Optional[int] = None
    num_observables: Optional[int] = None
    stim_circuit: Optional[object] = None

    def detectors_to_syndromes(self, dets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        sz = dets[:, mx:mx + mz].astype(np.uint8)
        return sx, sz


def _assert_css(Hx: np.ndarray, Hz: np.ndarray) -> None:
    check_css_commutation(Hx, Hz)


def _gf2_rank(mat: np.ndarray) -> int:
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


def _fault_map(Hx: np.ndarray, Hz: np.ndarray) -> List[List[int]]:
    mx, n = Hx.shape
    mz = Hz.shape[0]
    mapping: List[List[int]] = []
    for j in range(n):
        dets: List[int] = []
        if mx:
            dets.extend(np.flatnonzero(Hx[:, j]).tolist())
        if mz:
            dets.extend((mx + np.flatnonzero(Hz[:, j])).tolist())
        mapping.append(dets)
    return mapping


def _make_css(
    *,
    name: str,
    distance: int,
    Hx: np.ndarray,
    Hz: np.ndarray,
    layout: Dict[str, Any],
    detectors_per_fault: Optional[List[List[int]]] = None,
    fault_weights: Optional[List[float]] = None,
    num_observables: Optional[int] = None,
) -> CSSCode:
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
    )


def save_npz(hx: np.ndarray, hz: np.ndarray, path: str | np.ndarray, meta: Dict[str, Any], **extras: Any) -> None:
    data: Dict[str, Any] = {
        "hx": _ensure_binary(hx),
        "hz": _ensure_binary(hz),
    }
    data.update(meta)
    data.update(extras)
    np.savez_compressed(path, **data)


# ---------------------------------------------------------------------------
# Rotated surface code family
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def build_surface_rotated_H(d: int) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if d < 1 or d % 2 == 0:
        raise ValueError("Rotated surface code requires odd distance >= 1")
    n_data = d * d
    hz_rows = []
    hx_rows = []

    def q_index(r: int, c: int) -> int:
        return r * d + c

    center = d // 2

    for r in range(d):
        for c in range(d):
            parity = (r + c) % 2
            qubits = {q_index(r, c)}
            if r - 1 >= 0:
                qubits.add(q_index(r - 1, c))
            if r + 1 < d:
                qubits.add(q_index(r + 1, c))
            if c - 1 >= 0:
                qubits.add(q_index(r, c - 1))
            if c + 1 < d:
                qubits.add(q_index(r, c + 1))
            if parity == 1:
                row = np.zeros(n_data, dtype=np.uint8)
                row[list(qubits)] = 1
                hz_rows.append(row)
            elif parity == 0 and not (r == center and c == center):
                row = np.zeros(n_data, dtype=np.uint8)
                row[list(qubits)] = 1
                hx_rows.append(row)

    hz = np.vstack(hz_rows)
    hx = np.vstack(hx_rows)

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


@lru_cache(maxsize=None)
def default_surface_rotated_layout(d: int) -> Dict[str, Any]:
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


def logical_surface_rotated(d: int) -> Dict[str, np.ndarray]:
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
    hx, hz, meta = build_surface_rotated_H(d)
    meta_copy = dict(meta)
    return CodeSpec(name=f"surface_d{d}", n=hx.shape[1], hx=hx, hz=hz, d=d, meta=meta_copy)


# ---------------------------------------------------------------------------
# CSSCode builders for training / teacher integration
# ---------------------------------------------------------------------------


def build_surface(distance: int, *, rotated: bool = True) -> CSSCode:
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

def _bb_indices(l: int, m: int):
    def wrap_x(x: int) -> int:
        return x % l

    def wrap_y(y: int) -> int:
        return y % m

    def idx_h(x: int, y: int) -> int:
        return wrap_y(y) * l + wrap_x(x)

    def idx_v(x: int, y: int) -> int:
        return l * m + wrap_y(y) * l + wrap_x(x)

    return idx_h, idx_v, wrap_x, wrap_y


def bb_from_shifts(l: int, m: int, a: tuple[int, int], b: tuple[int, int]) -> CodeSpec:
    if l <= 0 or m <= 0:
        raise ValueError("l and m must be positive")
    ax, ay = a
    bx, by = b
    n_qubits = 2 * l * m
    n_checks = l * m
    hx = np.zeros((n_checks, n_qubits), dtype=np.uint8)
    hz = np.zeros_like(hx)
    idx_h, idx_v, wrap_x, wrap_y = _bb_indices(l, m)

    for x in range(l):
        for y in range(m):
            row = y * l + x
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
        name=f"bb_l{l}_m{m}_a{a}_b{b}",
        n=n_qubits,
        hx=hx,
        hz=hz,
        meta={"l": l, "m": m, "a": a, "b": b, "syndrome_order": "Z_first_then_X"},
    )


def bb_gross() -> CodeSpec:
    return bb_from_shifts(l=12, m=6, a=(3, -1), b=(-1, -3))


def build_bb(n1: int = 31, n2: int = 31, *, w1: int = 3, w2: int = 3) -> CSSCode:
    rng = np.random.default_rng(123)

    def circ_H(n: int, w: int) -> np.ndarray:
        H = np.zeros((n, n), dtype=np.uint8)
        base = rng.choice(n, size=w, replace=False)
        for r in range(n):
            idx = (base + r) % n
            H[r, idx] = 1
        return H

    H1 = circ_H(n1, w1)
    H2 = circ_H(n2, w2)
    code = build_hgp(H1, H2, name="bb")
    code.layout.update({"n1": n1, "n2": n2, "w1": w1, "w2": w2})
    code.distance = -1
    return code


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
    return CodeSpec(name="hgp", n=n, hx=hx, hz=hz,
                    meta={"H1_shape": H1.shape, "H2_shape": H2.shape, "syndrome_order": "Z_first_then_X"})


def build_hgp(H1: np.ndarray, H2: np.ndarray, *, name: str = "hgp") -> CSSCode:
    if H1 is None or H2 is None:
        raise ValueError("H1 and H2 must be provided for the HGP builder")
    H1 = _ensure_binary(np.asarray(H1, dtype=np.uint8))
    H2 = _ensure_binary(np.asarray(H2, dtype=np.uint8))
    m1, n1 = H1.shape
    m2, n2 = H2.shape
    I_n1 = np.eye(n1, dtype=np.uint8)
    I_m1 = np.eye(m1, dtype=np.uint8)
    I_n2 = np.eye(n2, dtype=np.uint8)
    I_m2 = np.eye(m2, dtype=np.uint8)

    Hx_left = np.kron(H1, I_n2).astype(np.uint8)
    Hx_right = np.kron(I_m1, H2.T).astype(np.uint8)
    Hx = np.concatenate([Hx_left, Hx_right], axis=1)

    Hz_left = np.kron(I_n1, H2).astype(np.uint8)
    Hz_right = np.kron(H1.T, I_m2).astype(np.uint8)
    Hz = np.concatenate([Hz_left, Hz_right], axis=1)
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
    H = np.array([
        [1, 1, 1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 1, 0],
        [1, 0, 1, 1, 0, 0, 1],
    ], dtype=np.uint8)
    return CodeSpec(name="steane", n=7, hx=H, hz=H, k=1, d=3, meta={"syndrome_order": "Z_first_then_X"})


def build_steane() -> CSSCode:
    H = np.array([
        [1, 0, 0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 1, 1, 1],
    ], dtype=np.uint8)
    layout = {"family": "steane"}
    return _make_css(
        name="steane",
        distance=3,
        Hx=H,
        Hz=H,
        layout=layout,
        fault_weights=[1.0] * 7,
        num_observables=1,
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
    return CodeSpec(name=f"qrm_hamming_m{m}", n=n, hx=H, hz=H, k=k, d=3,
                    meta={"m": m, "syndrome_order": "Z_first_then_X"})


def build_color(distance: int) -> CSSCode:
    d = int(distance)
    if d != 3:
        raise ValueError("Only distance=3 triangular color code is provided")
    H = np.array([
        [1, 0, 0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 1, 1, 1],
    ], dtype=np.uint8)
    layout = {"triangular": True, "distance": d}
    return _make_css(
        name="color",
        distance=d,
        Hx=H,
        Hz=H,
        layout=layout,
        fault_weights=[1.0] * 7,
        num_observables=1,
    )


REGISTRY = {
    "surface": lambda distance, **kw: build_surface(distance, **kw),
    "repetition": lambda distance, **kw: build_repetition(distance, **kw),
    "steane": lambda distance=None, **kw: build_steane(),
    "color": lambda distance, **kw: build_color(distance, **kw),
    "bb": lambda distance=None, n1=31, n2=31, w1=3, w2=3, **kw: build_bb(n1=n1, n2=n2, w1=w1, w2=w2),
    "hgp": lambda distance=None, H1=None, H2=None, name="hgp", **kw: build_hgp(H1, H2, name=name),
}

_OPTIONAL_DISTANCE = {"steane", "bb", "hgp"}


def get_code(family: str, distance: Optional[int] = None, **kw) -> CSSCode:
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
    return CodeSpec(name=f"repetition_{n_data}", n=n_data, hx=hx, hz=hz, k=1, d=n_data,
                    meta={"syndrome_order": "Z_first_then_X"})


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
    else:
        Hx = np.zeros((L - 1, n), dtype=np.uint8)
        for i in range(L - 1):
            Hx[i, [i, i + 1]] = 1
        Hz = np.zeros((0, n), dtype=np.uint8)
    layout = {"L": L, "basis": basis_upper}
    return _make_css(
        name="repetition",
        distance=L,
        Hx=Hx,
        Hz=Hz,
        layout=layout,
        fault_weights=[1.0] * n,
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
    with open(path, "r", encoding="utf-8") as fh:
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
