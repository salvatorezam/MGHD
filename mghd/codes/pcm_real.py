from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def rotated_surface_pcm(d: int, side: str) -> sp.csr_matrix:
    """Return rotated surface code parity-check matrix for odd distance d."""
    if d % 2 == 0 or d < 1:
        raise ValueError("rotated_surface_pcm requires odd d >= 1")

    side = side.upper()
    if side not in {"X", "Z"}:
        raise ValueError("side must be 'X' or 'Z'")

    n_qubits = d * d
    n_checks = (d * d - 1) // 2

    rows: list[int] = []
    cols: list[int] = []
    row_idx = 0

    def q_index(r: int, c: int) -> int:
        return r * d + c

    center = d // 2

    for r in range(d):
        for c in range(d):
            parity = (r + c) % 2
            include = False
            if side == "Z":
                include = (parity == 1)
            else:  # side == 'X'
                include = (parity == 0) and not (r == center and c == center)
            if not include:
                continue

            qubits = {q_index(r, c)}
            if r - 1 >= 0:
                qubits.add(q_index(r - 1, c))
            if r + 1 < d:
                qubits.add(q_index(r + 1, c))
            if c - 1 >= 0:
                qubits.add(q_index(r, c - 1))
            if c + 1 < d:
                qubits.add(q_index(r, c + 1))

            for q in sorted(qubits):
                rows.append(row_idx)
                cols.append(q)
            row_idx += 1

    if row_idx != n_checks:
        raise AssertionError(f"Constructed {row_idx} checks, expected {n_checks}")

    data = np.ones(len(rows), dtype=np.uint8)
    H = sp.coo_matrix((data, (rows, cols)), shape=(n_checks, n_qubits)).tocsr()
    return H
