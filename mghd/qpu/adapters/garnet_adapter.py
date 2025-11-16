from __future__ import annotations

"""Garnet adapter: convenience helpers around CUDA‑Q surface sampling.

This module provides a simple, self‑contained entrypoint (`sample_round`) that
returns CSS parity checks, single‑round detector bits (Z→X order), and lattice
coordinates for both data qubits and check operators. It also exposes
`split_components_for_side`, a utility that groups active checks into connected
components suitable for packing as training crops.

Design notes:
- Detector order is canonicalized to Z first, then X (consistent with DEM/Stim).
- Check coordinates preserve geometric half‑offsets and are emitted as float32.
- A synthetic fallback path keeps the pipeline alive when CUDA‑Q is unavailable.
"""

import os
from typing import Any

import numpy as np

from mghd.samplers.cudaq_sampler import cudaq_sample_surface_wrapper
from mghd.samplers.cudaq_backend.circuits import (
    build_H_rotated_general,
    make_surface_layout_general,
)

# Lazy CUDA-Q backend hooks; imported only when sampling is invoked
from mghd.samplers.cudaq_backend.garnet_noise import (
    FOUNDATION_DEFAULTS,  # single source of truth for noise
)

_USE_SYNTH = os.getenv("MGHD_SYNTHETIC", "0") == "1"
_MODE = os.getenv("MGHD_MODE", "foundation")  # {"foundation","student"}


def sample_round(d: int, p: float, seed: int, profile_path: str | None = None) -> dict[str, Any]:
    """Sample a single surface‑code round via CUDA‑Q (or synthetic fallback).

    Returns
    - Hx, Hz: uint8 parity‑check matrices trimmed to measured checks
    - synZ, synX: uint8 detector bits (Z then X ordering)
    - coords_q: int32 data‑qubit lattice coordinates
    - coords_c: float32 check coordinates (Z first then X), half‑offset preserved
    - dem_meta: metadata for downstream teachers (opaque)
    """
    if _USE_SYNTH:
        return _synthetic_sample_round(d, p, seed)

    # Create RNG and layout for CUDA-Q sampling
    rng = np.random.default_rng(seed)

    # Use general layout and matrix functions for arbitrary distances
    layout = make_surface_layout_general(d)
    Hx, Hz = build_H_rotated_general(d)

    try:
        # Sample syndrome data
        result = cudaq_sample_surface_wrapper(
            mode=_MODE,
            batch_size=1,
            T=3,  # Standard syndrome extraction rounds
            d=d,
            layout=layout,
            rng=rng,
            bitpack=False,
            profile_json=profile_path,
        )

        # Extract Z/X syndrome components from packed result
        # Backend format: [X_syndrome, 2*Z_syndrome, X_error + 2*Z_error]
        n_x_checks = len(layout["ancilla_x"])
        n_z_checks = len(layout["ancilla_z"])
        synX = result[0, :n_x_checks].astype(np.uint8)
        synZ = result[0, n_x_checks : n_x_checks + n_z_checks].astype(np.uint8)

        # Trim matrices to match actual measured ancilla (remove boundary checks)
        Hx_measured = Hx[:n_x_checks, :]
        Hz_measured = Hz[:n_z_checks, :]

        # Generate coordinate information for clustering (Z→X order for checks)
        coords_q = _generate_qubit_coords(d)
        coords_c = _generate_check_coords(d)

        dem_meta = {
            "backend": "cudaq",
            "mode": _MODE,
            "layout": layout,
            "noise_defaults": FOUNDATION_DEFAULTS,
            "T": 3,
            "d": d,
            "requested_p": p,
        }

        return {
            "Hx": Hx_measured.astype(np.uint8),
            "Hz": Hz_measured.astype(np.uint8),
            "synZ": synZ,
            "synX": synX,
            "coords_q": coords_q,
            "coords_c": coords_c,
            "dem_meta": dem_meta,
        }

    except Exception as e:
        print(f"CUDA-Q sampling failed for d={d}: {e}")
        print("Falling back to synthetic sampling")
        return _synthetic_sample_round(d, p, seed)


def _synthetic_sample_round(d: int, p: float, seed: int) -> dict[str, Any]:
    """
    Synthetic fallback when CUDA-Q unavailable or MGHD_SYNTHETIC=1.
    Higher effective error rate than real code for differentiation.
    """
    rng = np.random.default_rng(seed)

    # Surface code dimensions for distance d
    n_data = d * d
    # Rotated planar per-round check counts (each type)
    n_check_z = (d * d - 1) // 2
    n_check_x = (d * d - 1) // 2

    # Create simple parity-check matrices (placeholder structure)
    Hz = rng.integers(0, 2, (n_check_z, n_data), dtype=np.uint8)
    Hx = rng.integers(0, 2, (n_check_x, n_data), dtype=np.uint8)

    # Generate syndromes with higher effective error rate (3x for distinction)
    effective_p = min(0.1, 3.0 * p)
    synZ = rng.choice([0, 1], size=n_check_z, p=[1 - effective_p, effective_p]).astype(np.uint8)
    synX = rng.choice([0, 1], size=n_check_x, p=[1 - effective_p, effective_p]).astype(np.uint8)

    # Generate coordinate grids (checks are Z then X)
    coords_q = _generate_qubit_coords(d)
    coords_c = _generate_check_coords(d)  # Z first then X

    return {
        "Hx": Hx,
        "Hz": Hz,
        "synZ": synZ,
        "synX": synX,
        "coords_q": coords_q,
        "coords_c": coords_c,
        "dem_meta": {"synthetic": True, "effective_p": effective_p},
    }


def _generate_qubit_coords(d: int) -> np.ndarray:
    """Grid coordinates for data qubits on a rotated surface lattice (int32)."""
    coords = []
    for i in range(d):
        for j in range(d):
            # Rotated lattice positioning
            x = i + j
            y = i - j
            coords.append([x, y])
    return np.array(coords, dtype=np.int32)


def _generate_check_coords(d: int) -> np.ndarray:
    """Check coordinates (float32), Z checks first then X checks.

    Preserves half‑grid offsets so geometry is consistent with rotated planar
    embeddings and can be used directly for Hilbert ordering and bbox.
    """
    coords: list[list[float]] = []
    # Z checks first (canonical), then X checks
    for i in range(d):
        for j in range(d - 1):
            x = i + j + 0.5
            y = i - j - 0.5
            coords.append([x, y])
    for i in range(d - 1):
        for j in range(d):
            x = i + j + 0.5
            y = i - j + 0.5
            coords.append([x, y])
    return np.array(coords, dtype=np.float32)


def split_components_for_side(
    *,
    side: str,
    Hx: np.ndarray,
    Hz: np.ndarray,
    synZ: np.ndarray,
    synX: np.ndarray,
    coords_q: np.ndarray,
    coords_c: np.ndarray,
) -> list[dict[str, Any]]:
    """
    Build connected components + 1-hop halo for the chosen side ('Z' or 'X').
    Output list entries contain fields the crop packer expects:
      H_sub, xy_qubit, xy_check, synd_bits, bbox_xywh, k, r, kappa_stats
    Uses the in-repo clustering utilities (mghd.decoders.lsd.clustered).
    """
    try:
        import scipy.sparse as sp

        # clustered lives under mghd.decoders.lsd
        from mghd.decoders.lsd import clustered as cc

        # Select appropriate matrices and syndromes based on side
        if side == "Z":
            H = Hz
            synd_bits = synZ
            check_coords = coords_c[: len(synZ)]  # Z checks come first
        elif side == "X":
            H = Hx
            synd_bits = synX
            check_coords = coords_c[len(synZ) : len(synZ) + len(synX)]  # X after Z
        else:
            raise ValueError(f"Unknown side: {side}")

        # Convert to sparse for clustering functions
        H_sparse = sp.csr_matrix(H)

        # Use active_components to find connected components
        check_groups, qubit_groups = cc.active_components(H_sparse, synd_bits, halo=1)

        components = []
        for check_indices, qubit_indices in zip(check_groups, qubit_groups):
            if len(check_indices) == 0 or len(qubit_indices) == 0:
                continue

            # Extract submatrix
            H_sub = H[np.ix_(check_indices, qubit_indices)]

            # Get coordinates
            xy_qubit = coords_q[qubit_indices]
            xy_check = check_coords[check_indices]

            # Extract syndrome bits for this component
            synd_component = synd_bits[check_indices]

            # Compute bounding box
            all_coords = np.vstack([xy_qubit, xy_check])
            x_min, y_min = all_coords.min(axis=0)
            x_max, y_max = all_coords.max(axis=0)
            bbox_xywh = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]

            # Compute component statistics
            k = len(check_indices)  # number of checks
            r = len(qubit_indices)  # number of qubits
            kappa_stats = {
                "k": k,
                "r": r,
                "density": k / max(1, r),
                "syndrome_weight": int(synd_component.sum()),
                "component_id": len(components),
            }

            component = {
                "H_sub": H_sub.astype(np.uint8),
                "xy_qubit": np.asarray(xy_qubit),
                "xy_check": np.asarray(xy_check),
                "synd_bits": synd_component.astype(np.uint8),
                "bbox_xywh": bbox_xywh,
                "k": k,
                "r": r,
                "kappa_stats": kappa_stats,
                "qubit_indices": np.asarray(qubit_indices, dtype=np.int32),
                "check_indices": np.asarray(check_indices, dtype=np.int32),
            }

            components.append(component)

        return components

    except ImportError:
        # Fallback if clustering module not available
        return _fallback_split_components(
            side=side, Hx=Hx, Hz=Hz, synZ=synZ, synX=synX, coords_q=coords_q, coords_c=coords_c
        )


def _fallback_split_components(
    *,
    side: str,
    Hx: np.ndarray,
    Hz: np.ndarray,
    synZ: np.ndarray,
    synX: np.ndarray,
    coords_q: np.ndarray,
    coords_c: np.ndarray,
) -> list[dict[str, Any]]:
    """Fallback component splitting when clustering module not available"""

    # Select appropriate matrices and syndromes
    if side == "Z":
        H = Hz
        synd_bits = synZ
        check_coords = coords_c[: len(synZ)]
    elif side == "X":
        H = Hx
        synd_bits = synX
        check_coords = coords_c[len(synZ) : len(synZ) + len(synX)]
    else:
        raise ValueError(f"Unknown side: {side}")

    # Find connected components based on syndrome bits
    active_checks = np.where(synd_bits > 0)[0]

    if len(active_checks) == 0:
        # No active syndrome - return empty list
        return []

    # For simplicity, treat all active checks as one component
    # In practice, you'd do proper connected component analysis

    # Get involved qubits (columns with non-zero entries in active rows)
    involved_qubits = set()
    for check_idx in active_checks:
        qubit_indices = np.where(H[check_idx] > 0)[0]
        involved_qubits.update(qubit_indices)

    involved_qubits = sorted(list(involved_qubits))

    if len(involved_qubits) == 0:
        return []

    # Extract submatrix
    H_sub = H[np.ix_(active_checks, involved_qubits)]

    # Get coordinates
    xy_qubit = coords_q[involved_qubits]
    xy_check = check_coords[active_checks]

    # Compute bounding box
    all_coords = np.vstack([xy_qubit, xy_check])
    x_min, y_min = all_coords.min(axis=0)
    x_max, y_max = all_coords.max(axis=0)
    bbox_xywh = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]

    # Compute component statistics
    k = len(active_checks)  # number of checks
    r = len(involved_qubits)  # number of qubits
    kappa_stats = {
        "k": k,
        "r": r,
        "density": k / max(1, r),
        "syndrome_weight": int(synd_bits.sum()),
    }

    component = {
        "H_sub": H_sub.astype(np.uint8),
        "xy_qubit": xy_qubit.astype(np.int32),
        "xy_check": xy_check.astype(np.int32),
        "synd_bits": synd_bits[active_checks].astype(np.uint8),
        "bbox_xywh": bbox_xywh,
        "k": k,
        "r": r,
        "kappa_stats": kappa_stats,
        "qubit_indices": np.asarray(involved_qubits, dtype=np.int32),
        "check_indices": np.asarray(active_checks, dtype=np.int32),
    }

    return [component]


__all__ = ["sample_round", "split_components_for_side"]
