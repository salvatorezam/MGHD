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

try:
    import numba
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

from mghd.samplers.cudaq_sampler import cudaq_sample_surface_wrapper
from mghd.samplers.cudaq_backend.circuits import (
    build_H_rotated_general,
    make_surface_layout_general,
)

# Lazy CUDA-Q backend hooks; imported only when sampling is invoked
from mghd.samplers.cudaq_backend.garnet_noise import (
    FOUNDATION_DEFAULTS,  # single source of truth for noise
)

_MODE = os.getenv("MGHD_MODE", "foundation")  # {"foundation","student"}
_NEIGHBOR_CACHE: dict[tuple, np.ndarray] = {}


def _use_synth() -> bool:
    """Check if synthetic sampling should be used (evaluated at call time)."""
    return os.getenv("MGHD_SYNTHETIC", "0") == "1"


def sample_round(d: int, p: float, seed: int, profile_path: str | None = None) -> dict[str, Any]:
    """Sample a single surface‑code round via CUDA‑Q (or synthetic fallback).

    Returns
    - Hx, Hz: uint8 parity‑check matrices trimmed to measured checks
    - synZ, synX: uint8 detector bits (Z then X ordering)
    - ex_glob, ez_glob: uint8 oracle data‑qubit error indicators (X/Z), if available
    - coords_q: int32 data‑qubit lattice coordinates
    - coords_c: float32 check coordinates (Z first then X), half‑offset preserved
    - dem_meta: metadata for downstream teachers (opaque)
    """
    if _use_synth():
        return _synthetic_sample_round(d, p, seed)

    # Check if we should use Stim for circuit-level noise
    # This is triggered if MGHD_SAMPLER is set to "stim" (handled in train.py)
    if os.getenv("MGHD_SAMPLER", "cudaq") == "stim":
        return _stim_sample_round(d, p, seed)

    # Create RNG and layout for CUDA-Q sampling
    rng = np.random.default_rng(seed)

    # Use generic scalable lattice layout by default, and keep the legacy
    # d=3 hardware-aware map only when explicitly using the Garnet model.
    noise_model_kind = str(os.getenv("MGHD_NOISE_MODEL", "generic_cl")).strip().lower()
    hardware_aware_d3 = noise_model_kind in {"garnet", "hardware", "profile"}
    layout = make_surface_layout_general(d, hardware_aware_d3=hardware_aware_d3)
    # build_H_rotated_general returns (Hz, Hx)
    Hz, Hx = build_H_rotated_general(d)

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
            phys_p=float(p),
            noise_scale=None,
        )

        # Extract Z/X syndrome components from packed result
        # Backend format: [X_syndrome, 2*Z_syndrome, X_error + 2*Z_error]
        n_x_checks = len(layout["ancilla_x"])
        n_z_checks = len(layout["ancilla_z"])
        sx_raw = result[0, :n_x_checks].astype(np.uint8)
        sz_raw = result[0, n_x_checks : n_x_checks + n_z_checks].astype(np.uint8)
        synX = (sx_raw & 1).astype(np.uint8)
        synZ = ((sz_raw >> 1) & 1).astype(np.uint8)

        err_block = result[0, n_x_checks + n_z_checks :].astype(np.uint8)
        # Decode packed Pauli: X + 2Z => x_err = bit0, z_err = bit1
        ex_glob = (err_block & 1).astype(np.uint8)
        ez_glob = ((err_block >> 1) & 1).astype(np.uint8)

        # Trim matrices to match actual measured ancilla (remove boundary checks)
        Hx_measured = Hx[:n_x_checks, :]
        Hz_measured = Hz[:n_z_checks, :]

        # Generate coordinate information for clustering (Z→X order for checks)
        coords_q = _generate_qubit_coords(d)
        coords_c = _generate_check_coords(d)

        dem_meta = {
            "backend": "cudaq",
            "mode": _MODE,
            "noise_model": noise_model_kind,
            "layout": layout,
            "noise_defaults": FOUNDATION_DEFAULTS,
            "T": 3,
            "d": d,
            "requested_p": float(p),
            "effective_p": float(p),
            "noise_scale": None,
        }

        return {
            "Hx": Hx_measured.astype(np.uint8),
            "Hz": Hz_measured.astype(np.uint8),
            "synZ": synZ,
            "synX": synX,
            "ex_glob": ex_glob,
            "ez_glob": ez_glob,
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
    Synthetic fallback using proper phenomenological noise on valid surface code.
    Generates valid Hx/Hz matrices and consistent syndromes so MWPM works.
    """
    rng = np.random.default_rng(seed)

    # 1. Build valid surface code matrices
    # build_H_rotated_general returns (Hz, Hx)
    Hz, Hx = build_H_rotated_general(d)
    
    n_data = d * d
    
    # 2. Generate phenomenological noise (random data qubit errors)
    # X errors trigger Z checks (Hz)
    # Z errors trigger X checks (Hx)
    # We simulate X and Z errors independently with probability p
    
    # Generate error vectors
    err_x = rng.choice([0, 1], size=n_data, p=[1 - p, p]).astype(np.uint8)
    err_z = rng.choice([0, 1], size=n_data, p=[1 - p, p]).astype(np.uint8)
    
    # 3. Compute syndromes
    # Z-checks detect X-errors: synZ = Hz @ err_x
    synZ = (Hz @ err_x) % 2
    
    # X-checks detect Z-errors: synX = Hx @ err_z
    synX = (Hx @ err_z) % 2

    # Generate coordinate grids (checks are Z then X)
    coords_q = _generate_qubit_coords(d)
    coords_c = _generate_check_coords(d)  # Z first then X

    return {
        "Hx": Hx.astype(np.uint8),
        "Hz": Hz.astype(np.uint8),
        "synZ": synZ.astype(np.uint8),
        "synX": synX.astype(np.uint8),
        "ex_glob": err_x.astype(np.uint8),
        "ez_glob": err_z.astype(np.uint8),
        "coords_q": coords_q,
        "coords_c": coords_c,
        "dem_meta": {"synthetic": True, "effective_p": p, "model": "phenomenological"},
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


if _NUMBA_AVAILABLE:
    @njit(parallel=False, cache=True)
    def _flood_fill_single(syn: np.ndarray, neighbors: np.ndarray, n_checks: int):
        """Numba-compiled flood fill for connected components with halo."""
        visited = np.zeros(n_checks, dtype=np.bool_)
        components = []
        
        for start in range(n_checks):
            if syn[start] and not visited[start]:
                component = []
                halo = np.zeros(n_checks, dtype=np.bool_)
                stack = [start]
                visited[start] = True
                halo[start] = True
                
                while len(stack) > 0:
                    node = stack.pop()
                    component.append(node)
                    
                    # Visit active neighbors
                    for nb_idx in range(neighbors.shape[1]):
                        nb = neighbors[node, nb_idx]
                        if nb == -1:
                            break
                        if syn[nb] and not visited[nb]:
                            visited[nb] = True
                            stack.append(nb)
                        # Add all neighbors to halo
                        halo[nb] = True
                
                components.append((np.array(component, dtype=np.int32), np.where(halo)[0].astype(np.int32)))
        
        return components


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
    Uses Numba-accelerated clustering when available, falls back to scipy.
    """
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

    # Try fast Numba path first
    if _NUMBA_AVAILABLE:
        try:
            # Build or retrieve cached neighbor list
            global _NEIGHBOR_CACHE
            cache_key = (side, tuple(check_coords.flatten()))
            if cache_key not in _NEIGHBOR_CACHE:
                n = len(check_coords)
                neighbors = np.full((n, 12), -1, dtype=np.int32)  # max 12 neighbors in surface
                count = np.zeros(n, dtype=np.int32)
                for i in range(n):
                    for j in range(i + 1, n):
                        dist = np.linalg.norm(check_coords[i] - check_coords[j])
                        if dist < 1.6:  # threshold for adjacent checks
                            neighbors[i, count[i]] = j
                            neighbors[j, count[j]] = i
                            count[i] += 1
                            count[j] += 1
                _NEIGHBOR_CACHE[cache_key] = neighbors
            
            neighbors = _NEIGHBOR_CACHE[cache_key]
            
            # Run fast flood fill
            comp_list = _flood_fill_single(synd_bits, neighbors, len(check_coords))
            
            components = []
            for active_arr, halo_arr in comp_list:
                if len(halo_arr) == 0:
                    continue
                    
                # Determine qubit indices involved in this component
                qubit_indices = sorted(list(set(np.where(H[halo_arr, :].any(axis=0))[0])))
                
                if len(qubit_indices) == 0:
                    continue
                
                # Extract submatrix
                H_sub = H[np.ix_(halo_arr, qubit_indices)]
                
                # Get coordinates
                xy_qubit = coords_q[qubit_indices]
                xy_check = check_coords[halo_arr]
                
                # Extract syndrome bits for this component
                synd_component = synd_bits[halo_arr]
                
                # Compute bounding box
                all_coords = np.vstack([xy_qubit, xy_check])
                x_min, y_min = all_coords.min(axis=0)
                x_max, y_max = all_coords.max(axis=0)
                bbox_xywh = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
                
                # Compute component statistics
                k = len(halo_arr)
                r = len(qubit_indices)
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
                    "check_indices": halo_arr,
                }
                
                components.append(component)
            
            return components
            
        except Exception:
            pass  # Fall through to scipy path
    
    # Fallback to scipy path
    try:
        import scipy.sparse as sp
        from mghd.decoders.lsd import clustered as cc

        H_sparse = sp.csr_matrix(H)
        check_groups, qubit_groups = cc.active_components(H_sparse, synd_bits, halo=1)

        components = []
        for check_indices, qubit_indices in zip(check_groups, qubit_groups):
            if len(check_indices) == 0 or len(qubit_indices) == 0:
                continue

            H_sub = H[np.ix_(check_indices, qubit_indices)]
            xy_qubit = coords_q[qubit_indices]
            xy_check = check_coords[check_indices]
            synd_component = synd_bits[check_indices]

            all_coords = np.vstack([xy_qubit, xy_check])
            x_min, y_min = all_coords.min(axis=0)
            x_max, y_max = all_coords.max(axis=0)
            bbox_xywh = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]

            k = len(check_indices)
            r = len(qubit_indices)
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


# Cache for Stim circuit objects to avoid rebuilding every sample
_STIM_CIRCUIT_CACHE: dict[tuple[int, float], dict] = {}


def _get_stim_circuit_objects(d: int, p: float) -> dict:
    """
    Get or create cached Stim circuit objects for given (d, p).
    Caches: circuit, Hz, Hx, det_coords tuple, DEM, matcher, and other reusable objects.
    Does NOT cache sampler - that needs seed at compile time.
    
    Uses rounds=d for proper circuit-level noise simulation.
    """
    import stim
    
    key = (d, p)
    if key in _STIM_CIRCUIT_CACHE:
        return _STIM_CIRCUIT_CACHE[key]
    
    # Build matrices
    Hz, Hx = build_H_rotated_general(d)
    
    # Generate circuit with rounds=d for proper circuit-level noise
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=d,  # Use d rounds for proper circuit-level error correction
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p
    )
    
    # Get Detector Error Model for proper MWPM decoding
    dem = circuit.detector_error_model(decompose_errors=True)
    
    # Create pymatching matcher from DEM
    try:
        import pymatching
        matcher = pymatching.Matching.from_detector_error_model(dem)
    except Exception:
        matcher = None
    
    # Get detector coordinates for mapping
    det_coords = circuit.get_detector_coordinates()
    det_coords_tuple = tuple(sorted((k, tuple(v)) for k, v in det_coords.items()))
    
    # Pre-compute check counts
    n_z = Hz.shape[0]
    n_x = Hx.shape[0]
    
    # Pre-compute qubit and check coordinates
    coords_q = _generate_qubit_coords(d)
    coords_c = _generate_check_coords(d)
    
    # Pre-compute permutation
    perm = _get_stim_permutation(d, det_coords_tuple)
    
    cached = {
        "circuit": circuit,
        "dem": dem,
        "matcher": matcher,
        "Hz": Hz.astype(np.uint8),
        "Hx": Hx.astype(np.uint8),
        "det_coords_tuple": det_coords_tuple,
        "n_z": n_z,
        "n_x": n_x,
        "coords_q": coords_q,
        "coords_c": coords_c,
        "perm": perm,
    }
    
    _STIM_CIRCUIT_CACHE[key] = cached
    return cached


def _stim_sample_round(d: int, p: float, seed: int) -> dict[str, Any]:
    """
    Sample using Stim with circuit-level noise (rounds=d).
    Uses cached circuit objects for efficiency.
    Returns detector samples in STIM's NATIVE ordering (no permutation).
    The DEM-based teacher uses the same ordering, guaranteeing consistency.
    """
    # Get cached objects
    cached = _get_stim_circuit_objects(d, p)
    
    circuit = cached["circuit"]
    dem = cached["dem"]
    matcher = cached["matcher"]
    
    # Compile sampler with seed and sample
    sampler = circuit.compile_detector_sampler(seed=seed)
    det_sample, obs_sample = sampler.sample(shots=1, separate_observables=True)
    det_sample = det_sample[0]  # Shape: (num_detectors,)
    obs_sample = obs_sample[0]  # Shape: (num_observables,)
    
    # Get detector coordinates directly from Stim
    det_coords = circuit.get_detector_coordinates()
    n_detectors = len(det_sample)
    
    # Build coordinate arrays from Stim's detector coordinates
    # det_coords is {detector_idx: [x, y, t, ...]}
    coords_det = np.zeros((n_detectors, 2), dtype=np.float32)
    for idx, coord in det_coords.items():
        if idx < n_detectors:
            coords_det[idx, 0] = coord[0]  # x
            coords_det[idx, 1] = coord[1]  # y
    
    # For data qubit coordinates, use the standard grid
    n_data = d * d
    coords_q = _generate_qubit_coords(d)
    
    # Build H matrix from DEM for graph construction
    # Each error in DEM defines edges in the detector graph
    H_det = _build_H_from_dem(dem, n_detectors, n_data)
    
    dem_meta = {
        "backend": "stim_native",
        "mode": "circuit",
        "d": d,
        "p": p,
        "n_detectors": n_detectors,
        "dem": dem,
        "matcher": matcher,
        "observable": obs_sample.astype(np.uint8),
    }

    return {
        "detectors": det_sample.astype(np.uint8),  # Native Stim ordering
        "observable": obs_sample.astype(np.uint8),
        "coords_det": coords_det,
        "coords_q": coords_q,
        "H_det": H_det,
        "dem_meta": dem_meta,
        # Legacy fields for compatibility (will be removed)
        "Hx": cached["Hx"],
        "Hz": cached["Hz"],
        "synZ": det_sample[:cached["n_z"]].astype(np.uint8),  # Approximate split
        "synX": det_sample[cached["n_z"]:cached["n_z"]+cached["n_x"]].astype(np.uint8),
        "coords_c": cached["coords_c"],
    }


def _build_H_from_dem(dem, n_detectors: int, n_data: int) -> np.ndarray:
    """
    Build a detector-to-fault matrix from the DEM.
    This defines the graph structure for the GNN.
    
    Returns H_det: (n_detectors, n_faults) where H_det[d,f]=1 if fault f triggers detector d.
    """
    # Count faults in DEM
    faults = []
    for inst in dem.flattened():
        if inst.type == "error":
            targets = inst.targets_copy()
            det_ids = [t.val for t in targets if t.is_relative_detector_id()]
            obs_flip = any(t.is_logical_observable_id() for t in targets)
            faults.append({
                'detectors': det_ids,
                'flips_observable': obs_flip,
            })
    
    n_faults = len(faults)
    H_det = np.zeros((n_detectors, n_faults), dtype=np.uint8)
    
    for f_idx, fault in enumerate(faults):
        for d_idx in fault['detectors']:
            if d_idx < n_detectors:
                H_det[d_idx, f_idx] = 1
    
    return H_det

_STIM_PERM_CACHE = {}

def _get_stim_permutation(d, det_coords_tuple):
    key = (d, det_coords_tuple)
    if key in _STIM_PERM_CACHE:
        return _STIM_PERM_CACHE[key]

    # Stim-native path is not used by the MGHDv2 parity-check training flow.
    # Keep the mapping deterministic and explicit.
    n_checks = len(_generate_check_coords(d))
    perm = list(range(n_checks))
    _STIM_PERM_CACHE[key] = perm
    return perm


__all__ = ["sample_round", "split_components_for_side"]
