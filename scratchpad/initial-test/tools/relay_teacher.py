#!/usr/bin/env python3
"""
Relay-BP labeling script for quantum error correction.

- Uses relay_bp or MWPM to decode batch syndromes into data-qubit corrections
- Supports loading Hx/Hz (Astra matrices) or constructing surface code via Planar2DCode
- Supports both rotated and planar surface code layouts
- Strict parity validation for correctness
"""

import argparse
import sys
import os
import time
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import warnings

import numpy as np
import hashlib

# Add the parent directory to the path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cudaq_backend.circuits import build_H_rotated_d3_from_cfg

# GF(2) helpers
def _gf2_rref(A: np.ndarray):
    """Gauss-Jordan RREF over GF(2). A is uint8 (m,n). Returns (R, pivots)."""
    A = (A.copy().astype(np.uint8))  # defensive copy
    m, n = A.shape
    pivots = []
    r = 0
    for c in range(n):
        # find pivot row
        pr = None
        for rr in range(r, m):
            if A[rr, c] & 1:
                pr = rr
                break
        if pr is None:
            continue
        if pr != r:
            A[[r, pr]] = A[[pr, r]]
        # eliminate other rows in this column
        for rr in range(m):
            if rr != r and (A[rr, c] & 1):
                A[rr, :] ^= A[r, :]
        pivots.append(c)
        r += 1
        if r == m:
            break
    return A, pivots

def _gf2_solve_particular(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve A x = b over GF(2). A is (m,n) uint8, b is (m,) uint8.
    Returns one particular solution x (n,) uint8, raising if inconsistent.
    """
    A = A.astype(np.uint8, copy=False)
    b = b.astype(np.uint8, copy=False)
    m, n = A.shape
    Ab = np.concatenate([A, b.reshape(m,1)], axis=1).astype(np.uint8)
    R, pivots = _gf2_rref(Ab)
    # inconsistency: row all-zeros in A part but last col = 1
    for i in range(m):
        if not R[i, :n].any() and (R[i, n] & 1):
            raise ValueError("GF(2) system inconsistent.")
    x = np.zeros(n, dtype=np.uint8)
    # back-substitute from RREF: pivot rows have a single 1 at pivot col
    for i in range(len(pivots)-1, -1, -1):
        pc = pivots[i]
        row = R[i, :]
        rhs = row[n] & 1
        # sum over free columns j where row[j]==1
        s = 0
        # iterate all columns except pivot
        ones = np.where(row[:n] == 1)[0]
        for j in ones:
            if j == pc:
                continue
            s ^= (x[j] & 1)
        x[pc] = (rhs ^ s) & 1
    return x

def _gf2_nullspace_basis(A: np.ndarray):
    """
    Return a list of basis vectors for nullspace of A over GF(2).
    Each vector is (n,) uint8. Uses RREF structure to build free-variable basis.
    """
    A = A.astype(np.uint8, copy=False)
    m, n = A.shape
    R, pivots = _gf2_rref(A)
    pivset = set(pivots)
    free = [j for j in range(n) if j not in pivset]
    basis = []
    # Build basis vector for each free variable f
    for f in free:
        v = np.zeros(n, dtype=np.uint8)
        v[f] = 1
        # set pivot vars from RREF rows
        # RREF has pivot rows at top positions 0..len(pivots)-1
        for i, pc in enumerate(pivots):
            row = R[i, :n]
            # row has 1 at pc, possibly 1s at free columns
            s = 0
            if row[f]:
                s ^= 1
            # other free vars are 0 except f
            v[pc] = s & 1
        basis.append(v)
    return basis

def _pick_min_weight(vecs):
    """Pick minimal Hamming weight vector from list; tie-break by lexicographic order."""
    if not vecs:
        return np.zeros(0, dtype=np.uint8)
    weights = [int(v.sum()) for v in vecs]
    minw = min(weights)
    cand = [v for v, w in zip(vecs, weights) if w == minw]
    cand.sort(key=lambda x: x.tobytes())
    return cand[0].astype(np.uint8)

def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

# --- strict full-batch split-parity validator (used by all teachers) ---
def _strict_save_and_roundtrip_validate(*, Hz_u8, Hx_u8, synd_bin, labels_x, labels_z, meta, input_npz_path: str, out_npz_path: str, B: int) -> int:
    Hz_u8 = np.ascontiguousarray(Hz_u8.astype(np.uint8))
    Hx_u8 = np.ascontiguousarray(Hx_u8.astype(np.uint8))
    synd_bin = np.ascontiguousarray(synd_bin.astype(np.uint8))
    labels_x = np.ascontiguousarray(labels_x.astype(np.uint8))
    labels_z = np.ascontiguousarray(labels_z.astype(np.uint8))

    z_first_then_x = (meta.get('syndrome_order','Z_first_then_X') == 'Z_first_then_X')
    # FULL-BATCH split parity
    nz = Hz_u8.shape[0]; nx = Hx_u8.shape[0]
    sZ = synd_bin[:, :nz] if z_first_then_x else synd_bin[:, nx:]
    sX = synd_bin[:, nz:] if z_first_then_x else synd_bin[:, :nx]
    sZ_hat = (Hz_u8 @ labels_x.T) % 2
    sX_hat = (Hx_u8 @ labels_z.T) % 2
    mism = int((sZ_hat != sZ.T).sum() + (sX_hat != sX.T).sum())
    if mism != 0:
        print(f"[ERROR] Strict split parity mismatches (full batch): {mism}")
        return mism

    # derive hard_labels once
    hard_labels = np.ascontiguousarray((labels_x ^ labels_z).astype(np.uint8))

    # checksums of inputs and labels
    with open(input_npz_path, 'rb') as f:
        hash_in = _sha256_bytes(f.read())
    hash_labels = _sha256_bytes(np.concatenate([labels_x, labels_z, hard_labels], axis=1).tobytes())

    # checksums of matrices
    Hz_hash = hashlib.sha256(Hz_u8.tobytes()).hexdigest()
    Hx_hash = hashlib.sha256(Hx_u8.tobytes()).hexdigest()

    # save
    np.savez_compressed(
        out_npz_path,
        labels_x=labels_x, labels_z=labels_z, hard_labels=(labels_x ^ labels_z),
        Hz=Hz_u8, Hx=Hx_u8,
        meta=json.dumps(meta),
        hash_in=hash_in,
        hash_labels=hash_labels,
        Hz_hash=Hz_hash, Hx_hash=Hx_hash
    )

    # round-trip reload and re-validate on the same synd_bin
    z = np.load(out_npz_path, allow_pickle=False)
    lx_rt = np.ascontiguousarray(z['labels_x'].astype(np.uint8))
    lz_rt = np.ascontiguousarray(z['labels_z'].astype(np.uint8))
    hl_rt = np.ascontiguousarray(z['hard_labels'].astype(np.uint8))
    try:
        meta_rt = json.loads(z['meta'].item())
    except Exception:
        meta_rt = meta
    hash_in_rt = str(z['hash_in'].item()) if 'hash_in' in z else None
    hash_labels_rt = str(z['hash_labels'].item()) if 'hash_labels' in z else None

    with open(input_npz_path, 'rb') as f:
        hash_in_local = _sha256_bytes(f.read())
    if hash_in_rt != hash_in_local:
        print(f"[ERROR] hash_in mismatch after save: {hash_in_rt} != {hash_in_local}")
        return 10
    if _sha256_bytes(np.concatenate([lx_rt, lz_rt, hl_rt], axis=1).tobytes()) != hash_labels:
        print(f"[ERROR] hash_labels mismatch after save/reload.")
        return 11

    sZ_hat_rt = (Hz_u8 @ lx_rt.T) % 2
    sX_hat_rt = (Hx_u8 @ lz_rt.T) % 2
    mism_rt = int((sZ_hat_rt != sZ.T).sum() + (sX_hat_rt != sX.T).sum())
    if mism_rt != 0:
        print(f"[ERROR] Round-trip split parity mismatches (full batch): {mism_rt}")
        return mism_rt

    print(f"✓ Strict split parity validation passed for {B} samples (0 mismatches)")
    return 0

# Add parent directory to path for cudaq_backend imports
script_dir = Path(__file__).parent.absolute()
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Attempt to import relay_bp first (required for relay teacher)
try:
    import relay_bp
except Exception as e:
    print(f"[WARN] relay_bp failed to import: {e}")
    relay_bp = None

# PanQEC for surface code fallback when building Hx/Hz
try:
    from panqec.codes import surface_2d
    PANQEC_AVAILABLE = True
except Exception:
    PANQEC_AVAILABLE = False

# Define rotated d=3 matrices directly (verified to match CUDA-Q ordering)


CUDAQ_AVAILABLE = True

# Optional PyMatching for MWPM teacher
try:
    import pymatching as pm
    PYMATCHING_AVAILABLE = True
except Exception:
    PYMATCHING_AVAILABLE = False

# Optional MWPF and Stim for HyperBlossom teacher
try:
    import stim
    STIM_AVAILABLE = True
except Exception:
    STIM_AVAILABLE = False

try:
    from mwpf import construct_decoder_and_predictor, MwpfCompiledDecoder
    MWPF_AVAILABLE = True
except Exception:
    MWPF_AVAILABLE = False





def lift_logicals_separate(x0: np.ndarray, z0: np.ndarray,
                           ell_x: np.ndarray, ell_z: np.ndarray,
                           Lx: np.ndarray, Lz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sector-separated lifting:
      labels_x = x0 ⊕ (ell_x * Lx)
      labels_z = z0 ⊕ (ell_z * Lz)
    Vectorized over batch.
    """
    B = int(max(len(ell_x), len(ell_z)))
    Lx = np.ascontiguousarray(Lx.astype(np.uint8))
    Lz = np.ascontiguousarray(Lz.astype(np.uint8))
    x0 = np.ascontiguousarray(x0.astype(np.uint8))
    z0 = np.ascontiguousarray(z0.astype(np.uint8))
    ell_x = np.ascontiguousarray(ell_x.astype(np.uint8))
    ell_z = np.ascontiguousarray(ell_z.astype(np.uint8))

    # Broadcast: (B,1) & (9,) -> (B,9)
    add_x = (ell_x[:, None] & Lx[None, :]).astype(np.uint8)
    add_z = (ell_z[:, None] & Lz[None, :]).astype(np.uint8)

    labels_x = (x0 ^ add_x).astype(np.uint8)
    labels_z = (z0 ^ add_z).astype(np.uint8)
    return labels_x, labels_z


def mwpf_decode_logicals(dem, packed_synd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build compiled decoder with num_obs=2 (X_L,Z_L) and decode to get logical bits.

    Args:
        dem: stim.DetectorErrorModel
        packed_synd: bit-packed syndromes (B, 1) for 8 detectors

    Returns:
        (ell_x, ell_z): logical X and Z bits (B,) each
    """
    
    # Create MWPF decoder with 2 observables
    config = {"cluster_node_limit": 50}
    decoder, predictor = construct_decoder_and_predictor(
        model=dem,
        decoder_type="SolverSerialJointSingleHair", 
        config=config
    )
    
    # Create compiled decoder for batch processing
    compiled_decoder = MwpfCompiledDecoder(
        solver=decoder,
        predictor=predictor,
        num_dets=8,  # 8 syndrome bits
        num_obs=2,   # 2 logical observables (X_L, Z_L)
        panic_action=2,  # CATCH panics
        panic_cases=[],
        benchmark_suite=None,
        trace_filename=None,
        benchmark_suite_filename=None,
        bp_decoder=None,
        bp_weight_mix_ratio=1.0,
        floor_weight=None
    )
    
    # Decode to get logical observables
    obs_flips_packed = compiled_decoder.decode_shots_bit_packed(
        bit_packed_detection_event_data=packed_synd
    )
    
    # Extract logical bits
    B = packed_synd.shape[0]
    ell_x = np.zeros(B, dtype=np.uint8)
    ell_z = np.zeros(B, dtype=np.uint8)
    
    for i in range(B):
        obs_byte = obs_flips_packed[i, 0] if obs_flips_packed.ndim > 1 else obs_flips_packed[i]
        ell_x[i] = (obs_byte >> 0) & 1  # Bit 0: X_L
        ell_z[i] = (obs_byte >> 1) & 1  # Bit 1: Z_L
    
    return ell_x, ell_z


def lift_logicals_to_data_bits(x0: np.ndarray, ell_x: np.ndarray, ell_z: np.ndarray, 
                              Lx: np.ndarray, Lz: np.ndarray) -> np.ndarray:
    """
    Lift logical bits to data-bit corrections: x = x0 ⊕ (ell_x*Lx) ⊕ (ell_z*Lz).
    
    Args:
        x0: base solution (9,)
        ell_x: logical X bits (B,)
        ell_z: logical Z bits (B,)
        Lx: logical X representative (9,)
        Lz: logical Z representative (9,)
    
    Returns:
        x_data: data-bit corrections (B, 9)
    """
    B = len(ell_x)
    x_data = np.zeros((B, 9), dtype=np.uint8)
    
    for i in range(B):
        # x = x0 ⊕ (ell_x[i] * Lx) ⊕ (ell_z[i] * Lz)
        x_data[i, :] = x0.copy()
        if ell_x[i]:
            x_data[i, :] = (x_data[i, :] + Lx) % 2
        if ell_z[i]:
            x_data[i, :] = (x_data[i, :] + Lz) % 2
    
    return x_data


def bit_unpack_packed_rows(packed: np.ndarray, n_bits: int) -> np.ndarray:
    """Vectorized little-endian per-byte unpack of shape [B, N_bytes] -> [B, n_bits] uint8."""
    if packed.dtype != np.uint8 or packed.ndim != 2:
        raise ValueError("Packed syndromes must be uint8 array of shape [B, N_bytes]")
    B, n_bytes = packed.shape
    if n_bytes * 8 < n_bits:
        raise ValueError(f"Packed buffer has only {n_bytes*8} bits, needs {n_bits}")
    # Expand bits little-endian per byte, LSB-first
    bit_idx = np.arange(8, dtype=np.uint8)  # 0..7
    bits = ((packed[:, :, None] >> bit_idx[None, None, :]) & 1).astype(np.uint8)  # [B, N_bytes, 8]
    bits = bits.reshape(B, n_bytes * 8)
    return bits[:, :n_bits]

# Self-test to ensure bit unpacking is correct (run once at import)
if __name__ != "__main__":
    _test_packed = np.array([[0b10110011, 0b11001100]], dtype=np.uint8)  # [1, 2]
    _test_unpacked = bit_unpack_packed_rows(_test_packed, 16)  # [1, 16]
    _expected = np.array([[1,1,0,0,1,1,0,1, 0,0,1,1,0,0,1,1]], dtype=np.uint8)
    assert np.array_equal(_test_unpacked, _expected), f"Bit unpack self-test failed: got {_test_unpacked}, expected {_expected}"


def build_H_from_hx_hz(Hx: np.ndarray, Hz: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    """Build combined parity-check matrix H from Astra Hx/Hz matrices.
    Returns H (float32) and sizes.
    """
    if Hx.ndim != 2 or Hz.ndim != 2:
        raise ValueError("Hx and Hz must be 2D arrays")
    if Hx.shape[1] != Hz.shape[1]:
        raise ValueError(f"Hx and Hz must have same number of columns; got {Hx.shape} vs {Hz.shape}")
    num_bits = int(Hx.shape[1])
    H = np.vstack([Hz, Hx]).astype(np.float32)  # detectors: first Z, then X (convention)
    sizes = {
        "num_x_checks": int(Hx.shape[0]),
        "num_z_checks": int(Hz.shape[0]),
        "num_checks": int(H.shape[0]),
        "num_bits": num_bits,
        "num_edges": int(H.sum()),
    }
    return H, sizes


def build_H_surface(distance: int, surface_layout: str = "planar") -> Tuple[np.ndarray, Dict[str, int], Dict[str, Any]]:
    """Build H for surface code using PanQEC or rotated layout with metadata.
    Returns (H, sizes, meta)
    """
    if surface_layout == "rotated":
        if not CUDAQ_AVAILABLE:
            raise RuntimeError("CUDA-Q backend not available for rotated layout; pass --hx/--hz or install cudaq_backend")
        Hx, Hz, meta = build_H_rotated_d3_from_cfg(None)
        H, sizes = build_H_from_hx_hz(Hx, Hz)
        nz = sizes['num_z_checks']
        assert (H[:nz,:].sum() == Hz.sum()) and (H[nz:,:].sum() == Hx.sum()), "H stacking order bug"
        sizes.update(meta)
        return H, sizes, meta
    else:
        if not PANQEC_AVAILABLE:
            raise RuntimeError("PanQEC not available for planar layout; pass --hx/--hz or install panqec")
        code = surface_2d.Planar2DCode(distance)
        Hx = code.Hx.toarray().astype(np.int32)
        Hz = code.Hz.toarray().astype(np.int32)
        H, sizes = build_H_from_hx_hz(Hx, Hz)
        meta = {
            'surface_layout': 'planar',
            'distance': distance,
            'N_syn': sizes['num_checks'],
            'N_bits': sizes['num_bits'],
            'syndrome_order': 'Z_first_then_X'  # Default PanQEC convention
        }
        return H, sizes, meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Teacher labeling script for QEC")
    parser.add_argument("--code", choices=["surface", "bb"], required=True, help="Code type")
    parser.add_argument("--hx", type=str, help="Path to Hx (.npy)")
    parser.add_argument("--hz", type=str, help="Path to Hz (.npy)")
    parser.add_argument("--input-syndromes", type=str, required=True, help="Input syndromes .npz (key 'syndromes')")
    parser.add_argument("--packed", action="store_true", help="Input syndromes are bit-packed uint8")
    parser.add_argument("--num-sets", type=int, default=80)
    parser.add_argument("--set-max-iter", type=int, default=150)
    parser.add_argument("--gamma0", type=float, default=0.1)
    parser.add_argument("--gamma-dist-interval", nargs=2, type=float, default=[-0.24, 0.66])
    parser.add_argument("--timeout-ms", type=int, default=50)
    parser.add_argument("--out", type=str, required=True, help="Output .npz path")
    parser.add_argument("--distance", type=int, default=3, help="Distance (surface only)")
    parser.add_argument("--teacher", choices=["relay", "mwpm", "mwpf", "ensemble"], default="relay", 
                                                 help="Teacher type: relay (Relay-BP), mwpm (MWPM), mwpf (HyperBlossom), or ensemble (Relay+MWPF) (default: relay)")
    parser.add_argument("--surface-layout", choices=["rotated", "planar"], default="rotated",
                        help="Surface layout type (default: rotated)")

    args = parser.parse_args()

    # 1) Build detector graph / parity-check matrix H
    if args.hx and args.hz:
        print(f"Loading Hx from {args.hx}")
        Hx = np.load(args.hx)
        print(f"Loading Hz from {args.hz}")
        Hz = np.load(args.hz)
        H, sizes = build_H_from_hx_hz(Hx, Hz)
        build_source = "hx_hz"
        meta = {
            'surface_layout': 'custom',
            'N_syn': sizes['num_checks'],
            'N_bits': sizes['num_bits'],
            'syndrome_order': 'Z_first_then_X'  # Assume standard convention
        }
    else:
        if args.code != "surface":
            print("[ERROR] For non-surface codes, please supply --hx and --hz matrices.")
            return 2
        print(f"Building {args.surface_layout} surface code H for distance d={args.distance}")
        try:
            H, sizes, meta = build_H_surface(args.distance, args.surface_layout)
            build_source = f"surface_{args.surface_layout}"
        except Exception as e:
            print(f"[ERROR] Failed to build {args.surface_layout} surface code H: {e}")
            print("Provide --hx/--hz or ensure panqec is installed")
            return 2

    # Validate expected sizes
    if args.surface_layout == "rotated":
        expected_bits, expected_checks = 9, 8
    else:
        expected_bits, expected_checks = 13, 12
    if sizes["num_bits"] != expected_bits or sizes["num_checks"] != expected_checks:
        print(f"[ERROR] Size mismatch: bits={sizes['num_bits']} (exp {expected_bits}), checks={sizes['num_checks']} (exp {expected_checks})")
        return 2
    print(f"Graph sizes: checks={sizes['num_checks']} (X={sizes['num_x_checks']}, Z={sizes['num_z_checks']}), bits={sizes['num_bits']}, edges≈{sizes['num_edges']}")

    # Extract Hz, Hx according to frozen ordering
    nz = int(sizes["num_z_checks"])
    nx = int(sizes["num_x_checks"])
    Hz = H[:nz, :].astype(np.uint8)  # Z checks (detect X errors)
    Hx = H[nz:, :].astype(np.uint8)  # X checks (detect Z errors)

    # 2) Load syndromes
    data = np.load(args.input_syndromes)
    if "syndromes" not in data:
        print("[ERROR] Input NPZ must contain key 'syndromes'")
        return 3
    synd = data["syndromes"]

    if args.packed:
        expected_bytes = (sizes["num_checks"] + 7) // 8
        if synd.ndim != 2 or synd.shape[1] != expected_bytes or synd.dtype != np.uint8:
            print(f"[ERROR] Packed syndromes must be [B,{expected_bytes}] uint8; got {synd.shape} {synd.dtype}")
            return 3
        synd = bit_unpack_packed_rows(synd, sizes["num_checks"])  # [B, N_syn]
    else:
        if synd.ndim != 2 or synd.shape[1] != sizes["num_checks"]:
            print(f"[ERROR] Unpacked syndromes must be [B,{sizes['num_checks']}] but got {synd.shape}")
            return 3
        if synd.dtype != np.uint8:
            synd = synd.astype(np.uint8, copy=False)

    # Force canonical binary and contiguity; NO subsampling
    synd_bin = np.ascontiguousarray((synd != 0).astype(np.uint8))
    B = int(synd_bin.shape[0])

    if B <= 0:
        print("[ERROR] Empty batch (B=0)")
        return 3

    # Split syndromes (assume 'Z_first_then_X')
    sZ = synd_bin[:, :nz].astype(np.uint8)
    sX = synd_bin[:, nz:].astype(np.uint8)

    # 3) Configure decoders based on --teacher flag
    time_budget = max(0, int(args.timeout_ms)) / 1000.0  # seconds
    
    # Initialize sector-wise labels for all teachers
    labels_x = np.zeros((B, sizes["num_bits"]), dtype=np.uint8)
    labels_z = np.zeros((B, sizes["num_bits"]), dtype=np.uint8)
    
    if args.teacher == "relay":
        if relay_bp is None:
            print("[ERROR] relay_bp not available for Relay-BP teacher.")
            return 4
            
        print(f"Running Relay-BP decoder...")
        # Use optimal error priors for surface codes (very low = sparse errors expected)
        error_priors = np.full(sizes["num_bits"], 0.001, dtype=np.float32)  # Very low error priors
        gamma_interval = (float(args.gamma_dist_interval[0]), float(args.gamma_dist_interval[1]))

        # Try strict dtypes first; fall back to float64 if required
        last_err = None
        decoder = None
        for dt in (np.float32, np.float64):
            try:
                H_c = np.ascontiguousarray(H, dtype=dt)
                error_priors_c = np.ascontiguousarray(error_priors, dtype=dt)
                decoder = relay_bp.RelayDecoderF32(
                    H_c,
                    error_priors_c,
                    gamma0=float(args.gamma0),
                    num_sets=int(args.num_sets),
                    set_max_iter=int(args.set_max_iter),
                    gamma_dist_interval=gamma_interval,
                )
                break
            except Exception as e:
                last_err = e
                decoder = None
        if decoder is None:
            print(f"[ERROR] Failed to construct RelayDecoderF32: {last_err}")
            return 4

        # Decode
        shot_times = np.zeros(B, dtype=np.float32)
        
        # For relay, ensure sector labels exist via particular solutions
        for i in range(B):
            try:
                xX0 = _gf2_solve_particular(Hz, sZ[i])
                xZ0 = _gf2_solve_particular(Hx, sX[i])
                labels_x[i] = xX0
                labels_z[i] = xZ0
            except ValueError as e:
                print(f"[WARN] Inconsistent system for shot {i}: {e}")
                labels_x[i] = np.zeros(9, dtype=np.uint8)
                labels_z[i] = np.zeros(9, dtype=np.uint8)

        t0 = time.perf_counter()
        det_batch = np.ascontiguousarray(synd_bin, dtype=np.uint8)
        used_batch_decode = False
        try:
            res_batch = decoder.decode_batch(det_batch)
            if isinstance(res_batch, np.ndarray):
                if res_batch.shape != (B, sizes["num_bits"]):
                    raise RuntimeError(f"decode_batch returned wrong shape {res_batch.shape}")
                shot_times[:] = (time.perf_counter() - t0) / max(1, B)
            else:
                t_start = time.perf_counter()
                for i, r in enumerate(res_batch):
                    corr = np.asarray(r.correction, dtype=np.uint8)
                    if corr.shape[0] != sizes["num_bits"]:
                        raise RuntimeError("decode_batch element has mismatched length")
                    labels_x[i] = corr # Assuming Relay directly gives X error for Z-checks
                    labels_z[i] = corr # Assuming Relay directly gives Z error for X-checks
                t_end = time.perf_counter()
                shot_times[:] = (t_end - t_start) / max(1, B)
            used_batch_decode = True
        except Exception as e:
            print(f"[INFO] decode_batch not used ({e}); falling back to per-shot decode.")

        if not used_batch_decode:
            t_loop0 = time.perf_counter()
            for i in range(B):
                t_shot0 = time.perf_counter()
                try:
                    detectors_c = np.ascontiguousarray(det_batch[i], dtype=np.uint8)
                    res = decoder.decode(detectors_c)
                    corr = np.asarray(res.correction, dtype=np.uint8)
                    if corr.shape[0] != sizes["num_bits"]:
                        raise RuntimeError(f"Decoder returned wrong length {corr.shape[0]} != {sizes['num_bits']}")
                    hard_labels[i] = corr
                except Exception as e:
                    print(f"[WARN] Relay-BP decode failed at i={i}: {e}. Filling zeros.")
                    hard_labels[i].fill(0)
                shot_times[i] = time.perf_counter() - t_shot0
                if time_budget > 0 and shot_times[i] > time_budget:
                    print(f"[WARN] Shot {i} exceeded timeout {time_budget*1000:.1f} ms ({shot_times[i]*1000:.1f} ms)")
            total_time = time.perf_counter() - t_loop0
        else:
            total_time = time.perf_counter() - t0
            
    elif args.teacher == "mwpm":
        if not PYMATCHING_AVAILABLE:
            print("[ERROR] PyMatching not available for MWPM teacher.")
            return 4
            
        print(f"Running MWPM decoder using PyMatching on individual sectors...")
        try:
            # Use PyMatching on Hx (which has low degree) and simple decoder for Hz
            syndrome_order = meta.get('syndrome_order', 'Z_first_then_X')
            
            if syndrome_order == 'X_first_then_Z':
                # X checks first (0-3), Z checks (4-7)
                nx = sizes["num_x_checks"]
                nz = sizes["num_z_checks"]
                Hx = H[:nx, :].astype(np.uint8)
                Hz = H[nx:, :].astype(np.uint8)
            else:
                # Z checks first (0-3), X checks (4-7) 
                nz = sizes["num_z_checks"]
                nx = sizes["num_x_checks"]
                Hz = H[:nz, :].astype(np.uint8)
                Hx = H[nz:, :].astype(np.uint8)
            
            # Check if we can use PyMatching for Hx (should work for rotated codes)
            hx_max_weight = Hx.sum(axis=0).max()
            hz_max_weight = Hz.sum(axis=0).max()
            
            print(f"Matrix column weights: Hx max={hx_max_weight}, Hz max={hz_max_weight}")
            
            # Use PyMatching for Hx if possible
            if hx_max_weight <= 2:
                matcher_x = pm.Matching.from_check_matrix(Hx)
                use_pm_x = True
                print("✓ Using PyMatching for X-error correction")
            else:
                use_pm_x = False
                print("⚠ Using simple decoder for X-error correction (high degree)")
            
            # Use PyMatching for Hz if possible  
            if hz_max_weight <= 2:
                matcher_z = pm.Matching.from_check_matrix(Hz)
                use_pm_z = True
                print("✓ Using PyMatching for Z-error correction")
            else:
                use_pm_z = False
                print("⚠ Using simple decoder for Z-error correction (high degree)")
            
            hard_labels = np.zeros((B, sizes["num_bits"]), dtype=np.uint8)
            shot_times = np.zeros(B, dtype=np.float32)
            t_mwpm0 = time.perf_counter()
            
            for i in range(B):
                t_shot0 = time.perf_counter()
                
                if syndrome_order == 'X_first_then_Z':
                    sx = synd_bin[i, :nx].astype(np.uint8)
                    sz = synd_bin[i, nx:].astype(np.uint8)
                else:
                    sz = synd_bin[i, :nz].astype(np.uint8)
                    sx = synd_bin[i, nz:].astype(np.uint8)
                
                # Decode X errors (detected by Z checks)
                if use_pm_z and np.any(sz):
                    try:
                        correction_x = matcher_z.decode(sz)
                        error_x = np.asarray(correction_x, dtype=np.uint8)[:sizes["num_bits"]]
                    except:
                        error_x = np.zeros(sizes["num_bits"], dtype=np.uint8)
                else:
                    # Simple decoder for Z checks -> X errors
                    error_x = np.zeros(sizes["num_bits"], dtype=np.uint8)
                    for check_idx in range(nz):
                        if sz[check_idx]:
                            qubits_in_check = np.where(Hz[check_idx, :] > 0)[0]
                            if len(qubits_in_check) > 0:
                                error_x[qubits_in_check[0]] = 1
                
                # Decode Z errors (detected by X checks)
                if use_pm_x and np.any(sx):
                    try:
                        correction_z = matcher_x.decode(sx)
                        error_z = np.asarray(correction_z, dtype=np.uint8)[:sizes["num_bits"]]
                    except:
                        error_z = np.zeros(sizes["num_bits"], dtype=np.uint8)
                else:
                    # Simple decoder for X checks -> Z errors
                    error_z = np.zeros(sizes["num_bits"], dtype=np.uint8)
                    for check_idx in range(nx):
                        if sx[check_idx]:
                            qubits_in_check = np.where(Hx[check_idx, :] > 0)[0]
                            if len(qubits_in_check) > 0:
                                error_z[qubits_in_check[0]] = 1
                
                # Store sector-wise labels
                labels_x[i] = error_x
                labels_z[i] = error_z
                
                # Combine errors (X XOR Z for data qubits) for backward compatibility
                hard_labels[i] = (error_x ^ error_z)
                
                shot_times[i] = time.perf_counter() - t_shot0
                if time_budget > 0 and shot_times[i] > time_budget:
                    print(f"[WARN] Shot {i} exceeded timeout {time_budget*1000:.1f} ms ({shot_times[i]*1000:.1f} ms)")
            
            total_time = time.perf_counter() - t_mwpm0
            
        except Exception as e:
            print(f"[ERROR] MWPM setup failed: {e}")
            return 4
            
    elif args.teacher == "mwpf":
        if not MWPF_AVAILABLE:
            print("[ERROR] MWPF/Stim not available for HyperBlossom teacher.")
            return 4
            
        print(f"Running MWPF (HyperBlossom) decoder...")
        try:
            # For rotated d=3, use the new logical-first approach
            if args.surface_layout == "rotated" and args.distance == 3:
                # Import required functions
                try:
                    from cudaq_backend.circuits import logical_reps_rotated_d3, build_stim_circuit_rotated_d3_from_cfg
                except ImportError as e:
                    print(f"[ERROR] Cannot import required circuit functions: {e}")
                    return 4
                
                # Get H matrices and logical representatives
                Hx, Hz, meta = build_H_rotated_d3_from_cfg(None)
                logical_reps = logical_reps_rotated_d3()
                Lx, Lz = logical_reps['Lx'], logical_reps['Lz']
                
                # Build DEM manually for rotated d=3
                dem_lines = []
                
                # Add detector definitions
                for det_idx in range(8):
                    dem_lines.append(f"detector D{det_idx}")
                
                # Add logical observable definitions
                dem_lines.append("logical_observable L0")  # X_L
                dem_lines.append("logical_observable L1")  # Z_L
                
                # Add error instructions for each data qubit
                for data_qubit in range(9):
                    # X error on this data qubit affects Z checks
                    z_checks = np.where(Hz[:, data_qubit] > 0)[0]
                    if len(z_checks) > 0:
                        detectors = " ".join([f"D{check}" for check in z_checks])
                        dem_lines.append(f"error(0.001) {detectors}")
                    
                    # Z error on this data qubit affects X checks (offset by 4)
                    x_checks = np.where(Hx[:, data_qubit] > 0)[0] + 4
                    if len(x_checks) > 0:
                        detectors = " ".join([f"D{check}" for check in x_checks])
                        dem_lines.append(f"error(0.001) {detectors}")
                
                # Add logical error instructions
                # Logical X error affects logical observable L0
                dem_lines.append("error(0.001) L0")
                # Logical Z error affects logical observable L1
                dem_lines.append("error(0.001) L1")
                
                # Create DEM from string
                dem_string = "\n".join(dem_lines)
                dem = stim.DetectorErrorModel(dem_string)
                
                # Convert syndromes to bit-packed format (Z checks first, then X checks)
                packed_shots = np.zeros((B, 1), dtype=np.uint8)
                for i in range(B):
                    val = 0
                    for det_idx in range(8):
                        if synd_bin[i, det_idx]:
                            val |= (1 << det_idx)
                    packed_shots[i, 0] = val
                
                # MWPF teacher: sector-wise decoding with logical lifting
                print(f"Running MWPF (HyperBlossom) decoder...")

                t_mwpf0 = time.perf_counter()

                # Get logical representatives
                logical_reps = logical_reps_rotated_d3()
                Lx, Lz = logical_reps['Lx'], logical_reps['Lz']

                # Compute particular solutions for each sector (batch)
                x0_batch = np.zeros((B, sizes["num_bits"]), dtype=np.uint8)
                z0_batch = np.zeros((B, sizes["num_bits"]), dtype=np.uint8)

                for i in range(B):
                    x0_batch[i] = _gf2_solve_particular(Hz, sZ[i])  # Hx * x = sZ
                    z0_batch[i] = _gf2_solve_particular(Hx, sX[i])  # Hz * z = sX

                # Get MWPF logicals
                ell_x, ell_z = mwpf_decode_logicals(dem, packed_shots)

                # Use separated lifting
                labels_x, labels_z = lift_logicals_separate(x0_batch, z0_batch, ell_x, ell_z, Lx, Lz)

                total_time = time.perf_counter() - t_mwpf0
                shot_times = np.full(B, total_time / B, dtype=np.float32)

                print(f"MWPF completed: {B} shots in {total_time:.3f}s ({B/total_time:.0f} shots/s)")
                
            else:
                # Fallback to old implementation for non-rotated or non-d=3
                print("[WARN] Using fallback MWPF implementation for non-rotated d=3")
                # ... (keep old implementation for other cases)
                return 4  # For now, only support rotated d=3
                
        except Exception as e:
            print(f"[ERROR] MWPF setup failed: {e}")
            return 4
            
    elif args.teacher == "ensemble":
        # --- Ensemble = sector particular + logical lift from MWPF ---
        if not MWPF_AVAILABLE:
            print("[ERROR] mwpf not available; install mwpf[stim].")
            return 4

        print(f"Running ensemble teacher (sector particular + MWPF logical lift)...")

        # 1) Compute sector particular solutions (parity-exact)
        x0_batch = np.zeros((B, sizes["num_bits"]), dtype=np.uint8)
        z0_batch = np.zeros((B, sizes["num_bits"]), dtype=np.uint8)
        for i in range(B):
            x0_batch[i] = _gf2_solve_particular(Hz, sZ[i])  # Hx * x = sZ
            z0_batch[i] = _gf2_solve_particular(Hx, sX[i])  # Hz * z = sX

        # 2) Get MWPF logicals and logical representatives
        packed = (synd_bin.reshape(B, 8).astype(np.uint8) * (1 << np.arange(8, dtype=np.uint8))).sum(axis=1).astype(np.uint8)
        packed = packed.reshape(B, 1)

        # Build DEM using string-based approach that works
        dem_lines = []
        for det_idx in range(8):
            dem_lines.append(f"detector D{det_idx}")
        for q in range(sizes["num_bits"]):
            z_checks = np.where(Hz[:, q] > 0)[0]
            x_checks = np.where(Hx[:, q] > 0)[0] + nz
            if len(z_checks) > 0:
                detectors = " ".join([f"D{check}" for check in z_checks])
                dem_lines.append(f"error(0.001) {detectors}")
            if len(x_checks) > 0:
                detectors = " ".join([f"D{check}" for check in x_checks])
                dem_lines.append(f"error(0.001) {detectors}")
        dem_string = "\n".join(dem_lines)
        dem = stim.DetectorErrorModel(dem_string)

        ell_x, ell_z = mwpf_decode_logicals(dem, packed)

        from cudaq_backend.circuits import logical_reps_rotated_d3
        L = logical_reps_rotated_d3()
        Lx = L["Lx"].astype(np.uint8)
        Lz = L["Lz"].astype(np.uint8)

        # 3) Build candidates
        cand_x = []
        cand_z = []

        # C0: pure particular (guaranteed parity if syndromes are valid)
        cand_x.append(x0_batch)
        cand_z.append(z0_batch)

        # Cmwpf: lifted with MWPF logicals
        lx_mwpf, lz_mwpf = lift_logicals_separate(x0_batch, z0_batch, ell_x, ell_z, Lx, Lz)
        cand_x.append(lx_mwpf)
        cand_z.append(lz_mwpf)

        # 4) Evaluate parity and weight; pick first parity-valid with min weight, else fallback to C0
        labels_x = np.empty_like(x0_batch)
        labels_z = np.empty_like(z0_batch)
        for i in range(B):
            best_idx = None
            best_w = 1e9
            for k in range(len(cand_x)):
                xk = cand_x[k][i]
                zk = cand_z[k][i]
                okZ = ((Hz @ xk) % 2 == sZ[i]).all()
                okX = ((Hx @ zk) % 2 == sX[i]).all()
                if okZ and okX:
                    w = int(xk.sum() + zk.sum())
                    if w < best_w:
                        best_w = w
                        best_idx = k
            if best_idx is None:
                # Fallback to particular solution (parity-safe by construction)
                labels_x[i] = x0_batch[i]
                labels_z[i] = z0_batch[i]
            else:
                labels_x[i] = cand_x[best_idx][i]
                labels_z[i] = cand_z[best_idx][i]

        # 5) Single strict save & round-trip validate
        Hz_u8 = np.ascontiguousarray(Hz.astype(np.uint8))
        Hx_u8 = np.ascontiguousarray(Hx.astype(np.uint8))
        rc = _strict_save_and_roundtrip_validate(
            Hz_u8=Hz_u8, Hx_u8=Hx_u8,
            synd_bin=synd_bin,
            labels_x=labels_x, labels_z=labels_z,
            meta=meta,
            input_npz_path=args.input_syndromes,
            out_npz_path=args.out,
            B=B
        )
        return 0 if rc == 0 else 6
    else:
        print(f"[ERROR] Unknown teacher: {args.teacher}")
        return 4

    # ===== SINGLE exit path for ALL teachers: full-batch strict split parity + checksummed save + round-trip =====
    Hz_u8 = np.ascontiguousarray(Hz.astype(np.uint8))
    Hx_u8 = np.ascontiguousarray(Hx.astype(np.uint8))

    rc = _strict_save_and_roundtrip_validate(
        Hz_u8=Hz_u8, Hx_u8=Hx_u8,
        synd_bin=synd_bin,
        labels_x=labels_x, labels_z=labels_z,
        meta=meta,
        input_npz_path=args.input_syndromes,
        out_npz_path=args.out,
        B=B
    )
    return 0 if rc == 0 else 6


if __name__ == "__main__":
    sys.exit(main())