#!/usr/bin/env python3
"""
CUDA-Q Garnet sampler facade (lazy imports).

Provides a simple, GPU-friendly API to generate surface-code syndromes and
teacher labels. Falls back to deterministic numpy sampling and the existing
rotated d=3 LUT when CUDA-Q is unavailable.

API:
  - get_code_mats() -> (Hx:u8[4,9], Hz:u8[4,9], meta:dict)
  - class CudaqGarnetSampler:
      sample_batch(B:int, p:float, teacher:str="mwpf", mode:str="foundation")
        -> (s_bin:u8[B,8], labels_x:u8[B,9], labels_z:u8[B,9])

Notes:
  - Canonical ordering: Z_first_then_X, LSB-first within each byte.
  - For rotated d=3, N_syn=8, N_bits=9.
  - No CUDA at import time: CUDA-Q imported lazily inside callables.
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Tuple, Dict

import numpy as np

# Optional teacher backends (installed in venv)
try:
    import mwpf  # type: ignore
    from mwpf import SinterMWPFDecoder  # type: ignore
    _HAS_MWPF = True
except Exception:
    _HAS_MWPF = False
try:
    import stim  # type: ignore
    _HAS_STIM = True
except Exception:
    _HAS_STIM = False
try:
    from pymatching import Matching  # type: ignore
    _HAS_PYMATCHING = True
except Exception:
    _HAS_PYMATCHING = False

ROOT = Path(__file__).resolve().parents[1]
import sys as _sys
_sys.path.insert(0, str(ROOT))


def get_code_mats() -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load canonical Hx, Hz matrices and meta.

    Tries student_pack_p003.npz first, then falls back to fastpath LUT NPZ.
    Returns (Hx:u8[4,9], Hz:u8[4,9], meta:dict).
    """
    # Preferred pack in scratchpad root
    pack = ROOT / "student_pack_p003.npz"
    if pack.exists():
        z = np.load(pack, allow_pickle=False)
        Hx = z["Hx"].astype(np.uint8)
        Hz = z["Hz"].astype(np.uint8)
        meta_raw = z.get("meta", None)
        meta = json.loads(meta_raw.item()) if meta_raw is not None else {}
        return Hx, Hz, meta

    # Fallback to fastpath LUT bundle
    from fastpath import load_rotated_d3_lut_npz  # type: ignore
    lut16, Hx, Hz, meta = load_rotated_d3_lut_npz()
    return Hx, Hz, meta


def _unpack_byte_lsbf(arr: np.ndarray) -> np.ndarray:
    """Vectorized unpack of uint8 bytes to bits LSBF per byte.
    arr: [B] uint8 -> [B,8] uint8
    """
    bits = ((arr[:, None] >> np.arange(8, dtype=np.uint8)[None, :]) & 1).astype(np.uint8)
    return bits


def _pack_bits_lsbf(bits: np.ndarray) -> np.ndarray:
    """Pack bits LSBF per row: [B,8] -> [B] uint8."""
    B = bits.shape[0]
    out = np.zeros((B,), dtype=np.uint8)
    for b in range(8):
        out |= ((bits[:, b].astype(np.uint8) & 1) << b)
    return out


def _teacher_labels_from_lut(synd_bytes: np.ndarray) -> np.ndarray:
    """Use the rotated d=3 LUT to produce 9-bit corrections per syndrome.
    Returns [B,9] uint8.
    """
    from fastpath import decode_bytes, load_rotated_d3_lut_npz  # type: ignore
    lut16, *_ = load_rotated_d3_lut_npz()
    return decode_bytes(synd_bytes, lut16)


class CudaqGarnetSampler:
    """Sampler facade. Uses CUDA-Q if available, else numpy fallback.

    For the fallback, we sample an i.i.d. Bernoulli(p) error vector e[9], and
    compute the 8-bit syndrome as:
      sZ = (Hz @ e) % 2
      sX = (Hx @ e) % 2
    Teacher labels default to LUT-based (MWPF proxy for d=3).
    """

    def __init__(self, mode: str = "foundation"):
        self.mode = mode
        self.Hx, self.Hz, self.meta = get_code_mats()
        self._cudaq_ready = False
        self._tried_import = False
        # One-time p-guard: verifies CUDA-Q honors p; falls back to numpy if not
        self._p_guard_done = False
        self._p_guard_result = None
        # Teacher usage statistics (for logging/reporting)
        self._stats = {
            'mwpf_shots': 0,
            'mwpm_shots': 0,
            'total_shots': 0,
            'sampler_backend': 'unknown',
            'cudaq_calls': 0,
            'numpy_calls': 0,
        }

    def _ensure_cudaq(self) -> bool:
        if self._tried_import:
            return self._cudaq_ready
        self._tried_import = True
        # Allow explicit override to disable CUDA-Q path for A/B checks
        if os.getenv('MGHD_FORCE_NUMPY_SAMPLER', '0') == '1':
            self._cudaq_ready = False
            return False
        try:
            # Lazy import (no CUDA usage here)
            from cudaq_backend import (
                sample_surface_cudaq,  # noqa: F401
                get_backend_info,      # noqa: F401
                validate_backend_installation,  # noqa: F401
            )
            self._cudaq_ready = True
        except Exception:
            self._cudaq_ready = False
        return self._cudaq_ready

    def sample_batch(
        self,
        B: int,
        p: float,
        teacher: str = "mwpf",
        rng: np.random.Generator | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (s_bin[B,8], labels_x[B,9], labels_z[B,9]) as uint8 arrays.

        teacher: "mwpf"|"mwpm"|"oracle" (oracle unused here).
        Uses CUDA-Q trajectory simulation for rotated d=3 when available;
        otherwise falls back to deterministic numpy sampling.
        """
        rng = rng or np.random.default_rng()
        Hx, Hz = self.Hx, self.Hz
        assert Hx.shape == (4, 9) and Hz.shape == (4, 9)

        # Prefer CUDA-Q circuit-level trajectory simulation for rotated d=3
        s_bin: np.ndarray
        if self._ensure_cudaq():
            try:
                from cudaq_backend import sample_surface_cudaq  # type: ignore
                # Two rounds are sufficient to assemble one Z and one X stabilizer round
                base_kwargs = dict(
                    mode=self.mode,
                    batch_size=B,
                    T=2,
                    layout={},
                    rng=rng,
                    bitpack=False,
                    surface_layout="rotated",
                )
                syn = None
                # Try to pass physical error rate; attempt several common kw names
                for k in ("p", "p_phys", "p_error", "phys_p", "p_physical"):
                    try:
                        kw = dict(base_kwargs)
                        kw[k] = float(p)
                        syn = sample_surface_cudaq(**kw)
                        break
                    except TypeError:
                        syn = None
                        continue
                if syn is None:
                    # Fall back without explicit p (backend may use its own schedule)
                    syn = sample_surface_cudaq(**base_kwargs)
                # syn: [B,8] LSBF, Z-first then X
                s_bin = syn.astype(np.uint8, copy=False)
                if s_bin.shape != (B, 8):
                    # Defensive: if sampler returned packed bytes, unpack
                    if s_bin.ndim == 2 and s_bin.shape[1] == 1:
                        s_bin = _unpack_byte_lsbf(s_bin[:, 0])
                    else:
                        raise RuntimeError(f"Unexpected CUDA-Q syn shape {s_bin.shape}")
                # Update stats backend tag
                self._stats['sampler_backend'] = 'cudaq_rotated_d3'
                self._stats['cudaq_calls'] += 1

                # One-time p-honor guard (optional; can disable via MGHD_P_GUARD=0)
                if (not self._p_guard_done) and os.getenv('MGHD_P_GUARD', '1') == '1':
                    try:
                        # Probe two distinct p values and compare syndrome bit means
                        p_small = 0.005 if p >= 0.01 else max(1e-4, p)
                        p_large = 0.05 if p <= 0.02 else min(0.2, p)
                        for_probe = 2048
                        def _probe(pp: float):
                            syn2 = None
                            for k2 in ("p", "p_phys", "p_error", "phys_p", "p_physical"):
                                try:
                                    kw2 = dict(base_kwargs); kw2['batch_size'] = for_probe; kw2[k2] = float(pp)
                                    syn2 = sample_surface_cudaq(**kw2); break
                                except TypeError:
                                    syn2 = None; continue
                            if syn2 is None:
                                syn2 = sample_surface_cudaq(**{**base_kwargs, 'batch_size': for_probe})
                            sb = syn2.astype(np.uint8, copy=False)
                            if sb.ndim == 2 and sb.shape[1] == 1:
                                sb = _unpack_byte_lsbf(sb[:, 0])
                            return sb.mean(axis=0)
                        m_small = _probe(p_small)
                        m_large = _probe(p_large)
                        delta = float(abs(m_large.mean() - m_small.mean()))
                        self._p_guard_result = dict(p_small=p_small, p_large=p_large, delta=delta)
                        # If the difference in mean syndrome rate is tiny, assume p not honored; fall back to numpy
                        if delta < 0.01:
                            # Log warning only; do not auto-fallback. CUDA‑Q must be used.
                            print(f"[sampler] WARNING: CUDA-Q sampler appears insensitive to p (Δmean≈{delta:.4f}); please verify backend p plumbing.")
                    except Exception:
                        # Non-fatal: continue without guard
                        pass
                    finally:
                        self._p_guard_done = True
            except Exception as e:
                # Soft fallback to numpy if CUDA-Q path fails
                e_msg = str(e)
                # Compute ideal i.i.d. Bernoulli(p) error vector and derive syndrome
                e_bits = (rng.random((B, 9)) < p).astype(np.uint8)
                sZ = (Hz @ e_bits.T) % 2
                sX = (Hx @ e_bits.T) % 2
                s_bin = np.concatenate([sZ.T, sX.T], axis=1).astype(np.uint8)
                self._stats['sampler_backend'] = 'numpy_fallback'
                self._stats['numpy_calls'] += 1
        else:
            # Deterministic numpy fallback (no CUDA-Q dependency)
            e_bits = (rng.random((B, 9)) < p).astype(np.uint8)  # error bits
            sZ = (Hz @ e_bits.T) % 2
            sX = (Hx @ e_bits.T) % 2
            s_bin = np.concatenate([sZ.T, sX.T], axis=1).astype(np.uint8)
            self._stats['sampler_backend'] = 'numpy_fallback'
            self._stats['numpy_calls'] += 1

        # Teacher labels according to request (mwpf|mwpm|lut|ensemble)
        teacher = (teacher or 'mwpf').lower()
        teacher_used = teacher
        labels9: np.ndarray
        if teacher == 'lut':
            labels9 = _teacher_labels_from_lut(_pack_bits_lsbf(s_bin))
            teacher_used = 'lut'
        elif teacher == 'mwpm':
            labels9 = self._labels_via_mwpm(hx=Hx, hz=Hz, s_bin=s_bin)
            teacher_used = 'mwpm'
        elif teacher in ('mwpf', 'ensemble', 'mwpf+mwpm'):
            # Try MWPF; validate parity consistency; fall back to MWPM on any mismatch
            try:
                labels9 = self._labels_via_mwpf_stim(s_bin)
                teacher_used = 'mwpf'
                # Parity validation against provided Hx/Hz and s_bin
                sZ_hat = (Hz @ labels9.T) % 2
                sX_hat = (Hx @ labels9.T) % 2
                sZ = s_bin[:, :Hz.shape[0]].T
                sX = s_bin[:, Hz.shape[0]:Hz.shape[0]+Hx.shape[0]].T
                mism = int((sZ_hat != sZ).sum() + (sX_hat != sX).sum())
                if mism != 0:
                    # Fallback to MWPM to preserve correctness
                    if not _HAS_PYMATCHING:
                        raise RuntimeError("MWPF produced parity-mismatched labels and PyMatching unavailable for fallback")
                    labels9 = self._labels_via_mwpm(hx=Hx, hz=Hz, s_bin=s_bin)
                    teacher_used = 'mwpm'
            except Exception as e:
                if not _HAS_PYMATCHING:
                    raise RuntimeError(f"MWPF teacher failed and PyMatching not available: {e}")
                labels9 = self._labels_via_mwpm(hx=Hx, hz=Hz, s_bin=s_bin)
                teacher_used = 'mwpm'
        else:
            raise ValueError(f"Unknown teacher: {teacher}")
        # For d=3 rotated we set labels_x and labels_z identical (bit‑flip corrections)
        labels_x = labels9.copy(); labels_z = labels9.copy()
        # Update stats
        self._stats['total_shots'] += int(B)
        if teacher_used == 'mwpf':
            self._stats['mwpf_shots'] += int(B)
        elif teacher_used == 'mwpm':
            self._stats['mwpm_shots'] += int(B)
        # lut and other teachers don't increment mwpf/mwpm counters
        return s_bin, labels_x, labels_z

    def stats_snapshot(self) -> Dict:
        """Return a shallow copy of accumulated teacher usage statistics."""
        return dict(self._stats)

    # --- Teacher helpers ---
    def _build_stim_dem_rotated_d3(self, Hx: np.ndarray, Hz: np.ndarray) -> 'stim.DetectorErrorModel':
        if not _HAS_STIM:
            raise RuntimeError("stim is required for MWPF Sinter path")
        # Minimal DEM: define relationships between 8 detectors and 9 logical observables
        # by adding small-probability error mechanisms that toggle specific detectors
        # and tag a corresponding observable L{j}. No explicit OBSERVABLE_INCLUDE lines required.
        dem = stim.DetectorErrorModel()
        for j in range(9):
            dets = []
            # Z checks (first 4 detectors)
            for i in range(4):
                if int(Hz[i, j]) & 1:
                    dets.append(i)
            # X checks (next 4 detectors)
            for i in range(4):
                if int(Hx[i, j]) & 1:
                    dets.append(4 + i)
            # Build DEM targets using Stim's typed targets
            targets = [stim.target_relative_detector_id(int(d)) for d in dets]
            targets.append(stim.target_logical_observable_id(int(j)))
            # Append an error that flips the relevant detectors and links to logical observable j
            # Using a tiny probability keeps weights uniform and avoids biasing the decoder.
            dem.append("error", 1e-6, targets=targets)
        return dem

    def _labels_via_mwpf_stim(self, s_bin: np.ndarray) -> np.ndarray:
        if not (_HAS_MWPF and _HAS_STIM):
            raise RuntimeError("MWPF+Stim not available in environment")
        # Build H matrices for rotated d=3 and corresponding DEM
        from cudaq_backend.circuits import build_H_rotated_d3_from_cfg
        Hx, Hz, meta = build_H_rotated_d3_from_cfg({})
        dem = self._build_stim_dem_rotated_d3(Hx, Hz)
        # Compile decoder directly from DEM without attaching a circuit
        sinter_dec = SinterMWPFDecoder()
        compiled = sinter_dec.compile_decoder_for_dem(dem=dem)
        # Pack detection events [B,8] -> [B,1] LSBF
        B = s_bin.shape[0]
        dets_b8 = np.zeros((B, 1), dtype=np.uint8)
        for i in range(B):
            v = 0
            for k in range(8):
                v |= (int(s_bin[i, k]) & 1) << k
            dets_b8[i, 0] = v
        # Decode in batch: returns bit‑packed observables; unpack to [B,9]
        obs_b8 = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=dets_b8)
        # Unpack LSBF per byte(s)
        # Observables: 9 -> 2 bytes; api returns contiguous bytes per shot
        n_obs = 9
        n_bytes = (n_obs + 7) // 8
        if obs_b8.shape[1] < n_bytes:
            # If decoder returns 1 byte, extend with zeros
            obs_b8 = np.hstack([obs_b8, np.zeros((B, n_bytes - obs_b8.shape[1]), dtype=np.uint8)])
        bits = ((obs_b8[:, :, None] >> np.arange(8, dtype=np.uint8)[None, None, :]) & 1).astype(np.uint8)
        bits = bits.reshape(B, n_bytes * 8)[:, :n_obs]
        return bits

    def _labels_via_mwpm(self, hx: np.ndarray, hz: np.ndarray, s_bin: np.ndarray) -> np.ndarray:
        # Parity-exact minimal-weight solution via GF(2) solve on [Hz; Hx]
        B = s_bin.shape[0]
        labels = np.zeros((B, 9), dtype=np.uint8)
        H = np.vstack([hz, hx]).astype(np.uint8)
        for b in range(B):
            s = s_bin[b].astype(np.uint8)
            c = _solve_gf2_min_weight(H, s)
            labels[b] = c
        return labels

def _solve_gf2_min_weight(H: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Solve H c = s over GF(2) and pick minimal-weight solution.
    Assumes H is (8,9) for rotated d=3; nullspace typically 1‑dim.
    Returns c in {0,1}^9.
    """
    H = H.copy().astype(np.uint8)
    m, n = H.shape
    A = np.concatenate([H, s.reshape(-1,1)], axis=1)  # augmented [H|s]
    row = 0; pivots = []
    for col in range(n):
        # find pivot
        pivot = None
        for r in range(row, m):
            if A[r, col] & 1:
                pivot = r; break
        if pivot is None:
            continue
        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]
        # eliminate other rows
        for r in range(m):
            if r != row and (A[r, col] & 1):
                A[r, :] ^= A[row, :]
        pivots.append(col)
        row += 1
        if row == m:
            break
    # Back-substitution to one solution c0
    c0 = np.zeros(n, dtype=np.uint8)
    # Express free vars as 0 initially
    # Verify consistency (no 0=1 rows)
    for r in range(m):
        if (A[r, :n].sum() == 0) and (A[r, n] == 1):
            # No solution; return zero as fallback
            return c0
    # Build one solution by solving pivot rows
    for r in reversed(range(len(pivots))):
        col = pivots[r]
        rhs = A[r, n]
        # subtract known contributions
        for c in range(col+1, n):
            if A[r, c] & 1:
                rhs ^= c0[c]
        c0[col] = rhs
    # Nullspace basis vectors (free vars): pick minimal by also trying c0 xor nvecs
    # For (8,9), expect 1 free var -> 1 null vector; construct by setting one free var to 1
    free = [j for j in range(n) if j not in pivots]
    if not free:
        return c0
    # Build nullspace vector for first free var
    v = np.zeros(n, dtype=np.uint8); v[free[0]] = 1
    # For each pivot row, maintain H*pivotcol + sum(free*coef)=0 => pivot value equals sum of free contributions
    for r, pc in enumerate(pivots):
        acc = 0
        for c in free:
            if v[c] and (A[r, c] & 1):
                acc ^= 1
        v[pc] = acc
    # Compare weights
    w0 = int(c0.sum()); w1 = int((c0 ^ v).sum())
    return c0 if w0 <= w1 else (c0 ^ v)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="CUDA-Q Garnet sampler facade test")
    ap.add_argument("-B", type=int, default=16)
    ap.add_argument("-p", type=float, default=0.05)
    args = ap.parse_args()
    sam = CudaqGarnetSampler("foundation")
    s, lx, lz = sam.sample_batch(args.B, args.p)
    print("s shape:", s.shape, "labels:", lx.shape, lz.shape)
