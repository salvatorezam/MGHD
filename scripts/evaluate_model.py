#!/usr/bin/env python
"""
Evaluate a trained MGHD model over distance/noise sweeps.

The output JSON is intentionally metadata-rich so downstream plotting can keep a
clear x-axis contract (for example, circuit-level lambda scaling vs explicit
phenomenological p_data).
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch

from mghd.codes.registry import get_code
from mghd.core.core import MGHDDecoderPublic
from mghd.decoders.lsd.clustered import MGHDPrimaryClustered
from mghd.decoders.lsd_teacher import LSDTeacher
from mghd.decoders.mwpm_ctx import MWPMatchingContext
from mghd.decoders.mwpf_teacher import MWPFTeacher
from mghd.decoders.nvqldpc_teacher import NvQldpcTeacher
from mghd.samplers.cudaq_sampler import CudaQSampler
from mghd.samplers.stim_sampler import StimSampler
from mghd.utils.metrics import logical_error_rate

try:  # Optional CUDA-Q QEC python API (tensor-network decoder track)
    import cudaq_qec as qec  # type: ignore

    _HAVE_CUDAQ_QEC = True
except Exception:
    qec = None  # type: ignore
    _HAVE_CUDAQ_QEC = False


@dataclass
class PhenomenologicalBatch:
    """Sample batch from phenomenological noise model."""
    dets: np.ndarray  # Combined syndromes [synX | synZ] for compatibility
    obs: np.ndarray   # Logical observables
    synX: np.ndarray  # X-stabilizer syndromes (detect Z errors)
    synZ: np.ndarray  # Z-stabilizer syndromes (detect X errors)
    err_x: np.ndarray # X errors (for debugging)
    err_z: np.ndarray # Z errors (for debugging)
    meta: dict


class PhenomenologicalSampler:
    """Sampler that generates IID phenomenological noise on surface codes.
    
    This produces both X and Z syndromes from independent data qubit errors,
    matching the training setup used with CUDA-Q.
    """
    
    def __init__(self, p: float = 0.01):
        self.p = p
    
    def sample(self, code_obj, n_shots: int, seed: Optional[int] = None) -> PhenomenologicalBatch:
        """Sample phenomenological errors and compute syndromes."""
        rng = np.random.default_rng(seed)
        
        Hx = np.asarray(code_obj.Hx, dtype=np.uint8)
        Hz = np.asarray(code_obj.Hz, dtype=np.uint8)
        n_data = Hx.shape[1]
        mx = Hx.shape[0]
        mz = Hz.shape[0]
        
        # Generate IID errors
        err_x = (rng.random((n_shots, n_data)) < self.p).astype(np.uint8)
        err_z = (rng.random((n_shots, n_data)) < self.p).astype(np.uint8)
        
        # Compute syndromes
        # X checks (Hx) detect Z errors: synX = (Hx @ err_z.T).T % 2
        synX = (err_z @ Hx.T) % 2
        # Z checks (Hz) detect X errors: synZ = (Hz @ err_x.T).T % 2
        synZ = (err_x @ Hz.T) % 2
        
        # Combined detector array for compatibility with _resolve_syndromes
        # Format: [synX | synZ] so that detectors_to_syndromes works correctly
        dets = np.concatenate([synX, synZ], axis=1).astype(np.uint8)
        
        # Compute logical observables
        # For surface code: Lx detects Z logical errors, Lz detects X logical errors
        Lx = getattr(code_obj, 'Lx', None)
        Lz = getattr(code_obj, 'Lz', None)
        
        obs_list = []
        if Lz is not None:
            Lz_arr = np.asarray(Lz, dtype=np.uint8)
            if Lz_arr.ndim == 1:
                Lz_arr = Lz_arr.reshape(1, -1)
            # X errors flip Z logical: obs_z = (err_x @ Lz.T) % 2
            obs_z = (err_x @ Lz_arr.T) % 2
            obs_list.append(obs_z)
        if Lx is not None:
            Lx_arr = np.asarray(Lx, dtype=np.uint8)
            if Lx_arr.ndim == 1:
                Lx_arr = Lx_arr.reshape(1, -1)
            # Z errors flip X logical: obs_x = (err_z @ Lx.T) % 2  
            obs_x = (err_z @ Lx_arr.T) % 2
            obs_list.append(obs_x)
        
        if obs_list:
            obs = np.concatenate(obs_list, axis=1).astype(np.uint8)
        else:
            obs = np.zeros((n_shots, 1), dtype=np.uint8)
        
        return PhenomenologicalBatch(
            dets=dets,
            obs=obs,
            synX=synX.astype(np.uint8),
            synZ=synZ.astype(np.uint8),
            err_x=err_x,
            err_z=err_z,
            meta={'sampler': 'phenomenological', 'p': self.p},
        )


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return float(default)


def _resolve_noise_axis_and_params(
    *, sampler: str, p_input: float, noise_scale: float | None
) -> tuple[str, float, dict]:
    """Return canonical axis name/value and resolved noise parameters."""
    sampler = str(sampler).lower()
    if sampler == "phenomenological":
        resolved = {
            "noise_model_family": "phenomenological_iid",
            "p_data": float(p_input),
            "p_meas": float(p_input),
        }
        return "p_data", float(p_input), resolved

    if sampler == "stim":
        resolved = {
            "noise_model_family": "stim_generated_surface_memory_x",
            "after_clifford_depolarization": float(p_input),
            "rounds": 1,
        }
        return "p_data", float(p_input), resolved

    if sampler == "cudaq":
        family = str(os.getenv("MGHD_NOISE_MODEL", "generic_cl")).strip().lower()
        if family in {"generic", "generic_cl", "generic-circuit"}:
            if noise_scale is not None:
                lam = float(noise_scale)
            else:
                p_ref = 0.03
                lam = max(1e-3, min(5.0, float(p_input) / p_ref))
            resolved = {
                "noise_model_family": "generic_cl",
                "requested_phys_p": float(p_input),
                "lambda_scale": float(lam),
                "p_ref": 0.03,
                "p_1q": _env_float("MGHD_GENERIC_P1Q", 0.0015) * lam,
                "p_2q": _env_float("MGHD_GENERIC_P2Q", 0.01) * lam,
                "p_idle": _env_float("MGHD_GENERIC_PIDLE", 0.0008) * lam,
                "p_meas0": _env_float("MGHD_GENERIC_PMEAS0", 0.02) * lam,
                "p_meas1": _env_float("MGHD_GENERIC_PMEAS1", 0.02) * lam,
                "p_hook": _env_float("MGHD_GENERIC_PHOOK", 0.0) * lam,
                "p_crosstalk": _env_float("MGHD_GENERIC_PCROSSTALK", 0.0) * lam,
                "idle_ref_ns": _env_float("MGHD_GENERIC_IDLE_REF_NS", 20.0),
            }
            return "lambda_scale", float(lam), resolved

        # Hardware/profile modes keep requested p as axis by default.
        resolved = {
            "noise_model_family": family if family else "cudaq_profile",
            "requested_phys_p": float(p_input),
            "noise_scale": float(noise_scale) if noise_scale is not None else None,
        }
        return "p_phys_requested", float(p_input), resolved

    # Fallback for unknown sampler paths.
    return "p", float(p_input), {"noise_model_family": sampler}


def _wilson_ci(failures: float, shots: int, z: float = 1.959963984540054) -> dict[str, float] | None:
    if shots <= 0:
        return None
    k = float(np.clip(failures, 0.0, float(shots)))
    p_hat = k / float(shots)
    denom = 1.0 + (z * z) / float(shots)
    center = (p_hat + (z * z) / (2.0 * float(shots))) / denom
    radius = (z / denom) * np.sqrt(
        (p_hat * (1.0 - p_hat) / float(shots)) + ((z * z) / (4.0 * float(shots) * float(shots)))
    )
    return {"lo": float(max(0.0, center - radius)), "hi": float(min(1.0, center + radius))}


def _parse_tn_noise_model(spec: str | None, p_input: float, n_qubits: int) -> list[float]:
    if n_qubits <= 0:
        raise ValueError("tensor-network noise model requires n_qubits > 0")

    if spec is None or str(spec).strip().lower() in {"", "auto"}:
        p = float(np.clip(float(p_input), 1e-12, 0.499999))
        return [p] * int(n_qubits)

    vals = [float(x) for x in str(spec).split(",") if x.strip()]
    if len(vals) == 1:
        p = float(np.clip(float(vals[0]), 1e-12, 0.499999))
        return [p] * int(n_qubits)
    if len(vals) == n_qubits:
        return [float(np.clip(v, 1e-12, 0.499999)) for v in vals]
    if len(vals) == 3:
        # Allow pauli-style shorthand and collapse to a per-qubit error probability.
        p = float(np.clip(sum(max(float(v), 0.0) for v in vals), 1e-12, 0.499999))
        return [p] * int(n_qubits)
    raise ValueError(
        "--tn-noise-model must be 'auto', a single float, "
        "three comma-separated floats, or one float per qubit."
    )


def _first_logical_row(logical) -> np.ndarray | None:
    if logical is None:
        return None
    arr = np.asarray(logical, dtype=np.uint8)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[0] < 1:
        return None
    return arr[:1, :]


def _tn_decode_bit(decoder, syndrome_bits: np.ndarray) -> int:
    out = decoder.decode(syndrome_bits.astype(np.float32).tolist())
    raw = getattr(out, "result", None)
    arr = np.asarray(raw, dtype=np.float32).ravel() if raw is not None else np.zeros(1, dtype=np.float32)
    prob = float(arr[0]) if arr.size else 0.0
    return int(prob >= 0.5)


def align_preds(preds, obs_true):
    """Align predictions with ground truth observables."""
    # Squeeze middle dimension if present (B, 1, num_obs) -> (B, num_obs)
    if preds.ndim == 3 and preds.shape[1] == 1:
        preds = preds[:, 0, :]
        
    if preds.shape == obs_true.shape:
        return preds
    if obs_true.shape[1] == 1 and preds.shape[1] == 2:
        # Assume Stim surface code memory X -> logical X observable (2nd column of data_to_observables)
        # data_to_observables returns [Z_obs, X_obs]
        return preds[:, 1:]
    return preds

def evaluate(args):

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    device = "cuda" if args.cuda else "cpu"
    node_feat_dim = int(getattr(args, "node_feat_dim", 8))

    def _load_state(path: str):
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
            return payload["model"]
        if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
            return payload["state_dict"]
        return payload if isinstance(payload, dict) else {}

    try:
        decoder_public = MGHDDecoderPublic(
            args.checkpoint,
            device=device,
            profile=args.profile,
            node_feat_dim=node_feat_dim,
        )
    except RuntimeError as exc:
        # Retry once by inferring node feature width from checkpoint if mismatched.
        retried = False
        if "size mismatch for node_in.weight" in str(exc):
            state = _load_state(args.checkpoint)
            tensor = state.get("node_in.weight", None)
            if tensor is not None and hasattr(tensor, "shape"):
                inferred = int(tensor.shape[1])
                if inferred != node_feat_dim:
                    print(f"Warning: node_feat_dim={node_feat_dim} mismatch; retrying with {inferred}")
                    node_feat_dim = inferred
                    decoder_public = MGHDDecoderPublic(
                        args.checkpoint,
                        device=device,
                        profile=args.profile,
                        node_feat_dim=node_feat_dim,
                    )
                    retried = True
        if not retried:
            # If CUDA init fails, fall back to CPU but keep going.
            if args.cuda and ("cuda" in str(exc).lower() or "cudnn" in str(exc).lower()):
                print(f"Warning: CUDA failed ({exc}); falling back to CPU for evaluation.")
                device = "cpu"
                decoder_public = MGHDDecoderPublic(
                    args.checkpoint,
                    device=device,
                    profile=args.profile,
                    node_feat_dim=node_feat_dim,
                )
            else:
                raise
    
    results = []
    if Path(args.output).exists():
        try:
            with open(args.output, "r") as f:
                results = json.load(f)
            print(f"Loaded {len(results)} existing results from {args.output}")
        except json.JSONDecodeError:
            print(f"Could not load existing results from {args.output}, starting fresh.")

    distances = [int(d) for d in args.distances.split(",")]
    p_values = [float(p) for p in args.p_values.split(",")]
    
    for d in distances:
        print(f"\n=== Distance {d} ===")
        code = get_code(args.family, distance=d)
        
        # Ensure we have both sparse and dense versions
        if sp.issparse(code.Hx):
            Hx_sparse = code.Hx
            Hx_dense = code.Hx.toarray()
        else:
            Hx_sparse = sp.csr_matrix(code.Hx)
            Hx_dense = np.asarray(code.Hx)
            
        if sp.issparse(code.Hz):
            Hz_sparse = code.Hz
            Hz_dense = code.Hz.toarray()
        else:
            Hz_sparse = sp.csr_matrix(code.Hz)
            Hz_dense = np.asarray(code.Hz)
        
        # Bind code to decoder (needed for some internal checks, though PrimaryClustered passes H explicitly)
        decoder_public.bind_code(Hx_sparse, Hz_sparse)
        
        # Initialize decoders
        # Use non-batched decode path for robustness across distances.
        # The batched path can retain stale internal state on some larger-distance
        # component patterns and intermittently misindex samples.
        mghd_Z = MGHDPrimaryClustered(Hx_sparse, decoder_public, batched=False)
        mghd_X = MGHDPrimaryClustered(Hz_sparse, decoder_public, batched=False)
        
        # MWPM
        mwpm_ctx = MWPMatchingContext()
        mwpm_enabled = True

        # LSD
        lsd = LSDTeacher(Hx_dense, Hz_dense)

        # MWPF (hypergraph / fault-id proxy baseline)
        mwpf_teacher = None
        if not bool(getattr(args, "disable_mwpf", False)):
            try:
                mwpf_teacher = MWPFTeacher(code)
            except Exception as exc:
                print(f"  MWPFTeacher unavailable for d={d}: {exc}")

        # NvQldpc (GPU BP+OSD per-qubit baseline)
        nvqldpc_teacher = None
        if bool(getattr(args, "enable_nvqldpc", False)):
            try:
                nvqldpc_teacher = NvQldpcTeacher(Hx_dense, Hz_dense)
            except Exception as exc:
                print(f"  NvQldpcTeacher unavailable for d={d}: {exc}")

        # Tensor-network decoder requires one logical observable per decoder.
        lz_row = _first_logical_row(getattr(code, "Lz", None))
        lx_row = _first_logical_row(getattr(code, "Lx", None))
        
        for p in p_values:
            # Check if already computed
            if any(r["distance"] == d and r["p"] == p for r in results):
                print(f"  Skipping p={p} (already computed)")
                continue

            print(f"  Testing p={p}...")
            x_axis_name, x_axis_value, resolved_noise = _resolve_noise_axis_and_params(
                sampler=args.sampler,
                p_input=p,
                noise_scale=getattr(args, "noise_scale", None),
            )

            tn_decoder_z = None
            tn_decoder_x = None
            if bool(getattr(args, "enable_tn", False)):
                if not _HAVE_CUDAQ_QEC:
                    print("  Tensor-network decoder unavailable: cudaq_qec import failed.")
                else:
                    try:
                        tn_noise = _parse_tn_noise_model(
                            getattr(args, "tn_noise_model", "auto"),
                            p,
                            int(Hx_dense.shape[1]),
                        )
                        tn_device = str(getattr(args, "tn_device", "cpu"))
                        if lz_row is not None:
                            tn_decoder_z = qec.get_decoder(  # type: ignore[union-attr]
                                "tensor_network_decoder",
                                np.asarray(Hz_dense, dtype=np.uint8),
                                logical_obs=lz_row,
                                noise_model=tn_noise,
                                device=tn_device,
                            )
                        if lx_row is not None:
                            tn_decoder_x = qec.get_decoder(  # type: ignore[union-attr]
                                "tensor_network_decoder",
                                np.asarray(Hx_dense, dtype=np.uint8),
                                logical_obs=lx_row,
                                noise_model=tn_noise,
                                device=tn_device,
                            )
                    except Exception as exc:
                        print(f"  Tensor-network decoder unavailable at d={d}, p={p}: {exc}")
                        tn_decoder_z = None
                        tn_decoder_x = None
            
            # Setup sampler
            if args.sampler == "stim":
                sampler = StimSampler(rounds=1, dep=p)  # rounds=1 to match training (single-round phenomenological)
            elif args.sampler == "cudaq":
                # CudaQSampler expects phys_p/noise_scale in profile_kwargs
                pk = {"phys_p": p, "rounds": d}
                if getattr(args, "noise_scale", None) is not None:
                    pk["noise_scale"] = float(args.noise_scale)
                sampler = CudaQSampler(device_profile="garnet", profile_kwargs=pk)
            elif args.sampler == "phenomenological":
                # Phenomenological sampler generates IID X/Z errors with both syndrome types
                # This matches the training setup better than Stim's circuit-level simulation
                sampler = PhenomenologicalSampler(p=p)
            else:
                raise ValueError(f"Unknown sampler: {args.sampler}")
            
            total_shots = 0
            failures_mghd = 0.0
            failures_mwpm = 0.0 if mwpm_enabled else None
            failures_lsd = 0.0
            failures_nvqldpc = 0.0 if nvqldpc_teacher is not None else None
            failures_mwpf = 0.0 if mwpf_teacher is not None else None
            failures_tn = 0.0 if (tn_decoder_z is not None or tn_decoder_x is not None) else None
            mghd_lsd_agree = 0
            mghd_decode_errors = 0
            nvqldpc_decode_errors = 0
            tn_decode_errors = 0
            
            # Batched evaluation (ceil division so we always process >0 when shots>0)
            n_batches = (args.shots + args.batch_size - 1) // args.batch_size
            
            for b in range(n_batches):
                # Adjust batch size for final (possibly smaller) chunk
                this_batch = min(args.batch_size, args.shots - total_shots)
                if this_batch <= 0:
                    break

                # --- Sample data ---
                batch = sampler.sample(code, n_shots=this_batch, seed=args.seed + b)
                obs_true = batch.obs

                # For phenomenological sampler, use direct synX/synZ
                # For CUDA-Q, detectors are in canonical Z→X order.
                if hasattr(batch, 'synX') and hasattr(batch, 'synZ'):
                    sx = batch.synX
                    sz = batch.synZ
                elif args.sampler == "cudaq":
                    mz = int(code.Hz.shape[0])
                    mx = int(code.Hx.shape[0])
                    dets = np.asarray(batch.dets, dtype=np.uint8)
                    sz = dets[:, :mz].astype(np.uint8)
                    sx = dets[:, mz:mz + mx].astype(np.uint8)
                else:
                    sx, sz = _resolve_syndromes(code, batch.dets)
                
                # --- MGHD Decoding ---
                preds_mghd = []
                for i in range(this_batch):
                    try:
                        # Z errors (from sx - X checks detect Z errors)
                        res_z = mghd_Z.decode(sx[i])
                        ez = res_z["e"]

                        # X errors (from sz - Z checks detect X errors)
                        res_x = mghd_X.decode(sz[i])
                        ex = res_x["e"]

                        # Convert correction to observables
                        obs_pred = code.data_to_observables(ex, ez)
                    except Exception as exc:
                        mghd_decode_errors += 1
                        if getattr(args, "mghd_error_policy", "raise") == "zero":
                            # Use a canonical all-zero correction projected through
                            # code.data_to_observables so output shape matches
                            # successful decode paths exactly.
                            ex_zero = np.zeros(Hz_dense.shape[1], dtype=np.uint8)
                            ez_zero = np.zeros(Hx_dense.shape[1], dtype=np.uint8)
                            obs_pred = code.data_to_observables(ex_zero, ez_zero)
                        else:
                            raise RuntimeError(
                                f"MGHD decode failed at d={d}, p={p}, batch={b + 1}/{n_batches}, "
                                f"shot={i + 1}/{this_batch}"
                            ) from exc
                    preds_mghd.append(obs_pred)
                
                preds_mghd = np.array(preds_mghd, dtype=np.uint8)
                preds_mghd = align_preds(preds_mghd, obs_true)
                
                ler_res_mghd = logical_error_rate(obs_true, preds_mghd)
                if ler_res_mghd.ler_mean is None:
                    print(f"MGHD LER Error: {ler_res_mghd.notes}")
                    print(f"obs_true shape: {obs_true.shape}, preds_mghd shape: {preds_mghd.shape}")
                    sys.exit(1)
                failures_mghd += float(ler_res_mghd.ler_mean) * this_batch
                
                # --- MWPM Decoding ---
                if mwpm_enabled and mwpm_ctx is not None:
                    preds_mwpm = []
                    for i in range(this_batch):
                        ez_pm, _ = mwpm_ctx.decode(Hx_dense, sx[i], "Z")
                        ex_pm, _ = mwpm_ctx.decode(Hz_dense, sz[i], "X")
                        obs_pred = code.data_to_observables(ex_pm, ez_pm)
                        preds_mwpm.append(obs_pred)
                    preds_mwpm = np.array(preds_mwpm, dtype=np.uint8)
                    preds_mwpm = align_preds(preds_mwpm, obs_true)
                    ler_res_mwpm = logical_error_rate(obs_true, preds_mwpm)
                    if ler_res_mwpm.ler_mean is None:
                        print(f"MWPM LER Error: {ler_res_mwpm.notes}")
                        print(f"obs_true shape: {obs_true.shape}, preds_mwpm shape: {preds_mwpm.shape}")
                        sys.exit(1)
                    failures_mwpm += float(ler_res_mwpm.ler_mean) * this_batch
                
                # --- LSD Decoding ---
                # NOTE: LSDTeacher.decode_batch_xz returns (ex, ez) where:
                #   ex = Hx^{-1}(syndromes_x) - solves Hx @ ex = sx
                #   ez = Hz^{-1}(syndromes_z) - solves Hz @ ez = sz
                # But in CSS semantics: Hx detects Z errors, Hz detects X errors!
                # So the naming is misleading - we need to swap:
                #   The "ex" output is actually ez (Z errors from sx via Hx)
                #   The "ez" output is actually ex (X errors from sz via Hz)
                ez_lsd, ex_lsd = lsd.decode_batch_xz(sx, sz)  # Swapped!
                preds_lsd = []
                for i in range(this_batch):
                    obs_pred = code.data_to_observables(ex_lsd[i], ez_lsd[i])
                    preds_lsd.append(obs_pred)
                preds_lsd = np.array(preds_lsd, dtype=np.uint8)
                preds_lsd = align_preds(preds_lsd, obs_true)
                
                ler_res_lsd = logical_error_rate(obs_true, preds_lsd)
                if ler_res_lsd.ler_mean is None:
                    print(f"LSD LER Error: {ler_res_lsd.notes}")
                    print(f"obs_true shape: {obs_true.shape}, preds_lsd shape: {preds_lsd.shape}")
                    sys.exit(1)
                failures_lsd += float(ler_res_lsd.ler_mean) * this_batch
                if preds_lsd.shape == preds_mghd.shape:
                    mghd_lsd_agree += int(np.sum(np.all(preds_lsd == preds_mghd, axis=1)))

                # --- NvQldpc Decoding (GPU BP+OSD teacher) ---
                if nvqldpc_teacher is not None and failures_nvqldpc is not None:
                    try:
                        # decode_batch_xz(sx, sz) follows the same channel ordering as LSD:
                        # first output belongs to Hx/sx (Z-error channel), second to Hz/sz (X-error channel).
                        ez_nq, ex_nq = nvqldpc_teacher.decode_batch_xz(sx, sz)
                        preds_nvq = []
                        for i in range(this_batch):
                            obs_pred = code.data_to_observables(ex_nq[i], ez_nq[i])
                            preds_nvq.append(obs_pred)
                        preds_nvq = np.asarray(preds_nvq, dtype=np.uint8)
                        preds_nvq = align_preds(preds_nvq, obs_true)
                        ler_res_nvq = logical_error_rate(obs_true, preds_nvq)
                        if ler_res_nvq.ler_mean is not None:
                            failures_nvqldpc += float(ler_res_nvq.ler_mean) * this_batch
                    except Exception:
                        nvqldpc_decode_errors += this_batch
                        failures_nvqldpc = None

                # --- Tensor-Network Observable Decoder ---
                if failures_tn is not None and (tn_decoder_z is not None or tn_decoder_x is not None):
                    try:
                        preds_tn = []
                        for i in range(this_batch):
                            obs_cols = []
                            if tn_decoder_z is not None:
                                obs_cols.append(_tn_decode_bit(tn_decoder_z, sz[i]))
                            if tn_decoder_x is not None:
                                obs_cols.append(_tn_decode_bit(tn_decoder_x, sx[i]))
                            if not obs_cols:
                                continue
                            preds_tn.append(np.asarray(obs_cols, dtype=np.uint8))
                        if preds_tn:
                            preds_tn_arr = np.asarray(preds_tn, dtype=np.uint8)
                            preds_tn_arr = align_preds(preds_tn_arr, obs_true)
                            ler_res_tn = logical_error_rate(obs_true, preds_tn_arr)
                            if ler_res_tn.ler_mean is not None:
                                failures_tn += float(ler_res_tn.ler_mean) * this_batch
                    except Exception:
                        tn_decode_errors += this_batch
                        failures_tn = None

                # --- MWPF Decoding (approximate ex/ez from fault_ids) ---
                # NOTE: MWPF often crashes on large distances or complex syndromes.
                # We catch exceptions and disable it for the rest of this (d, p) combo.
                if mwpf_teacher is not None and failures_mwpf is not None:
                    try:
                        out_mwpf = mwpf_teacher.decode_batch(batch.dets)
                        fault_ids = np.asarray(out_mwpf.get("fault_ids"), dtype=np.int32)
                        if fault_ids.ndim == 2:
                            B = fault_ids.shape[0]
                            n = Hx_dense.shape[1]
                            ex_pf = np.zeros((B, n), dtype=np.uint8)
                            ez_pf = np.zeros((B, n), dtype=np.uint8)
                            for bi in range(B):
                                fids = fault_ids[bi]
                                fids = fids[fids >= 0]
                                if fids.size:
                                    # Interpret each fault id as a data-qubit index; flip both X and Z.
                                    ex_pf[bi, fids] ^= 1
                                    ez_pf[bi, fids] ^= 1

                            preds_mwpf = []
                            for i in range(this_batch):
                                obs_pred = code.data_to_observables(ex_pf[i], ez_pf[i])
                                preds_mwpf.append(obs_pred)
                            preds_mwpf = np.array(preds_mwpf, dtype=np.uint8)
                            preds_mwpf = align_preds(preds_mwpf, obs_true)

                            ler_res_mwpf = logical_error_rate(obs_true, preds_mwpf)
                            if ler_res_mwpf.ler_mean is not None:
                                failures_mwpf += float(ler_res_mwpf.ler_mean) * this_batch
                    except Exception as exc:
                        print(f"\n  MWPF decode failed at d={d}, p={p}: {type(exc).__name__}")
                        # Disable MWPF for the rest of this evaluation
                        failures_mwpf = None
                
                total_shots += this_batch
                
                mwpm_ratio = None if failures_mwpm is None or total_shots == 0 else failures_mwpm / total_shots
                mwpm_str = f"{mwpm_ratio:.10f}" if mwpm_ratio is not None else "NA"
                mwpf_ratio = None if failures_mwpf is None or total_shots == 0 else failures_mwpf / total_shots
                mwpf_str = f"{mwpf_ratio:.10f}" if mwpf_ratio is not None else "NA"
                nvq_ratio = (
                    None if failures_nvqldpc is None or total_shots == 0 else failures_nvqldpc / total_shots
                )
                nvq_str = f"{nvq_ratio:.10f}" if nvq_ratio is not None else "NA"
                tn_ratio = None if failures_tn is None or total_shots == 0 else failures_tn / total_shots
                tn_str = f"{tn_ratio:.10f}" if tn_ratio is not None else "NA"
                print(
                    f"    Batch {b+1}/{n_batches}: MGHD={failures_mghd/total_shots:.6f} "
                    f"MWPM={mwpm_str} MWPF={mwpf_str} NVQ={nvq_str} TN={tn_str} "
                    f"MGHD_err={mghd_decode_errors} NVQ_err={nvqldpc_decode_errors} TN_err={tn_decode_errors} "
                    f"LSD={failures_lsd/total_shots:.6f}",
                    end="\r",
                )
            
            mwpm_final = None if failures_mwpm is None or total_shots == 0 else failures_mwpm / total_shots
            mwpm_final_str = f"{mwpm_final:.10f}" if mwpm_final is not None else "NA"
            mwpf_final = None if failures_mwpf is None or total_shots == 0 else failures_mwpf / total_shots
            mwpf_final_str = f"{mwpf_final:.10f}" if mwpf_final is not None else "NA"
            nvq_final = None if failures_nvqldpc is None or total_shots == 0 else failures_nvqldpc / total_shots
            nvq_final_str = f"{nvq_final:.10f}" if nvq_final is not None else "NA"
            tn_final = None if failures_tn is None or total_shots == 0 else failures_tn / total_shots
            tn_final_str = f"{tn_final:.10f}" if tn_final is not None else "NA"
            print(
                f"    Final: MGHD={failures_mghd/total_shots:.6f} "
                f"MWPM={mwpm_final_str} MWPF={mwpf_final_str} NVQ={nvq_final_str} TN={tn_final_str} "
                f"MGHD_err={mghd_decode_errors} NVQ_err={nvqldpc_decode_errors} TN_err={tn_decode_errors} "
                f"LSD={failures_lsd/total_shots:.6f}"
            )
            
            results.append({
                "distance": d,
                "p": p,
                "x_axis_name": x_axis_name,
                "x_axis_value": x_axis_value,
                "resolved_noise": resolved_noise,
                "shots": total_shots,
                "ler_mghd": failures_mghd / total_shots,
                "ler_mwpm": mwpm_final,      # parity-check baseline on same shots
                "ler_mwpf": mwpf_final,
                "ler_lsd": failures_lsd / total_shots,
                "ler_nvqldpc": nvq_final,
                "ler_tn": tn_final,
                "decoder_output_classes": {
                    "mghd": "per_qubit",
                    "mwpm": "per_qubit",
                    "lsd": "per_qubit",
                    "mwpf": "fault_ids_proxy",
                    "nvqldpc": "per_qubit",
                    "tensor_network_decoder": "observable_prob",
                },
                "comparison_classes": {
                    "mghd": "per_qubit",
                    "mwpm": "per_qubit",
                    "lsd": "per_qubit",
                    "mwpf": "proxy_observable_from_fault_ids",
                    "nvqldpc": "per_qubit",
                    "tensor_network_decoder": "observable_prob",
                },
                "confidence_intervals_95": {
                    "mghd": _wilson_ci(failures_mghd, total_shots),
                    "mwpm": _wilson_ci(failures_mwpm, total_shots) if failures_mwpm is not None else None,
                    "mwpf": _wilson_ci(failures_mwpf, total_shots) if failures_mwpf is not None else None,
                    "lsd": _wilson_ci(failures_lsd, total_shots),
                    "nvqldpc": _wilson_ci(failures_nvqldpc, total_shots)
                    if failures_nvqldpc is not None
                    else None,
                    "tensor_network_decoder": _wilson_ci(failures_tn, total_shots)
                    if failures_tn is not None
                    else None,
                },
                "obs_agreement_mghd_lsd": (mghd_lsd_agree / total_shots) if total_shots else None,
                "mghd_decode_errors": int(mghd_decode_errors),
                "nvqldpc_decode_errors": int(nvqldpc_decode_errors),
                "tn_decode_errors": int(tn_decode_errors),
            })
            
            # Save results incrementally
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Saved progress to {args.output}")
            
    # Final save (redundant but safe)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")
    
    # Run sanity checks
    sanity_check_results(results)
    
    # Plot results
    plot_results(
        results,
        args.output,
        y_min=getattr(args, "y_min", None),
        y_max=getattr(args, "y_max", None),
    )


def sanity_check_results(results):
    """Validate LER trends follow expected QEC behavior.
    
    Expected:
    1. LER should increase with p at fixed d (more noise → more errors)
    2. LER should decrease with d at fixed p (below threshold) — larger codes correct more
    3. MGHD should be comparable to or better than teacher (LSD) on training distribution
    """
    print("\n" + "="*60)
    print("SANITY CHECK: Validating LER trends")
    print("="*60)
    
    warnings_found = []
    
    # Group by distance
    by_d = defaultdict(list)
    for r in results:
        by_d[r["distance"]].append(r)
    
    # Check 1: LER increases with p at fixed d
    print("\n[Check 1] LER should increase with p at fixed d:")
    for d in sorted(by_d.keys()):
        pts = sorted(by_d[d], key=lambda x: x["p"])
        mghd_lers = [x["ler_mghd"] for x in pts]
        lsd_lers = [x["ler_lsd"] for x in pts if x.get("ler_lsd") is not None]
        
        # Check MGHD trend
        mghd_monotonic = all(mghd_lers[i] <= mghd_lers[i+1] + 0.001 for i in range(len(mghd_lers)-1))
        status = "✓" if mghd_monotonic else "✗ WARNING"
        print(f"  d={d}: MGHD {status}")
        if not mghd_monotonic:
            warnings_found.append(f"d={d}: MGHD LER not monotonic with p")
        
        # Check LSD trend (teacher baseline)
        if lsd_lers:
            lsd_monotonic = all(lsd_lers[i] <= lsd_lers[i+1] + 0.001 for i in range(len(lsd_lers)-1))
            status = "✓" if lsd_monotonic else "✗ WARNING"
            print(f"  d={d}: LSD  {status}")
            if not lsd_monotonic:
                warnings_found.append(f"d={d}: LSD LER not monotonic with p (baseline issue!)")
    
    # Check 2: LER decreases with d at fixed p (below threshold ~1%)
    print("\n[Check 2] LER should decrease with d at fixed p (below threshold):")
    by_p = defaultdict(list)
    for r in results:
        by_p[r["p"]].append(r)
    
    for p in sorted(by_p.keys()):
        if p > 0.008:  # Skip above-threshold points
            continue
        pts = sorted(by_p[p], key=lambda x: x["distance"])
        if len(pts) < 2:
            continue
        
        mghd_lers = [x["ler_mghd"] for x in pts]
        distances = [x["distance"] for x in pts]
        
        # Check if generally decreasing (allow some noise)
        decreasing_count = sum(1 for i in range(len(mghd_lers)-1) if mghd_lers[i] >= mghd_lers[i+1] - 0.002)
        is_decreasing = decreasing_count >= len(mghd_lers) - 2  # Allow 1 violation
        
        status = "✓" if is_decreasing else "✗ WARNING"
        trend = " → ".join(f"d{d}:{ler:.4f}" for d, ler in zip(distances, mghd_lers))
        print(f"  p={p:.4f}: {status} ({trend})")
        if not is_decreasing:
            warnings_found.append(f"p={p}: MGHD LER not decreasing with d")
    
    # Check 3: MGHD vs Teacher comparison
    print("\n[Check 3] MGHD should match or beat LSD teacher:")
    mghd_better = 0
    mghd_worse = 0
    mghd_worse_cases = []
    
    for r in results:
        if r.get("ler_lsd") is None:
            continue
        diff = r["ler_mghd"] - r["ler_lsd"]
        if diff <= 0.001:  # MGHD is better or equal (within noise)
            mghd_better += 1
        else:
            mghd_worse += 1
            mghd_worse_cases.append((r["distance"], r["p"], r["ler_mghd"], r["ler_lsd"]))
    
    total = mghd_better + mghd_worse
    if total > 0:
        pct_better = 100 * mghd_better / total
        status = "✓" if pct_better >= 80 else "✗ WARNING"
        print(f"  {status} MGHD ≤ LSD in {mghd_better}/{total} cases ({pct_better:.1f}%)")
        
        if mghd_worse_cases:
            print("\n  Cases where MGHD > LSD:")
            for d, p, ler_mghd, ler_lsd in mghd_worse_cases[:5]:
                ratio = ler_mghd / max(ler_lsd, 1e-10)
                print(f"    d={d}, p={p:.4f}: MGHD={ler_mghd:.5f}, LSD={ler_lsd:.5f} (ratio={ratio:.2f}x)")
            if len(mghd_worse_cases) > 5:
                print(f"    ... and {len(mghd_worse_cases) - 5} more")
    
    # Summary
    print("\n" + "="*60)
    if warnings_found:
        print(f"⚠️  {len(warnings_found)} warnings found:")
        for w in warnings_found:
            print(f"   - {w}")
    else:
        print("✓ All sanity checks passed!")
    print("="*60 + "\n")

def plot_results(results, output_path, y_min=None, y_max=None):
    """Generate and save a clean LER plot from results."""
    # Organize data by distance
    x_axis_name = None
    data_by_d = defaultdict(
        lambda: {
            "x": [],
            "mghd": [],
            "mwpm": [],
            "mwpf": [],
            "lsd": [],
            "nvqldpc": [],
            "tn": [],
        }
    )
    
    for res in results:
        d = res["distance"]
        x_value = float(res.get("x_axis_value", res.get("p", 0.0)))
        data_by_d[d]["x"].append(x_value)
        data_by_d[d]["mghd"].append(res["ler_mghd"])
        if x_axis_name is None:
            x_axis_name = str(res.get("x_axis_name", "p"))
        data_by_d[d]["mwpm"].append(
            float(res["ler_mwpm"]) if res.get("ler_mwpm") is not None else np.nan
        )
        data_by_d[d]["mwpf"].append(
            float(res["ler_mwpf"]) if res.get("ler_mwpf") is not None else np.nan
        )
        data_by_d[d]["lsd"].append(
            float(res["ler_lsd"]) if res.get("ler_lsd") is not None else np.nan
        )
        data_by_d[d]["nvqldpc"].append(
            float(res["ler_nvqldpc"]) if res.get("ler_nvqldpc") is not None else np.nan
        )
        data_by_d[d]["tn"].append(
            float(res["ler_tn"]) if res.get("ler_tn") is not None else np.nan
        )

    if y_min is not None and y_min > 0:
        y_floor = float(y_min)
    else:
        y_floor = 1e-9
            
    # Use a professional style
    plt.style.use('seaborn-v0_8-paper')
    # Set global font sizes for publication
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10,
        'lines.linewidth': 2,
        'lines.markersize': 8
    })
    
    plt.figure(figsize=(10, 8))
    
    # Use a qualitative colormap
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in np.linspace(0, 1, len(data_by_d))]
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    
    for i, (d, data) in enumerate(sorted(data_by_d.items())):
        p_vals = np.array(data["x"])
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        # Sort by p
        idx = np.argsort(p_vals)
        p_vals = p_vals[idx]
        
        # MGHD
        mghd_vals = np.clip(np.array(data["mghd"])[idx], y_floor, None)
        plt.loglog(p_vals, mghd_vals, marker=marker, linestyle='-', color=color, label=f'MGHD d={d}')

        mwpm_vals = np.array(data["mwpm"])[idx]
        mwpm_mask = np.isfinite(mwpm_vals)
        if np.any(mwpm_mask):
            plt.loglog(
                p_vals[mwpm_mask],
                np.clip(mwpm_vals[mwpm_mask], y_floor, None),
                marker=marker,
                linestyle='--',
                color=color,
                alpha=0.5,
                label=f'MWPM d={d}',
            )

        # MWPF
        mwpf_vals = np.array(data["mwpf"])[idx]
        mwpf_mask = np.isfinite(mwpf_vals)
        if np.any(mwpf_mask):
            plt.loglog(
                p_vals[mwpf_mask],
                np.clip(mwpf_vals[mwpf_mask], y_floor, None),
                marker=marker,
                linestyle='-.',
                color=color,
                alpha=0.5,
                label=f'MWPF-proxy d={d}',
            )
            
        # LSD
        lsd_vals = np.array(data["lsd"])[idx]
        lsd_mask = np.isfinite(lsd_vals)
        if np.any(lsd_mask):
            plt.loglog(
                p_vals[lsd_mask],
                np.clip(lsd_vals[lsd_mask], y_floor, None),
                marker=marker,
                linestyle=':',
                color=color,
                alpha=0.4,
                label=f'LSD d={d}',
            )

        nvq_vals = np.array(data["nvqldpc"])[idx]
        nvq_mask = np.isfinite(nvq_vals)
        if np.any(nvq_mask):
            plt.loglog(
                p_vals[nvq_mask],
                np.clip(nvq_vals[nvq_mask], y_floor, None),
                marker=marker,
                linestyle=(0, (5, 1)),
                color=color,
                alpha=0.65,
                label=f'NvQldpc d={d}',
            )

        tn_vals = np.array(data["tn"])[idx]
        tn_mask = np.isfinite(tn_vals)
        if np.any(tn_mask):
            plt.loglog(
                p_vals[tn_mask],
                np.clip(tn_vals[tn_mask], y_floor, None),
                marker=marker,
                linestyle=(0, (1, 1)),
                color=color,
                alpha=0.65,
                label=f'TN(obs) d={d}',
            )

    plt.grid(True, which="major", ls="-", alpha=0.5)
    plt.grid(True, which="minor", ls=":", alpha=0.2)
    axis_label = x_axis_name if x_axis_name is not None else "p"
    plt.xlabel(f"{axis_label}")
    plt.ylabel("Logical Error Rate (LER)")
    plt.title("Logical Error Rate vs Noise Axis")
    if y_min is not None or y_max is not None:
        lo = y_min if y_min is not None else None
        hi = y_max if y_max is not None else None
        plt.ylim(lo, hi)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plot_file_png = str(Path(output_path).with_suffix('.png'))
    plot_file_pdf = str(Path(output_path).with_suffix('.pdf'))
    
    plt.savefig(plot_file_png, dpi=300, bbox_inches='tight')
    plt.savefig(plot_file_pdf, bbox_inches='tight')
    print(f"Plots saved to {plot_file_png} and {plot_file_pdf}")

def _resolve_syndromes(code_obj, dets):
    """Helper to map detectors to syndromes (copied from teacher_eval.py)"""
    mx = getattr(code_obj, "Hx", None)
    mz = getattr(code_obj, "Hz", None)
    B = dets.shape[0]
    if hasattr(code_obj, "detectors_to_syndromes"):
        sx, sz = code_obj.detectors_to_syndromes(dets)
        return sx.astype(np.uint8), sz.astype(np.uint8)
    if mx is not None:
        sx = np.zeros((B, mx.shape[0]), dtype=np.uint8)
    else:
        sx = np.zeros((B, 0), dtype=np.uint8)
    if mz is not None:
        sz = np.zeros((B, mz.shape[0]), dtype=np.uint8)
    else:
        sz = np.zeros((B, 0), dtype=np.uint8)
    return sx, sz

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
Evaluate a trained MGHD model against baselines (MWPM, LSD, optional MWPF).

Key samplers:
  - phenomenological: IID X/Z errors (matches synthetic training)  
  - stim: Legacy Stim sampler
  - cudaq: CUDA-Q backend
""")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--family", default="surface")
    parser.add_argument("--distances", default="3,5")
    parser.add_argument("--p-values", default="0.001,0.005,0.01")
    parser.add_argument("--shots", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="evaluation_results.json")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--profile", default="S")
    parser.add_argument("--node-feat-dim", type=int, default=9)
    parser.add_argument("--sampler", default="phenomenological", 
                        choices=["stim", "cudaq", "phenomenological"],
                        help="Sampler to use. "
                             "'phenomenological' (default) matches synthetic training setup with IID X/Z errors. "
                             "'stim' uses legacy Stim sampler. "
                             "'cudaq' uses CUDA-Q backend.")
    parser.add_argument(
        "--disable-mwpf",
        action="store_true",
        help="Disable MWPF proxy baseline (useful if local MWPF install is unstable).",
    )
    parser.add_argument(
        "--enable-nvqldpc",
        action="store_true",
        help="Enable NvQldpcTeacher baseline when CUDA-QEC GPU runtime is available.",
    )
    parser.add_argument(
        "--enable-tn",
        action="store_true",
        help="Enable tensor-network observable decoder baseline (observable-probability class).",
    )
    parser.add_argument(
        "--tn-noise-model",
        type=str,
        default="auto",
        help=(
            "Tensor-network noise model: 'auto', single float, three floats 'px,py,pz' "
            "(collapsed to per-qubit p), or one float per qubit."
        ),
    )
    parser.add_argument(
        "--tn-device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Execution device requested by tensor-network decoder backend.",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=None,
        help="Optional global noise scale for CUDA-Q Garnet sampler (cudaq only).",
    )
    parser.add_argument(
        "--mghd-error-policy",
        default="raise",
        choices=["raise", "zero"],
        help=(
            "How to handle MGHD decode exceptions. "
            "'raise' aborts evaluation (default, strict). "
            "'zero' records all-zero MGHD prediction for failed shots."
        ),
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=None,
        help="Optional lower y-limit for log-scale LER plots (e.g., 1e-7).",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Optional upper y-limit for log-scale LER plots.",
    )
    
    args = parser.parse_args()
    evaluate(args)
