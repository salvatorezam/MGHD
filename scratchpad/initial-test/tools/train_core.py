#!/usr/bin/env python
"""
MGHD training entrypoint (teacher-supervised). CUDA-Q trajectories are the default sampler.

Examples:
  # Surface code d=3..31, CUDA-Q sampler, 64 shots per batch, 10 batches per distance
  python -m tools.train_core --family surface --distances 3-31:2 --sampler cudaq \
      --shots-per-batch 64 --batches 10

Notes:
  - Teachers:
      * MWPF (hypergraph) takes detector streams; Python API used directly (no DEM).
        (See mwpf README / examples.)  [Ref]
      * LSD (BP+LSD) runs per-basis on Hx/Hz.                           [Ref]
      * MWPM fallback uses PyMatching v2 decode_batch from H.           [Ref]
  - CUDA-Q trajectories simulate general Kraus/coherent noise at circuit level;
    we keep this as the gold training path vs Pauli-only DEM approximations.   [Ref]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import warnings
from dataclasses import asdict
from typing import Any, Dict, Optional

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mghd.utils.graphlike import is_graphlike

from samplers import get_sampler
from teachers.mix import MixConfig, TeacherMix

from .code_loader import load_code
from .curriculum import parse_distances
from tools.metrics import LEResult, logical_error_rate, summary_line


def _resolve_syndromes(code_obj, dets):
    """
    BEST-EFFORT mapping from det-streams to CSS syndromes for LSD/MWPM teachers.
    If code_obj exposes a helper, use it. Else return zeros with correct shapes.
    """

    mx = getattr(code_obj, "Hx", None)
    mz = getattr(code_obj, "Hz", None)
    B = dets.shape[0]
    if hasattr(code_obj, "detectors_to_syndromes"):
        sx, sz = code_obj.detectors_to_syndromes(dets)
        return sx.astype(np.uint8), sz.astype(np.uint8)
    if mx is not None and getattr(mx, "ndim", 2) == 2:
        sx = np.zeros((B, mx.shape[0]), dtype=np.uint8)
    else:
        sx = np.zeros((B, 0), dtype=np.uint8)
    if mz is not None and getattr(mz, "ndim", 2) == 2:
        sz = np.zeros((B, mz.shape[0]), dtype=np.uint8)
    else:
        sz = np.zeros((B, 0), dtype=np.uint8)
    return sx, sz


def _infer_data_qubits(code_obj: Any) -> int:
    if hasattr(code_obj, "n"):
        try:
            return int(code_obj.n)
        except Exception:
            pass
    hx = getattr(code_obj, "Hx", None)
    if hx is not None and getattr(hx, "shape", None):
        return int(hx.shape[1])
    hz = getattr(code_obj, "Hz", None)
    if hz is not None and getattr(hz, "shape", None):
        return int(hz.shape[1])
    return 0


def _maybe_get_native_circuit(code_obj: Any) -> Optional[Any]:
    for attr in ("native_circuit", "reference_circuit", "circuit", "qc", "quantum_circuit"):
        if hasattr(code_obj, attr):
            circuit = getattr(code_obj, attr)
            if circuit is not None:
                return circuit
    meta = getattr(code_obj, "meta", None)
    if isinstance(meta, dict):
        for key in ("native_circuit", "qiskit_circuit", "circuit"):
            if meta.get(key) is not None:
                return meta[key]
    return None


def _build_schedule_ir(context_source: str, code_obj: Any) -> Any:
    native = _maybe_get_native_circuit(code_obj)
    if native is None:
        return []
    try:
        if context_source == "qiskit" and "qiskit" in sys.modules:
            from adapters import qiskit_adapter

            return qiskit_adapter.to_schedule_ir(native)
        if context_source == "cirq" and "cirq" in sys.modules:
            from adapters import cirq_adapter  # type: ignore

            return cirq_adapter.to_schedule_ir(native)
        if context_source == "cudaq" and "cudaq" in sys.modules:
            from adapters import cudaq_adapter  # type: ignore

            return cudaq_adapter.to_schedule_ir(native)
    except Exception as exc:  # pragma: no cover - best-effort optional deps
        warnings.warn(
            f"Context adapter for '{context_source}' failed: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
    return []


def _base_overrides_from_maps(weight_maps: Dict[str, Any], n_qubits: int) -> Dict[str, Any]:
    llr = np.zeros(int(n_qubits), dtype=np.float32)
    w_qubit = weight_maps.get("w_qubit", {}) or {}
    for layer in w_qubit.values():
        for q, weight in layer.items():
            idx = int(q)
            if 0 <= idx < llr.size:
                llr[idx] += float(weight)
    w_pair = weight_maps.get("w_pair", {}) or {}
    for layer in w_pair.values():
        for pair, weight in layer.items():
            for q in pair:
                idx = int(q)
                if 0 <= idx < llr.size:
                    llr[idx] += float(weight) * 0.5

    clipped_llr = np.clip(llr, -12.0, 12.0)
    probs = 1.0 / (1.0 + np.exp(clipped_llr))
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    override = {
        "llr": clipped_llr.astype(np.float32, copy=False),
        "mwpm": probs.astype(np.float32, copy=False),
        "mwpf": {
            idx: float(np.clip(prob / 0.5, 0.1, 10.0))
            for idx, prob in enumerate(probs)
        },
    }
    return override


def _materialize_overrides(base: Dict[str, Any], scale: float) -> Dict[str, Any]:
    if not base:
        return {}
    overrides: Dict[str, Any] = {}
    if base.get("llr") is not None:
        overrides["llr_per_qubit"] = np.asarray(base["llr"], dtype=np.float32) * float(scale)
    if base.get("mwpm") is not None:
        scaled = np.asarray(base["mwpm"], dtype=np.float32) * float(scale)
        overrides["mwpm_weights"] = np.clip(scaled, 1e-6, 1 - 1e-6)
    if base.get("mwpf") is not None:
        overrides["mwpf_scale"] = {
            idx: float(np.clip(weight * float(scale), 0.1, 10.0))
            for idx, weight in base["mwpf"].items()
        }
    return overrides


def _load_bandit_state(path: pathlib.Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _dump_bandit_state(path: pathlib.Path, state: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state))
    except Exception:  # pragma: no cover - best effort persistence
        warnings.warn(
            f"Could not persist RL state to {path}",
            RuntimeWarning,
            stacklevel=2,
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--family",
        default="surface",
        help="code family id in codes_registry (e.g., surface, bb, rm, steane, repetition, color)",
    )
    p.add_argument(
        "--families",
        default=None,
        help=(
            "Comma-separated list of families to sweep (overrides --family). "
            "Example: 'surface,color_666,color_488,steane,repetition,gb,bb,hgp'."
        ),
    )
    p.add_argument("--distances", default="3-31:2", help="e.g., '3,5,7' or '3-31:2'")
    p.add_argument("--sampler", default="cudaq", choices=["cudaq", "stim"])
    p.add_argument("--shots-per-batch", type=int, default=128)
    p.add_argument("--batches", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--p-mwpf", type=float, default=0.5)
    p.add_argument("--p-lsd", type=float, default=0.4)
    p.add_argument("--p-mwpm", type=float, default=0.1)
    p.add_argument("--dem-enable", action="store_true")
    p.add_argument("--dem-family", type=str, default=None)
    p.add_argument("--dem-rounds", type=int, default=5)
    p.add_argument("--dem-correlated", action="store_true")
    p.add_argument("--dem-cache-dir", type=str, default="dem_cache")
    p.add_argument("--dem-force-build", action="store_true")
    p.add_argument(
        "--allow-mwpm-non-graphlike",
        dest="mwpm_graphlike_only",
        action="store_false",
        help="Allow MWPM even when the parity-check matrix has columns with more than two ones.",
    )
    p.set_defaults(mwpm_graphlike_only=True)
    p.add_argument("--qpu-profile", type=str, default=None)
    p.add_argument(
        "--context-source",
        type=str,
        choices=["none", "qiskit", "cirq", "cudaq"],
        default="none",
    )
    p.add_argument("--rl-online", action="store_true")
    p.add_argument("--rl-state", type=str, default=None)
    p.add_argument("--rl-noise-var", type=float, default=1.0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.families:
        families = [f.strip() for f in args.families.split(",") if f.strip()]
        if not families:
            raise ValueError("--families specified but no valid entries parsed")
    else:
        families = [args.family]

    # Sampler (CUDA-Q is the default; actual CLN via trajectories)  [Ref CUDA-Q]
    sampler_name = args.sampler

    for family in families:
        distances = parse_distances(args.distances)
        for d in distances:
            print(f"\n=== Family={family}  d={d}  sampler={args.sampler} ===")
            # Load code object (Hx/Hz, detector metadata, hypergraph mapping if available)
            code = load_code(family, d)
            Hx = getattr(code, "Hx", None)
            Hz = getattr(code, "Hz", None)

            # Teacher stack
            graph_ok = (
                Hx is not None
                and Hz is not None
                and is_graphlike(Hx)
                and is_graphlike(Hz)
            )
            pymatching_ok = args.dem_enable or graph_ok
            if sampler_name == "cudaq" and not pymatching_ok:
                warnings.warn(
                    f"PyMatching disabled (sampler=cudaq, graphlike={graph_ok}, dem={args.dem_enable})",
                    RuntimeWarning,
                    stacklevel=2,
                )
                p_mwpf_effective = 0.0
                p_mwpm_effective = 0.0
                mwpm_guard = True
            else:
                p_mwpf_effective = args.p_mwpf
                p_mwpm_effective = args.p_mwpm
                mwpm_guard = args.mwpm_graphlike_only

            mix = TeacherMix(
                code,
                Hx,
                Hz,
                mix_cfg=MixConfig(
                    p_mwpf=p_mwpf_effective,
                    p_lsd=args.p_lsd,
                    p_mwpm=p_mwpm_effective,
                ),
                mwpm_graphlike_only=mwpm_guard,
            )
            context_payload: Optional[Dict[str, Any]] = None
            base_overrides: Dict[str, Any] = {}
            ctx_vec = None
            profile = None
            profile_dict: Dict[str, Any] = {}
            if args.qpu_profile and args.context_source != "none":
                try:
                    from mghd_main.qpu_profile import load_qpu_profile

                    profile = load_qpu_profile(args.qpu_profile)
                except Exception as exc:
                    warnings.warn(
                        f"Could not load QPU profile '{args.qpu_profile}': {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    profile = None
                if profile is not None:
                    schedule_ir = _build_schedule_ir(args.context_source, code)
                    try:
                        from tad import weighting as tad_weighting
                        from tad import context as tad_context
                    except Exception as exc:
                        warnings.warn(
                            f"TAD weighting/context unavailable: {exc}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        tad_weighting = None
                        tad_context = None
                    else:
                        n_qubits = _infer_data_qubits(code) or int(getattr(profile, "n_qubits", 0))
                        profile_dict = asdict(profile)
                        weight_maps = tad_weighting.schedule_to_weight_maps(
                            schedule_ir,
                            profile,
                            n_qubits,
                        )
                        base_overrides = _base_overrides_from_maps(weight_maps, n_qubits)
                        features = tad_weighting.feature_vector(schedule_ir)
                        gate_vocab = {gate: idx for idx, gate in enumerate(sorted(features.get("gate_hist", {}).keys()))}
                        ctx_vec = tad_context.context_vector(features, gate_vocab)
                        context_payload = {
                            "features": features,
                            "gate_vocab": gate_vocab,
                            "schedule_len": len(schedule_ir),
                            "profile": getattr(profile, "name", "unknown"),
                        }

            dem_teacher = None
            if args.dem_enable:
                target_family = args.dem_family or family
                if target_family not in {"surface"}:
                    warnings.warn(
                        f"DEM build currently supported for {{'surface'}}; skipping for family='{target_family}'.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                else:
                    try:
                        import stim  # type: ignore
                    except Exception as exc:
                        warnings.warn(
                            f"Stim not available for DEM construction: {exc}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    else:
                        from teachers.dem_utils import dem_cache_path, build_surface_memory_dem
                        try:
                            from teachers.dem_matching import DEMMatchingTeacher
                        except Exception as exc:
                            warnings.warn(
                                f"DEM matching unavailable: {exc}",
                                RuntimeWarning,
                                stacklevel=2,
                            )
                        else:
                            profile_payload = profile_dict if profile_dict else {}
                            cache_path = pathlib.Path(
                                dem_cache_path(
                                    args.dem_cache_dir,
                                    target_family,
                                    d,
                                    args.dem_rounds,
                                    profile_payload,
                                )
                            )
                            dem_obj = None
                            try:
                                if not args.dem_force_build and cache_path.exists():
                                    dem_obj = stim.DetectorErrorModel(cache_path.read_text())
                                else:
                                    dem_obj = build_surface_memory_dem(
                                        distance=d,
                                        rounds=args.dem_rounds,
                                        profile=profile_payload,
                                        decompose=not args.dem_correlated,
                                    )
                                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                                    cache_path.write_text(str(dem_obj))
                                dem_teacher = DEMMatchingTeacher(
                                    dem_obj,
                                    correlated=args.dem_correlated,
                                )
                            except Exception as exc:
                                warnings.warn(
                                    f"Failed to prepare DEM teacher: {exc}",
                                    RuntimeWarning,
                                    stacklevel=2,
                                )

            sampler_kwargs: Dict[str, Any] = {}
            if sampler_name == "stim":
                dep_value = None
                if profile_dict:
                    dep_value = (
                        profile_dict.get("gate_error", {})
                        .get("after_clifford_depolarization")
                    )
                if dep_value is None:
                    dep_value = 0.001
                sampler_kwargs["rounds"] = args.dem_rounds
                sampler_kwargs["dep"] = float(dep_value)
            sampler = get_sampler(sampler_name, **sampler_kwargs)

            totals = {"mwpf": 0, "lsd": 0, "mwpm": 0, "mwpm_fallback": 0}
            if dem_teacher is not None:
                totals["dem_matching"] = 0
            t0 = time.time()
            true_chunks = []
            pred_chunks = []
            dem_pred_chunks = []
            missing_obs_warned = False

            bandit = None
            bandit_ctx = None
            if args.rl_online:
                from tad_rl.lin_ts import LinTSBandit

                if ctx_vec is None:
                    ctx_vec = np.ones(1, dtype=np.float32)
                bandit_ctx = np.asarray(ctx_vec, dtype=np.float32)
                bandit = LinTSBandit(d=bandit_ctx.shape[0], noise_var=args.rl_noise_var)
                if args.rl_state:
                    path = pathlib.Path(args.rl_state)
                    saved = _load_bandit_state(path)
                    if saved is not None:
                        try:
                            A = np.asarray(saved.get("A"), dtype=np.float64)
                            b_vec = np.asarray(saved.get("b"), dtype=np.float64)
                            if A.shape == (bandit.d, bandit.d) and b_vec.shape == (bandit.d,):
                                bandit.A = A
                                bandit.b = b_vec
                            else:
                                raise ValueError("shape mismatch")
                        except Exception as exc:
                            warnings.warn(
                                f"Ignoring RL state due to mismatch: {exc}",
                                RuntimeWarning,
                                stacklevel=2,
                            )

            if ctx_vec is None:
                ctx_vec = np.ones(1, dtype=np.float32)

            for batch_idx in range(args.batches):
                batch = sampler.sample(
                    code,
                    n_shots=args.shots_per_batch,
                    seed=int(rng.integers(1 << 32) - 1),
                )

                if args.dem_enable and not missing_obs_warned:
                    if batch.obs is None or batch.obs.size == 0:
                        warnings.warn(
                            "Sampler did not return logical observables; LER will be NA. "
                            "Use --sampler stim for DEM validation or extend the sampler to emit obs.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        missing_obs_warned = True

                if (
                    dem_teacher is not None
                    and getattr(dem_teacher, "num_detectors", None) is not None
                    and batch.dets.shape[1] != dem_teacher.num_detectors
                ):
                    warnings.warn(
                        "DEM teacher detector count mismatch. Ensure the sampler mirrors the Stim generator "
                        "(--sampler stim) or align detectors (e.g., --dem-align).",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    dem_teacher = None

                sx, sz = _resolve_syndromes(code, batch.dets)

                scale = 1.0
                if bandit is not None and bandit_ctx is not None:
                    theta = bandit.sample_theta()
                    gain = float(np.dot(bandit_ctx, theta))
                    scale = float(np.clip(1.0 + 0.1 * gain, 0.5, 2.0))

                overrides = _materialize_overrides(base_overrides, scale)
                out = mix.route_batch(
                    dets=batch.dets,
                    syndromes_x=sx,
                    syndromes_z=sz,
                    rng=rng,
                    context=context_payload,
                    weight_overrides=overrides,
                    dem_teacher=dem_teacher,
                )
                totals[out["which"]] = totals.get(out["which"], 0) + 1
                dem_key = out.get("dem_teacher")
                if dem_key:
                    totals[dem_key] = totals.get(dem_key, 0) + 1

                true_obs = getattr(batch, "obs", None)
                true_arr = None
                if true_obs is not None:
                    true_arr = np.asarray(true_obs, dtype=np.uint8)
                    if true_arr.ndim == 1:
                        true_arr = true_arr[np.newaxis, :]
                    true_chunks.append(true_arr)

                pred_obs = None
                if out["which"] == "lsd":
                    ex = out.get("ex")
                    ez = out.get("ez")
                    if hasattr(code, "data_to_observables"):
                        pred_obs = code.data_to_observables(ex, ez)
                elif out["which"].startswith("mwpm"):
                    cx = out.get("cx")
                    cz = out.get("cz")
                    if hasattr(code, "data_to_observables"):
                        pred_obs = code.data_to_observables(cx, cz)
                elif out["which"] == "mwpf":
                    pred_obs = out.get("pred_obs")

                pred_arr = None
                if pred_obs is not None:
                    pred_arr = np.asarray(pred_obs, dtype=np.uint8)
                    if pred_arr.ndim == 1:
                        pred_arr = pred_arr[np.newaxis, :]
                    pred_chunks.append(pred_arr)

                dem_pred = out.get("dem_pred_obs")
                if dem_pred is not None:
                    dem_arr = np.asarray(dem_pred, dtype=np.uint8)
                    if dem_arr.ndim == 1:
                        dem_arr = dem_arr[np.newaxis, :]
                    dem_pred_chunks.append(dem_arr)

                if bandit is not None and bandit_ctx is not None and true_arr is not None and pred_arr is not None:
                    if true_arr.shape == pred_arr.shape:
                        reward = 1.0 if np.all(true_arr == pred_arr) else 0.0
                        bandit.update(bandit_ctx, reward)

            dt = time.time() - t0
            true_accum = np.concatenate(true_chunks, axis=0) if true_chunks else None
            pred_accum = np.concatenate(pred_chunks, axis=0) if pred_chunks else None
            if true_accum is not None and pred_accum is not None and true_accum.shape == pred_accum.shape:
                ler = logical_error_rate(true_accum, pred_accum)
            else:
                samples = int(true_accum.shape[0]) if true_accum is not None else 0
                ler = LEResult(None, None, samples, notes="obs unavailable")
            line = summary_line(family, d, args.batches, args.shots_per_batch, ler, dt, totals)

            if dem_teacher is not None:
                dem_accum = np.concatenate(dem_pred_chunks, axis=0) if dem_pred_chunks else None
                if (
                    true_accum is not None
                    and dem_accum is not None
                    and true_accum.shape == dem_accum.shape
                ):
                    dem_ler = logical_error_rate(true_accum, dem_accum)
                    if dem_ler.ler_mean is not None:
                        line += f" | LER_dem={dem_ler.ler_mean:.3e}"
                    else:
                        line += " | LER_dem=NA"
                else:
                    line += " | LER_dem=NA"

            print(line)

            if bandit is not None and args.rl_state:
                state = {"A": bandit.A.tolist(), "b": bandit.b.tolist()}
                _dump_bandit_state(pathlib.Path(args.rl_state), state)
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
