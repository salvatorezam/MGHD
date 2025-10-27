# NOTE: Initialize CUDA/CUDA-Q only in main().
from __future__ import annotations

import argparse
import json
import os
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from mghd.codes.registry import get_code
from mghd.core.core import pack_cluster
from mghd.tad import weighting as tad_weighting
from mghd.tad import context as tad_context
from mghd.codes.qpu_profile import load_qpu_profile
from mghd.decoders.ensemble import _check_parity_coset_valid
from mghd.decoders.lsd_teacher import LSDTeacher
from mghd.decoders.mwpf_ctx import MWPFContext
from mghd.decoders.mwpf_teacher import MWPFTeacher
from mghd.decoders.mwpm_ctx import MWPMatchingContext
from mghd.qpu.adapters.garnet_adapter import sample_round, split_components_for_side


@dataclass
class TeacherOutput:
    bits: np.ndarray
    weight: int
    teacher: str
    valid: bool


def short_sha(*objs) -> str:
    h = hashlib.sha1()
    for o in objs:
        if isinstance(o, (bytes, bytearray)):
            h.update(o)
        elif isinstance(o, str):
            h.update(o.encode())
        else:
            h.update(repr(o).encode())
    return h.hexdigest()[:8]


def _build_schedule_ir(context_source: str, code_obj: object):
    import sys
    native = None
    for attr in ("native_circuit", "reference_circuit", "circuit", "qc", "quantum_circuit"):
        if hasattr(code_obj, attr):
            native = getattr(code_obj, attr)
            if native is not None:
                break
    if native is None:
        return []
    try:
        if context_source == "qiskit" and "qiskit" in sys.modules:
            from mghd.qpu.adapters import qiskit_adapter
            return qiskit_adapter.to_schedule_ir(native)
        if context_source == "cirq" and "cirq" in sys.modules:
            from mghd.qpu.adapters import cirq_adapter  # type: ignore
            return cirq_adapter.to_schedule_ir(native)
        if context_source == "cudaq" and "cudaq" in sys.modules:
            from mghd.qpu.adapters import cudaq_adapter  # type: ignore
            return cudaq_adapter.to_schedule_ir(native)
    except Exception:
        pass
    return []


def parse_families(spec: str) -> List[str]:
    if not spec:
        return ["surface"]
    families = [part.strip() for part in spec.split(",") if part.strip()]
    return families or ["surface"]


def parse_distances(spec: str) -> List[int]:
    spec = (spec or "").strip()
    if not spec:
        return []
    if "," in spec:
        return [int(x.strip()) for x in spec.split(",") if x.strip()]
    m = re.fullmatch(r"(\d+)-(\d+):(\d+)", spec)
    if m:
        lo, hi, step = map(int, m.groups())
        return list(range(lo, hi + 1, step))
    return [int(spec)]


def parse_teacher_mix(spec: str) -> Dict[str, float]:
    weights = {"mwpf": 1.0, "mwpm": 0.0, "lsd": 0.0}
    if not spec:
        return weights
    for chunk in spec.split(","):
        if "=" not in chunk:
            continue
        name, value = chunk.split("=", 1)
        try:
            weights[name.strip().lower()] = float(value)
        except ValueError:
            continue
    # Ensure non-negative weights
    for key in weights:
        weights[key] = max(0.0, weights[key])
    if sum(weights.values()) == 0.0:
        weights["mwpf"] = 1.0
    return weights


def _parity_valid(bits: np.ndarray, synd_bits: np.ndarray, H_sub: np.ndarray) -> bool:
    return _check_parity_coset_valid(bits & 1, synd_bits, H_sub)


def _select_teacher(
    outputs: Dict[str, TeacherOutput],
    mix: Dict[str, float],
    rng: np.random.Generator,
) -> TeacherOutput:
    weighted = [
        (name, out, mix[name])
        for name, out in outputs.items()
        if mix.get(name, 0.0) > 0 and out.valid
    ]
    total = sum(weight for _, _, weight in weighted)
    if weighted and total > 0.0:
        r = float(rng.random() * total)
        acc = 0.0
        for name, out, weight in weighted:
            acc += weight
            if r <= acc:
                return out
    # Fallback: choose valid output with minimal weight, then any output.
    valids = [out for out in outputs.values() if out.valid]
    if valids:
        return min(valids, key=lambda o: (o.weight, o.teacher))
    return next(iter(outputs.values()))


def run(args) -> None:
    rng = np.random.default_rng(args.seed)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    families = parse_families(args.families)
    distances = parse_distances(args.distances)
    teacher_mix = parse_teacher_mix(args.teacher_mix)

    if not distances:
        raise ValueError("No distances parsed; provide --distances")
    if not args.ps:
        raise ValueError("No physical error rates provided via --ps")

    # MWPF teacher will be initialized per distance with code context
    mwpm_ctx = MWPMatchingContext()

    shots_per = args.shots_per_grid
    padN, padE, padS = args.N_max, args.E_max, args.S_max
    manifest: List[Dict[str, object]] = []

    print(f"Generating crops with MGHD_SYNTHETIC={os.getenv('MGHD_SYNTHETIC', '0')}")

    # Optionally compute TAD context and overrides per family/d using QPU profile
    qpu_profile = None
    ctx_vec: np.ndarray | None = None
    llr_overrides: np.ndarray | None = None
    if args.qpu_profile and args.context_source != "none":
        try:
            qpu_profile = load_qpu_profile(args.qpu_profile)
        except Exception:
            qpu_profile = None

    for family in families:
        if family != "surface":
            raise NotImplementedError(
                f"Family '{family}' is not yet supported by make_cluster_crops"
            )

        for d in distances:
            code = get_code(family, distance=d)
            # Initialize MWPF teacher for this code distance if requested
            mwpf_teacher = None
            if teacher_mix.get("mwpf", 0.0) > 0:
                try:
                    mwpf_teacher = MWPFTeacher(code)
                except Exception as exc:
                    print(f"Warning: MWPFTeacher unavailable ({exc}); disabling MWPF mix")
                    teacher_mix["mwpf"] = 0.0
                    mwpf_teacher = None
            # Build schedule IR and TAD features once per code distance
            ctx_vec = None
            llr_overrides = None
            if qpu_profile is not None:
                schedule_ir = _build_schedule_ir(args.context_source, code)
                try:
                    n_qubits = getattr(code, "n", None) or int(code.Hx.shape[1])
                except Exception:
                    n_qubits = 0
                try:
                    weight_maps = tad_weighting.schedule_to_weight_maps(schedule_ir, qpu_profile, n_qubits)
                    feats = tad_weighting.feature_vector(schedule_ir)
                    sorted_gates = sorted(feats.get("gate_hist", {}).keys())
                    gate_vocab = {gate: idx for idx, gate in enumerate(sorted_gates)}
                    ctx_vec = tad_context.context_vector(feats, gate_vocab)
                    # Base LLR override per qubit (use unit scale)
                    llr = np.zeros(int(n_qubits), dtype=np.float32)
                    for layer in (weight_maps.get("w_qubit", {}) or {}).values():
                        for q, w in layer.items():
                            idx = int(q)
                            if 0 <= idx < llr.size:
                                llr[idx] += float(w)
                    llr_overrides = llr if llr.size else None
                except Exception:
                    ctx_vec = None
                    llr_overrides = None
            lsd_teacher = None
            if teacher_mix.get("lsd", 0.0) > 0:
                try:
                    lsd_teacher = LSDTeacher(code.Hx, code.Hz)
                except Exception as exc:
                    print(f"Warning: LSDTeacher unavailable ({exc}); disabling LSD mix")
                    teacher_mix["lsd"] = 0.0
                    lsd_teacher = None

            for p in args.ps:
                print(f"Processing family={family}, d={d}, p={p:.5f}")
                shard_items: List[Dict[str, object]] = []
                for _ in range(shots_per):
                    seed = int(rng.integers(0, 2**31 - 1))
                    sample = sample_round(d=d, p=p, seed=seed, profile_path=args.qpu_profile if args.qpu_profile else None)

                    lsd_preds: Dict[str, np.ndarray] = {}
                    if lsd_teacher is not None:
                        try:
                            ex_glob, ez_glob = lsd_teacher.decode_batch_xz(
                                syndromes_x=sample["synX"][np.newaxis, :],
                                syndromes_z=sample["synZ"][np.newaxis, :],
                                llr_overrides=llr_overrides,
                            )
                            lsd_preds["X"] = ex_glob[0].astype(np.uint8)
                            lsd_preds["Z"] = ez_glob[0].astype(np.uint8)
                        except Exception as exc:
                            print(f"Warning: LSD decode failed ({exc}); disabling LSD")
                            teacher_mix["lsd"] = 0.0
                            lsd_teacher = None
                            lsd_preds.clear()

                    for side in ("Z", "X"):
                        components = split_components_for_side(
                            side=side,
                            Hx=sample["Hx"],
                            Hz=sample["Hz"],
                            synZ=sample["synZ"],
                            synX=sample["synX"],
                            coords_q=sample["coords_q"],
                            coords_c=sample["coords_c"],
                        )

                        for comp in components:
                            H_sub = comp["H_sub"]
                            synd_bits = comp["synd_bits"]
                            qubit_indices = comp["qubit_indices"]

                            outputs: Dict[str, TeacherOutput] = {}

                            if mwpf_teacher is not None:
                                # Use global MWPF fault_ids and map to local bits
                                dets_global = np.concatenate([
                                    sample["synX"][np.newaxis, :].astype(np.uint8),
                                    sample["synZ"][np.newaxis, :].astype(np.uint8),
                                ], axis=1)
                                # Per-fault scaling from LLR overrides if available
                                mwpf_scale = None
                                if llr_overrides is not None:
                                    try:
                                        probs = 1.0 / (1.0 + np.exp(llr_overrides))
                                        scale_full = np.clip(probs / 0.5, 0.1, 10.0)
                                        mwpf_scale = {int(i): float(s) for i, s in enumerate(scale_full)}
                                    except Exception:
                                        mwpf_scale = None
                                try:
                                    out_mwpf = mwpf_teacher.decode_batch(dets_global, mwpf_scale=mwpf_scale)
                                    fid_arr = np.asarray(out_mwpf.get("fault_ids"), dtype=np.int32)
                                    bits_pf_local = np.zeros(H_sub.shape[1], dtype=np.uint8)
                                    if fid_arr.ndim == 2 and fid_arr.shape[0] >= 1:
                                        valid_ids = fid_arr[0][fid_arr[0] >= 0]
                                        if valid_ids.size:
                                            mask = np.isin(qubit_indices, valid_ids)
                                            bits_pf_local[mask] = 1
                                    outputs["mwpf"] = TeacherOutput(
                                        bits=bits_pf_local,
                                        weight=int(bits_pf_local.sum()),
                                        teacher="mwpf",
                                        valid=_parity_valid(bits_pf_local, synd_bits, H_sub),
                                    )
                                except Exception:
                                    pass

                            bits_pm, w_pm = mwpm_ctx.decode(H_sub, synd_bits, side)
                            outputs["mwpm"] = TeacherOutput(
                                bits=bits_pm.astype(np.uint8),
                                weight=int(w_pm),
                                teacher="mwpm",
                                valid=_parity_valid(bits_pm, synd_bits, H_sub),
                            )

                            if teacher_mix.get("lsd", 0.0) > 0 and side in lsd_preds:
                                bits_global = lsd_preds[side]
                                if qubit_indices.size and bits_global.size > qubit_indices.max():
                                    bits_local = bits_global[qubit_indices].astype(np.uint8)
                                    outputs["lsd"] = TeacherOutput(
                                        bits=bits_local,
                                        weight=int(bits_local.sum()),
                                        teacher="lsd",
                                        valid=_parity_valid(bits_local, synd_bits, H_sub),
                                    )

                            chosen = _select_teacher(outputs, teacher_mix, rng)

                            packed = pack_cluster(
                                H_sub=H_sub,
                                xy_qubit=comp["xy_qubit"],
                                xy_check=comp["xy_check"],
                                synd_Z_then_X_bits=synd_bits,
                                k=int(comp["k"]),
                                r=int(comp["r"]),
                                bbox_xywh=tuple(int(v) for v in comp["bbox_xywh"]),
                                kappa_stats=comp.get("kappa_stats", {}),
                                y_bits_local=chosen.bits,
                                side=side,
                                d=d,
                                p=p,
                                seed=seed,
                                N_max=padN,
                                E_max=padE,
                                S_max=padS,
                                g_extra=ctx_vec,
                            )

                            item = {
                                "x_nodes": packed.x_nodes.numpy(),
                                "node_mask": packed.node_mask.numpy(),
                                "node_type": packed.node_type.numpy(),
                                "edge_index": packed.edge_index.numpy(),
                                "edge_attr": packed.edge_attr.numpy(),
                                "edge_mask": packed.edge_mask.numpy(),
                                "seq_idx": packed.seq_idx.numpy(),
                                "seq_mask": packed.seq_mask.numpy(),
                                "g_token": packed.g_token.numpy(),
                                "y_bits": packed.y_bits.numpy(),
                                "meta": {
                                    **packed.meta.__dict__,
                                    "side": side,
                                    "family": family,
                                    "distance": d,
                                    "p": p,
                                },
                                "H_sub": H_sub.astype(np.uint8),
                                "idx_data_local": np.arange(H_sub.shape[1], dtype=np.int32),
                                "idx_check_local": np.arange(H_sub.shape[0], dtype=np.int32),
                                "idx_data_global": qubit_indices.astype(np.int32),
                                "teacher": chosen.teacher,
                                "teacher_weight": int(chosen.weight),
                                "teacher_valid": bool(chosen.valid),
                                "teacher_matched_local_ml": False,
                            }
                            shard_items.append(item)

                shard_sha = short_sha(family, d, f"{p:.5f}", args.seed, len(shard_items))
                outp = out_root / f"{family}_d{d}_p{p:.5f}_seed{args.seed}_{shard_sha}.npz"
                np.savez_compressed(outp, packed=np.array(shard_items, dtype=object))
                manifest.append(
                    {
                        "file": str(outp),
                        "family": family,
                        "d": d,
                        "p": p,
                        "seed": args.seed,
                        "count": len(shard_items),
                        "sha": shard_sha,
                    }
                )
                print(json.dumps({"written": str(outp), "count": len(shard_items), "sha": shard_sha}))

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", type=str, default="surface", help="Comma-separated code families")
    ap.add_argument("--distances", type=str, default="3", help="Distance spec (e.g., '3,5,7' or '3-9:2')")
    ap.add_argument("--ps", type=float, nargs="+", required=True, help="Physical error rates")
    ap.add_argument("--shots-per-grid", type=int, default=64)
    ap.add_argument("--teacher-mix", type=str, default="mwpf=1.0,mwpm=0.0,lsd=0.0")
    ap.add_argument("--qpu-profile", type=str)
    ap.add_argument("--context-source", type=str, default="qiskit")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--N_max", type=int, default=512)
    ap.add_argument("--E_max", type=int, default=4096)
    ap.add_argument("--S_max", type=int, default=512)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
