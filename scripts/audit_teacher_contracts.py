#!/usr/bin/env python
"""Audit teacher output contracts for MGHD per-qubit supervision.

Reports, per teacher and (distance, p):
- output kind
- shape validity
- parity validity
- decode exception rate

This is a non-training diagnostic used to gate teacher mixes.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from mghd.decoders.lsd_teacher import LSDTeacher
from mghd.decoders.mwpf_teacher import MWPFTeacher
from mghd.decoders.mwpm_ctx import MWPMatchingContext
from mghd.decoders.nvqldpc_teacher import NvQldpcTeacher
from mghd.qpu.adapters.surface_sampler import sample_round
from mghd.qpu.adapters.surface_sampler import split_components_for_side


OUTPUT_KIND = {
    "lsd": "per_qubit",
    "mwpm": "per_qubit",
    "nvqldpc": "per_qubit",
    "oracle": "per_qubit",
    "mwpf": "fault_ids",
    "tensor_network_decoder": "observable_prob",
}


@dataclass
class TeacherStats:
    output_kind: str
    available: bool = True
    unavailable_reason: str | None = None
    components_total: int = 0
    shape_valid: int = 0
    parity_valid: int = 0
    side_valid: int = 0
    decode_exceptions: int = 0

    def to_dict(self) -> dict[str, Any]:
        comp = max(1, self.components_total)
        return {
            "output_kind": self.output_kind,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "components_total": self.components_total,
            "shape_valid": self.shape_valid,
            "shape_valid_rate": self.shape_valid / comp if self.components_total else None,
            "parity_valid": self.parity_valid,
            "parity_valid_rate": self.parity_valid / comp if self.components_total else None,
            "side_valid": self.side_valid,
            "side_valid_rate": self.side_valid / comp if self.components_total else None,
            "decode_exceptions": self.decode_exceptions,
            "decode_exception_rate": self.decode_exceptions / comp if self.components_total else None,
        }


def _parse_csv_int(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


def _parse_csv_float(spec: str) -> list[float]:
    return [float(x) for x in spec.split(",") if x.strip()]


def _shape_ok(bits: np.ndarray, n: int) -> bool:
    return bool(bits.ndim == 1 and bits.shape[0] == n)


def _parity_ok(H_sub: np.ndarray, bits: np.ndarray, synd_bits: np.ndarray) -> bool:
    lhs = (H_sub.astype(np.uint8) @ bits.astype(np.uint8)) & 1
    rhs = synd_bits.astype(np.uint8) & 1
    return bool(np.array_equal(lhs, rhs))


def _build_teachers(sample: dict[str, Any], enable_nvqldpc: bool) -> tuple[Any, Any, Any, Any]:
    Hx = np.asarray(sample["Hx"], dtype=np.uint8)
    Hz = np.asarray(sample["Hz"], dtype=np.uint8)
    mx = int(Hx.shape[0])
    dets_per_fault: list[list[int]] = []
    for col in range(Hx.shape[1]):
        dets = []
        dets.extend(np.flatnonzero(Hx[:, col]).tolist())
        dets.extend((mx + np.flatnonzero(Hz[:, col])).tolist())
        dets_per_fault.append(dets)
    code = SimpleNamespace(
        Hx=Hx,
        Hz=Hz,
        detectors_per_fault=dets_per_fault,
        fault_weights=[1.0] * Hx.shape[1],
        num_detectors=Hx.shape[0] + Hz.shape[0],
        n=Hx.shape[1],
    )
    mwpf = MWPFTeacher(code)
    mwpm = MWPMatchingContext()
    lsd = LSDTeacher(Hx, Hz)
    nvq = NvQldpcTeacher(Hx, Hz) if enable_nvqldpc else None
    return mwpf, mwpm, lsd, nvq


def _initialize_stats() -> dict[str, TeacherStats]:
    return {
        "lsd": TeacherStats(output_kind=OUTPUT_KIND["lsd"]),
        "mwpm": TeacherStats(output_kind=OUTPUT_KIND["mwpm"]),
        "nvqldpc": TeacherStats(output_kind=OUTPUT_KIND["nvqldpc"]),
        "mwpf": TeacherStats(output_kind=OUTPUT_KIND["mwpf"]),
        "oracle": TeacherStats(output_kind=OUTPUT_KIND["oracle"]),
    }


def _mark_unavailable(stats: dict[str, TeacherStats], key: str, reason: str) -> None:
    stats[key].available = False
    stats[key].unavailable_reason = reason


def _env_for_sampler(sampler: str) -> None:
    sampler = sampler.strip().lower()
    if sampler == "synthetic":
        os.environ["MGHD_SYNTHETIC"] = "1"
        os.environ.pop("MGHD_SAMPLER", None)
    elif sampler == "stim":
        os.environ["MGHD_SAMPLER"] = "stim"
        os.environ.pop("MGHD_SYNTHETIC", None)
    else:
        os.environ.pop("MGHD_SAMPLER", None)
        os.environ.pop("MGHD_SYNTHETIC", None)


def audit_contracts(
    *,
    distances: list[int],
    p_values: list[float],
    shots: int,
    seed: int,
    qpu_profile: str | None,
    sampler: str,
    enable_nvqldpc: bool,
) -> dict[str, Any]:
    _env_for_sampler(sampler)
    report: dict[str, Any] = {"runs": []}

    for d in distances:
        for p in p_values:
            stats = _initialize_stats()
            bootstrap = sample_round(d=d, p=p, seed=seed, profile_path=qpu_profile)

            try:
                mwpf_teacher, mwpm_ctx, lsd_teacher, nvq_teacher = _build_teachers(
                    bootstrap, enable_nvqldpc
                )
            except Exception as exc:
                raise RuntimeError(f"Teacher initialization failed for d={d}, p={p}: {exc}") from exc

            if nvq_teacher is None:
                _mark_unavailable(stats, "nvqldpc", "disabled or unavailable")

            for shot_idx in range(shots):
                sample = sample_round(
                    d=d,
                    p=p,
                    seed=seed + shot_idx,
                    profile_path=qpu_profile,
                )
                Hx = np.asarray(sample["Hx"], dtype=np.uint8)
                Hz = np.asarray(sample["Hz"], dtype=np.uint8)
                synX = np.asarray(sample["synX"], dtype=np.uint8)
                synZ = np.asarray(sample["synZ"], dtype=np.uint8)

                dets_global = np.concatenate([synZ[None, :], synX[None, :]], axis=1).astype(np.uint8)
                fault_ids = None
                try:
                    out = mwpf_teacher.decode_batch(dets_global)
                    fid_arr = np.asarray(out.get("fault_ids"), dtype=np.int32)
                    if fid_arr.ndim == 2 and fid_arr.shape[0] >= 1:
                        fault_ids = fid_arr[0]
                except Exception:
                    stats["mwpf"].decode_exceptions += 1

                ex_lsd = ez_lsd = None
                try:
                    ex_arr, ez_arr = lsd_teacher.decode_batch_xz(
                        syndromes_x=synX[None, :],
                        syndromes_z=synZ[None, :],
                    )
                    ex_lsd, ez_lsd = ex_arr[0], ez_arr[0]
                except Exception:
                    stats["lsd"].decode_exceptions += 1

                ex_nq = ez_nq = None
                if nvq_teacher is not None:
                    try:
                        ex_arr, ez_arr = nvq_teacher.decode_batch_xz(
                            syndromes_x=synX[None, :],
                            syndromes_z=synZ[None, :],
                        )
                        ex_nq, ez_nq = ex_arr[0], ez_arr[0]
                    except Exception:
                        stats["nvqldpc"].decode_exceptions += 1

                oracle_ex = sample.get("ex_glob")
                oracle_ez = sample.get("ez_glob")

                for side in ("Z", "X"):
                    comps = split_components_for_side(
                        side=side,
                        Hx=Hx,
                        Hz=Hz,
                        synZ=synZ,
                        synX=synX,
                        coords_q=sample["coords_q"],
                        coords_c=sample["coords_c"],
                    )

                    for comp in comps:
                        H_sub = np.asarray(comp["H_sub"], dtype=np.uint8)
                        synd_bits = np.asarray(comp["synd_bits"], dtype=np.uint8)
                        qubit_indices = np.asarray(comp["qubit_indices"], dtype=np.int32)
                        nq = int(H_sub.shape[1])

                        for key in ("lsd", "mwpm", "nvqldpc", "mwpf", "oracle"):
                            if not stats[key].available:
                                continue
                            stats[key].components_total += 1

                        # LSD
                        if stats["lsd"].available:
                            try:
                                if ex_lsd is None or ez_lsd is None:
                                    raise RuntimeError("lsd outputs unavailable")
                                bits_global = ez_lsd if side == "Z" else ex_lsd
                                side_ok = bool(qubit_indices.size == 0 or bits_global.size > qubit_indices.max())
                                bits = (
                                    bits_global[qubit_indices].astype(np.uint8)
                                    if side_ok
                                    else np.zeros(nq, dtype=np.uint8)
                                )
                                if side_ok:
                                    stats["lsd"].side_valid += 1
                                if _shape_ok(bits, nq):
                                    stats["lsd"].shape_valid += 1
                                    if _parity_ok(H_sub, bits, synd_bits):
                                        stats["lsd"].parity_valid += 1
                            except Exception:
                                stats["lsd"].decode_exceptions += 1

                        # MWPM
                        if stats["mwpm"].available:
                            try:
                                bits, _ = mwpm_ctx.decode(H_sub, synd_bits, side)
                                bits = np.asarray(bits, dtype=np.uint8)
                                stats["mwpm"].side_valid += 1
                                if _shape_ok(bits, nq):
                                    stats["mwpm"].shape_valid += 1
                                    if _parity_ok(H_sub, bits, synd_bits):
                                        stats["mwpm"].parity_valid += 1
                            except Exception:
                                stats["mwpm"].decode_exceptions += 1

                        # NVQLDPC
                        if stats["nvqldpc"].available:
                            try:
                                if ex_nq is None or ez_nq is None:
                                    raise RuntimeError("nvqldpc outputs unavailable")
                                bits_global = ez_nq if side == "Z" else ex_nq
                                side_ok = bool(qubit_indices.size == 0 or bits_global.size > qubit_indices.max())
                                bits = (
                                    bits_global[qubit_indices].astype(np.uint8)
                                    if side_ok
                                    else np.zeros(nq, dtype=np.uint8)
                                )
                                if side_ok:
                                    stats["nvqldpc"].side_valid += 1
                                if _shape_ok(bits, nq):
                                    stats["nvqldpc"].shape_valid += 1
                                    if _parity_ok(H_sub, bits, synd_bits):
                                        stats["nvqldpc"].parity_valid += 1
                            except Exception:
                                stats["nvqldpc"].decode_exceptions += 1

                        # MWPF fault-id -> local bits projection
                        if stats["mwpf"].available:
                            try:
                                if fault_ids is None:
                                    raise RuntimeError("fault_ids unavailable")
                                bits = np.zeros(nq, dtype=np.uint8)
                                valid_ids = fault_ids[fault_ids >= 0]
                                if valid_ids.size:
                                    mask = np.isin(qubit_indices, valid_ids)
                                    bits[mask] = 1
                                stats["mwpf"].side_valid += 1
                                if _shape_ok(bits, nq):
                                    stats["mwpf"].shape_valid += 1
                                    if _parity_ok(H_sub, bits, synd_bits):
                                        stats["mwpf"].parity_valid += 1
                            except Exception:
                                stats["mwpf"].decode_exceptions += 1

                        # Oracle
                        if stats["oracle"].available:
                            try:
                                bits_global = oracle_ex if side == "Z" else oracle_ez
                                if bits_global is None:
                                    raise RuntimeError("oracle bits unavailable")
                                side_ok = bool(qubit_indices.size == 0 or bits_global.size > qubit_indices.max())
                                bits = (
                                    bits_global[qubit_indices].astype(np.uint8)
                                    if side_ok
                                    else np.zeros(nq, dtype=np.uint8)
                                )
                                if side_ok:
                                    stats["oracle"].side_valid += 1
                                if _shape_ok(bits, nq):
                                    stats["oracle"].shape_valid += 1
                                    if _parity_ok(H_sub, bits, synd_bits):
                                        stats["oracle"].parity_valid += 1
                            except Exception:
                                stats["oracle"].decode_exceptions += 1

            run_entry = {
                "distance": d,
                "p": p,
                "shots": shots,
                "teachers": {key: value.to_dict() for key, value in stats.items()},
            }
            report["runs"].append(run_entry)

    return report


def _recommended_policy(
    report: dict[str, Any], parity_threshold: float, exception_rate_max: float
) -> dict[str, Any]:
    aggregate: dict[str, dict[str, float]] = {}
    for run in report.get("runs", []):
        for teacher, entry in run.get("teachers", {}).items():
            agg = aggregate.setdefault(
                teacher,
                {
                    "components": 0.0,
                    "parity_valid": 0.0,
                    "decode_exceptions": 0.0,
                    "available": 1.0,
                    "output_kind_fault_ids": 0.0,
                },
            )
            agg["components"] += float(entry.get("components_total", 0))
            agg["parity_valid"] += float(entry.get("parity_valid", 0))
            agg["decode_exceptions"] += float(entry.get("decode_exceptions", 0))
            if not bool(entry.get("available", True)):
                agg["available"] = 0.0
            if entry.get("output_kind") != "per_qubit":
                agg["output_kind_fault_ids"] = 1.0

    recommended: list[str] = []
    details: dict[str, Any] = {}
    for teacher, agg in aggregate.items():
        components = max(1.0, agg["components"])
        parity_rate = agg["parity_valid"] / components
        exc_rate = agg["decode_exceptions"] / components
        per_qubit = agg["output_kind_fault_ids"] == 0.0
        available = agg["available"] > 0.0
        eligible = available and per_qubit and parity_rate >= parity_threshold and exc_rate <= exception_rate_max
        if eligible:
            recommended.append(teacher)
        details[teacher] = {
            "available": available,
            "per_qubit_output": per_qubit,
            "parity_valid_rate": parity_rate,
            "decode_exception_rate": exc_rate,
            "eligible_for_per_qubit_supervision": eligible,
        }

    mix = {}
    if "oracle" in recommended:
        mix["oracle"] = 1.0
    else:
        candidates = [t for t in recommended if t in {"lsd", "nvqldpc", "mwpm"}]
        if candidates:
            weight = 1.0 / len(candidates)
            for name in candidates:
                mix[name] = weight

    return {
        "thresholds": {
            "parity_valid_rate_min": parity_threshold,
            "decode_exception_rate_max": exception_rate_max,
        },
        "details": details,
        "recommended_teachers": recommended,
        "recommended_teacher_mix": mix,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit teacher label contracts.")
    parser.add_argument("--distances", type=str, default="3,5,7")
    parser.add_argument("--p-values", type=str, default="0.001,0.003,0.005")
    parser.add_argument("--shots", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampler", choices=["cudaq", "synthetic", "stim"], default="cudaq")
    parser.add_argument("--qpu-profile", type=str, default=None)
    parser.add_argument("--enable-nvqldpc", action="store_true")
    parser.add_argument("--parity-threshold", type=float, default=0.999)
    parser.add_argument("--exception-rate-max", type=float, default=0.01)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("teacher_contract_report.json"),
        help="Output JSON report path.",
    )
    args = parser.parse_args()

    distances = _parse_csv_int(args.distances)
    p_values = _parse_csv_float(args.p_values)

    report = audit_contracts(
        distances=distances,
        p_values=p_values,
        shots=args.shots,
        seed=args.seed,
        qpu_profile=args.qpu_profile,
        sampler=args.sampler,
        enable_nvqldpc=bool(args.enable_nvqldpc),
    )
    report["config"] = {
        "distances": distances,
        "p_values": p_values,
        "shots": args.shots,
        "seed": args.seed,
        "sampler": args.sampler,
        "qpu_profile": args.qpu_profile,
        "output_kind_registry": OUTPUT_KIND,
    }
    report["policy"] = _recommended_policy(
        report,
        parity_threshold=float(args.parity_threshold),
        exception_rate_max=float(args.exception_rate_max),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote teacher contract report to {args.output}")


if __name__ == "__main__":
    main()
