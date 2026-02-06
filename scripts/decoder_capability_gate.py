#!/usr/bin/env python
"""Probe decoder availability/capabilities in the active environment.

This script is intentionally dependency-tolerant and never raises on decoder
initialization failures; instead it records explicit reasons in the output JSON.
"""

from __future__ import annotations

import argparse
import importlib
import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DecoderSpec:
    name: str
    output_kind: str
    comparison_class: str
    smoke_kwargs: dict[str, Any]


DECODER_SPECS: tuple[DecoderSpec, ...] = (
    DecoderSpec(
        name="single_error_lut",
        output_kind="per_qubit",
        comparison_class="per_qubit",
        smoke_kwargs={},
    ),
    DecoderSpec(
        name="nv-qldpc-decoder",
        output_kind="per_qubit",
        comparison_class="per_qubit",
        smoke_kwargs={"use_sparsity": True},
    ),
    DecoderSpec(
        name="tensor_network_decoder",
        output_kind="observable_prob",
        comparison_class="observable_prob",
        smoke_kwargs={
            "logical_obs": np.array([[1, 0, 1]], dtype=np.uint8),
            "noise_model": [0.01, 0.01, 0.01],
            "device": "cpu",
        },
    ),
)


def _check_module(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
        return {"available": True, "error": None, "module_file": getattr(module, "__file__", None)}
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"available": False, "error": f"{type(exc).__name__}: {exc}", "module_file": None}


def _decode_smoke(decoder: Any, n_checks: int) -> tuple[bool, str | None]:
    try:
        if hasattr(decoder, "decode"):
            result = decoder.decode([0.0] * n_checks)
            payload = getattr(result, "result", None)
            return True, f"type={type(payload).__name__}"
        return False, "decoder object has no decode()"
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"{type(exc).__name__}: {exc}"


def _probe_decoders() -> dict[str, Any]:
    out: dict[str, Any] = {
        "cudaq_qec_importable": False,
        "cudaq_qec_module": None,
        "decoder_matrix": {},
        "python_modules": {},
    }

    required_modules = ("quimb", "autoray", "cotengra", "cupy")
    for module_name in required_modules:
        out["python_modules"][module_name] = _check_module(module_name)

    try:
        import cudaq_qec as qec  # type: ignore

        out["cudaq_qec_importable"] = True
        out["cudaq_qec_module"] = getattr(qec, "__file__", None)
    except Exception as exc:  # pragma: no cover - environment dependent
        out["cudaq_qec_error"] = f"{type(exc).__name__}: {exc}"
        return out

    parity = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    for spec in DECODER_SPECS:
        entry = {
            "output_kind": spec.output_kind,
            "comparison_class": spec.comparison_class,
            "available": False,
            "init_error": None,
            "decode_smoke_ok": False,
            "decode_smoke_note": None,
        }
        try:
            decoder = qec.get_decoder(spec.name, parity, **spec.smoke_kwargs)  # type: ignore[attr-defined]
            entry["available"] = True
            smoke_ok, note = _decode_smoke(decoder, parity.shape[0])
            entry["decode_smoke_ok"] = smoke_ok
            entry["decode_smoke_note"] = note
        except Exception as exc:  # pragma: no cover - environment dependent
            entry["init_error"] = f"{type(exc).__name__}: {exc}"
        out["decoder_matrix"][spec.name] = entry

    try:
        plugin_mod = importlib.import_module("cudaq_qec.plugins.decoders")
        out["decoder_plugins"] = sorted(name for name in dir(plugin_mod) if not name.startswith("_"))
    except Exception as exc:  # pragma: no cover - environment dependent
        out["decoder_plugins_error"] = f"{type(exc).__name__}: {exc}"

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate decoder capability matrix.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("decoder_capability_matrix.json"),
        help="Path to write capability matrix JSON.",
    )
    args = parser.parse_args()

    payload = {
        "script": "decoder_capability_gate.py",
        "matrix": _probe_decoders(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote decoder capability matrix to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Capability probe failed: {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        raise
