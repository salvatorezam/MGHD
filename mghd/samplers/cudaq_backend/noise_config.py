from __future__ import annotations

"""Canonical noise-model resolution for MGHD CUDA-Q sampling/evaluation.

This module centralizes noise model naming, aliasing, default probabilities,
and lambda-scale mapping so train/eval/sampler metadata are consistent.
"""

from dataclasses import dataclass
import os
from typing import Any

import numpy as np


NOISE_MODEL_VERSION = "mghd_v4.0"

_NOISE_MODEL_ALIASES = {
    "generic": "circuit_standard",
    "generic_cl": "circuit_standard",
    "generic-circuit": "circuit_standard",
    "circuit": "circuit_standard",
    "circuit_level": "circuit_standard",
    "standard": "circuit_standard",
    "augmented": "circuit_augmented",
    "code-capacity": "code_capacity",
    "code_capacity": "code_capacity",
    "phenom": "phenomenological",
    "phenomenological": "phenomenological",
    "garnet": "circuit_standard",
    "auto": "circuit_standard",
}


def _clip_prob(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return float(default)


def _scaled_prob(base_p: float, scale: float) -> float:
    p = _clip_prob(base_p)
    scl = max(0.0, float(scale))
    return _clip_prob(1.0 - (1.0 - p) ** scl)


def _norm_model(name: str | None) -> str:
    key = str(name or "").strip().lower() or "circuit_standard"
    return _NOISE_MODEL_ALIASES.get(key, key)


@dataclass(frozen=True)
class CanonicalNoiseSpec:
    model_name: str
    model_version: str
    noise_ramp: str
    lambda_scale: float
    p_data: float
    p_meas: float
    p_1q: float
    p_2q: float
    p_idle: float
    p_meas0: float
    p_meas1: float
    p_hook: float
    p_xtalk: float
    p_erase: float
    p_long_range: float
    requested_phys_p: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "noise_model_name": self.model_name,
            "noise_model_version": self.model_version,
            "noise_ramp": self.noise_ramp,
            "lambda_scale": self.lambda_scale,
            "requested_phys_p": self.requested_phys_p,
            "p_data": self.p_data,
            "p_meas": self.p_meas,
            "p_1q": self.p_1q,
            "p_2q": self.p_2q,
            "p_idle": self.p_idle,
            "p_meas0": self.p_meas0,
            "p_meas1": self.p_meas1,
            "p_hook": self.p_hook,
            "p_xtalk": self.p_xtalk,
            "p_erase": self.p_erase,
            "p_long_range": self.p_long_range,
        }


def _resolve_base_param(
    overrides: dict[str, Any] | None,
    name_new: str,
    env_new: str,
    default: float,
    *,
    env_legacy: str | None = None,
) -> float:
    if overrides is not None and name_new in overrides and overrides[name_new] is not None:
        return _clip_prob(float(overrides[name_new]))
    if env_new in os.environ:
        return _clip_prob(_env_float(env_new, default))
    if env_legacy is not None and env_legacy in os.environ:
        return _clip_prob(_env_float(env_legacy, default))
    return _clip_prob(default)


def resolve_canonical_noise_spec(
    *,
    requested_phys_p: float | None,
    noise_scale: float | None,
    overrides: dict[str, Any] | None = None,
) -> CanonicalNoiseSpec:
    """Resolve canonical noise params from explicit args + env + defaults."""
    model_name_raw = None
    if overrides is not None:
        model_name_raw = overrides.get("noise_model", None)
    if model_name_raw is None:
        model_name_raw = os.getenv("MGHD_NOISE_MODEL", "circuit_standard")
    model_name = _norm_model(str(model_name_raw))

    if overrides is not None and overrides.get("lambda_scale", None) is not None:
        lam = float(overrides["lambda_scale"])
    elif noise_scale is not None:
        lam = float(noise_scale)
    elif "MGHD_LAMBDA_SCALE" in os.environ:
        lam = _env_float("MGHD_LAMBDA_SCALE", 1.0)
    elif requested_phys_p is not None:
        p_ref = _env_float("MGHD_P_REF", 0.03)
        lam = max(1e-3, min(5.0, float(requested_phys_p) / max(p_ref, 1e-9)))
    else:
        lam = 1.0
    lam = max(0.0, float(lam))

    noise_ramp = str(
        (overrides or {}).get("noise_ramp", os.getenv("MGHD_NOISE_RAMP", "ramp0"))
    ).strip().lower() or "ramp0"
    if noise_ramp not in {"ramp0", "ramp1", "ramp2", "ramp3"}:
        noise_ramp = "ramp0"

    p_data = _resolve_base_param(overrides, "p_data", "MGHD_P_DATA", requested_phys_p or 0.01)
    p_meas = _resolve_base_param(overrides, "p_meas", "MGHD_P_MEAS", p_data)
    p_1q_base = _resolve_base_param(
        overrides, "p_1q", "MGHD_P_1Q", 0.0015, env_legacy="MGHD_GENERIC_P1Q"
    )
    p_2q_base = _resolve_base_param(
        overrides, "p_2q", "MGHD_P_2Q", 0.01, env_legacy="MGHD_GENERIC_P2Q"
    )
    p_idle_base = _resolve_base_param(
        overrides, "p_idle", "MGHD_P_IDLE", 0.0008, env_legacy="MGHD_GENERIC_PIDLE"
    )
    p_meas0_base = _resolve_base_param(
        overrides, "p_meas0", "MGHD_P_MEAS0", 0.02, env_legacy="MGHD_GENERIC_PMEAS0"
    )
    p_meas1_base = _resolve_base_param(
        overrides, "p_meas1", "MGHD_P_MEAS1", 0.02, env_legacy="MGHD_GENERIC_PMEAS1"
    )
    p_hook_base = _resolve_base_param(
        overrides, "p_hook", "MGHD_P_HOOK", 0.0, env_legacy="MGHD_GENERIC_PHOOK"
    )
    p_xtalk_base = _resolve_base_param(
        overrides, "p_xtalk", "MGHD_P_XTALK", 0.0, env_legacy="MGHD_GENERIC_PCROSSTALK"
    )
    p_erase_base = _resolve_base_param(overrides, "p_erase", "MGHD_P_ERASE", 0.0)
    p_long_base = _resolve_base_param(overrides, "p_long_range", "MGHD_P_LONG_RANGE", 0.0)

    if model_name == "code_capacity":
        p_1q = _clip_prob(p_data)
        p_2q = 0.0
        p_idle = 0.0
        p_meas0 = 0.0
        p_meas1 = 0.0
        p_hook = 0.0
        p_xtalk = 0.0
        p_erase = 0.0
        p_long = 0.0
    elif model_name == "phenomenological":
        p_1q = _clip_prob(p_data)
        p_2q = 0.0
        p_idle = 0.0
        p_meas0 = _clip_prob(p_meas)
        p_meas1 = _clip_prob(p_meas)
        p_hook = 0.0
        p_xtalk = 0.0
        p_erase = 0.0
        p_long = 0.0
    else:
        p_1q = _scaled_prob(p_1q_base, lam)
        p_2q = _scaled_prob(p_2q_base, lam)
        p_idle = _scaled_prob(p_idle_base, lam)
        p_meas0 = _scaled_prob(p_meas0_base, lam)
        p_meas1 = _scaled_prob(p_meas1_base, lam)
        p_hook = _scaled_prob(p_hook_base, lam)
        p_xtalk = _scaled_prob(p_xtalk_base, lam)
        p_erase = _scaled_prob(p_erase_base, lam)
        p_long = _scaled_prob(p_long_base, lam)

    if noise_ramp == "ramp1":
        p_hook = max(p_hook, _clip_prob(0.25 * p_2q))
        p_xtalk = max(p_xtalk, _clip_prob(0.1 * p_2q))
    elif noise_ramp == "ramp2":
        p_hook = max(p_hook, _clip_prob(0.35 * p_2q))
        p_xtalk = max(p_xtalk, _clip_prob(0.15 * p_2q))
        p_erase = max(p_erase, _clip_prob(0.1 * p_meas0))
    elif noise_ramp == "ramp3":
        p_hook = max(p_hook, _clip_prob(0.5 * p_2q))
        p_xtalk = max(p_xtalk, _clip_prob(0.25 * p_2q))
        p_erase = max(p_erase, _clip_prob(0.2 * p_meas0))
        p_long = max(p_long, _clip_prob(0.1 * p_2q))

    return CanonicalNoiseSpec(
        model_name=model_name,
        model_version=NOISE_MODEL_VERSION,
        noise_ramp=noise_ramp,
        lambda_scale=float(lam),
        requested_phys_p=float(requested_phys_p) if requested_phys_p is not None else None,
        p_data=float(p_data),
        p_meas=float(p_meas),
        p_1q=float(p_1q),
        p_2q=float(p_2q),
        p_idle=float(p_idle),
        p_meas0=float(p_meas0),
        p_meas1=float(p_meas1),
        p_hook=float(p_hook),
        p_xtalk=float(p_xtalk),
        p_erase=float(p_erase),
        p_long_range=float(p_long),
    )


def axis_from_noise_spec(spec: CanonicalNoiseSpec, requested_p: float | None) -> tuple[str, float]:
    if spec.model_name == "code_capacity":
        return "p_data", float(spec.p_data)
    if spec.model_name == "phenomenological":
        return "p_data", float(spec.p_data)
    if requested_p is not None and spec.model_name not in {"circuit_standard", "circuit_augmented"}:
        return "p_phys_requested", float(requested_p)
    return "lambda_scale", float(spec.lambda_scale)

