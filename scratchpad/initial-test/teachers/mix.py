"""
Stochastic teacher mixer for robust supervision.
By default (p_mwpf=0.5, p_lsd=0.4, p_mwpm=0.1), pick a teacher per batch.
If a teacher fails (raises), fall back to MWPM.

All teachers consume raw detection streams or syndromes; no DEM required.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import warnings

from .mwpf_teacher import MWPFConfig, MWPFTeacher
from .lsd_teacher import LSDConfig, LSDTeacher
from .mwpm_fallback import MWPMFallback

try:
    from .erasure_surface_ml import ErasureSurfaceMLTeacher
except Exception as exc:  # pragma: no cover - optional dependency stack
    ErasureSurfaceMLTeacher = None
    warnings.warn(
        f"ErasureSurfaceMLTeacher unavailable ({exc}); continuing without erasure teacher.",
        RuntimeWarning,
        stacklevel=2,
    )

try:
    from .erasure_peeling import ErasureQLDPCPeelingTeacher
except Exception as exc:  # pragma: no cover - optional dependency stack
    ErasureQLDPCPeelingTeacher = None
    warnings.warn(
        f"ErasureQLDPCPeelingTeacher unavailable ({exc}); continuing without erasure teacher.",
        RuntimeWarning,
        stacklevel=2,
    )


@dataclass
class MixConfig:
    p_mwpf: float = 0.5
    p_lsd: float = 0.4
    p_mwpm: float = 0.1
    max_cluster: int = 256
    max_cluster: int = 256


class TeacherMix:
    """
    For CSS codes:
      - MWPF works on detector graph (fault hypergraph) -> returns fault_ids
      - LSD returns data-qubit flips ex, ez
      - MWPM returns fault_ids w.r.t. H columns
    The training loop can use either form:
      - convert fault_ids to data flips via code_obj map
      - or supervise per-fault logits (for MGHD) and parity projection
    """

    def __init__(
        self,
        code_obj: Any,
        Hx: np.ndarray,
        Hz: np.ndarray,
        *,
        mwpf_cfg: Optional[MWPFConfig] = None,
        lsd_cfg: Optional[LSDConfig] = None,
        mix_cfg: Optional[MixConfig] = None,
    ) -> None:
        self.mix = mix_cfg or MixConfig()
        probs = np.array([self.mix.p_mwpf, self.mix.p_lsd, self.mix.p_mwpm], dtype=float)
        if np.any(probs < 0):  # pragma: no cover - defensive
            raise ValueError("Mix probabilities must be non-negative")

        self.code_obj = code_obj
        self.mwpf = None
        try:
            self.mwpf = MWPFTeacher(
                code_obj,
                config=mwpf_cfg or MWPFConfig(cluster_node_limit=50),
            )
        except Exception as exc:  # pragma: no cover - degrade gracefully
            warnings.warn(
                f"MWPFTeacher unavailable ({exc}); disabling MWPF in mix.",
                RuntimeWarning,
                stacklevel=2,
            )
            probs[0] = 0.0

        self.lsd = LSDTeacher(Hx, Hz, cfg=lsd_cfg or LSDConfig())
        self.mwpm_x = MWPMFallback(Hx)
        self.mwpm_z = MWPMFallback(Hz)
        self.erasure_surface = None
        self.erasure_qldpc = None
        if getattr(code_obj, "name", None) == "surface" and ErasureSurfaceMLTeacher is not None:
            try:
                self.erasure_surface = ErasureSurfaceMLTeacher(code_obj)
            except Exception as exc:  # pragma: no cover - degrade gracefully
                warnings.warn(
                    f"ErasureSurfaceMLTeacher unavailable ({exc}); continuing without erasure teacher.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        elif ErasureQLDPCPeelingTeacher is not None:
            try:
                self.erasure_qldpc = ErasureQLDPCPeelingTeacher(
                    Hx,
                    Hz,
                    max_cluster=self.mix.max_cluster,
                )
            except Exception as exc:  # pragma: no cover - degrade gracefully
                warnings.warn(
                    f"ErasureQLDPCPeelingTeacher unavailable ({exc}); continuing without erasure teacher.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        max_cluster = self.mix.max_cluster
        total = probs.sum()
        if total <= 0:
            probs = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            probs = probs / total
        self.mix = MixConfig(probs[0], probs[1], probs[2], max_cluster)

    def route_batch(
        self,
        dets: np.ndarray,
        syndromes_x: np.ndarray,
        syndromes_z: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        *,
        erase_data_mask: Optional[np.ndarray] = None,
        erase_det_mask: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None,
        weight_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        rng = rng or np.random.default_rng()
        r = float(rng.uniform())
        has_erasure = erase_data_mask is not None and np.any(erase_data_mask)
        if has_erasure and self.erasure_surface is not None:
            try:
                return self.erasure_surface.decode_batch(
                    syndromes_x,
                    syndromes_z,
                    erase_data_mask,
                    erase_det_mask,
                )
            except Exception as exc:
                warnings.warn(
                    f"Erasure surface teacher failed ({exc}); falling back to stochastic mix.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if has_erasure and self.erasure_qldpc is not None:
            try:
                return self.erasure_qldpc.decode_batch(
                    syndromes_x,
                    syndromes_z,
                    erase_data_mask,
                    erase_det_mask,
                )
            except Exception as exc:
                warnings.warn(
                    f"Erasure qLDPC teacher failed ({exc}); falling back to stochastic mix.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        llr_override = None
        mwpf_scale = None
        mwpm_weights = None
        if weight_overrides:
            llr_override = weight_overrides.get("llr_per_qubit")
            mwpf_scale = weight_overrides.get("mwpf_scale")
            mwpm_weights = weight_overrides.get("mwpm_weights")

        col_w_x, col_w_z = _resolve_mwpm_weights(mwpm_weights)

        try:
            if self.mwpf is not None and r < self.mix.p_mwpf:
                out = self.mwpf.decode_batch(dets, mwpf_scale=mwpf_scale)
                out["which"] = "mwpf"
                return out
            if r < self.mix.p_mwpf + self.mix.p_lsd:
                ex, ez = self.lsd.decode_batch_xz(
                    syndromes_x,
                    syndromes_z,
                    llr_overrides=llr_override,
                    erase_mask=erase_data_mask,
                )
                return {"which": "lsd", "ex": ex, "ez": ez}
            cx = self.mwpm_x.decode_batch(syndromes_x, column_weights=col_w_x)
            cz = self.mwpm_z.decode_batch(syndromes_z, column_weights=col_w_z)
            return {"which": "mwpm", "cx": cx, "cz": cz}
        except Exception as exc:  # pragma: no cover - protective fallback
            cx = self.mwpm_x.decode_batch(syndromes_x, column_weights=col_w_x)
            cz = self.mwpm_z.decode_batch(syndromes_z, column_weights=col_w_z)
            return {
                "which": "mwpm_fallback",
                "cx": cx,
                "cz": cz,
                "error": str(exc),
            }


def _resolve_mwpm_weights(
    weights: Optional[Any],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Normalize optional MWPM column weights for X/Z bases."""

    if weights is None:
        return None, None

    def _to_array(value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=np.float32)
        except Exception:
            return None
        return arr

    if isinstance(weights, dict):
        w_x = _to_array(weights.get("x") or weights.get("X"))
        w_z = _to_array(weights.get("z") or weights.get("Z"))
        common = _to_array(weights.get("common") or weights.get("all"))
        if w_x is None:
            w_x = common
        if w_z is None:
            w_z = common
        return w_x, w_z

    arr = _to_array(weights)
    return arr, arr


__all__ = ["TeacherMix", "MixConfig"]
