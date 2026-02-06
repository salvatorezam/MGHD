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

from mghd.utils.graphlike import is_graphlike

from .mwpf_teacher import MWPFConfig, MWPFTeacher
from .lsd_teacher import LSDConfig, LSDTeacher
from .mwpm_fallback import MWPMFallback, MwpmNotGraphlike

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
    """Configuration of teacher selection probabilities and limits.

    p_mwpf/p_lsd/p_mwpm: selection probabilities per batch (normalized internally)
    max_cluster: cap used by erasure teachers for local component size
    """

    p_mwpf: float = 0.5
    p_lsd: float = 0.4
    p_mwpm: float = 0.1
    max_cluster: int = 256


class TeacherMix:
    """Stochastic router across MWPF/LSD/MWPM/erasure teachers for CSS codes.

    - MWPF (Hyperblossom) consumes detectors → returns fault_ids
    - LSD returns data-qubit flips (ex, ez)
    - MWPM returns fault_ids under graphlike assumptions
    - Erasure teachers handle explicit erasure masks

    The training loop can supervise per-fault or per-qubit heads depending on
    the teacher used; parity projection ensures consistency when needed.
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
        mwpm_graphlike_only: bool = True,
    ) -> None:
        """Initialize teacher stack and probabilities for a given CSS code."""
        base_mix = mix_cfg or MixConfig()
        self._max_cluster = base_mix.max_cluster
        self._mwpm_graphlike_only = bool(mwpm_graphlike_only)
        self._mwpm_enabled = True

        p_mwpf = base_mix.p_mwpf
        p_lsd = base_mix.p_lsd
        p_mwpm = base_mix.p_mwpm

        probs = np.array([p_mwpf, p_lsd, p_mwpm], dtype=float)
        if np.any(probs < 0):  # pragma: no cover - defensive
            raise ValueError("Mix probabilities must be non-negative")

        self.code_obj = code_obj
        self.flags: Dict[str, Any] = {}
        self.teachers: Dict[str, Any] = {}
        self.mwpf = None
        try:
            self.mwpf = MWPFTeacher(
                code_obj,
                config=mwpf_cfg or MWPFConfig(cluster_node_limit=50),
            )
            self.teachers["mwpf"] = self.mwpf
        except Exception as exc:  # pragma: no cover - degrade gracefully
            warnings.warn(
                f"MWPFTeacher unavailable ({exc}); disabling MWPF in mix.",
                RuntimeWarning,
                stacklevel=2,
            )
            p_mwpf = 0.0
            self.teachers["mwpf"] = None

        self.lsd = LSDTeacher(Hx, Hz, cfg=lsd_cfg or LSDConfig())
        self.teachers["lsd"] = self.lsd
        self.mwpm_x = None
        self.mwpm_z = None
        mwpm_ok = True
        mwpm_reason = "mwpm_not_graphlike"
        graphlike_ok = is_graphlike(Hx) and is_graphlike(Hz)
        if self._mwpm_graphlike_only and not graphlike_ok:
            mwpm_ok = False
        if mwpm_ok:
            try:
                self.mwpm_x = MWPMFallback(
                    code_obj,
                    basis="x",
                    require_graphlike=self._mwpm_graphlike_only,
                )
                self.mwpm_z = MWPMFallback(
                    code_obj,
                    basis="z",
                    require_graphlike=self._mwpm_graphlike_only,
                )
                self.teachers["mwpm"] = (self.mwpm_x, self.mwpm_z)
            except MwpmNotGraphlike as exc:
                mwpm_ok = False
                mwpm_reason = str(exc) or mwpm_reason
        if not mwpm_ok:
            self.flags["mwpm_not_graphlike"] = True
            try:
                self.mwpm_x = MWPMFallback(code_obj, basis="x", require_graphlike=False)
                self.mwpm_z = MWPMFallback(code_obj, basis="z", require_graphlike=False)
                self.teachers["mwpm"] = (self.mwpm_x, self.mwpm_z)
            except MwpmNotGraphlike:
                self.mwpm_x = None
                self.mwpm_z = None
                self.teachers["mwpm"] = None
        if not graphlike_ok and not self.flags.get("mwpm_not_graphlike"):
            self.flags["mwpm_not_graphlike"] = True
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
                    max_cluster=self._max_cluster,
                )
            except Exception as exc:  # pragma: no cover - degrade gracefully
                warnings.warn(
                    f"ErasureQLDPCPeelingTeacher unavailable ({exc}); continuing without erasure teacher.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        self._set_mix_probs(p_mwpf, p_lsd, p_mwpm)
        if not mwpm_ok:
            self._disable_mwpm(mwpm_reason)

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
        """Route one batch through a chosen teacher according to mix probs.

        Returns a payload tagged with 'which' ∈ {mwpf, lsd, mwpm, mwpm_fallback}.
        """
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
            mwpm_threshold = self.mix.p_mwpf + self.mix.p_lsd
            if r < mwpm_threshold or not self._mwpm_enabled or self.mix.p_mwpm <= 0:
                ex, ez = self.lsd.decode_batch_xz(
                    syndromes_x,
                    syndromes_z,
                    llr_overrides=llr_override,
                    erase_mask=erase_data_mask,
                )
                return {"which": "lsd", "ex": ex, "ez": ez}
            if self.mwpm_x is None or self.mwpm_z is None:
                self._disable_mwpm("mwpm_not_graphlike")
                ex, ez = self.lsd.decode_batch_xz(
                    syndromes_x,
                    syndromes_z,
                    llr_overrides=llr_override,
                    erase_mask=erase_data_mask,
                )
                return {"which": "lsd", "ex": ex, "ez": ez}
            try:
                cx = self.mwpm_x.decode_batch(syndromes_x, column_weights=col_w_x)
                cz = self.mwpm_z.decode_batch(syndromes_z, column_weights=col_w_z)
            except (ValueError, MwpmNotGraphlike) as exc:
                msg = str(exc)
                if "mwpm_not_graphlike" in msg or "two ones per column" in msg:
                    self._disable_mwpm(msg)
                    ex, ez = self.lsd.decode_batch_xz(
                        syndromes_x,
                        syndromes_z,
                        llr_overrides=llr_override,
                        erase_mask=erase_data_mask,
                    )
                    return {"which": "lsd", "ex": ex, "ez": ez}
                raise
            return {"which": "mwpm", "cx": cx, "cz": cz}
        except Exception as exc:  # pragma: no cover - protective fallback
            if self.mwpm_x is None or self.mwpm_z is None:
                ex, ez = self.lsd.decode_batch_xz(
                    syndromes_x,
                    syndromes_z,
                    llr_overrides=llr_override,
                    erase_mask=erase_data_mask,
                )
                return {"which": "lsd", "ex": ex, "ez": ez, "error": str(exc)}

            cx = self.mwpm_x._gf2_decode(syndromes_x)
            cz = self.mwpm_z._gf2_decode(syndromes_z)
            return {
                "which": "mwpm_fallback",
                "cx": cx,
                "cz": cz,
                "error": str(exc),
            }

    def _set_mix_probs(self, p_mwpf: float, p_lsd: float, p_mwpm: float) -> None:
        """Normalize and store mix probabilities, preserving total mass."""
        probs = np.array([float(p_mwpf), float(p_lsd), float(p_mwpm)], dtype=float)
        total = probs.sum()
        if total <= 0:
            probs = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            probs = probs / total
        self.mix = MixConfig(float(probs[0]), float(probs[1]), float(probs[2]), self._max_cluster)

    def _disable_mwpm(self, reason: str) -> None:
        """Disable MWPM branch and reassign its mass to LSD."""
        if not self._mwpm_enabled:
            return
        self._mwpm_enabled = False
        warnings.warn(
            f"MWPM disabled for mix (reason: {reason}); falling back to LSD/erasure teachers.",
            RuntimeWarning,
            stacklevel=2,
        )
        self.teachers["mwpm"] = None
        self.mwpm_x = None
        self.mwpm_z = None
        self._set_mix_probs(self.mix.p_mwpf, self.mix.p_lsd + self.mix.p_mwpm, 0.0)


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
            try:
                arr = np.asarray(list(value), dtype=np.float32)
            except Exception:
                return None
        return arr.astype(np.float32, copy=False)

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
