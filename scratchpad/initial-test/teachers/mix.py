"""
Stochastic teacher mixer for robust supervision.
By default (p_mwpf=0.5, p_lsd=0.4, p_mwpm=0.1), pick a teacher per batch.
If a teacher fails (raises), fall back to MWPM.

All teachers consume raw detection streams or syndromes; no DEM required.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import warnings

from .mwpf_teacher import MWPFConfig, MWPFTeacher
from .lsd_teacher import LSDConfig, LSDTeacher
from .mwpm_fallback import MWPMFallback


@dataclass
class MixConfig:
    p_mwpf: float = 0.5
    p_lsd: float = 0.4
    p_mwpm: float = 0.1


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

        total = probs.sum()
        if total <= 0:
            probs = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            probs = probs / total
        self.mix = MixConfig(*probs.tolist())

    def route_batch(
        self,
        dets: np.ndarray,
        syndromes_x: np.ndarray,
        syndromes_z: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        rng = rng or np.random.default_rng()
        r = float(rng.uniform())
        try:
            if self.mwpf is not None and r < self.mix.p_mwpf:
                out = self.mwpf.decode_batch(dets)
                out["which"] = "mwpf"
                return out
            if r < self.mix.p_mwpf + self.mix.p_lsd:
                ex, ez = self.lsd.decode_batch_xz(syndromes_x, syndromes_z)
                return {"which": "lsd", "ex": ex, "ez": ez}
            cx = self.mwpm_x.decode_batch(syndromes_x)
            cz = self.mwpm_z.decode_batch(syndromes_z)
            return {"which": "mwpm", "cx": cx, "cz": cz}
        except Exception as exc:  # pragma: no cover - protective fallback
            cx = self.mwpm_x.decode_batch(syndromes_x)
            cz = self.mwpm_z.decode_batch(syndromes_z)
            return {
                "which": "mwpm_fallback",
                "cx": cx,
                "cz": cz,
                "error": str(exc),
            }


__all__ = ["TeacherMix", "MixConfig"]
