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
        total = probs.sum()
        if not np.isclose(total, 1.0):
            # Normalise to avoid accidental drift; report via warning? keep simple.
            probs = probs / total if total > 0 else np.array([1.0, 0.0, 0.0], dtype=float)
            self.mix = MixConfig(*probs)

        self.code_obj = code_obj
        self.mwpf = MWPFTeacher(code_obj, config=mwpf_cfg or MWPFConfig(cluster_node_limit=50))
        self.lsd = LSDTeacher(Hx, Hz, cfg=lsd_cfg or LSDConfig())
        self.mwpm_x = MWPMFallback(Hx)
        self.mwpm_z = MWPMFallback(Hz)

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
            if r < self.mix.p_mwpf:
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
