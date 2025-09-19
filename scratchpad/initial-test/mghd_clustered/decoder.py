from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np
from ldpc.bplsd_decoder import BpLsdDecoder

from .adapter import MGHDAdapter


class MGHDClusteredDecoder:
    """Wrap LDPC BP+LSD decoder, optionally seeded with MGHD priors."""

    def __init__(
        self,
        pcm,
        n_bits: int,
        mghd_adapter: Optional[MGHDAdapter] = None,
        *,
        error_rate: Optional[float] = None,
        bp_method: str = "minimum_sum",
        max_iter: int = 1,
        schedule: str = "serial",
        lsd_method: str = "LSD_E",
        lsd_order: int = 0,
        bits_per_step: Optional[int] = None,
        initial_channel_probs: Optional[np.ndarray] = None,
    ) -> None:
        self.n_bits = int(n_bits)
        self.mghd = mghd_adapter
        self.decoder = BpLsdDecoder(
            pcm,
            error_rate=error_rate,
            bp_method=bp_method,
            max_iter=max_iter,
            schedule=schedule,
            lsd_method=lsd_method,
            lsd_order=lsd_order,
            bits_per_step=bits_per_step,
        )
        try:
            self.decoder.set_do_stats(True)
        except Exception:
            pass

        if initial_channel_probs is not None:
            self.decoder.update_channel_probs(np.asarray(initial_channel_probs, dtype=np.float64))

    @staticmethod
    def _prob_to_llr(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log((1.0 - p) / p)

    def decode(
        self,
        syndrome: np.ndarray,
        features: Optional[Dict[str, Any]] = None,
        *,
        use_mghd_priors: bool = True,
    ) -> np.ndarray:
        if use_mghd_priors and self.mghd is not None:
            probs = self.mghd.predict_error_probs(features)
            if probs.shape[0] == self.n_bits:
                try:
                    self.decoder.update_channel_probs(probs)
                except Exception:
                    pass

        recovery = self.decoder.decode(syndrome)
        return np.asarray(recovery, dtype=np.uint8)

    def get_stats(self) -> dict:
        stats = {"bp_converged": getattr(self.decoder, "converge", False)}
        detail = getattr(self.decoder, "statistics", None)
        if detail is not None:
            for key in (
                "cluster_count",
                "largest_cluster_size",
                "steps",
                "validated_clusters",
                "merged_clusters",
                "elapsed_time",
                "bp_iterations",
            ):
                stats[key] = getattr(detail, key, None)
        return stats
