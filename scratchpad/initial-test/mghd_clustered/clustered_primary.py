from __future__ import annotations

import time
import numpy as np
import scipy.sparse as sp
import torch
from typing import Any, Dict, List, Sequence, Tuple, TYPE_CHECKING

from .cluster_core import (
    active_components,
    extract_subproblem,
    ml_parity_project,
    solve_small_cluster_channel_ml,
    TIER0_K_MAX,
    TIER0_R_MAX,
)

if TYPE_CHECKING:  # pragma: no cover - hints only
    from mghd_public.infer import MGHDDecoderPublic


class MGHDPrimaryClustered:
    """
    Primary clustered decoder that works with both v1 (distance-specific) and v2 (distance-agnostic) MGHD models.
    CUDA graph capture and µs-precision timers are automatically applied to whichever model version is provided.
    """
    def __init__(
        self,
        H: sp.csr_matrix,
        mghd: MGHDDecoderPublic,
        *,
        halo: int = 0,
        thresh: float = 0.5,
        temp: float = 1.0,
        r_cap: int = 20,
        batched: bool = True,
        tier0_enable: bool = True,
        tier0_k_max: int = TIER0_K_MAX,
        tier0_r_max: int = TIER0_R_MAX,
        p_channel: float | None = None,
        default_p: float | None = None,
    ):
        self.H = H.tocsr()
        self.mghd = mghd
        self.halo = int(halo)
        self.thresh = float(thresh)
        self.temp = float(temp)
        self.r_cap = int(r_cap)
        self.mb_mode = "batched" if batched else "unbatched"
        self.side = self._infer_side()
        self.tier0_enable = bool(tier0_enable)
        self.tier0_k_max = int(tier0_k_max)
        self.tier0_r_max = int(tier0_r_max)
        self.p_channel = p_channel
        self.default_p = default_p

    @staticmethod
    def _histogram_from_sizes(sizes: Sequence[int]) -> Dict[str, int]:
        hist: Dict[str, int] = {}
        for size in sizes:
            bucket = 1
            while size > bucket:
                bucket *= 2
            key = str(bucket)
            hist[key] = hist.get(key, 0) + 1
        return hist

    def _sync_cuda(self) -> None:
        if torch.cuda.is_available() and getattr(self.mghd, "device", None) is not None:
            if getattr(self.mghd.device, "type", "cpu") == "cuda":
                torch.cuda.synchronize(self.mghd.device)

    def _infer_side(self) -> str | None:
        if not getattr(self.mghd, "_bound", False):
            return None
        try:
            if self.mghd._Hx is not None and self.H.shape == self.mghd._Hx.shape:
                if (self.H != self.mghd._Hx).nnz == 0:
                    return "X"
            if self.mghd._Hz is not None and self.H.shape == self.mghd._Hz.shape:
                if (self.H != self.mghd._Hz).nnz == 0:
                    return "Z"
        except Exception:
            return None
        return None

    def _resolve_mode(self, override) -> str:
        if override is None:
            mode = self.mb_mode
        elif isinstance(override, str):
            mode = override
        else:
            mode = "batched" if override else "unbatched"
        if mode not in {"batched", "unbatched"}:
            raise ValueError(f"Invalid micro-batch mode: {mode}")
        return mode

    def decode(self, s: np.ndarray, *, batched: bool | str | None = None) -> Dict[str, Any]:
        """Decode one syndrome vector with optional micro-batch override."""
        mode = self._resolve_mode(batched)
        return self._decode_impl(s, mode=mode)

    def decode_unbatched(self, s: np.ndarray) -> Dict[str, Any]:
        """Legacy single-cluster path (micro-batching disabled)."""
        return self._decode_impl(s, mode="unbatched")

    def _decode_impl(self, s: np.ndarray, *, mode: str) -> Dict[str, Any]:
        n = self.H.shape[1]
        s = np.asarray(s, dtype=np.uint8).ravel()
        t0 = time.perf_counter()
        checks_list, qubits_list = active_components(self.H, s, halo=self.halo)
        t_cluster = (time.perf_counter() - t0) * 1e6  # microseconds

        e = np.zeros(n, dtype=np.uint8)

        if len(checks_list) == 0:
            empty_report = {
                "fast_path_batches": 0,
                "fixed_d3_batches": 0,
                "fallback_loops": 0,
                "batch_sizes": [],
                "graph_used": False,
                "device": self.mghd._device_info() if hasattr(self.mghd, "_device_info") else {"device": str(self.mghd.device)},
                "avg_batch_size": 0.0,
                "batch_histogram": {},
            }
            return dict(
                e_hat=e,
                t_cluster_us=t_cluster,
                t_mghd_us=0.0,
                t_project_us=0.0,
                t_total_us=t_cluster,
                n_clusters=0,
                sizes=[],
                frac_empty=1.0,
                mb_stats=empty_report,
            )

        subproblems: List[Tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray]] = []
        sizes: List[int] = []
        tier0_clusters = 0
        tier0_qubits = 0
        tier0_p = self.p_channel if self.p_channel is not None else self.default_p
        tier0_active = self.tier0_enable and tier0_p is not None
        t_tier0 = 0.0  # Track tier-0 timing

        for ci, qi in zip(checks_list, qubits_list):
            sizes.append(int(qi.size))
            H_sub, s_sub, q_l2g, c_l2g = extract_subproblem(self.H, s, ci, qi)

            solved = False
            if tier0_active:
                t_t0 = time.perf_counter()
                tier0_sol = solve_small_cluster_channel_ml(
                    H_sub,
                    s_sub,
                    p_channel=float(tier0_p),
                    k_max=self.tier0_k_max,
                    r_cap=self.tier0_r_max,
                )
                t_tier0 += (time.perf_counter() - t_t0) * 1e6  # microseconds
                if tier0_sol is not None:
                    e[q_l2g] ^= tier0_sol.astype(np.uint8)
                    tier0_clusters += 1
                    tier0_qubits += len(q_l2g)
                    solved = True

            if not solved:
                subproblems.append((H_sub, s_sub, q_l2g, c_l2g))

        mb_report: Dict[str, Any]
        probs_list: List[np.ndarray]
        mghd_clusters = len(subproblems)

        if mode == "batched":
            if mghd_clusters > 0:
                # Timing and CUDA synchronization for both v1 and v2 models
                # Graph capture (if enabled in mghd) automatically applies here
                self._sync_cuda()
                t1 = time.perf_counter()
                probs_list, mb_report = self.mghd.priors_from_subgraphs_batched(
                    [(H_sub, s_sub, q_l2g, c_l2g) for H_sub, s_sub, q_l2g, c_l2g in subproblems],
                    temp=self.temp,
                    bucket=self.side,
                )
                self._sync_cuda()
                t_mghd = (time.perf_counter() - t1) * 1e6  # microseconds
            else:
                probs_list = []
                t_mghd = 0.0
                mb_report = self.mghd._init_mb_report() if hasattr(self.mghd, "_init_mb_report") else {
                    "fast_path_batches": 0,
                    "fixed_d3_batches": 0,
                    "fallback_loops": 0,
                    "batch_sizes": [],
                    "graph_used": False,
                    "device": self.mghd._device_info() if hasattr(self.mghd, "_device_info") else {"device": str(self.mghd.device)},
                }
        else:
            probs_list = []
            t_mghd = 0.0
            mb_report = {
                "fast_path_batches": 0,
                "fixed_d3_batches": 0,
                "fallback_loops": 0,
                "batch_sizes": [],
                "graph_used": False,
                "device": self.mghd._device_info() if hasattr(self.mghd, "_device_info") else {"device": str(self.mghd.device)},
            }
            for H_sub, s_sub, q_l2g, c_l2g in subproblems:
                # Per-subproblem timing (applies to both v1 and v2)
                self._sync_cuda()
                t1 = time.perf_counter()
                s_full = np.zeros(self.H.shape[0], dtype=np.float32)
                s_full[c_l2g] = s_sub.astype(np.float32)
                side = self.side if self.side in {"X", "Z"} else (
                    "X" if self.H.shape[0] == getattr(self.mghd, "_n_x", self.H.shape[0]) else "Z"
                )
                probs_full = self.mghd.priors_from_syndrome(s_full, side=side)
                self._sync_cuda()
                t_mghd += (time.perf_counter() - t1) * 1e6  # microseconds
                mb_report["fixed_d3_batches"] += 1
                mb_report["fallback_loops"] += 1
                mb_report["batch_sizes"].append(1)
                probs_list.append(np.asarray(probs_full, dtype=np.float64)[q_l2g])

        t_proj = 0.0
        for (H_sub, s_sub, q_l2g, _), probs in zip(subproblems, probs_list):
            if probs.shape[0] != H_sub.shape[1]:
                raise AssertionError("Probability vector length mismatch for subgraph")

            t_p0 = time.perf_counter()
            e_sub = ml_parity_project(H_sub, s_sub, probs, r_cap=self.r_cap)
            t_proj += (time.perf_counter() - t_p0) * 1e6  # microseconds

            parity = (H_sub @ e_sub) % 2
            parity = np.asarray(parity).ravel().astype(np.uint8) % 2
            if not np.array_equal(parity, s_sub % 2):
                raise AssertionError("ML parity projection failed to satisfy local checks")

            e[q_l2g] ^= e_sub.astype(np.uint8)

        sizes_for_hist = mb_report.get("batch_sizes", [])
        if sizes_for_hist:
            mb_report["avg_batch_size"] = float(sum(sizes_for_hist)) / float(len(sizes_for_hist))
        else:
            mb_report["avg_batch_size"] = 0.0
        mb_report["batch_histogram"] = self._histogram_from_sizes(sizes_for_hist)

        tier0_stats = dict(
            tier0_clusters=int(tier0_clusters),
            tier0_qubits=int(tier0_qubits),
            mghd_clusters=int(mghd_clusters),
            mghd_invoked=bool(mghd_clusters),
            p_channel_used=float(tier0_p) if tier0_active and tier0_p is not None else None,
        )

        t_total = t_cluster + t_mghd + t_proj + t_tier0
        return dict(
            e_hat=e,
            t_cluster_us=t_cluster,
            t_tier0_us=t_tier0,
            t_mghd_us=t_mghd,
            t_project_us=t_proj,
            t_total_us=t_total,
            n_clusters=len(sizes),
            sizes=sizes,
            frac_empty=0.0,
            mb_stats=mb_report,
            tier0_clusters=tier0_stats["tier0_clusters"],
            tier0_qubits=tier0_stats["tier0_qubits"],
            mghd_clusters=tier0_stats["mghd_clusters"],
            mghd_invoked=tier0_stats["mghd_invoked"],
            p_channel_used=tier0_stats["p_channel_used"],
            tier0_stats=tier0_stats,
        )
