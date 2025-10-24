from __future__ import annotations

import time
import numpy as np
import scipy.sparse as sp
import torch
from typing import Any, Dict, List, Sequence, Tuple, TYPE_CHECKING

from .cluster_core import (
    active_components,
    extract_subproblem,
    gf2_nullspace,
    ml_parity_project,
    solve_small_cluster_channel_ml,
    TIER0_K_MAX,
    TIER0_R_MAX,
)

if TYPE_CHECKING:  # pragma: no cover - hints only
    from mghd.core.infer import MGHDDecoderPublic


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
        tier0_k_max: int | None = None,
        tier0_r_max: int | None = None,
        tier0_mode: str = "mixed",
        p_channel: float | None = None,
        default_p: float | None = None,
        bucket_spec: Sequence[Tuple[int, int, int]] | None = None,
        microbatch: int = 64,
        flush_ms: float = 1.0,
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
        self.model_version = getattr(self.mghd, "model_version", "v1")
        self.distance = self._infer_distance()
        self.coords_qubit = self._compute_qubit_coords(self.distance)
        self.coords_check = self._compute_check_coords()

        # Production gating (balanced): force some MGHD without huge latency.
        # Defaults apply when user did not set explicit caps.
        if tier0_k_max is None and tier0_r_max is None:
            if tier0_mode in ("mixed", "mixed_tight", "aggressive"):
                # PROD DEFAULT: k_max=2, r_max=1 keeps X-side fast while ensuring MGHD engagement.
                self.tier0_k_max, self.tier0_r_max = 2, 1
            elif tier0_mode == "off":
                self.tier0_k_max, self.tier0_r_max = 0, 0
            else:
                # Fallback to original defaults
                self.tier0_k_max, self.tier0_r_max = TIER0_K_MAX, TIER0_R_MAX
        else:
            # Use explicit values provided by user
            self.tier0_k_max = int(tier0_k_max or TIER0_K_MAX)
            self.tier0_r_max = int(tier0_r_max or TIER0_R_MAX)
        
        self.p_channel = p_channel
        self.default_p = default_p
        self.bucket_spec = tuple(bucket_spec) if bucket_spec else None
        self.microbatch = max(1, int(microbatch))
        self.flush_ms = max(0.0, float(flush_ms))

    def _infer_distance(self) -> int:
        n_qubits = int(self.H.shape[1])
        d = int(round(float(np.sqrt(n_qubits))))
        if d * d != n_qubits:
            d = n_qubits  # fall back to identity distance for non-square codes
        return max(1, d)

    def _compute_qubit_coords(self, d: int) -> np.ndarray:
        coords = []
        if d * d != self.H.shape[1]:
            # Generic fallback: use unit grid indices
            for q in range(self.H.shape[1]):
                coords.append([float(q), 0.0])
        else:
            for r in range(d):
                for c in range(d):
                    coords.append([float(r + c), float(r - c)])
        return np.asarray(coords, dtype=np.float32)

    def _compute_check_coords(self) -> np.ndarray:
        m = self.H.shape[0]
        coords = np.zeros((m, 2), dtype=np.float32)
        for i in range(m):
            row = self.H.getrow(i)
            idx = row.indices
            if idx.size == 0:
                continue
            coords[i] = self.coords_qubit[idx].mean(axis=0)
        return coords

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

    def decode(
        self,
        s: np.ndarray,
        *,
        batched: bool | str | None = None,
        perf_only: bool = False,
    ) -> Dict[str, Any]:
        """Decode one syndrome vector with optional micro-batch override."""
        mode = self._resolve_mode(batched)
        return self._decode_impl(s, mode=mode, perf_only=perf_only)

    def decode_unbatched(self, s: np.ndarray, *, perf_only: bool = False) -> Dict[str, Any]:
        """Legacy single-cluster path (micro-batching disabled)."""
        return self._decode_impl(s, mode="unbatched", perf_only=perf_only)

    def _decode_impl(self, s: np.ndarray, *, mode: str, perf_only: bool = False) -> Dict[str, Any]:
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

        subproblems: List[Dict[str, Any]] = []
        sizes: List[int] = []
        tier0_clusters = 0
        tier0_qubits = 0
        tier0_p = self.p_channel if self.p_channel is not None else self.default_p
        tier0_active = self.tier0_enable and tier0_p is not None
        t_tier0 = 0.0  # Track tier-0 timing

        for ci, qi in zip(checks_list, qubits_list):
            sizes.append(int(qi.size))
            H_sub, s_sub, q_l2g, c_l2g = extract_subproblem(self.H, s, ci, qi)
            extra_meta: Dict[str, Any] | None = None

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
                if self.model_version == "v2":
                    extra_meta = self._build_v2_metadata(H_sub, s_sub, q_l2g, c_l2g)
                subproblems.append(
                    {
                        "H_sub": H_sub,
                        "s_sub": s_sub,
                        "q_l2g": q_l2g,
                        "c_l2g": c_l2g,
                        "extra": extra_meta,
                    }
                )

        mb_report: Dict[str, Any]
        probs_list: List[np.ndarray]
        mghd_clusters = len(subproblems)

        if mode == "batched":
            if mghd_clusters > 0:
                # Timing and CUDA synchronization for both v1 and v2 models
                # Graph capture (if enabled in mghd) automatically applies here
                self._sync_cuda()
                t1 = time.perf_counter()
                items = [
                    (
                        entry["H_sub"],
                        entry["s_sub"],
                        entry["q_l2g"],
                        entry["c_l2g"],
                        entry["extra"],
                    )
                    for entry in subproblems
                ]
                probs_list, mb_report = self.mghd.priors_from_subgraphs_batched(
                    items,
                    temp=self.temp,
                    bucket=self.side,
                    bucket_spec=self.bucket_spec,
                    microbatch=self.microbatch,
                    flush_ms=self.flush_ms,
                    use_graphs=self.mghd._graph_capture_enabled,
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
            for entry in subproblems:
                H_sub = entry["H_sub"]
                s_sub = entry["s_sub"]
                q_l2g = entry["q_l2g"]
                c_l2g = entry["c_l2g"]
                extra = entry.get("extra")

                # Per-subproblem timing (applies to both v1 and v2)
                self._sync_cuda()
                t1 = time.perf_counter()
                if self.model_version == "v2" and extra is not None:
                    probs_local, sub_report = self.mghd.priors_from_subgraphs_batched(
                        [(H_sub, s_sub, q_l2g, c_l2g, extra)],
                        temp=self.temp,
                        bucket=self.side,
                        bucket_spec=self.bucket_spec,
                        microbatch=self.microbatch,
                        flush_ms=self.flush_ms,
                        use_graphs=self.mghd._graph_capture_enabled,
                    )
                    probs = np.asarray(probs_local[0], dtype=np.float64)
                    for key, value in sub_report.items():
                        if key == "batch_sizes":
                            mb_report["batch_sizes"].extend(value)
                        elif key in {"fast_path_batches", "fixed_d3_batches", "fallback_loops"}:
                            mb_report[key] += value
                    mb_report["graph_used"] = mb_report.get("graph_used", False) or sub_report.get("graph_used", False)
                else:
                    s_full = np.zeros(self.H.shape[0], dtype=np.float32)
                    s_full[c_l2g] = s_sub.astype(np.float32)
                    side = self.side if self.side in {"X", "Z"} else (
                        "X" if self.H.shape[0] == getattr(self.mghd, "_n_x", self.H.shape[0]) else "Z"
                    )
                    probs_full = self.mghd.priors_from_syndrome(s_full, side=side)
                    probs = np.asarray(probs_full, dtype=np.float64)[q_l2g]
                    mb_report["fixed_d3_batches"] += 1
                    mb_report["fallback_loops"] += 1
                    mb_report["batch_sizes"].append(1)

                self._sync_cuda()
                t_mghd += (time.perf_counter() - t1) * 1e6  # microseconds
                probs_list.append(probs)

        t_proj = 0.0
        for entry, probs in zip(subproblems, probs_list):
            H_sub = entry["H_sub"]
            s_sub = entry["s_sub"]
            q_l2g = entry["q_l2g"]
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
            t_cluster_ms=t_cluster / 1000.0,
            t_tier0_us=t_tier0,
            t_tier0_ms=t_tier0 / 1000.0,
            t_mghd_us=t_mghd,
            t_mghd_ms=t_mghd / 1000.0,
            t_project_us=t_proj,
            t_project_ms=t_proj / 1000.0,
            t_total_us=t_total,
            t_total_ms=t_total / 1000.0,
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
            perf_only=perf_only,
        )

    def set_tier0_k_max(self, k: int):
        """Set tier0 cluster size threshold."""
        self.tier0_k_max = int(k)

    def set_tier0_r_max(self, r: int):
        """Set tier0 nullity (rank) threshold."""
        self.tier0_r_max = int(r)

    def get_tier0_caps(self) -> Tuple[int, int]:
        """Get current tier0 thresholds as (k_max, r_max) tuple."""
        return (self.tier0_k_max or 0), (self.tier0_r_max or 0)

    def _build_v2_metadata(
        self,
        H_sub: sp.csr_matrix,
        s_sub: np.ndarray,
        q_l2g: np.ndarray,
        c_l2g: np.ndarray,
    ) -> Dict[str, Any]:
        xy_qubit = np.round(self.coords_qubit[q_l2g]).astype(np.int32)
        xy_check = np.round(self.coords_check[c_l2g]).astype(np.int32)
        all_coords = np.vstack([xy_qubit, xy_check]) if xy_check.size else xy_qubit
        if all_coords.size == 0:
            bbox = (0, 0, 1, 1)
        else:
            mins = np.floor(all_coords.min(axis=0)).astype(int)
            maxs = np.ceil(all_coords.max(axis=0)).astype(int)
            bbox = (
                int(mins[0]),
                int(mins[1]),
                int(maxs[0] - mins[0] + 1),
                int(maxs[1] - mins[1] + 1),
            )

        n_checks, n_qubits = H_sub.shape
        nullity = int(gf2_nullspace(H_sub).shape[1])
        kappa_stats = {
            "size": int(n_checks + n_qubits),
            "density": float(n_checks) / float(max(1, n_qubits)),
            "syndrome_weight": int(np.asarray(s_sub, dtype=np.uint8).sum()),
        }
        p_channel = (
            self.default_p
            if self.default_p is not None
            else (self.p_channel if self.p_channel is not None else 0.0)
        )
        return {
            "xy_qubit": xy_qubit,
            "xy_check": xy_check,
            "bbox": bbox,
            "k": int(n_qubits),
            "r": nullity,
            "kappa_stats": kappa_stats,
            "side": self.side or "Z",
            "d": self.distance,
            "p": float(p_channel),
        }
