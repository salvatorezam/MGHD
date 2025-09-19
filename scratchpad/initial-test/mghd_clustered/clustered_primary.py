from __future__ import annotations
import time
import numpy as np
import scipy.sparse as sp
from typing import Dict, Any
from mghd_public.infer import MGHDDecoderPublic
from .cluster_core import active_components, extract_subproblem, greedy_parity_project, ml_parity_project

class MGHDPrimaryClustered:
    def __init__(self, H: sp.csr_matrix, mghd: MGHDDecoderPublic, *, halo: int = 0, thresh: float = 0.5, temp: float = 1.0, r_cap: int = 20):
        self.H = H.tocsr()
        self.mghd = mghd
        self.halo = int(halo)
        self.thresh = float(thresh)
        self.temp = float(temp)
        self.r_cap = int(r_cap)

    def decode(self, s: np.ndarray) -> Dict[str, Any]:
        """
        Returns dict with:
          - 'e_hat': uint8[n] correction
          - timings: t_cluster_ms, t_mghd_ms, t_project_ms, t_total_ms
          - counters: n_clusters, sizes, frac_empty
        """
        n = self.H.shape[1]
        s = np.asarray(s, dtype=np.uint8).ravel()
        t0 = time.perf_counter()
        checks_list, qubits_list = active_components(self.H, s, halo=self.halo)
        t_cluster = (time.perf_counter() - t0) * 1e3

        e = np.zeros(n, dtype=np.uint8)
        t_mghd = 0.0
        t_proj = 0.0

        if len(checks_list) == 0:
            return dict(e_hat=e, t_cluster_ms=t_cluster, t_mghd_ms=0.0, t_project_ms=0.0,
                        t_total_ms=t_cluster, n_clusters=0, sizes=[], frac_empty=1.0)

        sizes = []
        for ci, qi in zip(checks_list, qubits_list):
            sizes.append(int(qi.size))
            H_sub, s_sub, q_l2g, _ = extract_subproblem(self.H, s, ci, qi)

            # MGHD priors using full-graph method with masked syndrome
            t1 = time.perf_counter()
            # Create a full syndrome with zeros everywhere except for this cluster
            s_full = np.zeros_like(s, dtype=np.float32)
            s_full[ci] = s_sub.astype(np.float32)
            
            # Determine which side this is (X or Z) based on the mghd bound matrices
            if self.mghd._bound:
                # Use the existing syndrome method that handles full graphs
                if self.H.shape[0] == self.mghd._n_x:  # X syndrome
                    p_full = self.mghd.priors_from_syndrome(s_full, side="X")
                else:  # Z syndrome
                    p_full = self.mghd.priors_from_syndrome(s_full, side="Z")
                # Extract probabilities for qubits in this cluster
                p = p_full[q_l2g]
            else:
                # Fallback: use uniform probabilities if not bound
                p = np.full(len(q_l2g), 0.1, dtype=np.float64)
            
            t_mghd += (time.perf_counter() - t1) * 1e3

            # local parity projection (exact ML)
            t2 = time.perf_counter()
            e_sub = ml_parity_project(H_sub, s_sub, p, r_cap=self.r_cap)
            t_proj += (time.perf_counter() - t2) * 1e3

            # scatter
            e[q_l2g] ^= e_sub.astype(np.uint8)

        t_total = t_cluster + t_mghd + t_proj
        return dict(e_hat=e, t_cluster_ms=t_cluster, t_mghd_ms=t_mghd, t_project_ms=t_proj,
                    t_total_ms=t_total, n_clusters=len(sizes), sizes=sizes, frac_empty=0.0)