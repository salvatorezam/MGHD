from __future__ import annotations

import numpy as np

from mghd.utils.graphlike import is_graphlike

try:
    import pymatching as pm
except ImportError:
    pm = None


class MWPMatchingContext:
    def __init__(self):
        self._matcher_cache = {}
        self._failed_once = False

    def decode(self, H_sub: np.ndarray, synd_bits: np.ndarray, side: str):
        """
        MWPM decoder using pymatching.
        Returns (correction_bits, weight) where correction_bits are uint8.
        """
        if pm is None:
            return self._decode_fallback(H_sub, synd_bits, side)

        try:
            # Create cache key for this matrix
            cache_key = (H_sub.tobytes(), side)

            if cache_key not in self._matcher_cache:
                # Create pymatching Matching object
                # Convert to binary matrix for pymatching
                H_binary = (H_sub % 2).astype(np.uint8)
                # We allow non-graphlike codes (pymatching 2+ supports them)
                self._matcher_cache[cache_key] = pm.Matching(H_binary)

            matcher = self._matcher_cache[cache_key]

            # Decode syndrome
            syndrome_binary = (synd_bits % 2).astype(np.uint8)
            correction = matcher.decode(syndrome_binary)

            # Convert to uint8 and compute weight
            correction_uint8 = correction.astype(np.uint8)
            weight = int(correction_uint8.sum())

            return correction_uint8, weight

        except BaseException as e:
            # Fallback on any error; suppress noisy prints for non-graphlike cases.
            if not self._failed_once:
                if "mwpm_not_graphlike" not in str(e):
                    print(f"Warning: MWPM decode failed ({e}), using fallback (suppressing further warnings)")
                self._failed_once = True
            return self._decode_fallback(H_sub, synd_bits, side)

    def _decode_fallback(self, H_sub: np.ndarray, synd_bits: np.ndarray, side: str):
        """Fallback decoder when pymatching is not available"""
        n_qubits = H_sub.shape[1]

        # Return maximum weight correction as a conservative fallback
        correction = np.ones(n_qubits, dtype=np.uint8)
        weight = n_qubits

        return correction, weight
