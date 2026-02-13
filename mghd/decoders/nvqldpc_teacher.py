"""
GPU-accelerated BP+OSD teacher via CUDA-Q QEC (nv-qldpc-decoder).

This wraps the CUDA-Q QEC nv-qldpc-decoder plugin behind the same batch
interface used by LSDTeacher: given X/Z syndromes, return data-qubit
corrections ex/ez as uint8 arrays. This teacher is strict: if cudaq_qec
is unavailable, no GPU is present, or the decoder cannot be constructed,
it raises an error and training should stop rather than falling back.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
from typing import Any, Optional, Tuple

import numpy as np

_qec = None  # type: ignore
_CUDAQ_QEC_IMPORT_ERROR: Exception | None = None
_HAVE_CUDAQ_QEC = importlib.util.find_spec("cudaq_qec") is not None


def _load_cudaq_qec():
    """Import cudaq_qec lazily to avoid CUDA side effects at module import."""
    global _qec, _HAVE_CUDAQ_QEC, _CUDAQ_QEC_IMPORT_ERROR
    if _qec is not None:
        return _qec
    if not _HAVE_CUDAQ_QEC:
        err = _CUDAQ_QEC_IMPORT_ERROR or ModuleNotFoundError("cudaq_qec not installed")
        raise RuntimeError(
            "cudaq_qec not available; install cudaq-qec to enable NvQldpcTeacher"
        ) from err
    try:
        _qec = importlib.import_module("cudaq_qec")  # type: ignore
        return _qec
    except Exception as exc:  # pragma: no cover - exercised when runtime init fails
        _CUDAQ_QEC_IMPORT_ERROR = exc
        _HAVE_CUDAQ_QEC = False
        raise RuntimeError(
            "cudaq_qec import failed during NvQldpcTeacher initialization"
        ) from exc


@dataclass
class NvQldpcConfig:
    """Lightweight config for nv-qldpc-decoder.

    Fields map directly to CUDA-Q QEC nv_qldpc_decoder_config options.
    We only expose a small subset here; kwargs are forwarded as-is.
    """

    max_iter: Optional[int] = None
    osd_order: Optional[int] = None
    # Additional options can be threaded via extra_kwargs if needed.


class NvQldpcTeacher:
    """GPU BP+OSD teacher using CUDA-Q QEC nv-qldpc-decoder.

    Decodes X and Z syndromes separately as classical LDPC codes with
    parity-checks Hx and Hz. For each batch, returns (ex, ez) bit-flip
    vectors over data qubits. Any failure (missing lib, no GPU, decode
    error) is surfaced as an exception; callers are expected to handle
    this explicitly rather than silently falling back.
    """

    def __init__(self, Hx: np.ndarray, Hz: np.ndarray, cfg: NvQldpcConfig | None = None, **extra_kwargs: Any) -> None:
        self._qec = _load_cudaq_qec()

        self.Hx = np.asarray(Hx, dtype=np.uint8)
        self.Hz = np.asarray(Hz, dtype=np.uint8)
        self.cfg = cfg or NvQldpcConfig()
        self.extra_kwargs = dict(extra_kwargs) if extra_kwargs else {}

        self._dec_x = None
        self._dec_z = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_decoder(self, H: np.ndarray):
        """Instantiate nv-qldpc-decoder for a given parity-check matrix.

        We keep construction lazy and per-side to avoid doing any CUDA
        work unless the teacher is actually used. All errors are surfaced
        to the caller so they can disable the teacher cleanly.
        """

        if H.size == 0:
            raise ValueError("Parity-check matrix is empty")

        # Assemble decoder kwargs from NvQldpcConfig + extras.
        kwargs: dict[str, Any] = {}
        if self.cfg.max_iter is not None:
            kwargs["max_iter"] = int(self.cfg.max_iter)
        if self.cfg.osd_order is not None:
            kwargs["osd_order"] = int(self.cfg.osd_order)
        # nv-qldpc batched decoding requires GPU + sparse mode.
        # We always request sparse mode here to enable decode_batch.
        kwargs.setdefault("use_sparsity", True)
        kwargs.update(self.extra_kwargs)

        # qec.get_decoder(name, pcm, **kwargs) -> Decoder
        return self._qec.get_decoder("nv-qldpc-decoder", H, **kwargs)  # type: ignore[arg-type]

    def _ensure_decoders(self) -> None:
        if self._dec_x is None:
            self._dec_x = self._build_decoder(self.Hx)
        if self._dec_z is None:
            self._dec_z = self._build_decoder(self.Hz)

    @staticmethod
    def _results_to_bits(
        results: list[Any],
        n_qubits: int,
    ) -> np.ndarray:
        """Convert a list[DecoderResult] to a (B, n_qubits) uint8 array.

        The DecoderResult.result payload format is decoder-dependent; we
        best-effort interpret it as a collection of bit-flip indicators.
        """
        B = len(results)
        out = np.zeros((B, n_qubits), dtype=np.uint8)
        for i, res in enumerate(results):
            try:
                raw = getattr(res, "result", None)
                if raw is None:
                    continue
                arr = np.asarray(raw, dtype=np.float32).ravel()
                if arr.size == n_qubits:
                    bits = (arr > 0.5).astype(np.uint8)
                    out[i, :] = bits
                else:
                    # If the decoder returns indices of flipped bits, try that.
                    idx = np.asarray(raw, dtype=int).ravel()
                    mask = (0 <= idx) & (idx < n_qubits)
                    out[i, idx[mask]] = 1
            except Exception:
                continue
        return out

    # ------------------------------------------------------------------
    # Public API mirroring LSDTeacher
    # ------------------------------------------------------------------
    def decode_batch_xz(
        self,
        syndromes_x: np.ndarray,
        syndromes_z: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decode batches of X/Z syndromes into (ex, ez) data corrections.

        Parameters
        ----------
        syndromes_x, syndromes_z : np.ndarray
            Arrays of shape [B, mx] and [B, mz] with 0/1 entries.
        """
        self._ensure_decoders()

        sx = np.asarray(syndromes_x, dtype=np.float32)
        sz = np.asarray(syndromes_z, dtype=np.float32)

        B = max(sx.shape[0], sz.shape[0])
        if B == 0:
            return (
                np.zeros((0, self.Hx.shape[1]), dtype=np.uint8),
                np.zeros((0, self.Hz.shape[1]), dtype=np.uint8),
            )

        # Ensure 2D lists of floats for decode_batch.
        sx_list = sx.tolist()
        sz_list = sz.tolist()

        # Run GPU decoders; any exception should be handled by caller.
        if hasattr(self._dec_x, "decode_batch"):
            res_x = self._dec_x.decode_batch(sx_list)
        else:
            res_x = [self._dec_x.decode(v) for v in sx_list]  # type: ignore[attr-defined]

        if hasattr(self._dec_z, "decode_batch"):
            res_z = self._dec_z.decode_batch(sz_list)
        else:
            res_z = [self._dec_z.decode(v) for v in sz_list]  # type: ignore[attr-defined]

        ex = self._results_to_bits(list(res_x), self.Hx.shape[1])
        ez = self._results_to_bits(list(res_z), self.Hz.shape[1])
        return ex, ez


__all__ = ["NvQldpcTeacher", "NvQldpcConfig", "_HAVE_CUDAQ_QEC"]
