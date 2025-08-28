import torch, numpy as np, os
from torch.utils.cpp_extension import load as load_ext
from pathlib import Path

_THIS_DIR = Path(__file__).parent.resolve()
_EXT_NAME = "rotated_d3_fastlut"

# JIT build/load the extension
def _load_ext():
    extra_cuda_cflags = [
        "-O3",
        # Target H100 (sm_90a); keep a generic lower arch too if needed
        "-gencode=arch=compute_90,code=sm_90a",
        "-lineinfo"
    ]
    return load_ext(
        name=_EXT_NAME,
        sources=[str(_THIS_DIR / "fast_lut.cu")],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=True
    )

_ext = None
def _ensure_ext():
    global _ext
    if _ext is None:
        _ext = _load_ext()
    return _ext

# Load LUT file saved earlier
def load_rotated_d3_lut_npz(npz_path=None):
    import json
    if npz_path is None:
        npz_path = _THIS_DIR / "rotated_d3_lut_256.npz"
    z = np.load(npz_path, allow_pickle=False)
    lut16 = z["lut16"].astype(np.uint16, copy=False)
    Hx = z["Hx"].astype(np.uint8, copy=False)
    Hz = z["Hz"].astype(np.uint8, copy=False)
    meta = {}
    if "meta" in z:
        meta_item = z["meta"].item()
        if isinstance(meta_item, str):
            meta = json.loads(meta_item)
        else:
            meta = meta_item
    return lut16, Hx, Hz, meta

def decode_bytes(synd_bytes: np.ndarray, lut16: np.ndarray = None) -> np.ndarray:
    """
    Args:
      synd_bytes: np.uint8 shape [B], each is an 8-bit LSB-first syndrome byte.
      lut16: optional np.uint16 [256]; if None, loads default NPZ.
    Returns:
      np.uint8 corrections [B,9]
    """
    if lut16 is None:
        lut16, *_ = load_rotated_d3_lut_npz()
    assert synd_bytes.dtype == np.uint8 and synd_bytes.ndim == 1
    ext = _ensure_ext()
    # Move inputs to CUDA
    d_s = torch.from_numpy(synd_bytes).to("cuda", dtype=torch.uint8, non_blocking=True)
    lut_cpu = torch.from_numpy(lut16).contiguous()  # CPU tensor
    d_out = ext.fast_decode(d_s, lut_cpu)
    return d_out.cpu().numpy()

# Persistent LUT implementation
from torch.utils.cpp_extension import load as _load_ext_persist

_EXT_PERSIST = None

def _ensure_persist_ext():
    global _EXT_PERSIST
    if _EXT_PERSIST is not None:
        return _EXT_PERSIST
    src = os.path.join(os.path.dirname(__file__), "persist_lut.cu")
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    try:
        _EXT_PERSIST = _load_ext_persist(
            name="fastpath_persist_ext",
            sources=[src],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-Xptxas=-v", "-arch=sm_90a"],
            verbose=False
        )
    except Exception as e:
        _EXT_PERSIST = None
        print("[fastpath] WARN: persistent extension failed to build:", e)
    return _EXT_PERSIST

class PersistentLUT:
    def __init__(self, lut16, capacity=1024):
        self.lut16 = np.asarray(lut16, dtype=np.uint16)
        self.capacity = int(capacity)
        self._alive = False

    def __enter__(self):
        # calls into C++ persist_start; returns a handle id or warmup tensor
        _fastpath_ext = _ensure_persist_ext()
        _fastpath_ext.persist_start(torch.from_numpy(self.lut16), self.capacity)
        self._alive = True
        return self

    def decode_bytes(self, synd_bytes: np.ndarray) -> np.ndarray:
        _fastpath_ext = _ensure_persist_ext()
        t = _fastpath_ext.persist_submit(
            torch.from_numpy(np.asarray(synd_bytes, dtype=np.uint8))
        )
        # extension returns a CPU torch tensor [N,9]; convert to numpy
        return t.numpy()

    # Backward-compatible alias for realtime service
    def decode_batch(self, synd_bytes: np.ndarray) -> np.ndarray:
        return self.decode_bytes(synd_bytes)

    def __exit__(self, exc_type, exc, tb):
        if self._alive:
            _fastpath_ext = _ensure_persist_ext()
            _fastpath_ext.persist_stop()
            self._alive = False

def shared_library_path():
    """Get path to libfastpath.so for ctypes users."""
    so_path = _THIS_DIR / "c_api" / "build" / "libfastpath.so"
    return str(so_path)

def ctypes_load_fastpath():
    """Load fastpath shared library using ctypes for C API access."""
    import ctypes
    so_path = shared_library_path()
    if not os.path.exists(so_path):
        raise FileNotFoundError(f"libfastpath.so not found at {so_path}. Run tools/build_fastpath.sh first.")
    return ctypes.CDLL(so_path)
