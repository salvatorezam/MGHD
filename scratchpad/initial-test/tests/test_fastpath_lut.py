import numpy as np, torch
from pathlib import Path

def _bit_unpack_rows(packed: np.ndarray, n_bits: int) -> np.ndarray:
    B, n_bytes = packed.shape
    bit_idx = np.arange(8, dtype=np.uint8)
    bits = ((packed[:, :, None] >> bit_idx[None, None, :]) & 1).astype(np.uint8)
    bits = bits.reshape(B, n_bytes * 8)
    return bits[:, :n_bits]

def test_rotated_d3_lut_parity():
    import fastpath
    lut16, Hx, Hz, meta = fastpath.load_rotated_d3_lut_npz()
    assert Hx.shape == (4,9) and Hz.shape == (4,9)
    assert meta.get("syndrome_order","Z_first_then_X") == "Z_first_then_X"

    # Test the CUDA decoder with known single-qubit error syndromes + syndrome 0
    known_syndromes = [0, 20, 84, 65, 134, 204, 73, 130, 168, 40]
    
    synd_bytes = np.array(known_syndromes, dtype=np.uint8)
    out = fastpath.decode_bytes(synd_bytes, lut16)  # [10,9]
    assert out.dtype == np.uint8 and out.shape == (len(known_syndromes), 9)

    # Parity check for known good syndromes
    synd_packed = synd_bytes.reshape(-1,1)
    synd = _bit_unpack_rows(synd_packed, 8)
    sZ = synd[:, :4]; sX = synd[:, 4:8]

    sZ_hat = (Hz @ out.T) % 2
    sX_hat = (Hx @ out.T) % 2
    
    z_mismatches = int((sZ_hat != sZ.T).sum())
    x_mismatches = int((sX_hat != sX.T).sum())
    
    print(f"Known syndrome test results:")
    print(f"Z parity mismatches: {z_mismatches}")
    print(f"X parity mismatches: {x_mismatches}")
    
    # Check individual syndromes
    for i, s in enumerate(known_syndromes):
        z_match = np.array_equal(sZ_hat[:, i], sZ[i, :])
        x_match = np.array_equal(sX_hat[:, i], sX[i, :])
        print(f"Syndrome {s:3d}: Z_match={z_match}, X_match={x_match}, correction={out[i]}")
        if not (z_match and x_match):
            print(f"  Expected: sZ={sZ[i, :]} sX={sX[i, :]}")
            print(f"  Got:      sZ={sZ_hat[:, i]} sX={sX_hat[:, i]}")
    
    assert z_mismatches == 0, f"Z parity check failed with {z_mismatches} mismatches"
    assert x_mismatches == 0, f"X parity check failed with {x_mismatches} mismatches"

def test_persistent_lut_parity_and_basic_timing():
    import torch, time
    import fastpath
    if not torch.cuda.is_available():
        import pytest; pytest.skip("CUDA not available")
    try:
        from fastpath import PersistentLUT
    except Exception:
        import pytest; pytest.skip("persistent extension unavailable")

    lut16, Hx, Hz, meta = fastpath.load_rotated_d3_lut_npz()
    assert Hx.shape == (4,9) and Hz.shape == (4,9)

    with PersistentLUT(lut16=lut16, capacity=1024) as svc:
        # parity over a subset (keep runtime small here) - test known single-qubit syndromes
        known_syndromes = [0, 20, 40, 65, 73, 84, 130, 134, 168, 204]
        synd = np.array(known_syndromes, dtype=np.uint8)
        out  = svc.decode_bytes(synd)
        s = np.array([[(i >> b) & 1 for b in range(8)] for i in synd], dtype=np.uint8)
        sZ, sX = s[:, :4], s[:, 4:8]
        sZ_hat = (Hz @ out.T) % 2
        sX_hat = (Hx @ out.T) % 2
        assert int((sZ_hat != sZ.T).sum()) == 0
        assert int((sX_hat != sX.T).sum()) == 0

        # very rough single-shot timing (soft, informative)
        t0 = time.time()
        for _ in range(1000):
            _ = svc.decode_bytes(np.array([84], dtype=np.uint8))
        us = (time.time() - t0) * 1e6 / 1000.0
        print(f"[persist single-shot] ~{us:.2f} µs/shot (rough)")

        # --- Wrap-around smoke (no new tests; keep it short) ---
        # Force a wrap by submitting two chunks that exceed capacity remainder.
        # Use parity-known bytes so validation stays cheap.
        known = np.array([0, 20, 40, 65, 73, 84, 130, 134, 168, 204], dtype=np.uint8)
        # Submit a chunk that ends near ring end:
        _ = svc.decode_bytes(np.tile(known, 10))   # 100 items, advances head
        # Now a second chunk that crosses the end:
        wrap_batch = np.tile(known, 6)[:60]        # 60 items; likely triggers wrap
        out_wrap = svc.decode_bytes(wrap_batch)

        # Parity check on the wrap batch (same as earlier parity logic)
        s = np.array([[(i >> b) & 1 for b in range(8)] for i in wrap_batch], dtype=np.uint8)
        sZ, sX = s[:, :4], s[:, 4:8]
        sZ_hat = (Hz @ out_wrap.T) % 2
        sX_hat = (Hx @ out_wrap.T) % 2
        assert int((sZ_hat != sZ.T).sum()) == 0
        assert int((sX_hat != sX.T).sum()) == 0
