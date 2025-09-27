#!/usr/bin/env python3
import numpy as np, json, hashlib, os
from pathlib import Path
from itertools import combinations

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

def _unpack_byte_lsbf(b: int):
    return np.array([(b >> i) & 1 for i in range(8)], dtype=np.uint8)

def _gf2_solve_minimum_weight(H, syndrome):
    """Find minimum weight solution to H @ x = syndrome (mod 2)."""
    n = H.shape[1]
    
    # Try solutions of increasing weight
    for weight in range(n + 1):
        if weight == 0:
            # Try zero solution
            if np.array_equal((H @ np.zeros(n, dtype=np.uint8)) % 2, syndrome):
                return np.zeros(n, dtype=np.uint8)
        else:
            # Try all combinations of 'weight' positions
            for positions in combinations(range(n), weight):
                x = np.zeros(n, dtype=np.uint8)
                for pos in positions:
                    x[pos] = 1
                
                if np.array_equal((H @ x) % 2, syndrome):
                    return x
    
    # No solution found
    return None

def main():
    pack = Path("student_pack_p003.npz")
    z = np.load(pack, allow_pickle=False)
    Hx = z["Hx"].astype(np.uint8); Hz = z["Hz"].astype(np.uint8)
    assert Hx.shape == (4,9) and Hz.shape == (4,9)
    meta = json.loads(z["meta"].item())
    assert meta.get("syndrome_order","Z_first_then_X") == "Z_first_then_X"

    print("Building parity-complete LUT with direct mathematical approach...")
    
    # Create combined parity check matrix
    H_combined = np.vstack([Hz, Hx])  # (8, 9)
    
    # Build all 256 syndromes and find minimum weight corrections
    lut_corrections = np.zeros((256, 9), dtype=np.uint8)
    
    for s in range(256):
        syndrome_8bit = _unpack_byte_lsbf(s)
        
        # Find minimum weight solution
        correction = _gf2_solve_minimum_weight(H_combined, syndrome_8bit)
        
        if correction is not None:
            lut_corrections[s] = correction
            print(f"Syndrome {s:3d}: found weight-{int(correction.sum())} correction")
        else:
            print(f"Syndrome {s:3d}: no solution found, using zero")
            # This shouldn't happen for a valid surface code

    print("Verifying all corrections...")
    errors = 0
    for s in range(256):
        expected_syndrome = _unpack_byte_lsbf(s)
        correction = lut_corrections[s]
        computed_syndrome = (H_combined @ correction) % 2
        
        if not np.array_equal(computed_syndrome, expected_syndrome):
            sZ_exp = expected_syndrome[:4]; sX_exp = expected_syndrome[4:8]
            sZ_got = computed_syndrome[:4]; sX_got = computed_syndrome[4:8]
            print(f"ERROR: Syndrome {s}: expected sZ={sZ_exp} sX={sX_exp}, got sZ={sZ_got} sX={sX_got}")
            errors += 1

    if errors > 0:
        print(f"❌ {errors} verification errors found")
        return 1

    print("✅ All 256 syndromes verified successfully!")

    # Pack 9 bits per row into uint16 (LSB-first)
    lut16 = np.zeros((256,), dtype=np.uint16)
    for i in range(256):
        v = 0
        row = lut_corrections[i]
        for j in range(9):
            v |= (int(row[j]) & 1) << j
        lut16[i] = v

    out_lut = Path("fastpath") / "rotated_d3_lut_256.npz"
    out_lut.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_lut,
        lut16=lut16, Hx=Hx, Hz=Hz, meta=z["meta"],
        hx_hash=hashlib.sha256(Hx.tobytes()).hexdigest(),
        hz_hash=hashlib.sha256(Hz.tobytes()).hexdigest()
    )
    print("Saved parity-complete LUT to", out_lut)
    return 0

if __name__ == "__main__":
    exit(main())
