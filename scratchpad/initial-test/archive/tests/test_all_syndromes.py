#!/usr/bin/env python3
import os, sys
sys.path.insert(0, "."); os.chdir(".")
import numpy as np
import fastpath

def test_all_syndromes():
    """Test comprehensive parity verification for all 256 syndromes."""
    Hx, Hz = fastpath.get_H_matrices()
    decode_fn = fastpath.decode_bytes
    
    print("Testing parity for all 256 syndromes...")
    errors = 0
    
    for s in range(256):
        syndrome_bytes = np.array([s], dtype=np.uint8)
        corrections = decode_fn(syndrome_bytes)
        correction = corrections[0]
        
        # Unpack syndrome (LSB-first)
        syndrome_8bit = np.array([(s >> i) & 1 for i in range(8)], dtype=np.uint8)
        
        # Verify parity: H @ correction should equal syndrome
        H_combined = np.vstack([Hz, Hx])  # (8, 9)
        computed_syndrome = (H_combined @ correction) % 2
        
        if not np.array_equal(computed_syndrome, syndrome_8bit):
            sZ_exp, sX_exp = syndrome_8bit[:4], syndrome_8bit[4:8]
            sZ_got, sX_got = computed_syndrome[:4], computed_syndrome[4:8]
            print(f"ERROR: Syndrome {s:3d}: expected sZ={sZ_exp} sX={sX_exp}, got sZ={sZ_got} sX={sX_got}")
            errors += 1
        elif s % 50 == 0 or s < 20:
            print(f"✓ Syndrome {s:3d}: parity correct (weight {correction.sum()})")
    
    if errors == 0:
        print(f"✅ All 256 syndromes have correct parity!")
    else:
        print(f"❌ {errors} parity errors found")
    
    return errors == 0

if __name__ == "__main__":
    success = test_all_syndromes()
    sys.exit(0 if success else 1)
