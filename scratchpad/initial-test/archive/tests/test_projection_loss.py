import numpy as np, torch
import scipy.sparse as sp
from training.cluster_crops_train import projection_aware_logits_to_bits

def test_projection_monotonicity():
    logits = torch.zeros((8,2), dtype=torch.float32)
    logits[:,1] = torch.linspace(-4, 4, 8)
    
    # Create a dummy H_sub and s_sub for projection
    H_sub = sp.csr_matrix(np.eye(4, 8, dtype=np.uint8))  # 4 checks, 8 qubits
    s_sub = np.zeros(4, dtype=np.uint8)
    
    bits = projection_aware_logits_to_bits(logits, projector_kwargs={"H_sub": H_sub, "s_sub": s_sub})
    # more positive logits -> more ones expected overall under threshold fallback
    assert bits.sum() >= 3

def test_projection_consistency():
    # Test that projection gives consistent results for same input
    logits = torch.randn((5,2), dtype=torch.float32)
    logits[:,1] = torch.tensor([2.0, -1.0, 0.5, -2.0, 1.5])  # deterministic
    
    # Create dummy constraints
    H_sub = sp.csr_matrix(np.array([[1,1,0,0,0],[0,0,1,1,1]], dtype=np.uint8))
    s_sub = np.array([0, 1], dtype=np.uint8)
    
    bits1 = projection_aware_logits_to_bits(logits.clone(), projector_kwargs={"H_sub": H_sub, "s_sub": s_sub})
    bits2 = projection_aware_logits_to_bits(logits.clone(), projector_kwargs={"H_sub": H_sub, "s_sub": s_sub})
    
    # Should get identical results
    assert np.array_equal(bits1, bits2), "Projection should be deterministic"

def test_projection_bounds():
    # Test that projection always returns valid bits
    logits = torch.randn((10,2), dtype=torch.float32) * 5  # large range
    
    # Create constraints
    H_sub = sp.csr_matrix(np.random.randint(0, 2, (3, 10), dtype=np.uint8))
    s_sub = np.random.randint(0, 2, 3, dtype=np.uint8)
    
    bits = projection_aware_logits_to_bits(logits, projector_kwargs={"H_sub": H_sub, "s_sub": s_sub})
    
    # All bits should be 0 or 1
    assert np.all((bits == 0) | (bits == 1)), "All projected bits should be 0 or 1"
    assert bits.dtype == np.uint8, "Projected bits should be uint8"

if __name__ == "__main__":
    test_projection_monotonicity()
    test_projection_consistency()
    test_projection_bounds()
    print("test_projection_loss.py: PASSED")