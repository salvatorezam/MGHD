"""
Test suite for p_depol mapping validation

Validates that the fidelity-to-depolarizing-probability conversion
follows the correct formula: p = (1 - F) * (d+1) / d
"""

import sys
import os
import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cudaq_backend.garnet_noise import GarnetNoiseModel


class TestPDepolMapping:
    """Test fidelity to depolarizing probability conversion."""
    
    def test_single_qubit_depol_formula(self):
        """Test d=2 (single-qubit) depolarizing formula."""
        # For single-qubit gates: F_avg = 1 - 3p/4, so p = 4(1-F)/3
        test_cases = [
            (1.0, 0.0),      # Perfect fidelity -> no error
            (0.99, 4/3 * 0.01),  # F=0.99 -> p=4/3 * 0.01
            (0.9989, 4/3 * 0.0011),  # Garnet median F1Q
            (0.95, 4/3 * 0.05),  # F=0.95 -> p=4/3 * 0.05
            (0.75, 4/3 * 0.25),  # F=0.75 -> p=1/3
        ]
        
        for F_avg, expected_p in test_cases:
            actual_p = GarnetNoiseModel.p_depol_from_F(F_avg, d=2)
            np.testing.assert_almost_equal(actual_p, expected_p, decimal=6,
                err_msg=f"F={F_avg}: expected p={expected_p:.6f}, got p={actual_p:.6f}")
    
    def test_two_qubit_depol_formula(self):
        """Test d=4 (two-qubit) depolarizing formula."""
        # For two-qubit gates: F_avg = 1 - 15p/16, so p = 16(1-F)/15
        test_cases = [
            (1.0, 0.0),      # Perfect fidelity -> no error
            (0.99, 16/15 * 0.01),  # F=0.99 -> p=16/15 * 0.01
            (0.9906, 16/15 * 0.0094),  # Garnet median F2Q
            (0.95, 16/15 * 0.05),   # F=0.95 -> p=16/15 * 0.05
            (0.9228, 16/15 * 0.0772),  # Bad Garnet edge (10,11)
        ]
        
        for F_avg, expected_p in test_cases:
            actual_p = GarnetNoiseModel.p_depol_from_F(F_avg, d=4)
            np.testing.assert_almost_equal(actual_p, expected_p, decimal=6,
                err_msg=f"F={F_avg}: expected p={expected_p:.6f}, got p={actual_p:.6f}")
    
    def test_boundary_conditions(self):
        """Test boundary conditions for fidelity conversion."""
        # Test F=0 (worst possible fidelity)
        p_1q_worst = GarnetNoiseModel.p_depol_from_F(0.0, d=2)
        p_2q_worst = GarnetNoiseModel.p_depol_from_F(0.0, d=4)
        
        assert p_1q_worst == 1.0, f"Expected p=1 for F=0 (1Q), got p={p_1q_worst}"
        assert p_2q_worst == 1.0, f"Expected p=1 for F=0 (2Q), got p={p_2q_worst}"
        
        # Test F=1 (perfect fidelity)
        p_1q_perfect = GarnetNoiseModel.p_depol_from_F(1.0, d=2)
        p_2q_perfect = GarnetNoiseModel.p_depol_from_F(1.0, d=4)
        
        assert p_1q_perfect == 0.0, f"Expected p=0 for F=1 (1Q), got p={p_1q_perfect}"
        assert p_2q_perfect == 0.0, f"Expected p=0 for F=1 (2Q), got p={p_2q_perfect}"
        
        # Test clamping for invalid inputs
        p_invalid_high = GarnetNoiseModel.p_depol_from_F(1.5, d=2)  # F > 1
        p_invalid_low = GarnetNoiseModel.p_depol_from_F(-0.1, d=2)  # F < 0
        
        assert 0.0 <= p_invalid_high <= 1.0, f"p should be clamped to [0,1], got {p_invalid_high}"
        assert 0.0 <= p_invalid_low <= 1.0, f"p should be clamped to [0,1], got {p_invalid_low}"
    
    def test_garnet_specific_values(self):
        """Test conversion for specific Garnet calibration values."""
        from cudaq_backend.garnet_noise import FOUNDATION_DEFAULTS, GARNET_COUPLER_F2
        
        # Test median values
        F1Q_median = FOUNDATION_DEFAULTS["F1Q_median"]
        F2Q_median = FOUNDATION_DEFAULTS["F2Q_median"]
        
        p1q_median = GarnetNoiseModel.p_depol_from_F(F1Q_median, d=2)
        p2q_median = GarnetNoiseModel.p_depol_from_F(F2Q_median, d=4)
        
        # Sanity checks: probabilities should be small for high fidelities
        assert 0.0 < p1q_median < 0.02, f"F1Q median p={p1q_median:.6f} seems unreasonable"
        assert 0.0 < p2q_median < 0.02, f"F2Q median p={p2q_median:.6f} seems unreasonable"
        
        # Test bad edge
        bad_edge_fidelity = GARNET_COUPLER_F2[(10, 11)]
        p_bad_edge = GarnetNoiseModel.p_depol_from_F(bad_edge_fidelity, d=4)
        
        # Bad edge should have higher error probability
        assert p_bad_edge > p2q_median, f"Bad edge p={p_bad_edge:.6f} should exceed median p={p2q_median:.6f}"
        assert p_bad_edge < 0.1, f"Bad edge p={p_bad_edge:.6f} seems too high"
    
    def test_noise_model_integration(self):
        """Test that GarnetNoiseModel correctly uses the conversion formula."""
        from cudaq_backend.garnet_noise import GarnetStudentCalibration
        
        # Create noise model with known parameters
        calibration = GarnetStudentCalibration()
        params = calibration.to_dict()
        noise_model = GarnetNoiseModel(params)
        
        # Test single-qubit gates
        for q in range(5):  # Test first 5 qubits
            p_actual = noise_model.depol_1q_p(q)
            F1Q = params['F1Q'][q]
            p_expected = GarnetNoiseModel.p_depol_from_F(F1Q, d=2)
            
            np.testing.assert_almost_equal(p_actual, p_expected, decimal=10,
                err_msg=f"Qubit {q}: noise model returned wrong 1Q depol prob")
        
        # Test two-qubit gates
        test_edges = [(0, 1), (10, 11), (5, 6)]  # Mix of good and bad edges
        for edge in test_edges:
            if edge in params['F2Q']:
                p_actual = noise_model.depol_2q_p(edge)
                F2Q = params['F2Q'][edge]
                p_expected = GarnetNoiseModel.p_depol_from_F(F2Q, d=4)
                
                np.testing.assert_almost_equal(p_actual, p_expected, decimal=10,
                    err_msg=f"Edge {edge}: noise model returned wrong 2Q depol prob")


if __name__ == '__main__':
    # Run tests directly
    test_suite = TestPDepolMapping()
    
    print("Running p_depol mapping validation tests...")
    
    try:
        test_suite.test_single_qubit_depol_formula()
        print("✓ Single-qubit depolarizing formula")
    except Exception as e:
        print(f"✗ Single-qubit depolarizing formula: {e}")
    
    try:
        test_suite.test_two_qubit_depol_formula()
        print("✓ Two-qubit depolarizing formula")
    except Exception as e:
        print(f"✗ Two-qubit depolarizing formula: {e}")
    
    try:
        test_suite.test_boundary_conditions()
        print("✓ Boundary conditions")
    except Exception as e:
        print(f"✗ Boundary conditions: {e}")
    
    try:
        test_suite.test_garnet_specific_values()
        print("✓ Garnet-specific values")
    except Exception as e:
        print(f"✗ Garnet-specific values: {e}")
    
    try:
        test_suite.test_noise_model_integration()
        print("✓ Noise model integration")
    except Exception as e:
        print(f"✗ Noise model integration: {e}")
    
    print("p_depol mapping validation complete!")
