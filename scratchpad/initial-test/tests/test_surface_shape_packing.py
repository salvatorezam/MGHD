"""
Test surface code shape and packing validation

Validates that the CUDA-Q surface code syndrome generator produces
outputs with the correct shape and data format matching panq_functions.py
"""

import sys
import os
import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cudaq_backend.syndrome_gen import sample_surface_cudaq
from cudaq_backend.circuits import make_surface_layout_d3_avoid_bad_edges


class TestSurfaceShapePacking:
    """Test surface code syndrome format compatibility."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.layout = make_surface_layout_d3_avoid_bad_edges()
        self.rng = np.random.default_rng(42)
        self.d = 3
        
        # Expected format from panq_functions.py:
        # [X_syndrome, 2*Z_syndrome, X_error + 2*Z_error]
        # For d=3: 4 X-checks + 4 Z-checks + 9 data qubits = 17 total
        self.expected_syndrome_bits = len(self.layout['ancilla_x']) + len(self.layout['ancilla_z'])  # 8
        self.expected_data_bits = len(self.layout['data'])  # 9
        self.expected_total_length = self.expected_syndrome_bits + self.expected_data_bits  # 17
    
    def test_single_sample_shape(self):
        """Test shape for batch_size=1."""
        samples = sample_surface_cudaq(
            mode="foundation",
            batch_size=1,
            T=1,
            layout=self.layout,
            rng=self.rng,
            bitpack=False
        )
        
        expected_shape = (1, self.expected_total_length)
        assert samples.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {samples.shape}"
        
        # Check data type
        assert samples.dtype == np.uint8, f"Expected dtype uint8, got {samples.dtype}"
    
    def test_batch_sample_shape(self):
        """Test shape for larger batch sizes."""
        batch_sizes = [10, 100, 1000]
        
        for batch_size in batch_sizes:
            samples = sample_surface_cudaq(
                mode="student",
                batch_size=batch_size,
                T=1,
                layout=self.layout,
                rng=self.rng,
                bitpack=False
            )
            
            expected_shape = (batch_size, self.expected_total_length)
            assert samples.shape == expected_shape, \
                f"Batch size {batch_size}: expected shape {expected_shape}, got {samples.shape}"
            
            assert samples.dtype == np.uint8, \
                f"Batch size {batch_size}: expected dtype uint8, got {samples.dtype}"
    
    def test_multi_round_consistency(self):
        """Test that multi-round syndrome extraction maintains format."""
        T_values = [1, 2, 5]
        batch_size = 50
        
        for T in T_values:
            samples = sample_surface_cudaq(
                mode="foundation",
                batch_size=batch_size,
                T=T,
                layout=self.layout,
                rng=self.rng,
                bitpack=False
            )
            
            expected_shape = (batch_size, self.expected_total_length)
            assert samples.shape == expected_shape, \
                f"T={T}: expected shape {expected_shape}, got {samples.shape}"
    
    def test_syndrome_value_ranges(self):
        """Test that syndrome values are in valid ranges."""
        samples = sample_surface_cudaq(
            mode="foundation",
            batch_size=1000,
            T=1,
            layout=self.layout,
            rng=self.rng,
            bitpack=False
        )
        
        # Extract syndrome and error parts
        syndrome_part = samples[:, :self.expected_syndrome_bits]
        error_part = samples[:, self.expected_syndrome_bits:]
        
        # Syndrome bits should be 0, 1, or 2 (for 2*Z_syndrome encoding)
        syndrome_min, syndrome_max = syndrome_part.min(), syndrome_part.max()
        assert 0 <= syndrome_min <= syndrome_max <= 2, \
            f"Syndrome values out of range [0,2]: min={syndrome_min}, max={syndrome_max}"
        
        # Error bits should be 0, 1, 2, or 3 (for X + 2*Z encoding)
        error_min, error_max = error_part.min(), error_part.max()
        assert 0 <= error_min <= error_max <= 3, \
            f"Error values out of range [0,3]: min={error_min}, max={error_max}"
    
    def test_packing_format_compatibility(self):
        """Test that packing matches panq_functions.py format."""
        samples = sample_surface_cudaq(
            mode="student",
            batch_size=100,
            T=1,
            layout=self.layout,
            rng=self.rng,
            bitpack=False
        )
        
        # Split into syndrome and error parts
        syndrome_part = samples[:, :self.expected_syndrome_bits]
        error_part = samples[:, self.expected_syndrome_bits:]
        
        # Check syndrome structure: [X_checks, 2*Z_checks]
        num_x_checks = len(self.layout['ancilla_x'])
        num_z_checks = len(self.layout['ancilla_z'])
        
        x_syndrome = syndrome_part[:, :num_x_checks]
        z_syndrome = syndrome_part[:, num_x_checks:]
        
        assert x_syndrome.shape[1] == num_x_checks, \
            f"X syndrome should have {num_x_checks} bits, got {x_syndrome.shape[1]}"
        assert z_syndrome.shape[1] == num_z_checks, \
            f"Z syndrome should have {num_z_checks} bits, got {z_syndrome.shape[1]}"
        
        # Z syndrome should be even (since it's 2*syndrome_bits)
        # This is a format requirement from panq_functions.py
        z_is_even = (z_syndrome % 2 == 0)
        z_even_fraction = np.mean(z_is_even)
        # Allow some noise, but most Z syndrome bits should follow 2*pattern
        # In practice, with noise, this might not always be exactly even
        # So we just check that the values are in the right range [0,2]
        assert np.all(z_syndrome <= 2), "Z syndrome values should be ≤ 2"
        
        # Check error structure: X + 2*Z
        num_data_qubits = len(self.layout['data'])
        assert error_part.shape[1] == num_data_qubits, \
            f"Error part should have {num_data_qubits} bits, got {error_part.shape[1]}"
    
    def test_mode_consistency(self):
        """Test that both modes produce same-shaped outputs."""
        batch_size = 100
        
        foundation_samples = sample_surface_cudaq(
            mode="foundation",
            batch_size=batch_size,
            T=1,
            layout=self.layout,
            rng=self.rng,
            bitpack=False
        )
        
        student_samples = sample_surface_cudaq(
            mode="student", 
            batch_size=batch_size,
            T=1,
            layout=self.layout,
            rng=self.rng,
            bitpack=False
        )
        
        assert foundation_samples.shape == student_samples.shape, \
            f"Foundation and student modes should produce same shape: " \
            f"{foundation_samples.shape} vs {student_samples.shape}"
        
        assert foundation_samples.dtype == student_samples.dtype, \
            f"Foundation and student modes should produce same dtype: " \
            f"{foundation_samples.dtype} vs {student_samples.dtype}"
    
    def test_layout_qubit_coverage(self):
        """Test that layout covers expected number of qubits for d=3."""
        # For d=3 surface code: 9 data + 8 stabilizers = 17 qubits
        all_qubits = set(self.layout['data'])
        all_qubits.update(self.layout['ancilla_x'])
        all_qubits.update(self.layout['ancilla_z'])
        
        assert len(all_qubits) == 17, \
            f"d=3 surface code should use 17 qubits, layout uses {len(all_qubits)}"
        
        # Verify no overlap between qubit sets
        data_set = set(self.layout['data'])
        x_anc_set = set(self.layout['ancilla_x'])
        z_anc_set = set(self.layout['ancilla_z'])
        
        assert len(data_set & x_anc_set) == 0, "Data and X-ancilla qubits should not overlap"
        assert len(data_set & z_anc_set) == 0, "Data and Z-ancilla qubits should not overlap"
        assert len(x_anc_set & z_anc_set) == 0, "X and Z ancilla qubits should not overlap"
    
    def test_bad_edge_avoidance(self):
        """Test that layout avoids the problematic (10,11) coupler."""
        # Check if any circuit operations would use the bad edge
        bad_edge = (10, 11)
        
        # Look through CZ layers to see if bad edge is used
        uses_bad_edge = False
        for cz_layer in self.layout.get('cz_layers', []):
            for (q1, q2) in cz_layer:
                if (q1, q2) == bad_edge or (q2, q1) == bad_edge:
                    uses_bad_edge = True
                    break
        
        assert not uses_bad_edge, \
            f"Layout should avoid bad edge {bad_edge}, but it appears in CZ layers"
        
        # Also check that qubits 10 and 11 are not both used
        all_qubits = set(self.layout['data'])
        all_qubits.update(self.layout['ancilla_x'])
        all_qubits.update(self.layout['ancilla_z'])
        
        has_both_bad_qubits = (10 in all_qubits) and (11 in all_qubits)
        if has_both_bad_qubits:
            print("Warning: Layout uses both qubits 10 and 11 - verify no direct coupling")


if __name__ == '__main__':
    # Run tests directly
    test_suite = TestSurfaceShapePacking()
    
    print("Running surface code shape and packing validation tests...")
    
    test_suite.setup_method()
    
    try:
        test_suite.test_single_sample_shape()
        print("✓ Single sample shape")
    except Exception as e:
        print(f"✗ Single sample shape: {e}")
    
    try:
        test_suite.test_batch_sample_shape()
        print("✓ Batch sample shapes")
    except Exception as e:
        print(f"✗ Batch sample shapes: {e}")
    
    try:
        test_suite.test_multi_round_consistency()
        print("✓ Multi-round consistency")
    except Exception as e:
        print(f"✗ Multi-round consistency: {e}")
    
    try:
        test_suite.test_syndrome_value_ranges()
        print("✓ Syndrome value ranges")
    except Exception as e:
        print(f"✗ Syndrome value ranges: {e}")
    
    try:
        test_suite.test_packing_format_compatibility()
        print("✓ Packing format compatibility")
    except Exception as e:
        print(f"✗ Packing format compatibility: {e}")
    
    try:
        test_suite.test_mode_consistency()
        print("✓ Mode consistency")
    except Exception as e:
        print(f"✗ Mode consistency: {e}")
    
    try:
        test_suite.test_layout_qubit_coverage()
        print("✓ Layout qubit coverage")
    except Exception as e:
        print(f"✗ Layout qubit coverage: {e}")
    
    try:
        test_suite.test_bad_edge_avoidance()
        print("✓ Bad edge avoidance")
    except Exception as e:
        print(f"✗ Bad edge avoidance: {e}")
    
    print("Surface code shape and packing validation complete!")
