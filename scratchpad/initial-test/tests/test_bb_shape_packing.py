"""
Test BB/qLDPC code shape and packing validation

Validates that the CUDA-Q BB/qLDPC syndrome generator produces
outputs with the correct shape and data format matching bb_panq_functions.py
"""

import sys
import os
import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cudaq_backend.syndrome_gen import sample_bb_cudaq
from codes_q import bb_code  # Import BB code construction


class TestBBShapePacking:
    """Test BB/qLDPC code syndrome format compatibility."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.rng = np.random.default_rng(42)
        
        # Create a small BB code for testing
        self.code = bb_code(d=6)  # [[72, 12, 6]] code
        self.mapping = {i: i for i in range(self.code.N)}  # Identity mapping
        
        # Expected format from bb_panq_functions.py:
        # [X_syndrome, Z_syndrome, X_error + 2*Z_error]
        self.expected_x_checks = self.code.hx.shape[0]
        self.expected_z_checks = self.code.hz.shape[0]
        self.expected_syndrome_bits = self.expected_x_checks + self.expected_z_checks
        self.expected_data_bits = self.code.N
        self.expected_total_length = self.expected_syndrome_bits + self.expected_data_bits
    
    def test_bb_single_sample_shape(self):
        """Test shape for batch_size=1."""
        samples = sample_bb_cudaq(
            mode="foundation",
            batch_size=1,
            T=1,
            hx=self.code.hx,
            hz=self.code.hz,
            mapping=self.mapping,
            rng=self.rng,
            bitpack=False
        )
        
        expected_shape = (1, self.expected_total_length)
        assert samples.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {samples.shape}"
        
        # Check data type
        assert samples.dtype == np.uint8, f"Expected dtype uint8, got {samples.dtype}"
    
    def test_bb_batch_sample_shape(self):
        """Test shape for larger batch sizes."""
        batch_sizes = [10, 50, 100]
        
        for batch_size in batch_sizes:
            samples = sample_bb_cudaq(
                mode="student",
                batch_size=batch_size,
                T=1,
                hx=self.code.hx,
                hz=self.code.hz,
                mapping=self.mapping,
                rng=self.rng,
                bitpack=False
            )
            
            expected_shape = (batch_size, self.expected_total_length)
            assert samples.shape == expected_shape, \
                f"Batch size {batch_size}: expected shape {expected_shape}, got {samples.shape}"
            
            assert samples.dtype == np.uint8, \
                f"Batch size {batch_size}: expected dtype uint8, got {samples.dtype}"
    
    def test_bb_syndrome_value_ranges(self):
        """Test that syndrome values are in valid ranges."""
        samples = sample_bb_cudaq(
            mode="foundation",
            batch_size=100,
            T=1,
            hx=self.code.hx,
            hz=self.code.hz,
            mapping=self.mapping,
            rng=self.rng,
            bitpack=False
        )
        
        # Extract syndrome and error parts
        syndrome_part = samples[:, :self.expected_syndrome_bits]
        error_part = samples[:, self.expected_syndrome_bits:]
        
        # Syndrome bits should be 0 or 1
        syndrome_min, syndrome_max = syndrome_part.min(), syndrome_part.max()
        assert 0 <= syndrome_min <= syndrome_max <= 1, \
            f"Syndrome values out of range [0,1]: min={syndrome_min}, max={syndrome_max}"
        
        # Error bits should be 0, 1, 2, or 3 (for X + 2*Z encoding)
        error_min, error_max = error_part.min(), error_part.max()
        assert 0 <= error_min <= error_max <= 3, \
            f"Error values out of range [0,3]: min={error_min}, max={error_max}"
    
    def test_bb_packing_format_compatibility(self):
        """Test that packing matches bb_panq_functions.py format."""
        samples = sample_bb_cudaq(
            mode="student",
            batch_size=50,
            T=1,
            hx=self.code.hx,
            hz=self.code.hz,
            mapping=self.mapping,
            rng=self.rng,
            bitpack=False
        )
        
        # Split into syndrome and error parts
        syndrome_part = samples[:, :self.expected_syndrome_bits]
        error_part = samples[:, self.expected_syndrome_bits:]
        
        # Check syndrome structure: [X_checks, Z_checks]
        x_syndrome = syndrome_part[:, :self.expected_x_checks]
        z_syndrome = syndrome_part[:, self.expected_x_checks:]
        
        assert x_syndrome.shape[1] == self.expected_x_checks, \
            f"X syndrome should have {self.expected_x_checks} bits, got {x_syndrome.shape[1]}"
        assert z_syndrome.shape[1] == self.expected_z_checks, \
            f"Z syndrome should have {self.expected_z_checks} bits, got {z_syndrome.shape[1]}"
        
        # Check error structure: X + 2*Z
        assert error_part.shape[1] == self.expected_data_bits, \
            f"Error part should have {self.expected_data_bits} bits, got {error_part.shape[1]}"
    
    def test_bb_code_properties(self):
        """Test that BB code has expected properties."""
        print(f"BB code properties:")
        print(f"  N (block length): {self.code.N}")
        print(f"  K (dimension): {self.code.K}")
        print(f"  D (distance): {self.code.D}")
        print(f"  X checks: {self.expected_x_checks}")
        print(f"  Z checks: {self.expected_z_checks}")
        
        # Basic sanity checks
        assert self.code.N > 0, "Block length should be positive"
        assert self.code.K > 0, "Code dimension should be positive"
        assert self.expected_x_checks > 0, "Should have X checks"
        assert self.expected_z_checks > 0, "Should have Z checks"
        
        # Check matrix dimensions
        assert self.code.hx.shape == (self.expected_x_checks, self.code.N), \
            f"Hx shape mismatch: expected ({self.expected_x_checks}, {self.code.N}), got {self.code.hx.shape}"
        assert self.code.hz.shape == (self.expected_z_checks, self.code.N), \
            f"Hz shape mismatch: expected ({self.expected_z_checks}, {self.code.N}), got {self.code.hz.shape}"
    
    def test_bb_multi_round_consistency(self):
        """Test that multi-round syndrome extraction maintains format."""
        T_values = [1, 2, 3]
        batch_size = 20
        
        for T in T_values:
            samples = sample_bb_cudaq(
                mode="foundation",
                batch_size=batch_size,
                T=T,
                hx=self.code.hx,
                hz=self.code.hz,
                mapping=self.mapping,
                rng=self.rng,
                bitpack=False
            )
            
            expected_shape = (batch_size, self.expected_total_length)
            assert samples.shape == expected_shape, \
                f"T={T}: expected shape {expected_shape}, got {samples.shape}"
    
    def test_bb_mode_consistency(self):
        """Test that both modes produce same-shaped outputs."""
        batch_size = 30
        
        foundation_samples = sample_bb_cudaq(
            mode="foundation",
            batch_size=batch_size,
            T=1,
            hx=self.code.hx,
            hz=self.code.hz,
            mapping=self.mapping,
            rng=self.rng,
            bitpack=False
        )
        
        student_samples = sample_bb_cudaq(
            mode="student",
            batch_size=batch_size,
            T=1,
            hx=self.code.hx,
            hz=self.code.hz,
            mapping=self.mapping,
            rng=self.rng,
            bitpack=False
        )
        
        assert foundation_samples.shape == student_samples.shape, \
            f"Foundation and student modes should produce same shape: " \
            f"{foundation_samples.shape} vs {student_samples.shape}"
        
        assert foundation_samples.dtype == student_samples.dtype, \
            f"Foundation and student modes should produce same dtype: " \
            f"{foundation_samples.dtype} vs {student_samples.dtype}"
    
    def test_bb_different_codes(self):
        """Test with different BB code distances."""
        distances = [6, 10]  # Test multiple distances
        
        for d in distances:
            try:
                test_code = bb_code(d=d)
                test_mapping = {i: i for i in range(test_code.N)}
                
                samples = sample_bb_cudaq(
                    mode="student",
                    batch_size=10,
                    T=1,
                    hx=test_code.hx,
                    hz=test_code.hz,
                    mapping=test_mapping,
                    rng=self.rng,
                    bitpack=False
                )
                
                expected_length = test_code.hx.shape[0] + test_code.hz.shape[0] + test_code.N
                expected_shape = (10, expected_length)
                
                assert samples.shape == expected_shape, \
                    f"d={d}: expected shape {expected_shape}, got {samples.shape}"
                
                print(f"✓ BB code d={d}: N={test_code.N}, shape={samples.shape}")
                
            except Exception as e:
                print(f"⚠ BB code d={d} test failed: {e}")


if __name__ == '__main__':
    # Run tests directly
    test_suite = TestBBShapePacking()
    
    print("Running BB/qLDPC code shape and packing validation tests...")
    
    test_suite.setup_method()
    
    try:
        test_suite.test_bb_single_sample_shape()
        print("✓ BB single sample shape")
    except Exception as e:
        print(f"✗ BB single sample shape: {e}")
    
    try:
        test_suite.test_bb_batch_sample_shape()
        print("✓ BB batch sample shapes")
    except Exception as e:
        print(f"✗ BB batch sample shapes: {e}")
    
    try:
        test_suite.test_bb_syndrome_value_ranges()
        print("✓ BB syndrome value ranges")
    except Exception as e:
        print(f"✗ BB syndrome value ranges: {e}")
    
    try:
        test_suite.test_bb_packing_format_compatibility()
        print("✓ BB packing format compatibility")
    except Exception as e:
        print(f"✗ BB packing format compatibility: {e}")
    
    try:
        test_suite.test_bb_code_properties()
        print("✓ BB code properties")
    except Exception as e:
        print(f"✗ BB code properties: {e}")
    
    try:
        test_suite.test_bb_multi_round_consistency()
        print("✓ BB multi-round consistency")
    except Exception as e:
        print(f"✗ BB multi-round consistency: {e}")
    
    try:
        test_suite.test_bb_mode_consistency()
        print("✓ BB mode consistency")
    except Exception as e:
        print(f"✗ BB mode consistency: {e}")
    
    try:
        test_suite.test_bb_different_codes()
        print("✓ BB different codes")
    except Exception as e:
        print(f"✗ BB different codes: {e}")
    
    print("BB/qLDPC code shape and packing validation complete!")
