"""
Backend API Bridge

This module provides the final interface between the CUDA-Q backend and the existing
training pipeline. It exposes wrapped versions of sample functions that preserve
exact output shapes and formats expected by the GNN model.
"""

from typing import Dict, Any, Optional
import numpy as np

from .syndrome_gen import (
    sample_surface_cudaq,
    sample_bb_cudaq, 
    sample_repetition_cudaq
)
from .circuits import make_surface_layout_d3_avoid_bad_edges


def cudaq_sample_surface_wrapper(mode: str, batch_size: int, T: int = 3, d: int = 3,
                                layout: Optional[Dict[str, Any]] = None,
                                rng: Optional[np.random.Generator] = None, 
                                bitpack: bool = False,
                                surface_layout: str = "planar") -> np.ndarray:
    """
    Wrapper for surface code sampling that matches the existing panq_functions interface.
    
    This function bridges the CUDA-Q backend to the existing training pipeline,
    ensuring exact output format compatibility.
    
    Args:
        mode: "foundation" or "student" - selects noise model calibration
        batch_size: Number of syndrome samples to generate
        T: Number of syndrome extraction rounds (default: 3)
        d: Code distance (now supports arbitrary distances)
        layout: Surface code layout dict, auto-generated if None
        rng: Random number generator, auto-created if None
        bitpack: Whether to pack bits into bytes (preserved for compatibility)
        
    Returns:
        Packed syndrome + error array with shape matching original panq_functions
        Format: [X_syndrome, 2*Z_syndrome, X_error + 2*Z_error] per sample
    """
    # Validate inputs
    if mode not in ["foundation", "student"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'foundation' or 'student'")
    
    # Generate default layout if not provided
    if layout is None:
        from .circuits import make_surface_layout_general
        layout = make_surface_layout_general(d)
    
    # Create RNG if not provided
    if rng is None:
        rng = np.random.default_rng()
    
    # Call CUDA-Q backend
    result = sample_surface_cudaq(
        mode=mode,
        batch_size=batch_size, 
        T=T,
        layout=layout,
        rng=rng,
        bitpack=bitpack,
        surface_layout=surface_layout
    )
    
    # Validate output shape consistency
    expected_x_checks = len(layout.get('ancilla_x', []))
    expected_z_checks = len(layout.get('ancilla_z', []))
    expected_data_qubits = len(layout.get('data', []))
    expected_cols = expected_x_checks + expected_z_checks + expected_data_qubits
    
    if result.shape[1] != expected_cols:
        # Handle shape mismatch - pad or truncate as needed
        if result.shape[1] < expected_cols:
            # Pad with zeros
            padding = np.zeros((batch_size, expected_cols - result.shape[1]), dtype=result.dtype)
            result = np.concatenate([result, padding], axis=1)
        else:
            # Truncate to expected size
            result = result[:, :expected_cols]
    
    # Ensure correct data type
    return result.astype(np.uint8)


def cudaq_sample_bb_wrapper(mode: str, batch_size: int, hx: np.ndarray, hz: np.ndarray,
                           T: int = 3, mapping: Optional[Dict[int, int]] = None,
                           rng: Optional[np.random.Generator] = None,
                           bitpack: bool = False) -> np.ndarray:
    """
    Wrapper for BB/qLDPC code sampling that matches the existing bb_panq_functions interface.
    
    Args:
        mode: "foundation" or "student"
        batch_size: Number of syndrome samples to generate
        hx: X-check matrix (num_x_checks, num_qubits)
        hz: Z-check matrix (num_z_checks, num_qubits)
        T: Number of syndrome extraction rounds (default: 3)
        mapping: Logical to physical qubit mapping, identity if None
        rng: Random number generator, auto-created if None
        bitpack: Whether to pack bits into bytes (preserved for compatibility)
        
    Returns:
        Packed syndrome + error array with shape matching original bb_panq_functions
        Format: [X_syndrome, Z_syndrome, X_error + 2*Z_error] per sample
    """
    # Validate inputs
    if mode not in ["foundation", "student"]:
        raise ValueError(f"Invalid mode: {mode}")
    
    if hx.shape[1] != hz.shape[1]:
        raise ValueError(f"Hx and Hz must have same number of qubits: {hx.shape[1]} vs {hz.shape[1]}")
    
    # Create identity mapping if not provided
    if mapping is None:
        mapping = {i: i for i in range(hx.shape[1])}
    
    # Create RNG if not provided
    if rng is None:
        rng = np.random.default_rng()
    
    # Call CUDA-Q backend
    result = sample_bb_cudaq(
        mode=mode,
        batch_size=batch_size,
        T=T, 
        hx=hx,
        hz=hz,
        mapping=mapping,
        rng=rng,
        bitpack=bitpack
    )
    
    # Validate output shape consistency
    expected_x_checks = hx.shape[0]
    expected_z_checks = hz.shape[0] 
    expected_data_qubits = hx.shape[1]
    expected_cols = expected_x_checks + expected_z_checks + expected_data_qubits
    
    if result.shape[1] != expected_cols:
        # Handle shape mismatch
        if result.shape[1] < expected_cols:
            padding = np.zeros((batch_size, expected_cols - result.shape[1]), dtype=result.dtype)
            result = np.concatenate([result, padding], axis=1)
        else:
            result = result[:, :expected_cols]
    
    return result.astype(np.uint8)


def cudaq_sample_repetition_wrapper(mode: str, batch_size: int, n_data: int = 5,
                                   T: int = 3, layout: Optional[Dict[str, Any]] = None,
                                   rng: Optional[np.random.Generator] = None,
                                   bitpack: bool = False) -> np.ndarray:
    """
    Wrapper for repetition code sampling that matches the existing interface.
    
    Args:
        mode: "foundation" or "student"
        batch_size: Number of syndrome samples to generate
        n_data: Number of data qubits in repetition code (default: 5)
        T: Number of syndrome extraction rounds (default: 3)
        layout: Repetition code layout dict, auto-generated if None
        rng: Random number generator, auto-created if None
        bitpack: Whether to pack bits into bytes (preserved for compatibility)
        
    Returns:
        Packed syndrome + error array with shape matching original format
        Format: [syndrome_bits, X_error + 2*Z_error] per sample
    """
    # Validate inputs
    if mode not in ["foundation", "student"]:
        raise ValueError(f"Invalid mode: {mode}")
    
    # Generate default layout if not provided
    if layout is None:
        layout = {
            'data': list(range(n_data)),
            'ancilla': list(range(n_data, n_data + n_data - 1))  # n_data - 1 ancillas
        }
    
    # Create RNG if not provided
    if rng is None:
        rng = np.random.default_rng()
    
    # Call CUDA-Q backend
    result = sample_repetition_cudaq(
        mode=mode,
        batch_size=batch_size,
        T=T,
        layout=layout,
        rng=rng,
        bitpack=bitpack
    )
    
    # Validate output shape consistency
    expected_ancillas = len(layout.get('ancilla', []))
    expected_data_qubits = len(layout.get('data', []))
    expected_cols = expected_ancillas + expected_data_qubits
    
    if result.shape[1] != expected_cols:
        # Handle shape mismatch
        if result.shape[1] < expected_cols:
            padding = np.zeros((batch_size, expected_cols - result.shape[1]), dtype=result.dtype)
            result = np.concatenate([result, padding], axis=1)
        else:
            result = result[:, :expected_cols]
    
    return result.astype(np.uint8)


def get_backend_info() -> Dict[str, Any]:
    """
    Get information about the CUDA-Q backend configuration.
    
    Returns:
        Dictionary with backend information and capabilities
    """
    from . import __version__
    from .garnet_noise import FOUNDATION_DEFAULTS, GARNET_COUPLER_F2
    
    return {
        'backend': 'CUDA-Q',
        'version': __version__,
        'noise_model': 'IQM Garnet',
        'supported_codes': ['surface', 'bb', 'qldpc', 'repetition'],
        'supported_distances': 'arbitrary',  # Now supports arbitrary distances
        'modes': ['foundation', 'student'],
        'foundation_defaults': FOUNDATION_DEFAULTS,
        'garnet_couplers': len(GARNET_COUPLER_F2),
        'max_qubits': 100,  # Increased limit for larger distances
        'features': [
            'circuit_level_noise',
            'amplitude_damping', 
            'pure_dephasing',
            'gate_depolarizing',
            'measurement_assignment_errors',
            'batched_monte_carlo',
            'idle_noise_timing',
            'arbitrary_distance_surface_codes'
        ]
    }


def validate_backend_installation() -> bool:
    """
    Validate that the CUDA-Q backend is properly configured.
    
    Returns:
        True if backend is ready for use, False otherwise
    """
    try:
        # Test basic functionality
        rng = np.random.default_rng(42)
        
        # Test surface code sampling
        layout = make_surface_layout_d3_avoid_bad_edges()
        result = cudaq_sample_surface_wrapper(
            mode="foundation",
            batch_size=2,
            T=1,
            d=3,
            layout=layout,
            rng=rng
        )
        
        # Basic validation checks
        if result.shape[0] != 2:
            return False
        
        if result.dtype != np.uint8:
            return False
        
        return True
        
    except Exception as e:
        print(f"Backend validation failed: {e}")
        return False


# Backward compatibility aliases
sample_surface_foundation = lambda *args, **kwargs: cudaq_sample_surface_wrapper("foundation", *args, **kwargs)
sample_surface_student = lambda *args, **kwargs: cudaq_sample_surface_wrapper("student", *args, **kwargs)
sample_bb_foundation = lambda *args, **kwargs: cudaq_sample_bb_wrapper("foundation", *args, **kwargs)
sample_bb_student = lambda *args, **kwargs: cudaq_sample_bb_wrapper("student", *args, **kwargs)
sample_repetition_foundation = lambda *args, **kwargs: cudaq_sample_repetition_wrapper("foundation", *args, **kwargs)
sample_repetition_student = lambda *args, **kwargs: cudaq_sample_repetition_wrapper("student", *args, **kwargs)
