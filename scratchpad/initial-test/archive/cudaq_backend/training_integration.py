"""
Training Integration Module

This module modifies the training pipeline to use CUDA-Q backend while preserving
exact compatibility with existing model interfaces and loss computation.

Key features:
- Drop-in replacement for existing sampling functions
- Preserves all data shapes and formats
- Optional fallback to original implementations
- Performance monitoring and validation
"""

import time
import warnings
from typing import Dict, Any, Optional, Callable, Union
import numpy as np
import torch

# Try to import CUDA-Q backend
try:
    from cudaq_backend import (
        cudaq_sample_surface_wrapper,
        cudaq_sample_bb_wrapper,
        cudaq_sample_repetition_wrapper,
        validate_backend_installation,
        get_backend_info
    )
    CUDAQ_AVAILABLE = True
    CUDAQ_VALIDATED = validate_backend_installation()
except ImportError as e:
    CUDAQ_AVAILABLE = False
    CUDAQ_VALIDATED = False
    print(f"CUDA-Q backend not available: {e}")


class CudaQSamplingMode:
    """Configuration class for CUDA-Q backend usage."""
    
    def __init__(self, 
                 enabled: bool = True,
                 mode: str = "foundation",
                 fallback_on_error: bool = True,
                 validate_outputs: bool = True,
                 performance_logging: bool = False,
                 surface_layout: str = "planar"):
        """
        Initialize CUDA-Q sampling configuration.
        
        Args:
            enabled: Whether to use CUDA-Q backend
            mode: "foundation" or "student" noise model
            fallback_on_error: Whether to fall back to original implementation on errors
            validate_outputs: Whether to validate output shapes and ranges
            performance_logging: Whether to log timing comparisons
        """
        self.enabled = enabled and CUDAQ_AVAILABLE and CUDAQ_VALIDATED
        self.mode = mode
        self.fallback_on_error = fallback_on_error
        self.validate_outputs = validate_outputs
        self.performance_logging = performance_logging
        self.surface_layout = surface_layout
        
        if enabled and not CUDAQ_AVAILABLE:
            warnings.warn("CUDA-Q backend requested but not available. Using fallback.")
        
        if enabled and CUDAQ_AVAILABLE and not CUDAQ_VALIDATED:
            warnings.warn("CUDA-Q backend failed validation. Using fallback.")


def create_cudaq_surface_sampler(original_sampler: Callable, config: CudaQSamplingMode) -> Callable:
    """
    Create a CUDA-Q-enabled surface code sampler that wraps the original function.
    
    Args:
        original_sampler: The original sampling function to wrap/replace
        config: CUDA-Q configuration settings
        
    Returns:
        Wrapped sampling function with CUDA-Q backend
    """
    def cudaq_surface_sampler(*args, **kwargs):
        """CUDA-Q surface code sampler with fallback."""
        
        if not config.enabled:
            return original_sampler(*args, **kwargs)
        
        # Extract parameters (adapt to actual original_sampler signature)
        batch_size = args[0] if len(args) > 0 else kwargs.get('batch_size', 1000)
        T = args[1] if len(args) > 1 else kwargs.get('T', 3)
        d = args[2] if len(args) > 2 else kwargs.get('d', 3)
        bitpack = kwargs.get('bitpack', False)
        surface_layout = kwargs.get('surface_layout', config.surface_layout)
        
        # Create RNG if seed provided
        rng = None
        if 'seed' in kwargs:
            rng = np.random.default_rng(kwargs['seed'])
        
        try:
            start_time = time.time() if config.performance_logging else None
            
            # Call CUDA-Q backend
            result = cudaq_sample_surface_wrapper(
                mode=config.mode,
                batch_size=batch_size,
                T=T,
                d=d,
                layout=None,  # Use default d=3 layout
                rng=rng,
                bitpack=bitpack,
                surface_layout=surface_layout
            )
            
            if config.performance_logging and start_time is not None:
                cudaq_time = time.time() - start_time
                print(f"CUDA-Q surface sampling: {cudaq_time:.3f}s for {batch_size} samples")
            
            # Validate output if requested
            if config.validate_outputs:
                if result.shape[0] != batch_size:
                    raise ValueError(f"Batch size mismatch: expected {batch_size}, got {result.shape[0]}")
                
                if result.dtype != np.uint8:
                    raise ValueError(f"Data type mismatch: expected uint8, got {result.dtype}")
                
                # Check for reasonable syndrome/error ranges
                if np.any(result > 3):  # Max value should be 3 (X + 2*Z)
                    warnings.warn("CUDA-Q output contains unexpectedly large values")
            
            return result
            
        except Exception as e:
            if config.fallback_on_error:
                warnings.warn(f"CUDA-Q surface sampling failed, using fallback: {e}")
                return original_sampler(*args, **kwargs)
            else:
                raise
    
    return cudaq_surface_sampler


def create_cudaq_bb_sampler(original_sampler: Callable, config: CudaQSamplingMode) -> Callable:
    """
    Create a CUDA-Q-enabled BB/qLDPC code sampler that wraps the original function.
    
    Args:
        original_sampler: The original BB sampling function to wrap/replace
        config: CUDA-Q configuration settings
        
    Returns:
        Wrapped sampling function with CUDA-Q backend
    """
    def cudaq_bb_sampler(*args, **kwargs):
        """CUDA-Q BB code sampler with fallback."""
        
        if not config.enabled:
            return original_sampler(*args, **kwargs)
        
        # Extract parameters (adapt to actual bb_sampler signature)
        batch_size = args[0] if len(args) > 0 else kwargs.get('batch_size', 1000)
        hx = args[1] if len(args) > 1 else kwargs.get('hx')
        hz = args[2] if len(args) > 2 else kwargs.get('hz')
        T = args[3] if len(args) > 3 else kwargs.get('T', 3)
        mapping = kwargs.get('mapping', None)
        bitpack = kwargs.get('bitpack', False)
        
        if hx is None or hz is None:
            raise ValueError("BB sampler requires hx and hz matrices")
        
        # Create RNG if seed provided
        rng = None
        if 'seed' in kwargs:
            rng = np.random.default_rng(kwargs['seed'])
        
        try:
            start_time = time.time() if config.performance_logging else None
            
            # Call CUDA-Q backend
            result = cudaq_sample_bb_wrapper(
                mode=config.mode,
                batch_size=batch_size,
                hx=hx,
                hz=hz,
                T=T,
                mapping=mapping,
                rng=rng,
                bitpack=bitpack
            )
            
            if config.performance_logging and start_time is not None:
                cudaq_time = time.time() - start_time
                print(f"CUDA-Q BB sampling: {cudaq_time:.3f}s for {batch_size} samples")
            
            # Validate output
            if config.validate_outputs:
                if result.shape[0] != batch_size:
                    raise ValueError(f"Batch size mismatch: expected {batch_size}, got {result.shape[0]}")
                
                expected_cols = hx.shape[0] + hz.shape[0] + hx.shape[1]
                if result.shape[1] != expected_cols:
                    warnings.warn(f"BB output shape unexpected: {result.shape[1]} vs {expected_cols}")
            
            return result
            
        except Exception as e:
            if config.fallback_on_error:
                warnings.warn(f"CUDA-Q BB sampling failed, using fallback: {e}")
                return original_sampler(*args, **kwargs)
            else:
                raise
    
    return cudaq_bb_sampler


def create_cudaq_repetition_sampler(original_sampler: Callable, config: CudaQSamplingMode) -> Callable:
    """
    Create a CUDA-Q-enabled repetition code sampler that wraps the original function.
    
    Args:
        original_sampler: The original repetition sampling function to wrap/replace
        config: CUDA-Q configuration settings
        
    Returns:
        Wrapped sampling function with CUDA-Q backend
    """
    def cudaq_repetition_sampler(*args, **kwargs):
        """CUDA-Q repetition code sampler with fallback."""
        
        if not config.enabled:
            return original_sampler(*args, **kwargs)
        
        # Extract parameters
        batch_size = args[0] if len(args) > 0 else kwargs.get('batch_size', 1000)
        n_data = args[1] if len(args) > 1 else kwargs.get('n_data', 5)
        T = args[2] if len(args) > 2 else kwargs.get('T', 3)
        bitpack = kwargs.get('bitpack', False)
        
        # Create RNG if seed provided
        rng = None
        if 'seed' in kwargs:
            rng = np.random.default_rng(kwargs['seed'])
        
        try:
            start_time = time.time() if config.performance_logging else None
            
            # Call CUDA-Q backend
            result = cudaq_sample_repetition_wrapper(
                mode=config.mode,
                batch_size=batch_size,
                n_data=n_data,
                T=T,
                layout=None,  # Use default layout
                rng=rng,
                bitpack=bitpack
            )
            
            if config.performance_logging and start_time is not None:
                cudaq_time = time.time() - start_time
                print(f"CUDA-Q repetition sampling: {cudaq_time:.3f}s for {batch_size} samples")
            
            return result
            
        except Exception as e:
            if config.fallback_on_error:
                warnings.warn(f"CUDA-Q repetition sampling failed, using fallback: {e}")
                return original_sampler(*args, **kwargs)
            else:
                raise
    
    return cudaq_repetition_sampler


def inject_cudaq_backend(module_or_dict: Union[object, Dict[str, Any]], 
                        config: CudaQSamplingMode,
                        sampler_mappings: Optional[Dict[str, str]] = None) -> Dict[str, Callable]:
    """
    Inject CUDA-Q backend into existing training code by replacing sampling functions.
    
    Args:
        module_or_dict: Module or dictionary containing original sampling functions
        config: CUDA-Q configuration settings  
        sampler_mappings: Custom mapping of {function_name: replacement_type}
                         Default: {'sample_surface': 'surface', 'sample_bb': 'bb', 'sample_repetition': 'repetition'}
        
    Returns:
        Dictionary of original functions that were replaced (for restoration)
    """
    if sampler_mappings is None:
        sampler_mappings = {
            'sample_surface': 'surface',
            'sample_bb': 'bb', 
            'sample_repetition': 'repetition'
        }
    
    replaced_functions = {}
    
    for func_name, replacement_type in sampler_mappings.items():
        # Get original function
        if isinstance(module_or_dict, dict):
            original_func = module_or_dict.get(func_name)
        else:
            original_func = getattr(module_or_dict, func_name, None)
        
        if original_func is None:
            continue
        
        # Store original for potential restoration
        replaced_functions[func_name] = original_func
        
        # Create CUDA-Q replacement
        if replacement_type == 'surface':
            cudaq_func = create_cudaq_surface_sampler(original_func, config)
        elif replacement_type == 'bb':
            cudaq_func = create_cudaq_bb_sampler(original_func, config)
        elif replacement_type == 'repetition':
            cudaq_func = create_cudaq_repetition_sampler(original_func, config)
        else:
            warnings.warn(f"Unknown replacement type: {replacement_type}")
            continue
        
        # Inject replacement
        if isinstance(module_or_dict, dict):
            module_or_dict[func_name] = cudaq_func
        else:
            setattr(module_or_dict, func_name, cudaq_func)
        
        print(f"Injected CUDA-Q backend for {func_name}")
    
    return replaced_functions


def restore_original_functions(module_or_dict: Union[object, Dict[str, Any]], 
                              original_functions: Dict[str, Callable]):
    """
    Restore original sampling functions after CUDA-Q injection.
    
    Args:
        module_or_dict: Module or dictionary to restore functions in
        original_functions: Dictionary of original functions to restore
    """
    for func_name, original_func in original_functions.items():
        if isinstance(module_or_dict, dict):
            module_or_dict[func_name] = original_func
        else:
            setattr(module_or_dict, func_name, original_func)
        
        print(f"Restored original {func_name}")


def get_cudaq_performance_stats() -> Dict[str, Any]:
    """
    Get performance and usage statistics for CUDA-Q backend.
    
    Returns:
        Dictionary with performance metrics and backend status
    """
    stats = {
        'cudaq_available': CUDAQ_AVAILABLE,
        'cudaq_validated': CUDAQ_VALIDATED,
        'backend_info': get_backend_info() if CUDAQ_AVAILABLE else None
    }
    
    return stats


# Convenience function for easy integration
def enable_cudaq_backend(training_module_or_globals: Union[object, Dict[str, Any]],
                        mode: str = "foundation",
                        fallback_on_error: bool = True,
                        performance_logging: bool = False) -> Optional[Dict[str, Callable]]:
    """
    Convenience function to enable CUDA-Q backend in training code.
    
    Usage:
        # In training script:
        from cudaq_backend.training_integration import enable_cudaq_backend
        original_functions = enable_cudaq_backend(globals())
        
        # Now all sample_* calls will use CUDA-Q backend
        
    Args:
        training_module_or_globals: Module or globals() dict to inject into
        mode: "foundation" or "student" noise model
        fallback_on_error: Whether to fall back on errors
        performance_logging: Whether to log performance metrics
        
    Returns:
        Dictionary of replaced functions for later restoration, or None if injection failed
    """
    if not CUDAQ_AVAILABLE:
        print("CUDA-Q backend not available. No injection performed.")
        return None
    
    config = CudaQSamplingMode(
        enabled=True,
        mode=mode,
        fallback_on_error=fallback_on_error,
        performance_logging=performance_logging
    )
    
    try:
        replaced = inject_cudaq_backend(training_module_or_globals, config)
        print(f"Successfully enabled CUDA-Q backend in {mode} mode")
        return replaced
    except Exception as e:
        print(f"Failed to enable CUDA-Q backend: {e}")
        return None
