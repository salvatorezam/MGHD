#!/usr/bin/env python3
"""
CUDA-Q Syndrome Generator Benchmark

This script benchmarks the performance of the CUDA-Q syndrome generator
for d=3 surface codes with different batch sizes and configurations.

Usage:
    python tools/bench_cudaq_gen.py [--mode foundation|student] [--batch-size N] [--rounds T]
"""

import sys
import os
import time
import argparse
import numpy as np
import subprocess
import platform
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cudaq_backend.syndrome_gen import sample_surface_cudaq
from cudaq_backend.circuits import make_surface_layout_d3_avoid_bad_edges
from cudaq_backend.garnet_noise import FOUNDATION_DEFAULTS, GARNET_COUPLER_F2


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU and CUDA information."""
    gpu_info = {
        "cuda_available": False,
        "gpu_count": 0,
        "gpu_names": [],
        "driver_versions": [],
        "memory_total": [],
        "cuda_runtime": None,
        "cudaq_version": None
    }
    
    try:
        # Try to get CUDA-Q version
        try:
            import cudaq
            gpu_info["cudaq_version"] = getattr(cudaq, '__version__', 'Unknown')
        except ImportError:
            gpu_info["cudaq_version"] = "Not available (using fallback)"
        
        # Try nvidia-smi
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=name,driver_version,memory.total', 
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 3:
                            gpu_info["gpu_names"].append(parts[0])
                            gpu_info["driver_versions"].append(parts[1])
                            gpu_info["memory_total"].append(f"{parts[2]} MB")
                            gpu_info["gpu_count"] += 1
                            gpu_info["cuda_available"] = True
        except:
            pass
        
        # Try to get CUDA runtime version
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'release' in line.lower():
                        gpu_info["cuda_runtime"] = line.strip()
                        break
        except:
            pass
        
    except Exception as e:
        print(f"Warning: Could not get full GPU info: {e}")
    
    return gpu_info


def print_system_info():
    """Print comprehensive system and GPU information."""
    print("="*60)
    print("SYSTEM AND GPU INFORMATION")
    print("="*60)
    
    # Basic platform info
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Architecture: {platform.machine()}")
    
    # GPU information
    gpu_info = get_gpu_info()
    
    print(f"\nCUDA-Q Version: {gpu_info['cudaq_version']}")
    
    if gpu_info["cuda_runtime"]:
        print(f"CUDA Runtime: {gpu_info['cuda_runtime']}")
    else:
        print("CUDA Runtime: Not detected")
    
    if gpu_info["cuda_available"] and gpu_info["gpu_count"] > 0:
        print(f"\nGPU Count: {gpu_info['gpu_count']}")
        for i, (name, driver, memory) in enumerate(zip(
            gpu_info["gpu_names"], 
            gpu_info["driver_versions"], 
            gpu_info["memory_total"]
        )):
            print(f"  GPU {i}: {name}")
            print(f"    Driver: {driver}")
            print(f"    Memory: {memory}")
    else:
        print("\nGPU: Not available or not detected")
    
    print("="*60)
    return gpu_info


def benchmark_cudaq_generator(mode: str = "foundation", 
                            batch_size: int = 100000,
                            T: int = 1,
                            num_trials: int = 5) -> Dict[str, float]:
    """
    Benchmark the CUDA-Q syndrome generator performance.
    
    Args:
        mode: "foundation" or "student"
        batch_size: Number of syndrome samples per trial
        T: Number of syndrome extraction rounds
        num_trials: Number of benchmark trials to average
        
    Returns:
        Dictionary with performance metrics
    """
    print(f"Benchmarking CUDA-Q syndrome generator...")
    print(f"  Mode: {mode}")
    print(f"  Batch size: {batch_size:,}")
    print(f"  Syndrome rounds: {T}")
    print(f"  Trials: {num_trials}")
    print(f"  Distance: 3 (surface code)")
    
    # Setup configuration
    layout = make_surface_layout_d3_avoid_bad_edges()
    rng = np.random.default_rng(42)
    
    times = []
    
    for trial in range(num_trials):
        print(f"  Trial {trial + 1}/{num_trials}...", end=" ", flush=True)
        
        start_time = time.perf_counter()
        
        # Generate syndrome samples
        samples = sample_surface_cudaq(
            mode=mode,
            batch_size=batch_size,
            T=T,
            layout=layout,
            rng=rng,
            bitpack=False
        )
        
        end_time = time.perf_counter()
        trial_time = end_time - start_time
        times.append(trial_time)
        
        # Validate output format
        expected_shape = (batch_size, len(layout['ancilla_x']) + len(layout['ancilla_z']) + len(layout['data']))
        if samples.shape != expected_shape:
            print(f"Warning: Unexpected sample shape {samples.shape}, expected {expected_shape}")
        
        samples_per_sec = batch_size / trial_time
        print(f"{samples_per_sec:,.0f} samples/sec")
    
    # Calculate statistics
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    mean_samples_per_sec = batch_size / mean_time
    
    metrics = {
        'mean_time_seconds': mean_time,
        'std_time_seconds': std_time,
        'mean_samples_per_second': mean_samples_per_sec,
        'total_samples': batch_size * num_trials,
        'batch_size': batch_size,
        'rounds': T,
        'mode': mode
    }
    
    return metrics


def print_performance_report(metrics: Dict[str, float]):
    """Print a formatted performance report."""
    print("\n" + "="*60)
    print("CUDA-Q SYNDROME GENERATOR PERFORMANCE REPORT")
    print("="*60)
    print(f"Configuration:")
    print(f"  Mode: {metrics['mode']}")
    print(f"  Batch size: {metrics['batch_size']:,}")
    print(f"  Syndrome rounds: {metrics['rounds']}")
    print(f"  Total samples: {metrics['total_samples']:,}")
    print()
    print(f"Performance:")
    print(f"  Mean time per batch: {metrics['mean_time_seconds']:.3f} ± {metrics['std_time_seconds']:.3f} seconds")
    print(f"  Mean throughput: {metrics['mean_samples_per_second']:,.0f} samples/second")
    print(f"  Time per sample: {1e6 * metrics['mean_time_seconds'] / metrics['batch_size']:.2f} microseconds")
    print()
    
    # Compare to target performance
    target_samples_per_sec = 50000  # Target: 50k samples/sec
    if metrics['mean_samples_per_second'] >= target_samples_per_sec:
        print(f"✓ Performance target MET ({target_samples_per_sec:,} samples/sec)")
    else:
        shortfall = target_samples_per_sec - metrics['mean_samples_per_second']
        print(f"✗ Performance target MISSED by {shortfall:,.0f} samples/sec")
    
    print("="*60)


def compare_modes():
    """Compare Foundation vs Student mode performance."""
    print("Comparing Foundation vs Student modes...")
    
    foundation_metrics = benchmark_cudaq_generator(mode="foundation", batch_size=50000, T=1, num_trials=3)
    student_metrics = benchmark_cudaq_generator(mode="student", batch_size=50000, T=1, num_trials=3)
    
    print(f"\nPerformance Comparison:")
    print(f"  Foundation: {foundation_metrics['mean_samples_per_second']:,.0f} samples/sec")
    print(f"  Student:    {student_metrics['mean_samples_per_second']:,.0f} samples/sec")
    
    if foundation_metrics['mean_samples_per_second'] > student_metrics['mean_samples_per_second']:
        faster_mode = "Foundation"
        speedup = foundation_metrics['mean_samples_per_second'] / student_metrics['mean_samples_per_second']
    else:
        faster_mode = "Student"
        speedup = student_metrics['mean_samples_per_second'] / foundation_metrics['mean_samples_per_second']
    
    print(f"  {faster_mode} mode is {speedup:.2f}x faster")


def validate_noise_model():
    """Validate that the noise model produces expected parameter ranges."""
    print("Validating noise model parameters...")
    
    from cudaq_backend.garnet_noise import GarnetFoundationPriors, GarnetStudentCalibration
    
    # Test Foundation mode
    priors = GarnetFoundationPriors()
    rng = np.random.default_rng(42)
    foundation_params = priors.sample_pseudo_device(rng)
    
    print(f"Foundation mode validation:")
    f1q_values = list(foundation_params['F1Q'].values())
    f2q_values = list(foundation_params['F2Q'].values())
    print(f"  F1Q range: {min(f1q_values):.4f} - {max(f1q_values):.4f}")
    print(f"  F2Q range: {min(f2q_values):.4f} - {max(f2q_values):.4f}")
    print(f"  Bad edge (10,11) avoided: {(10,11) not in foundation_params['F2Q']}")
    
    # Test Student mode
    calibration = GarnetStudentCalibration()
    student_params = calibration.to_dict()
    
    print(f"Student mode validation:")
    print(f"  F1Q uniform: {student_params['F1Q'][0]:.4f}")
    print(f"  Bad edge F2Q: {student_params['F2Q'].get((10,11), 'N/A')}")
    print(f"  Best edge F2Q: {max(student_params['F2Q'].values()):.4f}")
    
    # Verify bad edge is included in student mode
    has_bad_edge = (10, 11) in student_params['F2Q']
    print(f"  Bad edge (10,11) present: {has_bad_edge}")
    if has_bad_edge:
        print(f"  Bad edge fidelity: {student_params['F2Q'][(10,11)]:.4f}")


def main():
    """Main benchmark script."""
    parser = argparse.ArgumentParser(description='Benchmark CUDA-Q syndrome generator')
    parser.add_argument('--mode', choices=['foundation', 'student'], default='foundation',
                        help='Noise model mode (default: foundation)')
    parser.add_argument('--batch-size', type=int, default=100000,
                        help='Batch size for benchmarking (default: 100000)')
    parser.add_argument('--rounds', type=int, default=1,
                        help='Number of syndrome extraction rounds (default: 1)')
    parser.add_argument('--trials', type=int, default=5,
                        help='Number of benchmark trials (default: 5)')
    parser.add_argument('--compare-modes', action='store_true',
                        help='Compare Foundation vs Student performance')
    parser.add_argument('--validate', action='store_true',
                        help='Validate noise model parameters')
    parser.add_argument('--system-info', action='store_true',
                        help='Print detailed system and GPU information')
    
    args = parser.parse_args()
    
    print("CUDA-Q Syndrome Generator Benchmark")
    print("="*40)
    
    # Always print system info for verification purposes
    gpu_info = print_system_info()
    print()
    
    if args.validate:
        validate_noise_model()
        print()
    
    if args.compare_modes:
        compare_modes()
    else:
        metrics = benchmark_cudaq_generator(
            mode=args.mode,
            batch_size=args.batch_size,
            T=args.rounds,
            num_trials=args.trials
        )
        
        # Add GPU info to metrics
        metrics.update({
            'gpu_info': gpu_info,
            'cudaq_version': gpu_info['cudaq_version'],
            'cuda_available': gpu_info['cuda_available'],
            'gpu_count': gpu_info['gpu_count']
        })
        
        print_performance_report(metrics)


if __name__ == '__main__':
    main()
