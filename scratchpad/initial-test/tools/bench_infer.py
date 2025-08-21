#!/usr/bin/env python3
"""
Batch-1 latency microbenchmark for MGHD inference.

This script measures end-to-end inference latency across different backends:
- PyTorch eager
- TorchScript (script/trace)
- ONNX Runtime
- TensorRT
- CUDA Graph replay

Usage:
    python tools/bench_infer.py --backend eager --repeats 5000 --warmup 500 --n-syn 24 --n-bits 17 --packed
"""

import argparse
import json
import time
import torch
import torch.nn as nn
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    import onnxruntime as ort
    ONNXRT_AVAILABLE = True
except ImportError:
    ONNXRT_AVAILABLE = False
    warnings.warn("ONNX Runtime not available. Install with: pip install onnxruntime-gpu")

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    warnings.warn("TensorRT not available. Install with: pip install tensorrt")

from poc_my_models import MGHD
from panq_functions import GNNDecoder


class LatencyBenchmark:
    """Benchmark class for measuring inference latency across different backends."""
    
    def __init__(self, model_path: Optional[str], n_syn: int, n_bits: int, device: str = "cuda"):
        self.model_path = model_path
        self.n_syn = n_syn
        self.n_bits = n_bits
        self.device = device
        self.model = None
        self.scripted_model = None
        self.traced_model = None
        self.onnx_session = None
        self.trt_engine = None
        self.cuda_graph = None
        self.cuda_graph_inputs = None
        self.cuda_graph_outputs = None
        
        # Calculate distance from syndrome size
        self.dist = int(np.sqrt(n_syn + 1))
        
        # Initialize model if path provided
        if model_path and os.path.exists(model_path):
            self._load_model()
    
    def _load_model(self):
        """Load the MGHD model from checkpoint."""
        print(f"Loading model from {self.model_path}")
        
        # Create model with same parameters as training
        gnn_params = {
            'dist': self.dist,
            'n_node_inputs': 9,
            'n_node_outputs': 9,
            'n_iters': 7,
            'n_node_features': 10,
            'n_edge_features': 11,
            'msg_net_size': 96,
            'msg_net_dropout_p': 0.0,
            'gru_dropout_p': 0.0
        }
        
        mamba_params = {
            'd_model': 64,
            'd_state': 16,
            'd_conv': 4,
            'expand': 2,
            'attention_mechanism': 'none'
        }
        
        self.model = MGHD(gnn_params=gnn_params, mamba_params=mamba_params).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()
        
        print(f"Model loaded successfully. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _create_dummy_model(self):
        """Create a dummy model for testing when no checkpoint is provided."""
        print("Creating dummy model for testing")
        
        gnn_params = {
            'dist': self.dist,
            'n_node_inputs': 9,
            'n_node_outputs': 9,
            'n_iters': 7,
            'n_node_features': 10,
            'n_edge_features': 11,
            'msg_net_size': 96,
            'msg_net_dropout_p': 0.0,
            'gru_dropout_p': 0.0
        }
        
        mamba_params = {
            'd_model': 64,
            'd_state': 16,
            'd_conv': 4,
            'expand': 2,
            'attention_mechanism': 'none'
        }
        
        self.model = MGHD(gnn_params=gnn_params, mamba_params=mamba_params).to(self.device)
        self.model.eval()
    
    def _generate_synthetic_syndrome(self, packed: bool = False) -> torch.Tensor:
        """Generate synthetic syndrome data for benchmarking."""
        if packed:
            # Generate packed syndrome [N_bytes]
            N_bytes = (self.n_syn + 7) // 8
            syndrome = torch.randint(0, 256, (N_bytes,), dtype=torch.uint8, device=self.device)
        else:
            # Generate unpacked syndrome [N_syn]
            syndrome = torch.randint(0, 2, (self.n_syn,), dtype=torch.uint8, device=self.device)
        
        return syndrome
    
    def _measure_eager_latency(self, syndrome: torch.Tensor, repeats: int) -> List[float]:
        """Measure PyTorch eager inference latency."""
        latencies = []
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model.decode_one(syndrome, device=self.device)
        
        torch.cuda.synchronize()
        
        # Benchmark
        for _ in range(repeats):
            start = time.perf_counter()
            with torch.no_grad():
                _ = self.model.decode_one(syndrome, device=self.device)
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        return latencies
    
    def _measure_torchscript_latency(self, syndrome: torch.Tensor, repeats: int, use_trace: bool = False) -> List[float]:
        """Measure TorchScript inference latency."""
        if self.model is None:
            return []
        
        # Create TorchScript model if not exists
        if use_trace:
            if self.traced_model is None:
                # Create dummy input for tracing
                dummy_input = self._generate_synthetic_syndrome(packed=False)
                self.traced_model = torch.jit.trace(self.model.decode_one, (dummy_input,))
                self.traced_model.eval()
            model = self.traced_model
        else:
            if self.scripted_model is None:
                self.scripted_model = torch.jit.script(self.model.decode_one)
                self.scripted_model.eval()
            model = self.scripted_model
        
        latencies = []
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(syndrome, device=self.device)
        
        torch.cuda.synchronize()
        
        # Benchmark
        for _ in range(repeats):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(syndrome, device=self.device)
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        return latencies
    
    def _measure_onnx_latency(self, syndrome: torch.Tensor, repeats: int) -> List[float]:
        """Measure ONNX Runtime inference latency."""
        if not ONNXRT_AVAILABLE or self.model is None:
            return []
        
        # Export to ONNX if not exists
        onnx_path = f"temp_model_{self.dist}.onnx"
        if not os.path.exists(onnx_path):
            self.model.export_onnx_int8_ready(onnx_path, self.n_syn, self.n_bits)
        
        # Create ONNX session
        if self.onnx_session is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.onnx_session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Prepare input
        # Note: ONNX expects the full node inputs, not just syndrome
        num_check_nodes = self.dist**2 - 1
        num_qubit_nodes = self.dist**2
        nodes_per_graph = num_check_nodes + num_qubit_nodes
        
        # Convert syndrome to full node inputs
        node_inputs = torch.zeros(1, nodes_per_graph, 9, device=self.device, dtype=torch.float32)
        if syndrome.dtype == torch.uint8 and syndrome.dim() == 1:
            # Unpack if needed
            unpacked = torch.zeros(1, self.n_syn, dtype=torch.float32, device=self.device)
            for i in range(self.n_syn):
                byte_idx = i // 8
                bit_idx = i % 8
                unpacked[0, i] = (syndrome[byte_idx] >> bit_idx) & 1
            syndrome = unpacked
        
        for i in range(num_check_nodes):
            node_inputs[0, i, 0] = syndrome[0, i]
        
        node_inputs_flat = node_inputs.view(-1, 9).cpu().numpy()
        
        # Get static indices
        self.model._ensure_static_indices(self.device)
        src_ids = self.model._src_ids.cpu().numpy()
        dst_ids = self.model._dst_ids.cpu().numpy()
        
        latencies = []
        
        # Warmup
        for _ in range(10):
            _ = self.onnx_session.run(
                None,
                {
                    'node_inputs': node_inputs_flat,
                    'src_ids': src_ids,
                    'dst_ids': dst_ids
                }
            )
        
        # Benchmark
        for _ in range(repeats):
            start = time.perf_counter()
            _ = self.onnx_session.run(
                None,
                {
                    'node_inputs': node_inputs_flat,
                    'src_ids': src_ids,
                    'dst_ids': dst_ids
                }
            )
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        return latencies
    
    def _measure_cuda_graph_latency(self, syndrome: torch.Tensor, repeats: int) -> List[float]:
        """Measure CUDA Graph replay latency."""
        if self.model is None:
            return []
        
        # Capture CUDA graph if not exists
        if self.cuda_graph is None:
            print("Capturing CUDA graph...")
            
            # Create CUDA stream for graph capture
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                # Start graph capture
                torch.cuda.cudart().cudaStreamBeginCapture(
                    stream.cuda_stream, torch.cuda.cudart().cudaStreamCaptureMode.cudaStreamCaptureModeGlobal
                )
                
                # Run inference
                with torch.no_grad():
                    outputs = self.model.decode_one(syndrome, device=self.device)
                
                # End graph capture
                graph = torch.cuda.cudart().cudaStreamEndCapture(stream.cuda_stream)
                
                # Create executable graph
                self.cuda_graph = torch.cuda.cudart().cudaGraphInstantiate(graph, 0)
                
                # Create input/output tensors for graph execution
                self.cuda_graph_inputs = [syndrome]
                self.cuda_graph_outputs = [outputs]
            
            print("CUDA graph captured successfully")
        
        latencies = []
        
        # Warmup
        for _ in range(10):
            torch.cuda.cudart().cudaGraphLaunch(self.cuda_graph, stream.cuda_stream)
            stream.synchronize()
        
        # Benchmark
        for _ in range(repeats):
            start = time.perf_counter()
            torch.cuda.cudart().cudaGraphLaunch(self.cuda_graph, stream.cuda_stream)
            stream.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        return latencies
    
    def benchmark(self, backend: str, repeats: int, warmup: int, packed: bool = False) -> Dict:
        """Run benchmark for specified backend."""
        print(f"\nBenchmarking {backend} backend...")
        
        # Ensure model exists
        if self.model is None:
            self._create_dummy_model()
        
        # Generate synthetic syndrome
        syndrome = self._generate_synthetic_syndrome(packed=packed)
        
        # Run warmup
        if warmup > 0:
            print(f"Running {warmup} warmup iterations...")
            if backend == "eager":
                self._measure_eager_latency(syndrome, warmup)
            elif backend == "ts":
                self._measure_torchscript_latency(syndrome, warmup, use_trace=False)
            elif backend == "trace":
                self._measure_torchscript_latency(syndrome, warmup, use_trace=True)
            elif backend == "onnxrt":
                self._measure_onnx_latency(syndrome, warmup)
            elif backend == "graph":
                self._measure_cuda_graph_latency(syndrome, warmup)
        
        # Run benchmark
        print(f"Running {repeats} benchmark iterations...")
        if backend == "eager":
            latencies = self._measure_eager_latency(syndrome, repeats)
        elif backend == "ts":
            latencies = self._measure_torchscript_latency(syndrome, repeats, use_trace=False)
        elif backend == "trace":
            latencies = self._measure_torchscript_latency(syndrome, repeats, use_trace=True)
        elif backend == "onnxrt":
            latencies = self._measure_onnx_latency(syndrome, repeats)
        elif backend == "graph":
            latencies = self._measure_cuda_graph_latency(syndrome, repeats)
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        # Calculate statistics
        latencies = np.array(latencies)
        stats = {
            'backend': backend,
            'repeats': repeats,
            'warmup': warmup,
            'packed': packed,
            'n_syn': self.n_syn,
            'n_bits': self.n_bits,
            'dist': self.dist,
            'p50': float(np.percentile(latencies, 50)),
            'p90': float(np.percentile(latencies, 90)),
            'p99': float(np.percentile(latencies, 99)),
            'mean': float(np.mean(latencies)),
            'std': float(np.std(latencies)),
            'min': float(np.min(latencies)),
            'max': float(np.max(latencies)),
            'latencies': latencies.tolist()
        }
        
        # Print results
        print(f"Results for {backend}:")
        print(f"  P50: {stats['p50']:.3f} ms")
        print(f"  P90: {stats['p90']:.3f} ms")
        print(f"  P99: {stats['p99']:.3f} ms")
        print(f"  Mean: {stats['mean']:.3f} ms")
        print(f"  Std: {stats['std']:.3f} ms")
        
        return stats


def main():
    parser = argparse.ArgumentParser(description="MGHD Inference Latency Benchmark")
    parser.add_argument("--model", type=str, help="Path to model checkpoint (optional)")
    parser.add_argument("--backend", type=str, choices=["eager", "ts", "trace", "onnxrt", "trt", "graph"], 
                       default="eager", help="Backend to benchmark")
    parser.add_argument("--repeats", type=int, default=10000, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=1000, help="Number of warmup iterations")
    parser.add_argument("--n-syn", type=int, required=True, help="Number of syndrome bits")
    parser.add_argument("--n-bits", type=int, required=True, help="Number of output bits")
    parser.add_argument("--packed", action="store_true", help="Use packed syndrome input")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.n_syn <= 0 or args.n_bits <= 0:
        print("Error: n-syn and n-bits must be positive")
        return 1
    
    # Check backend availability
    if args.backend == "onnxrt" and not ONNXRT_AVAILABLE:
        print("Error: ONNX Runtime not available")
        return 1
    
    if args.backend == "trt" and not TENSORRT_AVAILABLE:
        print("Error: TensorRT not available")
        return 1
    
    # Create benchmark instance
    benchmark = LatencyBenchmark(args.model, args.n_syn, args.n_bits, args.device)
    
    # Run benchmark
    try:
        results = benchmark.benchmark(args.backend, args.repeats, args.warmup, args.packed)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(__file__).parent.parent / "reports"
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / f"bench_infer_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        
        return 0
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
