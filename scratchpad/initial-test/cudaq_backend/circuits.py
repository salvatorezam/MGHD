"""
Circuit Construction Module

This module builds per-round native PRX/CZ measurement circuits for each code family
and produces layered schedules to compute idle windows correctly.

Implements circuit construction for:
- Repetition codes
- Surface codes (rotated lattice)
- BB/qLDPC codes

All circuits are constructed as CUDA-Q kernels without baked-in noise.
Noise is applied by the syndrome generator between layers.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from collections import defaultdict


# For now, we'll use a simplified approach that doesn't depend on CUDA-Q installation
# In a real implementation, these would be actual cudaq.Kernel objects
class MockCudaQKernel:
    """Mock CUDA-Q kernel for development without CUDA-Q installation."""
    
    def __init__(self, name: str, operations: List[Dict[str, Any]]):
        self.name = name
        self.operations = operations  # List of operations with timing info
        self.layers = []  # Will store parallel layers
        
    def add_layer(self, layer_ops: List[Dict[str, Any]], layer_type: str):
        """Add a parallel layer of operations."""
        self.layers.append({
            'type': layer_type,
            'operations': layer_ops,
            'duration_ns': layer_ops[0].get('duration_ns', 20.0) if layer_ops else 0.0
        })
    
    def get_layer_schedule(self) -> List[Dict[str, Any]]:
        """Return the layered schedule for idle time calculation."""
        return self.layers


def make_surface_layout_d3_avoid_bad_edges() -> Dict[str, Any]:
    """
    Create a d=3 surface code layout that avoids the bad (10,11) coupler.
    
    Maps a 17-qubit rotated surface code patch onto the Garnet 20-qubit device,
    preferring high-fidelity edges and avoiding the problematic (10,11) coupler.
    
    Returns:
        Dictionary with layout information for d=3 surface code
    """
    # For d=3 rotated surface code: 9 data qubits + 8 stabilizer qubits = 17 total
    # Avoid coupler (10,11) which has F2Q = 0.9228 (worst in device)
    
    # Use a good patch of 17 qubits avoiding the bad edge
    # This is a simplified mapping - in practice you'd optimize placement
    qubit_mapping = {
        # Data qubits (9 total for d=3)
        'data': [0, 1, 2, 3, 4, 5, 6, 7, 8],
        # X-stabilizer ancillas (4 total)
        'ancilla_x': [12, 13, 14, 15],
        # Z-stabilizer ancillas (4 total) 
        'ancilla_z': [16, 17, 18, 19]
    }
    
    # Define CZ layers that respect the rotated surface code connectivity
    # These would be computed from the actual surface code graph structure
    cz_layers = [
        # Layer 1: X-stabilizer CZs
        [(12, 0), (13, 1), (14, 2), (15, 3)],  # Example connectivity
        # Layer 2: Z-stabilizer CZs  
        [(16, 4), (17, 5), (18, 6), (19, 7)]   # Example connectivity
    ]
    
    # PRX layers for stabilizer measurements
    prx_layers = [
        # X-basis measurements for X-stabilizers
        [12, 13, 14, 15],
        # Z-basis measurements for Z-stabilizers (identity, no PRX needed)
        []
    ]
    
    return {
        'data': qubit_mapping['data'],
        'ancilla_x': qubit_mapping['ancilla_x'],
        'ancilla_z': qubit_mapping['ancilla_z'],
        'cz_layers': cz_layers,
        'prx_layers': prx_layers,
        'total_qubits': 17,
        'distance': 3
    }


def build_round_repetition(layout: Dict[str, Any], round_idx: int) -> MockCudaQKernel:
    """
    Build a repetition code syndrome extraction round.
    
    Args:
        layout: Dictionary with repetition code layout information
        round_idx: Round index (for potential round-dependent variations)
        
    Returns:
        CUDA-Q kernel for the repetition code round
    """
    operations = []
    kernel = MockCudaQKernel(f"rep_round_{round_idx}", operations)
    
    # For repetition code: CZ between adjacent data qubits and ancilla
    n_data = len(layout.get('data', []))
    ancillas = layout.get('ancilla', [])
    
    if not ancillas or not layout.get('data'):
        return kernel
    
    # Layer 1: CZ gates between data qubits and ancillas
    cz_ops = []
    for i, data_q in enumerate(layout['data'][:-1]):
        if i < len(ancillas):
            ancilla_q = ancillas[i]
            cz_ops.append({
                'gate': 'CZ',
                'qubits': (data_q, ancilla_q),
                'duration_ns': 40.0
            })
    
    if cz_ops:
        kernel.add_layer(cz_ops, 'CZ')
    
    # Layer 2: Measurement of ancillas (Z-basis)
    meas_ops = [{
        'gate': 'MEASURE_Z',
        'qubits': (q,),
        'duration_ns': 20.0
    } for q in ancillas]
    
    if meas_ops:
        kernel.add_layer(meas_ops, 'MEASURE')
    
    return kernel


def build_round_surface(layout: Dict[str, Any], round_idx: int) -> MockCudaQKernel:
    """
    Build a surface code syndrome extraction round.
    
    Alternates between X and Z stabilizer measurements, using the provided
    CZ and PRX layer schedule from the layout.
    
    Args:
        layout: Dictionary with surface code layout (from make_surface_layout_d3_avoid_bad_edges)
        round_idx: Round index - even rounds do X stabilizers, odd do Z stabilizers
        
    Returns:
        CUDA-Q kernel for the surface code round
    """
    operations = []
    kernel = MockCudaQKernel(f"surface_round_{round_idx}", operations)
    
    # Alternate between X and Z stabilizer types
    is_x_round = (round_idx % 2 == 0)
    
    if is_x_round:
        # X-stabilizer round
        active_ancillas = layout.get('ancilla_x', [])
        stabilizer_type = 'X'
    else:
        # Z-stabilizer round  
        active_ancillas = layout.get('ancilla_z', [])
        stabilizer_type = 'Z'
    
    # Get relevant CZ layers for this stabilizer type
    cz_layers = layout.get('cz_layers', [])
    prx_layers = layout.get('prx_layers', [])
    
    # For X-stabilizers: Apply PRX to ancillas first (H gate equivalent)
    if is_x_round and prx_layers and len(prx_layers) > 0:
        prx_ops = [{
            'gate': 'PRX',
            'qubits': (q,),
            'duration_ns': 20.0
        } for q in prx_layers[0] if q in active_ancillas]
        
        if prx_ops:
            kernel.add_layer(prx_ops, 'PRX_PREP')
    
    # Add CZ layers (typically 2 layers for surface codes)
    for layer_idx, cz_layer in enumerate(cz_layers):
        if layer_idx >= 2:  # Limit to 2 CZ layers per round
            break
            
        # Filter CZ operations relevant to current stabilizer type
        relevant_czs = []
        for (q1, q2) in cz_layer:
            if (is_x_round and q1 in active_ancillas) or \
               (not is_x_round and q1 in active_ancillas):
                relevant_czs.append({
                    'gate': 'CZ',
                    'qubits': (q1, q2),
                    'duration_ns': 40.0
                })
        
        if relevant_czs:
            kernel.add_layer(relevant_czs, f'CZ_{layer_idx+1}')
    
    # For X-stabilizers: Apply PRX to measure in X-basis
    if is_x_round and active_ancillas:
        prx_meas_ops = [{
            'gate': 'PRX',
            'qubits': (q,),
            'duration_ns': 20.0
        } for q in active_ancillas]
        
        if prx_meas_ops:
            kernel.add_layer(prx_meas_ops, 'PRX_MEAS')
    
    # Final measurement (Z-basis for all)
    meas_ops = [{
        'gate': 'MEASURE_Z',
        'qubits': (q,),
        'duration_ns': 20.0
    } for q in active_ancillas]
    
    if meas_ops:
        kernel.add_layer(meas_ops, 'MEASURE')
    
    return kernel


def build_round_bb(hx: np.ndarray, hz: np.ndarray, mapping: Dict[int, int], round_idx: int) -> MockCudaQKernel:
    """
    Build a BB/qLDPC code syndrome extraction round.
    
    Derives check circuits from (Hx, Hz) matrices and schedules CZ operations
    to minimize idle time while respecting hardware constraints.
    
    Args:
        hx: X-check matrix (num_x_checks, num_qubits)
        hz: Z-check matrix (num_z_checks, num_qubits)  
        mapping: Logical to physical qubit mapping
        round_idx: Round index - even rounds do X checks, odd do Z checks
        
    Returns:
        CUDA-Q kernel for the BB/qLDPC round
    """
    operations = []
    kernel = MockCudaQKernel(f"bb_round_{round_idx}", operations)
    
    # Alternate between X and Z check measurements
    is_x_round = (round_idx % 2 == 0)
    
    if is_x_round:
        check_matrix = hx
        check_type = 'X'
    else:
        check_matrix = hz  
        check_type = 'Z'
    
    num_checks, num_qubits = check_matrix.shape
    
    # Build ancilla allocation (assume ancillas start after data qubits)
    max_physical_qubit = max(mapping.values()) if mapping else num_qubits - 1
    ancilla_start = max_physical_qubit + 1
    
    # Create CZ schedule: group by ancilla degree to minimize layers
    check_schedules = defaultdict(list)
    
    for check_idx in range(num_checks):
        # Find data qubits involved in this check
        data_qubits = [mapping.get(q, q) for q in range(num_qubits) if check_matrix[check_idx, q] == 1]
        
        if not data_qubits:
            continue
            
        ancilla_qubit = ancilla_start + check_idx
        
        # Create CZ operations for this check
        for data_q in data_qubits:
            check_schedules[len(data_qubits)].append({
                'gate': 'CZ',
                'qubits': (ancilla_qubit, data_q),
                'duration_ns': 40.0,
                'check_idx': check_idx,
                'ancilla': ancilla_qubit
            })
    
    # For X-checks: PRX preparation layer
    if is_x_round:
        ancillas = [ancilla_start + i for i in range(num_checks)]
        prx_prep_ops = [{
            'gate': 'PRX',
            'qubits': (q,),
            'duration_ns': 20.0
        } for q in ancillas]
        
        if prx_prep_ops:
            kernel.add_layer(prx_prep_ops, 'PRX_PREP')
    
    # Schedule CZ operations in layers (aim for ≤3 layers)
    max_layers = 3
    layer_idx = 0
    
    # Sort by degree (higher degree checks first for better parallelization)
    for degree in sorted(check_schedules.keys(), reverse=True):
        if layer_idx >= max_layers:
            break
            
        cz_ops = check_schedules[degree]
        if cz_ops:
            kernel.add_layer(cz_ops, f'CZ_{layer_idx+1}')
            layer_idx += 1
    
    # For X-checks: PRX measurement layer
    if is_x_round:
        ancillas = [ancilla_start + i for i in range(num_checks)]
        prx_meas_ops = [{
            'gate': 'PRX',
            'qubits': (q,),
            'duration_ns': 20.0
        } for q in ancillas]
        
        if prx_meas_ops:
            kernel.add_layer(prx_meas_ops, 'PRX_MEAS')
    
    # Final measurement layer
    ancillas = [ancilla_start + i for i in range(num_checks)]
    meas_ops = [{
        'gate': 'MEASURE_Z',
        'qubits': (q,),
        'duration_ns': 20.0
    } for q in ancillas]
    
    if meas_ops:
        kernel.add_layer(meas_ops, 'MEASURE')
    
    return kernel


def analyze_idle_qubits(kernel: MockCudaQKernel, total_qubits: int) -> Dict[int, List[Tuple[int, float]]]:
    """
    Analyze which qubits are idle during each layer and for how long.
    
    Args:
        kernel: Circuit kernel with layer information
        total_qubits: Total number of qubits in the circuit
        
    Returns:
        Dictionary mapping layer_idx -> list of (qubit, idle_duration_ns) tuples
    """
    idle_analysis = {}
    
    for layer_idx, layer in enumerate(kernel.get_layer_schedule()):
        active_qubits = set()
        
        # Collect all qubits active in this layer
        for op in layer['operations']:
            if 'qubits' in op:
                qubits = op['qubits']
                if isinstance(qubits, (list, tuple)):
                    active_qubits.update(qubits)
                else:
                    active_qubits.add(qubits)
        
        # Find idle qubits
        idle_qubits = []
        layer_duration = layer.get('duration_ns', 20.0)
        
        for q in range(total_qubits):
            if q not in active_qubits:
                idle_qubits.append((q, layer_duration))
        
        idle_analysis[layer_idx] = idle_qubits
    
    return idle_analysis


def optimize_layout_for_device(code_distance: int, device_couplers: List[Tuple[int, int]], 
                              coupler_fidelities: Dict[Tuple[int, int], float]) -> Dict[str, Any]:
    """
    Optimize qubit layout for a given code distance and device topology.
    
    This function would implement graph embedding algorithms to find the best
    mapping of code connectivity to device connectivity, considering fidelities.
    
    Args:
        code_distance: Distance of the quantum error correcting code
        device_couplers: List of available coupler pairs on device  
        coupler_fidelities: Fidelity for each coupler pair
        
    Returns:
        Optimized layout dictionary
    """
    # For now, return the hardcoded d=3 layout
    # In practice, this would implement sophisticated graph embedding
    if code_distance == 3:
        return make_surface_layout_d3_avoid_bad_edges()
    else:
        # Fallback for other distances
        raise NotImplementedError(f"Layout optimization for d={code_distance} not implemented")
