"""
Syndrome Generation Module

This module implements batched Monte-Carlo trajectories in CUDA-Q with circuit-level 
noise applied between layers. Provides high-level interfaces for surface, BB/qLDPC, 
and repetition code syndrome generation.

The implementation follows the IQM Garnet noise model with proper idle noise,
gate depolarizing noise, and measurement assignment errors.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import time
from collections import defaultdict

from .garnet_noise import (
    GarnetFoundationPriors, 
    GarnetStudentCalibration, 
    GarnetNoiseModel,
    FOUNDATION_DEFAULTS
)
from .circuits import (
    build_round_surface,
    build_round_bb, 
    build_round_repetition,
    analyze_idle_qubits,
    make_surface_layout_d3_avoid_bad_edges,
    build_H_rotated_d3,
    place_rotated_d3_on_garnet,
)


class CudaQSimulator:
    """
    CUDA-Q quantum simulator for circuit-level noise simulation.
    
    Provides batched Monte-Carlo trajectory simulation with the IQM Garnet noise model,
    including gate depolarizing noise, idle amplitude damping and dephasing, and 
    measurement assignment errors.
    """
    
    def __init__(self, noise_model: GarnetNoiseModel, batch_size: int, rng: np.random.Generator):
        """
        Initialize simulator with noise model and batch configuration.
        
        Args:
            noise_model: Garnet noise model for gate and idle errors
            batch_size: Number of parallel trajectories to simulate
            rng: Random number generator for seeded simulation
        """
        self.noise_model = noise_model
        self.batch_size = batch_size
        self.rng = rng
        
        # Track quantum state for each trajectory (simplified)
        self.n_qubits = 20  # Default to Garnet size
        
        # For trajectory simulation, track:
        # - Qubit states (|0⟩ or |1⟩ for each trajectory)
        # - Error accumulation (Pauli X and Z errors)
        self.reset_state()
    
    def reset_state(self, n_qubits: Optional[int] = None):
        """Reset all trajectories to |0...0⟩ state."""
        if n_qubits is not None:
            self.n_qubits = n_qubits
            
        # State tracking: 0 = |0⟩, 1 = |1⟩ for each (trajectory, qubit)
        self.qubit_states = np.zeros((self.batch_size, self.n_qubits), dtype=np.int8)
        
        # Error tracking: accumulated Pauli errors
        self.pauli_x_errors = np.zeros((self.batch_size, self.n_qubits), dtype=np.int8)
        self.pauli_z_errors = np.zeros((self.batch_size, self.n_qubits), dtype=np.int8)
    
    def apply_idle_noise(self, qubits: List[int], duration_ns: float):
        """
        Apply amplitude damping and pure dephasing to idle qubits.
        
        Args:
            qubits: List of qubit indices experiencing idle noise
            duration_ns: Idle duration in nanoseconds
        """
        for q in qubits:
            p_amp, p_dephase = self.noise_model.idle_params(q, duration_ns)
            
            # Amplitude damping: |1⟩ -> |0⟩ with probability p_amp
            excited_mask = (self.qubit_states[:, q] == 1)
            damp_events = self.rng.random(self.batch_size) < p_amp
            flip_to_ground = excited_mask & damp_events
            self.qubit_states[flip_to_ground, q] = 0
            # Track X error when amplitude damping occurs
            self.pauli_x_errors[flip_to_ground, q] ^= 1
            
            # Pure dephasing: Apply Z error with probability p_dephase
            dephase_events = self.rng.random(self.batch_size) < p_dephase
            self.pauli_z_errors[dephase_events, q] ^= 1
    
    def apply_prx_gate(self, qubit: int):
        """
        Apply PRX (X rotation) gate with depolarizing noise.
        
        Args:
            qubit: Target qubit index
        """
        # Perfect PRX operation: bit flip
        self.qubit_states[:, qubit] ^= 1
        
        # Apply depolarizing noise
        p_depol = self.noise_model.depol_1q_p(qubit)
        
        # Depolarizing channel: apply random Pauli with probability p_depol
        error_events = self.rng.random(self.batch_size) < p_depol
        if np.any(error_events):
            # Random Pauli: I, X, Y, Z with equal probability
            pauli_choice = self.rng.integers(0, 4, size=np.sum(error_events))
            
            trajectory_indices = np.where(error_events)[0]
            for i, pauli in enumerate(pauli_choice):
                traj_idx = trajectory_indices[i]
                if pauli == 1:  # X error
                    self.pauli_x_errors[traj_idx, qubit] ^= 1
                    self.qubit_states[traj_idx, qubit] ^= 1  # Apply X error immediately
                elif pauli == 2:  # Y error = XZ
                    self.pauli_x_errors[traj_idx, qubit] ^= 1
                    self.pauli_z_errors[traj_idx, qubit] ^= 1
                    self.qubit_states[traj_idx, qubit] ^= 1  # Apply X part
                elif pauli == 3:  # Z error
                    self.pauli_z_errors[traj_idx, qubit] ^= 1
                # pauli == 0 is identity, no action needed
    
    def apply_cz_gate(self, control: int, target: int):
        """
        Apply CZ gate with depolarizing noise.
        
        Args:
            control: Control qubit index
            target: Target qubit index
        """
        # Perfect CZ operation: phase flip if both qubits are |1⟩
        both_excited = (self.qubit_states[:, control] == 1) & (self.qubit_states[:, target] == 1)
        # CZ is diagonal, so no state change needed (we track phase errors separately)
        
        # Apply 2-qubit depolarizing noise
        edge = (control, target)
        p_depol = self.noise_model.depol_2q_p(edge)
        
        error_events = self.rng.random(self.batch_size) < p_depol
        if np.any(error_events):
            # 16 possible 2-qubit Paulis: II, IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ
            pauli_choice = self.rng.integers(0, 16, size=np.sum(error_events))
            
            trajectory_indices = np.where(error_events)[0]
            for i, pauli in enumerate(pauli_choice):
                traj_idx = trajectory_indices[i]
                
                # Decode 2-qubit Pauli
                control_pauli = pauli // 4  # 0=I, 1=X, 2=Y, 3=Z
                target_pauli = pauli % 4
                
                # Apply to control qubit
                if control_pauli == 1:  # X
                    self.pauli_x_errors[traj_idx, control] ^= 1
                    self.qubit_states[traj_idx, control] ^= 1
                elif control_pauli == 2:  # Y = XZ
                    self.pauli_x_errors[traj_idx, control] ^= 1
                    self.pauli_z_errors[traj_idx, control] ^= 1
                    self.qubit_states[traj_idx, control] ^= 1
                elif control_pauli == 3:  # Z
                    self.pauli_z_errors[traj_idx, control] ^= 1
                
                # Apply to target qubit
                if target_pauli == 1:  # X
                    self.pauli_x_errors[traj_idx, target] ^= 1
                    self.qubit_states[traj_idx, target] ^= 1
                elif target_pauli == 2:  # Y = XZ
                    self.pauli_x_errors[traj_idx, target] ^= 1
                    self.pauli_z_errors[traj_idx, target] ^= 1
                    self.qubit_states[traj_idx, target] ^= 1
                elif target_pauli == 3:  # Z
                    self.pauli_z_errors[traj_idx, target] ^= 1
    
    def measure_qubit(self, qubit: int) -> np.ndarray:
        """
        Measure qubit in Z-basis with assignment errors.
        
        Args:
            qubit: Qubit index to measure
            
        Returns:
            Array of measurement results (0 or 1) for each trajectory
        """
        # Perfect measurement outcome
        perfect_results = self.qubit_states[:, qubit].copy()
        
        # Apply measurement assignment errors
        eps0, eps1 = self.noise_model.meas_asym_errors(qubit)
        
        # Apply eps0: P(measure 1 | state 0)
        state_0_mask = (perfect_results == 0)
        flip_to_1 = state_0_mask & (self.rng.random(self.batch_size) < eps0)
        
        # Apply eps1: P(measure 0 | state 1)  
        state_1_mask = (perfect_results == 1)
        flip_to_0 = state_1_mask & (self.rng.random(self.batch_size) < eps1)
        
        # Construct final measurement results
        measured_results = perfect_results.copy()
        measured_results[flip_to_1] = 1
        measured_results[flip_to_0] = 0
        
        return measured_results
    
    def get_final_errors(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get final accumulated Pauli errors.
        
        Returns:
            Tuple of (X_errors, Z_errors) arrays of shape (batch_size, n_qubits)
        """
        return self.pauli_x_errors.copy(), self.pauli_z_errors.copy()


def simulate_circuit_with_noise(kernel, simulator: CudaQSimulator) -> Dict[str, np.ndarray]:
    """
    Simulate a circuit kernel with the given noise simulator.
    
    Args:
        kernel: Circuit kernel with layered operations
        simulator: CUDA-Q noise simulator
        
    Returns:
        Dictionary with measurement results and final error state
    """
    layers = kernel.get_layer_schedule()
    measurement_results = {}
    
    for layer_idx, layer in enumerate(layers):
        layer_type = layer['type']
        operations = layer['operations']
        duration_ns = layer.get('duration_ns', 20.0)
        
        # Determine which qubits are active in this layer
        active_qubits = set()
        for op in operations:
            active_qubits.update(op['qubits'])
        
        # Apply idle noise to qubits NOT participating in this layer
        all_qubits = set(range(simulator.n_qubits))
        idle_qubits = list(all_qubits - active_qubits)
        
        if idle_qubits and duration_ns > 0:
            simulator.apply_idle_noise(idle_qubits, duration_ns)
        
        # Apply gate operations with their specific noise
        for op in operations:
            gate_type = op['gate']
            qubits = op['qubits']
            
            if gate_type == 'PRX':
                simulator.apply_prx_gate(qubits[0])
            elif gate_type == 'CZ':
                simulator.apply_cz_gate(qubits[0], qubits[1])
            elif gate_type == 'MEASURE_Z':
                qubit = qubits[0]
                results = simulator.measure_qubit(qubit)
                measurement_results[f"layer_{layer_idx}_qubit_{qubit}"] = results
    
    # Get final error states
    x_errors, z_errors = simulator.get_final_errors()
    
    return {
        'measurements': measurement_results,
        'x_errors': x_errors,
        'z_errors': z_errors
    }


def extract_syndrome_from_measurements(measurements: Dict[str, np.ndarray], 
                                     layout: Dict[str, Any],
                                     code_type: str) -> np.ndarray:
    """
    Extract syndrome bits from measurement results based on code structure.
    
    Args:
        measurements: Raw measurement results from simulation
        layout: Code layout information
        code_type: Type of code ('surface', 'bb', 'repetition')
        
    Returns:
        Syndrome array of shape (batch_size, num_syndrome_bits)
    """
    # Extract measurement results by ancilla type
    if not measurements:
        return np.array([])
    
    # Get batch size from first measurement
    batch_size = len(next(iter(measurements.values())))
    
    if code_type == 'surface':
        # For surface codes: extract X and Z stabilizer measurements
        x_ancillas = layout.get('ancilla_x', [])
        z_ancillas = layout.get('ancilla_z', [])
        
        syndrome_bits = []
        
        # Extract X-stabilizer results (from odd rounds)
        for ancilla in x_ancillas:
            found_result = None
            for key, results in measurements.items():
                if f"qubit_{ancilla}" in key:
                    found_result = results
                    break
            if found_result is not None:
                syndrome_bits.append(found_result)
        
        # Extract Z-stabilizer results (from even rounds)  
        for ancilla in z_ancillas:
            found_result = None
            for key, results in measurements.items():
                if f"qubit_{ancilla}" in key:
                    found_result = results
                    break
            if found_result is not None:
                syndrome_bits.append(found_result)
        
        if syndrome_bits:
            return np.column_stack(syndrome_bits)
        else:
            return np.zeros((batch_size, len(x_ancillas) + len(z_ancillas)), dtype=np.uint8)
    
    elif code_type == 'bb':
        # For BB codes: extract all check measurements
        num_checks = layout.get('num_checks', 0)
        syndrome_bits = []
        
        for check_idx in range(num_checks):
            found_result = None
            for key, results in measurements.items():
                # Look for measurements from ancillas associated with this check
                if f"check_{check_idx}" in key or any(f"qubit_{q}" in key for q in range(20, 20+num_checks)):
                    found_result = results
                    break
            
            if found_result is not None:
                syndrome_bits.append(found_result)
            else:
                syndrome_bits.append(np.zeros(batch_size, dtype=np.uint8))
        
        if syndrome_bits:
            return np.column_stack(syndrome_bits)
        else:
            return np.zeros((batch_size, num_checks), dtype=np.uint8)
    
    elif code_type == 'repetition':
        # For repetition codes: extract ancilla measurements
        ancillas = layout.get('ancilla', [])
        syndrome_bits = []
        
        for ancilla in ancillas:
            found_result = None
            for key, results in measurements.items():
                if f"qubit_{ancilla}" in key:
                    found_result = results
                    break
            if found_result is not None:
                syndrome_bits.append(found_result)
        
        if syndrome_bits:
            return np.column_stack(syndrome_bits)
        else:
            return np.zeros((batch_size, len(ancillas)), dtype=np.uint8)
    
    # Fallback
    return np.zeros((batch_size, 1), dtype=np.uint8)


def pack_syndrome_and_errors(syndrome: np.ndarray, x_errors: np.ndarray, z_errors: np.ndarray,
                           layout: Dict[str, Any], code_type: str) -> np.ndarray:
    """
    Pack syndrome and error data to match existing pipeline formats.
    
    Args:
        syndrome: Raw syndrome measurements
        x_errors: X Pauli errors on data qubits
        z_errors: Z Pauli errors on data qubits  
        layout: Code layout information
        code_type: Type of code for format-specific packing
        
    Returns:
        Packed array matching existing format expectations
    """
    batch_size = syndrome.shape[0]
    data_qubits = layout.get('data', [])
    
    if not data_qubits:
        return syndrome
    
    # Extract errors on data qubits only
    x_data_errors = x_errors[:, data_qubits]
    z_data_errors = z_errors[:, data_qubits]
    
    if code_type == 'surface':
        # Surface code format: [X_syndrome, 2*Z_syndrome, X_error + 2*Z_error]
        num_x_checks = len(layout.get('ancilla_x', []))
        num_z_checks = len(layout.get('ancilla_z', []))
        
        if syndrome.shape[1] >= num_x_checks + num_z_checks:
            x_syndrome = syndrome[:, :num_x_checks]
            z_syndrome = syndrome[:, num_x_checks:num_x_checks + num_z_checks]
            
            # Pack syndrome: [X_checks, 2*Z_checks]
            syndromexz = np.concatenate([x_syndrome, 2 * z_syndrome], axis=1)
        else:
            syndromexz = syndrome
        
        # Pack data errors: X + 2*Z
        errorxz = x_data_errors + 2 * z_data_errors
        
        # Final format: [syndromexz | errorxz]
        return np.concatenate([syndromexz, errorxz], axis=1).astype(np.uint8)
    
    elif code_type == 'bb':
        # BB format: [X_syndrome, Z_syndrome, X_error + 2*Z_error]
        num_checks = syndrome.shape[1]
        half_checks = num_checks // 2
        
        # Split syndrome into X and Z parts
        x_syndrome = syndrome[:, :half_checks] if half_checks > 0 else np.zeros((batch_size, 0), dtype=np.uint8)
        z_syndrome = syndrome[:, half_checks:] if half_checks > 0 else np.zeros((batch_size, 0), dtype=np.uint8)
        
        # Pack data errors
        errorxz = x_data_errors + 2 * z_data_errors
        
        # Final format: [X_syndrome, Z_syndrome, errorxz]
        return np.concatenate([x_syndrome, z_syndrome, errorxz], axis=1).astype(np.uint8)
    
    elif code_type == 'repetition':
        # Repetition format: [syndrome | error]
        errorxz = x_data_errors + 2 * z_data_errors
        return np.concatenate([syndrome, errorxz], axis=1).astype(np.uint8)
    
    # Default: just concatenate
    errorxz = x_data_errors + 2 * z_data_errors
    return np.concatenate([syndrome, errorxz], axis=1).astype(np.uint8)


def sample_surface_cudaq(mode: str, batch_size: int, T: int, layout: Dict[str, Any],
                        rng: np.random.Generator, bitpack: bool = False,
                        surface_layout: str = "planar",
                        phys_p: float = None, noise_scale: float = None) -> np.ndarray:
    """
    Sample surface code syndromes using CUDA-Q with circuit-level noise.
    
    Args:
        mode: "foundation" or "student"
        batch_size: Number of syndrome samples to generate
        T: Number of syndrome extraction rounds
        layout: Surface code layout dictionary
        rng: Random number generator for reproducible sampling
        bitpack: Whether to pack bits into bytes (currently ignored)
        
    Returns:
        Packed syndrome + error array matching panq_functions format
    """
    # Initialize noise model based on mode
    if mode == "foundation":
        priors = GarnetFoundationPriors()
        params = priors.sample_pseudo_device(rng, n_qubits=20)
    elif mode == "student":
        calibration = GarnetStudentCalibration(n_qubits=20)
        params = calibration.to_dict()
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'foundation' or 'student'")
    
    # Determine noise scaling from requested phys_p or explicit noise_scale
    scale = 1.0
    if noise_scale is not None:
        try:
            scale = float(noise_scale)
        except Exception:
            scale = 1.0
    elif phys_p is not None:
        # Map requested physical error rate to a global noise scaling.
        # Use a heuristic baseline p_ref for d=3, e.g., 0.03 (mid acceptance grid).
        p_ref = 0.03
        try:
            scale = max(0.1, min(5.0, float(phys_p) / p_ref))
        except Exception:
            scale = 1.0

    noise_model = GarnetNoiseModel(params, scale=scale)
    simulator = CudaQSimulator(noise_model, batch_size, rng)

    # Optional rotated layout override for d=3
    if surface_layout == "rotated":
        # Place rotated d=3 onto Garnet avoiding bad edges
        data_map, anc_map, cz_layers_phys = place_rotated_d3_on_garnet(params)
        # Build layout dict
        layout = {
            'data': [data_map[i] for i in sorted(data_map.keys())],
            'ancilla_x': [anc_map['X'][i] for i in sorted(anc_map['X'].keys())],
            'ancilla_z': [anc_map['Z'][i] for i in sorted(anc_map['Z'].keys())],
            'cz_layers': cz_layers_phys['X'] + cz_layers_phys['Z'],
            'prx_layers': [list(anc_map['X'].values()), []],
            'total_qubits': max(max(data_map.values()), max(anc_map['X'].values()), max(anc_map['Z'].values())) + 1,
            'distance': 3,
            'surface_layout': 'rotated'
        }
    
    # Determine total qubits needed
    all_qubits = set(layout.get('data', []))
    all_qubits.update(layout.get('ancilla_x', []))
    all_qubits.update(layout.get('ancilla_z', []))
    max_qubit = max(all_qubits) if all_qubits else 19
    simulator.reset_state(n_qubits=max_qubit + 1)
    
    # For surface codes, we need measurements from both X and Z stabilizers
    # Force at least 2 rounds to get both types, or use last X and Z rounds
    effective_T = max(T, 2)
    all_measurements = {}
    
    for round_idx in range(effective_T):
        # Build circuit for this round
        kernel = build_round_surface(layout, round_idx)
        
        # Simulate with noise
        round_results = simulate_circuit_with_noise(kernel, simulator)
        
        # Accumulate measurements (prefix with round number)
        for key, results in round_results['measurements'].items():
            all_measurements[f"round_{round_idx}_{key}"] = results
    
    # Extract syndrome measurements from both X and Z rounds
    # Find the most recent X round (even) and Z round (odd)
    x_round_idx = effective_T - 1 if (effective_T - 1) % 2 == 0 else effective_T - 2
    z_round_idx = effective_T - 1 if (effective_T - 1) % 2 == 1 else effective_T - 2
    
    # Ensure we have valid rounds
    x_round_idx = max(0, x_round_idx)
    z_round_idx = max(1, z_round_idx)
    
    # Collect X-stabilizer measurements
    x_measurements = {}
    for key, results in all_measurements.items():
        if key.startswith(f"round_{x_round_idx}_"):
            clean_key = key.replace(f"round_{x_round_idx}_", "")
            x_measurements[clean_key] = results
    
    # Collect Z-stabilizer measurements  
    z_measurements = {}
    for key, results in all_measurements.items():
        if key.startswith(f"round_{z_round_idx}_"):
            clean_key = key.replace(f"round_{z_round_idx}_", "")
            z_measurements[clean_key] = results
    
    # Combine measurements for syndrome extraction
    combined_measurements = {}
    combined_measurements.update(x_measurements)
    combined_measurements.update(z_measurements)
    
    syndrome = extract_syndrome_from_measurements(combined_measurements, layout, 'surface')
    
    # Get final error state
    x_errors, z_errors = simulator.get_final_errors()
    
    if surface_layout == "rotated":
        # For rotated d=3, expose raw [B, 8] syndrome or bitpack to 1 byte if requested
        # Here we return just the 8-bit syndrome for downstream teacher usage
        syn = syndrome.astype(np.uint8)
        if bitpack:
            # Pack 8 bits into 1 byte per sample
            B, N = syn.shape
            assert N == 8, "Rotated d=3 expects 8 syndrome bits"
            bytes_out = np.zeros((B, 1), dtype=np.uint8)
            for b in range(B):
                val = 0
                for k in range(8):
                    val |= (int(syn[b, k]) & 1) << k
                bytes_out[b, 0] = val
            return bytes_out
        else:
            return syn
    else:
        # Pack in planar-like surface code format
        packed_result = pack_syndrome_and_errors(syndrome, x_errors, z_errors, layout, 'surface')
        return packed_result


def sample_bb_cudaq(mode: str, batch_size: int, T: int, hx: np.ndarray, hz: np.ndarray,
                   mapping: Dict[int, int], rng: np.random.Generator, bitpack: bool = False) -> np.ndarray:
    """
    Sample BB/qLDPC code syndromes using CUDA-Q with circuit-level noise.
    
    Args:
        mode: "foundation" or "student"
        batch_size: Number of syndrome samples to generate
        T: Number of syndrome extraction rounds
        hx: X-check matrix
        hz: Z-check matrix
        mapping: Logical to physical qubit mapping
        rng: Random number generator
        bitpack: Whether to pack bits into bytes (currently ignored)
        
    Returns:
        Packed syndrome + error array matching bb_panq_functions format
    """
    # Initialize noise model
    if mode == "foundation":
        priors = GarnetFoundationPriors()
        params = priors.sample_pseudo_device(rng, n_qubits=20)
    elif mode == "student":
        calibration = GarnetStudentCalibration(n_qubits=20)
        params = calibration.to_dict()
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    noise_model = GarnetNoiseModel(params)
    simulator = CudaQSimulator(noise_model, batch_size, rng)
    
    # Determine qubits needed
    num_data_qubits = max(hx.shape[1], hz.shape[1])
    num_x_checks, num_z_checks = hx.shape[0], hz.shape[0]
    total_qubits = max(mapping.values()) + num_x_checks + num_z_checks + 1 if mapping else num_data_qubits + num_x_checks + num_z_checks
    
    simulator.reset_state(n_qubits=total_qubits)
    
    # Create layout info for BB code
    layout = {
        'data': list(range(num_data_qubits)),
        'num_checks': num_x_checks + num_z_checks,
        'hx': hx,
        'hz': hz
    }
    
    # Run T rounds
    all_measurements = {}
    
    for round_idx in range(T):
        kernel = build_round_bb(hx, hz, mapping, round_idx)
        round_results = simulate_circuit_with_noise(kernel, simulator)
        
        for key, results in round_results['measurements'].items():
            all_measurements[f"round_{round_idx}_{key}"] = results
    
    # Extract syndrome from last round
    last_round_measurements = {
        key.replace(f"round_{T-1}_", ""): results
        for key, results in all_measurements.items()
        if key.startswith(f"round_{T-1}_")
    }
    
    syndrome = extract_syndrome_from_measurements(last_round_measurements, layout, 'bb')
    x_errors, z_errors = simulator.get_final_errors()
    
    # Pack in BB format
    packed_result = pack_syndrome_and_errors(syndrome, x_errors, z_errors, layout, 'bb')
    
    return packed_result


def sample_repetition_cudaq(mode: str, batch_size: int, T: int, layout: Dict[str, Any],
                           rng: np.random.Generator, bitpack: bool = False) -> np.ndarray:
    """
    Sample repetition code syndromes using CUDA-Q with circuit-level noise.
    
    Args:
        mode: "foundation" or "student" 
        batch_size: Number of syndrome samples to generate
        T: Number of syndrome extraction rounds
        layout: Repetition code layout dictionary
        rng: Random number generator
        bitpack: Whether to pack bits into bytes (currently ignored)
        
    Returns:
        Packed syndrome + error array matching expected format
    """
    # Initialize noise model
    if mode == "foundation":
        priors = GarnetFoundationPriors()
        params = priors.sample_pseudo_device(rng, n_qubits=20)
    elif mode == "student":
        calibration = GarnetStudentCalibration(n_qubits=20)
        params = calibration.to_dict()
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    noise_model = GarnetNoiseModel(params)
    simulator = CudaQSimulator(noise_model, batch_size, rng)
    
    # Determine qubits needed
    all_qubits = set(layout.get('data', []))
    all_qubits.update(layout.get('ancilla', []))
    max_qubit = max(all_qubits) if all_qubits else 10
    simulator.reset_state(n_qubits=max_qubit + 1)
    
    # Run T rounds
    all_measurements = {}
    
    for round_idx in range(T):
        kernel = build_round_repetition(layout, round_idx)
        round_results = simulate_circuit_with_noise(kernel, simulator)
        
        for key, results in round_results['measurements'].items():
            all_measurements[f"round_{round_idx}_{key}"] = results
    
    # Extract syndrome from last round
    last_round_measurements = {
        key.replace(f"round_{T-1}_", ""): results
        for key, results in all_measurements.items()
        if key.startswith(f"round_{T-1}_")
    }
    
    syndrome = extract_syndrome_from_measurements(last_round_measurements, layout, 'repetition')
    x_errors, z_errors = simulator.get_final_errors()
    
    # Pack in repetition format
    packed_result = pack_syndrome_and_errors(syndrome, x_errors, z_errors, layout, 'repetition')
    
    return packed_result
