"""
Garnet Noise Model Implementation

This module holds hardcoded preprint parameters and exposes a noise model API
for both FOUNDATION (device-agnostic) and STUDENT (device-specific) modes.

Based on IQM Garnet quantum computer calibration data from the preprint.
All parameters are hardcoded as exact values from Table I and Appendix A.
"""

from typing import Dict, Tuple, Union, Optional
import numpy as np


# Hardcoded constants from the preprint (Table I medians/means + timings)
FOUNDATION_DEFAULTS = {
    # Table I (medians)
    "F1Q_median": 0.9989,        # single-qubit average gate fidelity
    "F2Q_median": 0.9906,        # two-qubit average gate fidelity
    "T1_median_us": 43.1,        # microseconds
    "T2_median_us": 2.8,         # microseconds
    "eps0_median": 0.0243,       # P(0->1)
    "eps1_median": 0.0363,       # P(1->0)
    # Gate durations (fixed)
    "t_prx_ns": 20.0,            # PRX gate duration (ns)
    "t_cz_ns": 40.0,             # CZ gate duration (ns)
    # Suggested default jitters for priors (hardcode conservative values)
    "jitter": {
        "F1Q": 0.002,   # ±0.2% abs around median
        "F2Q": 0.005,   # ±0.5% abs around median
        "T1": 0.15,     # ±15% relative
        "T2": 0.25,     # ±25% relative
        "eps": 0.005    # ±0.5% abs
    }
}

# Per-coupler 2Q fidelities (hardcoded from Appendix A), as fractions:
GARNET_COUPLER_F2 = {
    (0, 1): 0.9929, (0, 3): 0.9902, (2, 3): 0.9587, (2, 7): 0.9913,
    (1, 4): 0.9931, (3, 4): 0.9732, (3, 8): 0.9880, (7, 8): 0.9922,
    (7, 12): 0.9910, (4, 5): 0.9938, (4, 9): 0.9923, (8, 9): 0.9938,
    (8, 13): 0.9912, (12, 13): 0.9928, (5, 6): 0.9947, (5, 10): 0.9913,
    (9, 10): 0.9925, (9, 14): 0.9871, (13, 14): 0.9883, (13, 17): 0.9910,
    (6, 11): 0.9769, (10, 11): 0.9228, (10, 15): 0.9881, (14, 15): 0.9854,
    (14, 18): 0.9854, (17, 18): 0.9902, (11, 16): 0.9791, (15, 16): 0.9929,
    (15, 19): 0.9887, (18, 19): 0.9897
}

# Student defaults (hardcode per-device values):
STUDENT_DEFAULTS = {
    # Use per-coupler F2 above; per-qubit parameters default to medians (uniform across qubits).
    "F1Q_per_qubit": "use_median_uniform",
    "T1_per_qubit_us": "use_median_uniform",
    "T2_per_qubit_us": "use_median_uniform",
    "eps0_per_qubit": "use_median_uniform",
    "eps1_per_qubit": "use_median_uniform",
    "t_prx_ns": FOUNDATION_DEFAULTS["t_prx_ns"],
    "t_cz_ns": FOUNDATION_DEFAULTS["t_cz_ns"]
}


class GarnetFoundationPriors:
    """
    Samples pseudo-device parameters for the FOUNDATION model from hardcoded medians + jitter.
    
    This represents device-agnostic sampling where we model uncertainty around the 
    published median values from the Garnet preprint.
    """
    
    def __init__(self, defaults: dict = None):
        """
        Initialize Foundation priors with default or custom parameters.
        
        Args:
            defaults: Parameter dictionary, uses FOUNDATION_DEFAULTS if None
        """
        self.defaults = defaults if defaults is not None else FOUNDATION_DEFAULTS.copy()
    
    def sample_pseudo_device(self, rng: np.random.Generator, n_qubits: int = 20) -> Dict[str, Union[Dict, float]]:
        """
        Returns dict with keys:
          'F1Q': {q: float}, 'F2Q': {(i,j): float}, 'T1_us': {q: float},
          'T2_us': {q: float}, 'eps0': {q: float}, 'eps1': {q: float},
          't_prx_ns': float, 't_cz_ns': float
        Where:
          - F2Q per coupler sampled around F2Q_median (preserve bad edge tails e.g. allow min near 0.92).
          - All per-qubit params sampled around medians with given jitter.
        
        Args:
            rng: NumPy random generator for seeded sampling
            n_qubits: Number of qubits in the device (default: 20 for Garnet)
            
        Returns:
            Dictionary of sampled device parameters
        """
        jitter = self.defaults["jitter"]
        
        # Sample per-qubit F1Q around median with absolute jitter
        F1Q_base = self.defaults["F1Q_median"]
        F1Q = {}
        for q in range(n_qubits):
            jittered = F1Q_base + rng.uniform(-jitter["F1Q"], jitter["F1Q"])
            F1Q[q] = np.clip(jittered, 0.95, 0.9999)  # Reasonable bounds
        
        # Sample per-coupler F2Q around median, preserving bad edge characteristics
        F2Q_base = self.defaults["F2Q_median"]
        F2Q = {}
        for edge in GARNET_COUPLER_F2.keys():
            # Include some correlation with actual Garnet values for realism
            actual_f2 = GARNET_COUPLER_F2[edge]
            # Blend actual value with median-based sampling
            blend_weight = 0.3  # 30% influence from actual, 70% from median sampling
            median_sample = F2Q_base + rng.uniform(-jitter["F2Q"], jitter["F2Q"])
            blended = blend_weight * actual_f2 + (1 - blend_weight) * median_sample
            F2Q[edge] = np.clip(blended, 0.92, 0.995)  # Preserve tail behavior
        
        # Sample per-qubit T1 with relative jitter
        T1_base = self.defaults["T1_median_us"]
        T1_us = {}
        for q in range(n_qubits):
            relative_jitter = rng.uniform(-jitter["T1"], jitter["T1"])
            T1_us[q] = T1_base * (1 + relative_jitter)
            T1_us[q] = np.clip(T1_us[q], 10.0, 100.0)  # Reasonable bounds
        
        # Sample per-qubit T2 with relative jitter
        T2_base = self.defaults["T2_median_us"]
        T2_us = {}
        for q in range(n_qubits):
            relative_jitter = rng.uniform(-jitter["T2"], jitter["T2"])
            T2_us[q] = T2_base * (1 + relative_jitter)
            # Ensure T2 <= 2*T1 (physics constraint)
            T2_us[q] = np.clip(T2_us[q], 0.5, min(2 * T1_us[q], 10.0))
        
        # Sample per-qubit measurement errors with absolute jitter
        eps0_base = self.defaults["eps0_median"]
        eps1_base = self.defaults["eps1_median"]
        eps0, eps1 = {}, {}
        for q in range(n_qubits):
            eps0[q] = np.clip(
                eps0_base + rng.uniform(-jitter["eps"], jitter["eps"]),
                0.005, 0.1
            )
            eps1[q] = np.clip(
                eps1_base + rng.uniform(-jitter["eps"], jitter["eps"]),
                0.005, 0.1
            )
        
        return {
            'F1Q': F1Q,
            'F2Q': F2Q,
            'T1_us': T1_us,
            'T2_us': T2_us,
            'eps0': eps0,
            'eps1': eps1,
            't_prx_ns': self.defaults["t_prx_ns"],
            't_cz_ns': self.defaults["t_cz_ns"]
        }


class GarnetStudentCalibration:
    """
    Holds the hardcoded per-device calibration used for STUDENT model.
    
    This represents the actual measured device parameters from the Garnet preprint,
    with exact per-coupler 2Q fidelities and uniform per-qubit medians.
    """
    
    def __init__(self, n_qubits: int = 20):
        """
        Initialize with hardcoded Garnet calibration data.
        
        Args:
            n_qubits: Number of qubits (default: 20 for Garnet)
        """
        self.n_qubits = n_qubits
        
        # Build dicts with uniform per-qubit medians and exact per-coupler F2:
        self.F1Q = {q: FOUNDATION_DEFAULTS["F1Q_median"] for q in range(n_qubits)}
        self.F2Q = dict(GARNET_COUPLER_F2)
        self.T1_us = {q: FOUNDATION_DEFAULTS["T1_median_us"] for q in range(n_qubits)}
        self.T2_us = {q: FOUNDATION_DEFAULTS["T2_median_us"] for q in range(n_qubits)}
        self.eps0 = {q: FOUNDATION_DEFAULTS["eps0_median"] for q in range(n_qubits)}
        self.eps1 = {q: FOUNDATION_DEFAULTS["eps1_median"] for q in range(n_qubits)}
        self.t_prx_ns = STUDENT_DEFAULTS["t_prx_ns"]
        self.t_cz_ns = STUDENT_DEFAULTS["t_cz_ns"]
    
    def to_dict(self) -> Dict[str, Union[Dict, float]]:
        """Convert calibration to dictionary format matching sample_pseudo_device output."""
        return {
            'F1Q': self.F1Q,
            'F2Q': self.F2Q,
            'T1_us': self.T1_us,
            'T2_us': self.T2_us,
            'eps0': self.eps0,
            'eps1': self.eps1,
            't_prx_ns': self.t_prx_ns,
            't_cz_ns': self.t_cz_ns
        }


class GarnetNoiseModel:
    """
    Maps fidelities to depolarizing probabilities and timings + T1/T2 to idle Kraus parameters.
    
    This class converts the device parameter dictionaries into concrete noise probabilities
    suitable for circuit-level Monte-Carlo simulation.
    """
    
    def __init__(self, params: Dict[str, Union[Dict, float]]):
        """
        Initialize noise model from device parameters.
        
        Args:
            params: Device parameters from sample_pseudo_device() or GarnetStudentCalibration
                   Keys: 'F1Q', 'F2Q', 'T1_us', 'T2_us', 'eps0', 'eps1', 't_prx_ns', 't_cz_ns'
        """
        self.params = params
        
        # Extract timing parameters
        self.t_prx_ns = params['t_prx_ns']
        self.t_cz_ns = params['t_cz_ns']
        
        # Store parameter dictionaries
        self.F1Q = params['F1Q']
        self.F2Q = params['F2Q']
        self.T1_us = params['T1_us']
        self.T2_us = params['T2_us']
        self.eps0 = params['eps0']
        self.eps1 = params['eps1']
    
    @staticmethod
    def p_depol_from_F(F_avg: float, d: int) -> float:
        """
        Convert average gate fidelity to depolarizing probability.
        
        Correct formulas:
        - 1Q: F_avg = 1 - 3p/4  =>  p = 4(1 - F_avg)/3
        - 2Q: F_avg = 1 - 15p/16 => p = 16(1 - F_avg)/15
        
        Args:
            F_avg: Average gate fidelity
            d: Dimension of Hilbert space (2 for single-qubit, 4 for two-qubit)
            
        Returns:
            Depolarizing channel probability p
        """
        if F_avg >= 1.0:
            return 0.0
        if F_avg <= 0.0:
            return 1.0
        
        if d == 2:  # Single-qubit
            p = 4.0 * (1.0 - F_avg) / 3.0
        elif d == 4:  # Two-qubit
            p = 16.0 * (1.0 - F_avg) / 15.0
        else:
            # Generic formula for other dimensions
            p = (1.0 - F_avg) * (d + 1) / d
        
        return np.clip(p, 0.0, 1.0)
    
    def depol_1q_p(self, q: int) -> float:
        """
        Get depolarizing probability for single-qubit gate on qubit q.
        
        Args:
            q: Qubit index
            
        Returns:
            Depolarizing probability for PRX gate
        """
        F_1q = self.F1Q.get(q, FOUNDATION_DEFAULTS["F1Q_median"])
        return self.p_depol_from_F(F_1q, d=2)
    
    def depol_2q_p(self, edge: Tuple[int, int]) -> float:
        """
        Get depolarizing probability for two-qubit gate on edge.
        
        Args:
            edge: Tuple of qubit indices (i, j)
            
        Returns:
            Depolarizing probability for CZ gate
        """
        # Handle both (i,j) and (j,i) orderings
        if edge in self.F2Q:
            F_2q = self.F2Q[edge]
        elif (edge[1], edge[0]) in self.F2Q:
            F_2q = self.F2Q[(edge[1], edge[0])]
        else:
            # Fallback to median if edge not in calibration
            F_2q = FOUNDATION_DEFAULTS["F2Q_median"]
        
        return self.p_depol_from_F(F_2q, d=4)
    
    def idle_params(self, q: int, dt_ns: float) -> Tuple[float, float]:
        """
        Return (p_amp, p_dephase) for amplitude damping and pure dephasing over dt_ns.
        
        Args:
            q: Qubit index
            dt_ns: Idle duration in nanoseconds
            
        Returns:
            Tuple of (amplitude_damping_prob, pure_dephasing_prob)
        """
        T1_us = self.T1_us.get(q, FOUNDATION_DEFAULTS["T1_median_us"])
        T2_us = self.T2_us.get(q, FOUNDATION_DEFAULTS["T2_median_us"])
        
        # Convert to nanoseconds (T*_us * 1e3)
        T1_ns = T1_us * 1e3
        T2_ns = T2_us * 1e3
        
        # Amplitude damping probability
        p_amp = 1.0 - np.exp(-dt_ns / T1_ns)
        
        # Pure dephasing: 1/T2 = 1/(2*T1) + 1/T_phi
        # So T_phi = 1 / (1/T2 - 1/(2*T1))
        den = 1.0 / T2_ns - 1.0 / (2.0 * T1_ns)
        if den <= 0:
            Tphi_ns = 1e12  # Large sentinel for unphysical regime
        else:
            Tphi_ns = 1.0 / den
        
        p_dephase = 1.0 - np.exp(-dt_ns / Tphi_ns)
        
        # Clamp to [0,1]
        return np.clip(p_amp, 0.0, 1.0), np.clip(p_dephase, 0.0, 1.0)
    
    def meas_asym_errors(self, q: int) -> Tuple[float, float]:
        """
        Return (eps0, eps1) for readout assignment errors.
        
        Args:
            q: Qubit index
            
        Returns:
            Tuple of (P(measure_1|state_0), P(measure_0|state_1))
        """
        eps0 = self.eps0.get(q, FOUNDATION_DEFAULTS["eps0_median"])
        eps1 = self.eps1.get(q, FOUNDATION_DEFAULTS["eps1_median"])
        return eps0, eps1
