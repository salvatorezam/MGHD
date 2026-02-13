from __future__ import annotations

"""Native CUDA-Q trajectory sampling with Kraus-operator noise.

This module builds a repeated surface-code memory circuit directly with CUDA-Q,
attaches a ``cudaq.NoiseModel`` composed of Kraus channels, and samples
shot-resolved measurement trajectories.
"""

from dataclasses import dataclass
import os
from typing import Any
import warnings

import numpy as np
from mghd.samplers.cudaq_backend.noise_config import resolve_canonical_noise_spec


_NOISE_UNSUPPORTED_TARGETS = {"qpp-cpu", "stim"}
_GPU_TRAJECTORY_TARGETS = ("nvidia", "tensornet", "tensornet-mps")

def _configure_batched_trajectory_env() -> None:
    """Apply CUDA-Q batched-trajectory env knobs before importing cudaq.

    NVIDIA docs use `CUDAQ_MGPU__BATCH_SIZE` / `CUDAQ_MGPU__NUM_GPUS` for
    batched trajectory simulation on multi-GPU targets.
    """
    batch = str(os.getenv("MGHD_CUDAQ_TRAJ_BATCH_SIZE", "")).strip()
    ngpus = str(os.getenv("MGHD_CUDAQ_TRAJ_NUM_GPUS", "")).strip()
    if batch:
        os.environ.setdefault("CUDAQ_MGPU__BATCH_SIZE", batch)
    if ngpus:
        os.environ.setdefault("CUDAQ_MGPU__NUM_GPUS", ngpus)


def _gpu_runtime_available() -> bool:
    """Best-effort CUDA runtime check without hard-failing on CUDA init issues."""
    try:
        import torch

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            available = bool(torch.cuda.is_available())
            count = int(torch.cuda.device_count()) if available else 0
        return available and count > 0
    except Exception:
        return False


def _ensure_noise_capable_target() -> str:
    """Select a CUDA-Q GPU trajectory target that applies noise channels."""
    _configure_batched_trajectory_env()
    import cudaq

    requested = str(os.getenv("MGHD_CUDAQ_TARGET", "nvidia")).strip().lower() or "nvidia"
    requested_option = str(os.getenv("MGHD_CUDAQ_TARGET_OPTION", "")).strip()
    strict_target = os.getenv("MGHD_CUDAQ_TARGET_STRICT", "0") == "1"
    allow_density = os.getenv("MGHD_CUDAQ_ALLOW_DENSITY_MATRIX", "0") == "1"
    gpu_available = _gpu_runtime_available()

    if requested.startswith("density-matrix") and not allow_density:
        raise RuntimeError(
            "Density-matrix targets are disabled for trajectory mode. "
            "Set MGHD_CUDAQ_TARGET to nvidia/tensornet (or set "
            "MGHD_CUDAQ_ALLOW_DENSITY_MATRIX=1 to override)."
        )
    if requested in _GPU_TRAJECTORY_TARGETS and not gpu_available:
        raise RuntimeError(
            "No visible CUDA GPU runtime for CUDA-Q trajectory targets. "
            f"Requested target '{requested}'."
        )

    candidates = [requested]
    if not strict_target:
        for target in _GPU_TRAJECTORY_TARGETS:
            if target not in candidates:
                candidates.append(target)

    errors: list[str] = []
    for candidate in candidates:
        if candidate in _GPU_TRAJECTORY_TARGETS and not gpu_available:
            errors.append(f"{candidate}: no visible CUDA GPU runtime")
            continue
        option = requested_option if candidate == requested and requested_option else None
        try:
            if option:
                cudaq.set_target(candidate, option=option)
            else:
                cudaq.set_target(candidate)
            active = cudaq.get_target().name
            if active in _NOISE_UNSUPPORTED_TARGETS:
                raise RuntimeError(f"target '{active}' does not apply noise channels")
            if active.startswith("density-matrix") and not allow_density:
                raise RuntimeError(
                    f"target '{active}' is density-matrix (disabled for trajectory mode)"
                )
            return active
        except Exception as exc:
            suffix = f" option='{option}'" if option else ""
            errors.append(f"{candidate}{suffix}: {exc}")

    raise RuntimeError(
        "No usable CUDA-Q trajectory target found. "
        f"Tried: {', '.join(candidates)}. "
        f"Errors: {' | '.join(errors)}"
    )


def _guard_runtime_cost(*, layout: dict[str, Any], batch_size: int, cudaq_target: str) -> None:
    """Fail fast on known-infeasible local configurations."""
    if not str(cudaq_target).startswith("density-matrix"):
        return

    env = os.getenv
    try:
        max_qubits = int(env("MGHD_CUDAQ_TRAJ_MAX_QUBITS", "12"))
    except Exception:
        max_qubits = 12
    try:
        max_shots = int(env("MGHD_CUDAQ_TRAJ_MAX_SHOTS", "256"))
    except Exception:
        max_shots = 256

    total_qubits = int(layout.get("total_qubits", 0))
    if total_qubits > max_qubits:
        raise RuntimeError(
            "Native CUDA-Q trajectory backend on density-matrix targets is disabled "
            f"for total_qubits={total_qubits} (> {max_qubits}). "
            "Set MGHD_CUDAQ_TRAJ_MAX_QUBITS higher to override, or use "
            "MGHD_CUDAQ_BACKEND=legacy for scalable training."
        )
    if int(batch_size) > max_shots:
        raise RuntimeError(
            "Native CUDA-Q trajectory backend on density-matrix targets is disabled "
            f"for batch_size={int(batch_size)} (> {max_shots}). "
            "Set MGHD_CUDAQ_TRAJ_MAX_SHOTS higher to override."
        )


@dataclass(frozen=True)
class TrajectoryNoiseConfig:
    model_name: str
    model_version: str
    noise_ramp: str
    lambda_scale: float
    p_data: float
    p_meas: float
    p_1q: float
    p_2q: float
    p_idle_amp: float
    p_idle_phase: float
    p_meas0: float
    p_meas1: float
    p_hook: float
    p_xtalk: float
    p_erase: float
    p_long_range: float
    requested_phys_p: float | None


def resolve_trajectory_noise_config(
    *,
    phys_p: float | None,
    noise_scale: float | None,
    noise_overrides: dict[str, Any] | None = None,
) -> TrajectoryNoiseConfig:
    """Resolve canonical circuit-level noise probabilities."""
    spec = resolve_canonical_noise_spec(
        requested_phys_p=phys_p,
        noise_scale=noise_scale,
        overrides=noise_overrides,
    )

    # Idle decomposition defaults to splitting total idle equally unless
    # explicit Kraus components are provided.
    idle_total = float(spec.p_idle)
    p_idle_amp_env = os.getenv("MGHD_GENERIC_PIDLE_AMP", None)
    p_idle_phase_env = os.getenv("MGHD_GENERIC_PIDLE_PHASE", None)
    if p_idle_amp_env is not None or p_idle_phase_env is not None:
        p_idle_amp = float(np.clip(float(p_idle_amp_env or 0.0), 0.0, 1.0))
        p_idle_phase = float(np.clip(float(p_idle_phase_env or 0.0), 0.0, 1.0))
    else:
        p_idle_amp = 0.5 * idle_total
        p_idle_phase = 0.5 * idle_total

    return TrajectoryNoiseConfig(
        model_name=str(spec.model_name),
        model_version=str(spec.model_version),
        noise_ramp=str(spec.noise_ramp),
        lambda_scale=float(spec.lambda_scale),
        p_data=float(spec.p_data),
        p_meas=float(spec.p_meas),
        p_1q=float(np.clip(spec.p_1q, 0.0, 1.0)),
        p_2q=float(np.clip(spec.p_2q, 0.0, 1.0)),
        p_idle_amp=float(np.clip(p_idle_amp, 0.0, 1.0)),
        p_idle_phase=float(np.clip(p_idle_phase, 0.0, 1.0)),
        p_meas0=float(np.clip(spec.p_meas0, 0.0, 1.0)),
        p_meas1=float(np.clip(spec.p_meas1, 0.0, 1.0)),
        p_hook=float(np.clip(spec.p_hook, 0.0, 1.0)),
        p_xtalk=float(np.clip(spec.p_xtalk, 0.0, 1.0)),
        p_erase=float(np.clip(spec.p_erase, 0.0, 1.0)),
        p_long_range=float(np.clip(spec.p_long_range, 0.0, 1.0)),
        requested_phys_p=spec.requested_phys_p,
    )


def _append_idle_markers(kernel: Any, qureg: Any, all_qubits: list[int]) -> None:
    # Use RZ(0) as an explicit no-op marker so idle Kraus channels can be bound
    # to a concrete operator in CUDA-Q noise-model APIs.
    for qidx in all_qubits:
        kernel.rz(0.0, qureg[qidx])


def _build_surface_memory_kernel(
    *,
    layout: dict[str, Any],
    rounds: int,
) -> tuple[Any, dict[str, Any]]:
    import cudaq

    data = [int(q) for q in layout.get("data", [])]
    anc_x = [int(q) for q in layout.get("ancilla_x", [])]
    anc_z = [int(q) for q in layout.get("ancilla_z", [])]
    cz_layers = [[(int(a), int(d)) for (a, d) in layer] for layer in layout.get("cz_layers", [])]
    if not data or (not anc_x and not anc_z):
        raise ValueError("Surface layout missing data/ancilla sets for trajectory sampling.")

    total_qubits = int(layout.get("total_qubits", max(data + anc_x + anc_z) + 1))
    all_qubits = sorted(set(data + anc_x + anc_z))
    effective_rounds = max(2, int(rounds))

    kernel = cudaq.make_kernel()
    qreg = kernel.qalloc(total_qubits)

    round_specs: list[tuple[str, int]] = []
    for ridx in range(effective_rounds):
        is_x_round = (ridx % 2) == 0
        ancillas = anc_x if is_x_round else anc_z
        anc_set = set(ancillas)
        basis = "X" if is_x_round else "Z"

        for anc in ancillas:
            kernel.reset(qreg[anc])
        _append_idle_markers(kernel, qreg, all_qubits)

        if is_x_round:
            for anc in ancillas:
                kernel.h(qreg[anc])
            _append_idle_markers(kernel, qreg, all_qubits)

        for layer in cz_layers:
            for anc, dat in layer:
                if anc in anc_set:
                    kernel.cz(qreg[anc], qreg[dat])
            _append_idle_markers(kernel, qreg, all_qubits)

        if is_x_round:
            for anc in ancillas:
                kernel.h(qreg[anc])
            _append_idle_markers(kernel, qreg, all_qubits)

        for anc in ancillas:
            kernel.mz(qreg[anc])
        round_specs.append((basis, len(ancillas)))

    # Final data readout used as a proxy error witness for legacy packed format.
    for dq in data:
        kernel.mz(qreg[dq])

    meta = {
        "round_specs": round_specs,
        "num_data": len(data),
        "num_x_checks": len(anc_x),
        "num_z_checks": len(anc_z),
        "effective_rounds": effective_rounds,
    }
    return kernel, meta


def _build_kraus_noise_model(cfg: TrajectoryNoiseConfig) -> Any:
    import cudaq

    noise = cudaq.NoiseModel()

    if cfg.p_1q > 0.0:
        dep1 = cudaq.DepolarizationChannel(cfg.p_1q)
        for op in ("h", "x", "y", "z", "s", "t", "rx", "ry", "rz"):
            noise.add_all_qubit_channel(op, dep1)

    if cfg.p_2q > 0.0:
        # CUDA-Q's built-in DepolarizationChannel is single-qubit. For
        # controlled 2-qubit operations we construct a 2-qubit Kraus channel:
        # K0 = sqrt(1-p) I4, and the 15 non-identity Pauli products each with
        # weight sqrt(p/15).
        p2 = float(np.clip(cfg.p_2q, 0.0, 1.0))
        i2 = np.array([[1, 0], [0, 1]], dtype=np.complex128)
        x2 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        y2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        z2 = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        paulis = [i2, x2, y2, z2]
        ops = [np.kron(a, b) for a in paulis for b in paulis]
        kraus_ops = [np.sqrt(max(0.0, 1.0 - p2)) * ops[0]]
        if p2 > 0.0:
            scale = np.sqrt(p2 / 15.0)
            for op in ops[1:]:
                kraus_ops.append(scale * op)
        dep2 = cudaq.KrausChannel(kraus_ops)
        # Controlled operations in CUDA-Q noise APIs are represented as
        # base op + num_controls.
        noise.add_all_qubit_channel("x", dep2, 1)  # CX
        noise.add_all_qubit_channel("z", dep2, 1)  # CZ

    if cfg.p_idle_amp > 0.0:
        noise.add_all_qubit_channel("rz", cudaq.AmplitudeDampingChannel(cfg.p_idle_amp))
    if cfg.p_idle_phase > 0.0:
        noise.add_all_qubit_channel("rz", cudaq.PhaseFlipChannel(cfg.p_idle_phase))

    # Readout noise approximation: symmetric bit flips + asymmetric 1->0 damping.
    p_meas_avg = 0.5 * (cfg.p_meas0 + cfg.p_meas1)
    if p_meas_avg > 0.0:
        noise.add_all_qubit_channel("mz", cudaq.BitFlipChannel(float(np.clip(p_meas_avg, 0.0, 1.0))))
    if cfg.p_meas1 > 0.0:
        noise.add_all_qubit_channel("mz", cudaq.AmplitudeDampingChannel(cfg.p_meas1))

    return noise


def _sample_sequential_bits(
    *,
    kernel: Any,
    noise_model: Any,
    shots: int,
    seed: int | None = None,
) -> np.ndarray:
    import cudaq

    if seed is not None:
        try:
            cudaq.set_random_seed(int(seed))
        except Exception:
            pass

    result = cudaq.sample(
        kernel,
        shots_count=int(shots),
        noise_model=noise_model,
        explicit_measurements=True,
    )
    seq = list(result.get_sequential_data())
    if len(seq) != int(shots):
        raise RuntimeError(
            f"Trajectory sampler returned {len(seq)} sequential samples, expected {shots}."
        )
    if not seq:
        return np.zeros((0, 0), dtype=np.uint8)
    width = len(seq[0])
    out = np.zeros((len(seq), width), dtype=np.uint8)
    for i, bitstring in enumerate(seq):
        if len(bitstring) != width:
            raise RuntimeError("Inconsistent sequential-measurement width in CUDA-Q sample result.")
        out[i] = np.fromiter((1 if c == "1" else 0 for c in bitstring), dtype=np.uint8, count=width)
    return out


def _apply_augmented_channels(
    *,
    layout: dict[str, Any],
    syn_x: np.ndarray,
    syn_z: np.ndarray,
    data_readout: np.ndarray,
    cfg: TrajectoryNoiseConfig,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply augmented detector/data perturbations for circuit_augmented sweeps."""
    batch_size = int(syn_x.shape[0])
    n_x = int(syn_x.shape[1])
    n_z = int(syn_z.shape[1])
    n_data = int(data_readout.shape[1])
    if batch_size <= 0:
        return syn_x, syn_z, data_readout, np.zeros((0, n_data), dtype=np.uint8), np.zeros(
            (0, n_x + n_z), dtype=np.uint8
        )

    rng = np.random.default_rng(seed if seed is not None else 0)
    syn_x = np.asarray(syn_x, dtype=np.uint8).copy()
    syn_z = np.asarray(syn_z, dtype=np.uint8).copy()
    data_readout = np.asarray(data_readout, dtype=np.uint8).copy()
    erase_data_mask = np.zeros((batch_size, n_data), dtype=np.uint8)
    erase_det_mask = np.zeros((batch_size, n_z + n_x), dtype=np.uint8)

    # Hook-like correlated events: pair-flip one Z-check and one X-check.
    if cfg.p_hook > 0.0 and n_x > 0 and n_z > 0:
        hook_events = rng.random(batch_size) < cfg.p_hook
        hook_idx = np.where(hook_events)[0]
        if hook_idx.size:
            z_idx = rng.integers(0, n_z, size=hook_idx.size)
            x_idx = rng.integers(0, n_x, size=hook_idx.size)
            syn_z[hook_idx, z_idx] ^= 1
            syn_x[hook_idx, x_idx] ^= 1

    # Crosstalk: spectator-correlated detector flips.
    if cfg.p_xtalk > 0.0 and (n_x + n_z) > 0:
        z_events = rng.random((batch_size, n_z)) < cfg.p_xtalk if n_z > 0 else None
        x_events = rng.random((batch_size, n_x)) < cfg.p_xtalk if n_x > 0 else None
        if z_events is not None:
            syn_z ^= z_events.astype(np.uint8)
        if x_events is not None:
            syn_x ^= x_events.astype(np.uint8)

    # Long-range bursts: paired distant data flips with weak detector echoes.
    if cfg.p_long_range > 0.0 and n_data >= 2:
        burst = rng.random(batch_size) < cfg.p_long_range
        bidx = np.where(burst)[0]
        if bidx.size:
            q0 = rng.integers(0, n_data, size=bidx.size)
            min_sep = max(1, n_data // 3)
            q1 = (q0 + rng.integers(min_sep, n_data, size=bidx.size)) % n_data
            data_readout[bidx, q0] ^= 1
            data_readout[bidx, q1] ^= 1
            if n_z > 0:
                zz = rng.integers(0, n_z, size=bidx.size)
                syn_z[bidx, zz] ^= 1
            if n_x > 0:
                xx = rng.integers(0, n_x, size=bidx.size)
                syn_x[bidx, xx] ^= 1

    # Erasure mask generation (decoder-visible side channel).
    if cfg.p_erase > 0.0 and n_data > 0:
        em = rng.random((batch_size, n_data)) < cfg.p_erase
        erase_data_mask = em.astype(np.uint8)
        if n_z + n_x > 0:
            det_any = rng.random((batch_size, n_z + n_x)) < (0.5 * cfg.p_erase)
            erase_det_mask = det_any.astype(np.uint8)

    return syn_x, syn_z, data_readout, erase_data_mask, erase_det_mask


def sample_surface_trajectory_kraus(
    *,
    layout: dict[str, Any],
    batch_size: int,
    rounds: int,
    phys_p: float | None,
    noise_scale: float | None,
    seed: int | None = None,
    noise_overrides: dict[str, Any] | None = None,
) -> dict[str, np.ndarray | dict[str, Any]]:
    """Run native CUDA-Q shot trajectories and return last-round syndromes.

    Returns a dict with keys:
      - ``syn_x``: uint8 [B, n_x_checks]
      - ``syn_z``: uint8 [B, n_z_checks]
      - ``data_readout``: uint8 [B, n_data]
      - ``meta``: parser metadata and resolved noise parameters
    """
    active_target = _ensure_noise_capable_target()
    _guard_runtime_cost(layout=layout, batch_size=batch_size, cudaq_target=active_target)
    cfg = resolve_trajectory_noise_config(
        phys_p=phys_p,
        noise_scale=noise_scale,
        noise_overrides=noise_overrides,
    )
    kernel, kmeta = _build_surface_memory_kernel(layout=layout, rounds=rounds)
    noise_model = _build_kraus_noise_model(cfg)
    bits = _sample_sequential_bits(
        kernel=kernel,
        noise_model=noise_model,
        shots=batch_size,
        seed=seed,
    )

    n_data = int(kmeta["num_data"])
    n_x = int(kmeta["num_x_checks"])
    n_z = int(kmeta["num_z_checks"])
    round_specs = list(kmeta["round_specs"])

    syn_x_last = np.zeros((batch_size, n_x), dtype=np.uint8)
    syn_z_last = np.zeros((batch_size, n_z), dtype=np.uint8)

    offset = 0
    for basis, count in round_specs:
        chunk = bits[:, offset : offset + int(count)]
        offset += int(count)
        if basis == "X":
            syn_x_last = chunk.astype(np.uint8, copy=False)
        else:
            syn_z_last = chunk.astype(np.uint8, copy=False)

    data_readout = bits[:, offset : offset + n_data].astype(np.uint8, copy=False)
    if data_readout.shape[1] != n_data:
        raise RuntimeError(
            f"Trajectory parser expected {n_data} final data bits, got {data_readout.shape[1]}."
        )

    syn_x_last, syn_z_last, data_readout, erase_data_mask, erase_det_mask = _apply_augmented_channels(
        layout=layout,
        syn_x=syn_x_last,
        syn_z=syn_z_last,
        data_readout=data_readout,
        cfg=cfg,
        seed=seed,
    )

    return {
        "syn_x": syn_x_last.astype(np.uint8, copy=False),
        "syn_z": syn_z_last.astype(np.uint8, copy=False),
        "data_readout": data_readout.astype(np.uint8, copy=False),
        "erase_data_mask": erase_data_mask.astype(np.uint8, copy=False),
        "erase_det_mask": erase_det_mask.astype(np.uint8, copy=False),
        "meta": {
            "num_x_checks": n_x,
            "num_z_checks": n_z,
            "num_data": n_data,
            "effective_rounds": int(kmeta["effective_rounds"]),
            "noise_model_name": cfg.model_name,
            "noise_model_version": cfg.model_version,
            "noise_ramp": cfg.noise_ramp,
            "noise_params_resolved": {
                "requested_phys_p": cfg.requested_phys_p,
                "lambda_scale": cfg.lambda_scale,
                "p_data": cfg.p_data,
                "p_meas": cfg.p_meas,
                "p_1q": cfg.p_1q,
                "p_2q": cfg.p_2q,
                "p_idle": cfg.p_idle_amp + cfg.p_idle_phase,
                "p_meas0": cfg.p_meas0,
                "p_meas1": cfg.p_meas1,
                "p_hook": cfg.p_hook,
                "p_xtalk": cfg.p_xtalk,
                "p_erase": cfg.p_erase,
                "p_long_range": cfg.p_long_range,
            },
            "cudaq_target": active_target,
            "seed": seed,
        },
    }
