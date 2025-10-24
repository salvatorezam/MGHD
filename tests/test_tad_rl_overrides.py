from __future__ import annotations

import numpy as np
import pytest

from mghd.cli.train_core import _base_overrides_from_maps, _materialize_overrides
from mghd.codes.qpu_profile import GateError, QPUProfile
from mghd.tad.context import context_vector
from mghd.tad.rl.lin_ts import LinTSBandit
from mghd.tad.weighting import feature_vector, schedule_to_weight_maps


def _toy_schedule_ir() -> list[tuple[int, str, tuple[int, ...], None]]:
    """
    Minimal schedule with both 1Q and 2Q gates to exercise weighting.
    Matches the tuple format consumed by schedule_to_weight_maps.
    """
    return [
        (0, "rx90", (0,), None),
        (1, "cx", (0, 1), None),
        (2, "rx90", (1,), None),
        (3, "cx", (0, 1), None),
    ]


def _toy_profile() -> QPUProfile:
    gate_error = GateError(
        p_1q={"rx90": 3e-4, "rz": 1e-4},
        p_2q={"cx": {"default": 1.1e-3}},
        p_meas=2e-3,
        t1_us={},
        t2_us={},
        idle_us=0.1,
        crosstalk_pairs={},
    )
    return QPUProfile(
        name="toy",
        n_qubits=3,
        coupling=[(0, 1)],
        gate_error=gate_error,
        meta={},
    )


def test_weight_maps_and_overrides_have_expected_shape():
    profile = _toy_profile()
    schedule_ir = _toy_schedule_ir()
    maps = schedule_to_weight_maps(schedule_ir, profile, profile.n_qubits)
    assert "w_qubit" in maps and "w_pair" in maps
    assert maps["w_qubit"], "Expected non-empty per-qubit map"

    overrides = _base_overrides_from_maps(maps, profile.n_qubits)
    llr = overrides["llr"]
    mwpm = overrides["mwpm"]

    assert llr.shape == (profile.n_qubits,)
    assert mwpm.shape == (profile.n_qubits,)
    assert np.isfinite(llr).all()
    assert np.isfinite(mwpm).all()
    # Confirm the overrides are non-trivial
    assert not np.allclose(llr, 0.0)
    assert not np.allclose(mwpm, 0.5)


def test_lin_ts_scaling_modifies_overrides():
    profile = _toy_profile()
    schedule_ir = _toy_schedule_ir()
    maps = schedule_to_weight_maps(schedule_ir, profile, profile.n_qubits)
    base_overrides = _base_overrides_from_maps(maps, profile.n_qubits)

    feats = feature_vector(schedule_ir)
    gate_vocab = {gate: idx for idx, gate in enumerate(sorted(feats["gate_hist"].keys()))}
    ctx_vec = context_vector(feats, gate_vocab)
    assert ctx_vec.ndim == 1 and ctx_vec.size == len(gate_vocab) + 1

    bandit = LinTSBandit(d=ctx_vec.size, prior_var=5.0, noise_var=0.5)
    theta = bandit.sample_theta()
    scale = float(np.clip(1.0 + 0.1 * float(np.dot(ctx_vec, theta)), 0.5, 2.0))

    overrides = _materialize_overrides(base_overrides, scale)
    assert set(overrides) >= {"llr_per_qubit", "mwpm_weights"}
    assert not np.allclose(overrides["llr_per_qubit"], base_overrides["llr"])
    assert not np.allclose(overrides["mwpm_weights"], base_overrides["mwpm"])

    # Posterior update should change subsequent samples
    bandit.update(ctx_vec, reward=1.0)
    theta2 = bandit.sample_theta()
    assert not np.allclose(theta2, theta)


@pytest.mark.parametrize("scale", [0.6, 1.5])
def test_materialize_overrides_monotone_effect(scale: float):
    base = {
        "llr": np.array([0.2, 0.0, -0.5], dtype=np.float32),
        "mwpm": np.full(3, 0.5, dtype=np.float32),
        "mwpf": {0: 1.0, 1: 0.4, 2: 1.8},
    }
    overrides = _materialize_overrides(base, scale)
    assert overrides["llr_per_qubit"].shape == (3,)
    assert overrides["mwpm_weights"].shape == (3,)
    if scale > 1.0:
        assert overrides["llr_per_qubit"][0] > base["llr"][0]
    else:
        assert overrides["llr_per_qubit"][0] < base["llr"][0]
