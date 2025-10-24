from __future__ import annotations

from mghd.codes.qpu_profile import GateError, QPUProfile
from mghd.tad.context import context_vector
from mghd.tad.rl.lin_ts import LinTSBandit
from mghd.tad.weighting import feature_vector, schedule_to_weight_maps


def test_weight_maps_and_bandit_update():
    profile = QPUProfile(
        name="test",
        n_qubits=3,
        coupling=[(0, 1)],
        gate_error=GateError(
            p_1q={"rx90": 3e-4},
            p_2q={"cx": {"default": 1.2e-3}},
            p_meas=2e-3,
            t1_us={},
            t2_us={},
            idle_us=0.1,
            crosstalk_pairs={},
        ),
        meta={},
    )

    schedule = [
        (0, "rx90", (0,), None),
        (1, "cx", (0, 1), None),
        (2, "rx90", (1,), None),
    ]
    weights = schedule_to_weight_maps(schedule, profile, n_qubits=3)
    assert 0 in weights["w_qubit"] and 1 in weights["w_pair"]

    feat = feature_vector(schedule, window=4)
    gate_vocab = {"rx90": 0, "cx": 1}
    ctx = context_vector(feat, gate_vocab)
    assert ctx.shape[0] == len(gate_vocab) + 1

    bandit = LinTSBandit(d=ctx.shape[0])
    theta = bandit.sample_theta()
    assert theta.shape == ctx.shape
    bandit.update(ctx, reward=1.0)
    theta2 = bandit.sample_theta()
    assert theta2.shape == ctx.shape
