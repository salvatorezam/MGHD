import pytest

pytest.importorskip("torch")

from tools.bench_clustered_sweep_surface import wilson_ci_upper


def test_wilson_upper_bound_hits_target():
    target = 1e-2
    max_shots = 10_000
    check_interval = 250
    shots = 0
    early = False

    while shots < max_shots:
        shots += 1
        if shots % check_interval == 0:
            upper = wilson_ci_upper(0, shots)
            if upper <= target:
                early = True
                break

    assert early, "Wilson upper bound never dropped below target"
    assert shots < max_shots
