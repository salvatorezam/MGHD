import numpy as np

from mghd.utils.metrics import LEResult, summary_line


def test_summary_line_includes_wilson_ci_text():
    per = np.array([0.1, 0.2], dtype=float)
    ler = LEResult(ler_per_logical=per, ler_mean=float(per.mean()), n_shots=1000)
    line = summary_line(
        family="surface",
        distance=3,
        batches=10,
        shots_per_batch=100,
        ler=ler,
        elapsed_s=1.0,
        teacher_usage={"mwpf": 0, "lsd": 10},
    )
    # Expect "LER=<mean>±<pm>" in scientific notation
    assert "LER=" in line and "±" in line
    assert "per-logical=[" in line

