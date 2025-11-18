import numpy as np

from mghd.utils.metrics import (
    logical_error_rate,
    throughput,
    _wilson_interval,
    summary_line,
    LEResult,
)


def test_metrics_edges_and_summary_na():
    # None inputs path
    res = logical_error_rate(None, None)
    assert res.ler_mean is None and res.notes == "obs unavailable"
    # Shape mismatch path
    res2 = logical_error_rate(np.zeros((1, 2), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8))
    assert res2.ler_mean is None and res2.notes == "shape mismatch"
    # _wilson_interval n<=0
    lo, hi = _wilson_interval(0, 0)
    assert lo == 0.0 and hi == 0.0
    # throughput
    assert throughput(100, 0.0) > 0
    # summary_line NA branch
    line = summary_line(
        "surface", 3, 1, 10, LEResult(None, None, 0, "obs unavailable"), 1.0, {"lsd": 1}
    )
    assert "LER=NA" in line
