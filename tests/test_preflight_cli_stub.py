from types import SimpleNamespace

import subprocess

from mghd.cli import preflight_mghd as pf


def test_preflight_main_stubbed(monkeypatch, tmp_path):
    # Stub run() to avoid heavy subprocess calls; return LER lines
    class CP(subprocess.CompletedProcess):
        def __init__(self, args, code=0, out=""):
            super().__init__(args, code, out, "")

    def fake_run(cmd, log_path):
        return CP(cmd, 0, "LER_dem=0.001 LER_mix=0.002")

    monkeypatch.setattr(pf, "run", fake_run)

    # Run with --no-pytest and --skip-cudaq to exercise flow quickly
    args = [
        "--families",
        "surface",
        "--distances",
        "3",
        "--shots-per-batch",
        "2",
        "--batches",
        "1",
        "--no-pytest",
        "--skip-cudaq",
    ]
    rc = pf.main(args)
    assert rc in (0, 1)  # accept either, just ensure path executes

