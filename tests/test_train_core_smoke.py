import os
import subprocess
import sys


def test_train_core_smoke():
    # dry-run single small distance; use sampler placeholder if CUDA-Q not present
    cmd = [
        sys.executable,
        "-m",
        "mghd.cli.train_core",
        "--family",
        "surface",
        "--distances",
        "3",
        "--sampler",
        "stim",
        "--shots-per-batch",
        "4",
        "--batches",
        "1",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    cp = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert cp.returncode == 0, cp.stderr
    assert "teacher-usage" in cp.stdout
