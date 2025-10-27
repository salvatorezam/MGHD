import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def _has_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_module("qecsim"), reason="qecsim not installed")
def test_precompute_script_runs(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "color_cache").mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    project_root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", "mghd.cli.precompute_color_codes", "--max-d", "3", "--which", "666"]
    cp = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert cp.returncode == 0, cp.stderr
    cache_file = tmp_path / "color_cache" / "color_666_d3.npz"
    assert cache_file.exists()
    data = np.load(cache_file)
    Hx, Hz = data["Hx"], data["Hz"]
    assert Hx.shape[1] == Hz.shape[1]
    assert np.all((Hx.astype(np.uint8) @ Hz.astype(np.uint8).T) % 2 == 0)


def test_cli_multifamily_smoke(monkeypatch, tmp_path):
    # ensure PYTHONPATH includes project root when running from tmp
    env = os.environ.copy()
    project_root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable,
        "-m",
        "mghd.tools.teacher_eval",
        "--families",
        "surface,steane,repetition",
        "--distances",
        "3",
        "--sampler",
        "stim",
        "--shots-per-batch",
        "4",
        "--batches",
        "1",
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert cp.returncode == 0, cp.stderr
    assert "teacher-usage" in cp.stdout
