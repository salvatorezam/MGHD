import subprocess
from types import SimpleNamespace
import json

import pytest

torch = pytest.importorskip("torch")

from mghd.cli import train as train_mod


def test_train_post_eval_writes_report(tmp_path, monkeypatch):
    # Stub teacher_eval subprocess
    class CP:
        def __init__(self):
            self.stdout = "LER=1.0e-02 teacher-usage={'lsd': 10}"
            self.stderr = ""

    def fake_run(cmd, capture_output=True, text=True, env=None):
        return CP()

    monkeypatch.setattr(subprocess, "run", fake_run)

    # Use synthetic online run for speed
    monkeypatch.setenv("MGHD_SYNTHETIC", "1")

    save_root = tmp_path / "runs"
    ns = SimpleNamespace()
    ns.online = True
    ns.family = "surface"
    ns.distance = 3
    ns.p = 0.01
    ns.p_curriculum = None
    ns.epochs_per_p = 1
    ns.qpu_profile = None
    ns.context_source = "none"
    ns.shots_per_epoch = 1
    ns.erasure_frac = 0.0
    ns.teacher_mix = "lsd=1.0,mwpf=0.0,mwpm=0.0"
    ns.online_rl = False
    ns.profile = "S"
    ns.ema = 0.999
    ns.lr = 1e-4
    ns.wd = 1e-6
    ns.epochs = 1
    ns.batch = 1
    ns.parity_lambda = 0.0
    ns.projection_aware = 0
    ns.label_smoothing = 0.0
    ns.noise_injection = 0.0
    ns.grad_clip = 1.0
    ns.hparams = None
    ns.save = None
    ns.save_root = str(save_root)
    ns.save_auto = True
    ns.seed = 42
    ns.data_root = ""
    ns.post_eval = True
    ns.post_eval_sampler = "stim"
    ns.post_eval_shots_per_batch = 1
    ns.post_eval_batches = 1

    out_path = train_mod.train_inprocess(ns)
    # teacher_eval.txt should exist under the auto run directory
    report_files = list(save_root.rglob("teacher_eval.txt"))
    assert report_files, "Expected teacher_eval.txt to be written"
    txt = report_files[0].read_text()
    assert "LER=" in txt

