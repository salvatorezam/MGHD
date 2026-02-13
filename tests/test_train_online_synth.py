import json
from types import SimpleNamespace

import pytest

# Ensure torch is present before importing the training module
torch = pytest.importorskip("torch")
from mghd.cli import train as train_mod


def test_train_inprocess_online_synth(tmp_path, monkeypatch):
    # Force synthetic CUDA-Q path for fast online sampling
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
    ns.shots_per_epoch = 2
    ns.erasure_frac = 0.0
    ns.teacher_mix = "lsd=1.0,mwpf=0.0,mwpm=0.0"
    ns.online_rl = False
    ns.distance_curriculum = None
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
    ns.seed = 123
    # Provide data_root to avoid argparse consuming pytest's flags
    ns.data_root = ""
    # run
    out_path = train_mod.train_inprocess(ns)
    # Check outputs exist
    assert out_path.endswith("best.pt")
    run_dir = save_root
    # The actual auto dir is nested; search
    found = list(run_dir.rglob("train_log.json"))
    assert found, "expected a train_log.json in the save tree"
    # quick sanity: train_log has at least one entry
    log = json.loads(found[0].read_text())
    assert isinstance(log, list) and log, "train_log should have entries"
