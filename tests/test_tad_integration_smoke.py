import importlib
from types import SimpleNamespace

import numpy as np

from mghd.decoders.mix import MixConfig, TeacherMix


def _dummy_code(n: int = 4):
    hx = np.zeros((1, n), dtype=np.uint8)
    hz = np.zeros((1, n), dtype=np.uint8)
    return SimpleNamespace(name="dummy", Hx=hx, Hz=hz, n=n)


def test_route_batch_passes_llr_overrides(monkeypatch):
    code = _dummy_code()
    mix = TeacherMix(code, code.Hx, code.Hz, mix_cfg=MixConfig(p_mwpf=0.0, p_lsd=1.0, p_mwpm=0.0))

    captured = {}

    def fake_decode(sx, sz, *, llr_overrides=None, erase_mask=None):
        captured["llr"] = llr_overrides
        ex = np.zeros((sx.shape[0], code.Hx.shape[1]), dtype=np.uint8)
        ez = np.zeros((sz.shape[0], code.Hz.shape[1]), dtype=np.uint8)
        return ex, ez

    monkeypatch.setattr(mix, "lsd", SimpleNamespace(decode_batch_xz=fake_decode))
    dets = np.zeros((1, 0), dtype=np.uint8)
    sx = np.zeros((1, code.Hx.shape[0]), dtype=np.uint8)
    sz = np.zeros((1, code.Hz.shape[0]), dtype=np.uint8)
    llr = np.linspace(-1.0, 1.0, code.Hx.shape[1], dtype=np.float32)

    mix.route_batch(
        dets,
        sx,
        sz,
        rng=np.random.default_rng(0),
        weight_overrides={"llr_per_qubit": llr},
    )

    assert "llr" in captured
    assert np.allclose(captured["llr"], llr)


def test_route_batch_passes_mwpm_weights(monkeypatch):
    code = _dummy_code()
    mix = TeacherMix(code, code.Hx, code.Hz, mix_cfg=MixConfig(p_mwpf=0.0, p_lsd=0.0, p_mwpm=1.0))

    weights_seen = {"x": None, "z": None}

    class StubMWPM:
        def __init__(self, key):
            self.key = key

        def decode_batch(self, syndromes, *, column_weights=None):
            weights_seen[self.key] = column_weights
            return np.zeros((syndromes.shape[0], code.Hx.shape[1]), dtype=np.uint8)

    mix.mwpm_x = StubMWPM("x")
    mix.mwpm_z = StubMWPM("z")

    dets = np.zeros((1, 0), dtype=np.uint8)
    sx = np.zeros((1, code.Hx.shape[0]), dtype=np.uint8)
    sz = np.zeros((1, code.Hz.shape[0]), dtype=np.uint8)
    weights = np.ones(code.Hx.shape[1], dtype=np.float32)

    mix.route_batch(
        dets,
        sx,
        sz,
        rng=np.random.default_rng(0),
        weight_overrides={"mwpm_weights": weights},
    )

    assert np.allclose(weights_seen["x"], weights)
    assert np.allclose(weights_seen["z"], weights)


def test_route_batch_passes_mwpf_scale(monkeypatch):
    code = _dummy_code()
    captured_scale = {}

    class StubMWPF:
        def __init__(self, *args, **kwargs):
            pass

        def decode_batch(self, dets, *, mwpf_scale=None):
            captured_scale["scale"] = mwpf_scale
            return {
                "fault_ids": np.zeros((1, 1), dtype=np.int32),
                "weights": np.zeros(1, dtype=np.float32),
            }

    monkeypatch.setattr("mghd.decoders.mix.MWPFTeacher", StubMWPF)

    mix = TeacherMix(code, code.Hx, code.Hz, mix_cfg=MixConfig(p_mwpf=1.0, p_lsd=0.0, p_mwpm=0.0))

    dets = np.zeros((1, 0), dtype=np.uint8)
    sx = np.zeros((1, code.Hx.shape[0]), dtype=np.uint8)
    sz = np.zeros((1, code.Hz.shape[0]), dtype=np.uint8)
    scale = {0: 1.5, 2: 0.7}

    mix.route_batch(
        dets,
        sx,
        sz,
        rng=np.random.default_rng(0),
        weight_overrides={"mwpf_scale": scale},
    )

    assert captured_scale["scale"] == scale


def test_weighting_modules_importable():
    mix = importlib.import_module("mghd.decoders.mix")
    lsd = importlib.import_module("mghd.decoders.lsd_teacher")
    mwpm = importlib.import_module("mghd.decoders.mwpm_fallback")
    assert hasattr(mix, "TeacherMix")
    assert hasattr(lsd, "LSDTeacher")
    assert hasattr(mwpm, "MWPMFallback")


def test_rl_bandit_update_smoke():
    from mghd.tad.rl.lin_ts import LinTSBandit

    bandit = LinTSBandit(d=5)
    x = np.arange(5.0)
    theta = bandit.sample_theta()
    bandit.update(x, 1.0)
    assert theta.shape == (5,)
