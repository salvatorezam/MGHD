import numpy as np

from mghd.codes.registry import get_code
from mghd.decoders.mix import TeacherMix, MixConfig
from mghd.decoders.mwpm_fallback import MWPMFallback


def test_mix_mwpm_fallback_outer_except(monkeypatch):
    code = get_code("surface", distance=3)
    mix = TeacherMix(code, code.Hx, code.Hz, mix_cfg=MixConfig(p_mwpf=0.0, p_lsd=0.0, p_mwpm=1.0), mwpm_graphlike_only=False)
    # Force MWPM branch by setting rng to return > thresholds
    rng = np.random.default_rng(123)
    dets = np.zeros((1, code.Hx.shape[0] + code.Hz.shape[0]), dtype=np.uint8)
    sx = np.zeros((1, code.Hx.shape[0]), dtype=np.uint8)
    sz = np.zeros((1, code.Hz.shape[0]), dtype=np.uint8)

    # Cause decode_batch to raise a non-graphlike error to hit outer except
    def boom(*args, **kwargs):
        raise ValueError("some runtime error")

    # Monkeypatch both X and Z to raise
    mix.mwpm_x.decode_batch = boom  # type: ignore[attr-defined]
    mix.mwpm_z.decode_batch = boom  # type: ignore[attr-defined]

    # Also patch GF(2) fallback to known outputs to assert shape
    def fake_gf2(self, syn):
        return np.ones((syn.shape[0], self._gf2_cols), dtype=np.uint8)

    monkeypatch.setattr(MWPMFallback, "_gf2_decode", fake_gf2)

    out = mix.route_batch(dets, sx, sz, rng=rng)
    assert out["which"] == "mwpm_fallback"
    assert out["cx"].shape == (1, code.Hx.shape[1])
    assert out["cz"].shape == (1, code.Hz.shape[1])
    assert "error" in out


def test_mix_disable_mwpm_reassigns_mass():
    code = get_code("surface", distance=3)
    mix = TeacherMix(code, code.Hx, code.Hz, mix_cfg=MixConfig(p_mwpf=0.0, p_lsd=0.2, p_mwpm=0.8))
    mix._disable_mwpm("test_reason")
    assert mix.mix.p_mwpm == 0.0
    # Remaining mass should be on LSD (since MWPF=0)
    assert abs(mix.mix.p_lsd - 1.0) < 1e-9
    assert mix.teachers.get("mwpm") is None
