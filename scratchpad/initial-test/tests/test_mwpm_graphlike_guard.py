from types import SimpleNamespace

import numpy as np
import pytest

from mghd.utils.graphlike import is_graphlike
from teachers.mwpm_fallback import MWPMFallback, MwpmNotGraphlike


def test_is_graphlike_detection():
    H = np.array([[1, 1, 0], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    assert not is_graphlike(H)


def test_mwpm_init_raises_on_non_graphlike():
    H = np.array([[1, 1, 0], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    code = SimpleNamespace(Hx=H, Hz=H)
    with pytest.raises(MwpmNotGraphlike):
        MWPMFallback(code, basis="x", require_graphlike=True)


def test_teacher_mix_disables_non_graphlike_mwpm(monkeypatch):
    from teachers import mix as mix_mod

    H = np.array([[1, 1, 0], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

    class DummyMWPF:
        def __init__(self, *args, **kwargs):
            pass

        def decode_batch(self, dets, *, mwpf_scale=None):
            return {"fault_ids": np.zeros((dets.shape[0], 1), dtype=np.int32)}

    class DummyLSD:
        def __init__(self, *args, **kwargs):
            pass

        def decode_batch_xz(self, sx, sz, llr_overrides=None, erase_mask=None):
            cols = H.shape[1]
            ex = np.zeros((sx.shape[0], cols), dtype=np.uint8)
            ez = np.zeros((sz.shape[0], cols), dtype=np.uint8)
            return ex, ez

    monkeypatch.setattr(mix_mod, "MWPFTeacher", DummyMWPF, raising=False)
    monkeypatch.setattr(mix_mod, "LSDTeacher", DummyLSD, raising=False)
    monkeypatch.setattr(mix_mod, "ErasureSurfaceMLTeacher", None, raising=False)
    monkeypatch.setattr(mix_mod, "ErasureQLDPCPeelingTeacher", None, raising=False)

    code = SimpleNamespace(name="dummy", Hx=H, Hz=H)

    teacher_mix = mix_mod.TeacherMix(
        code,
        H,
        H,
        mix_cfg=mix_mod.MixConfig(p_mwpf=0.3, p_lsd=0.4, p_mwpm=0.3),
    )

    assert not teacher_mix._mwpm_enabled
    assert teacher_mix.mwpm_x is None
    assert teacher_mix.mix.p_mwpm == 0.0
    assert teacher_mix.teachers.get("mwpm") is None
