from types import SimpleNamespace

import numpy as np
import pytest

import mghd.decoders.dem_matching as dem_matching


def test_dem_matching_teacher_fallback(monkeypatch):
    class DummyMatching:
        def __init__(self):
            self.calls = []

        def decode_batch(self, dets):
            self.calls.append(dets.copy())
            return dets[:, :1]

    class MatchingFactory:
        def __init__(self):
            self.instance = DummyMatching()

        def from_detector_error_model(self, dem, correlated=False):
            if correlated:
                raise TypeError("correlated kw not supported")
            return self.instance

    monkeypatch.setattr(dem_matching, "_HAVE_PM", True)
    monkeypatch.setattr(dem_matching, "_PM_IMPORT_ERROR", None)
    monkeypatch.setattr(dem_matching, "_pm", SimpleNamespace(Matching=MatchingFactory()))

    dem = SimpleNamespace(num_detectors=2, num_observables=1)
    teacher = dem_matching.DEMMatchingTeacher(dem, correlated=True)

    dets = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    out = teacher.decode_batch(dets)
    assert out["which"] == "dem_matching"
    assert out["pred_obs"].shape == (2, 1)

    with pytest.raises(ValueError):
        teacher.decode_batch(np.ones(3, dtype=np.uint8))


def test_dem_matching_teacher_requires_pymatching(monkeypatch):
    monkeypatch.setattr(dem_matching, "_HAVE_PM", False)
    monkeypatch.setattr(dem_matching, "_PM_IMPORT_ERROR", RuntimeError("missing pymatching"))
    with pytest.raises(RuntimeError):
        dem_matching.DEMMatchingTeacher(SimpleNamespace())
