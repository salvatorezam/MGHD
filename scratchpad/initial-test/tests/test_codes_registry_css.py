import numpy as np
import pytest
import importlib


def _css_ok(code):
    Hx, Hz = code.Hx, code.Hz
    A = (Hx.astype(np.uint8) @ Hz.astype(np.uint8).T) % 2
    assert not (A != 0).any()


def test_surface_small_commutes():
    cr = importlib.import_module("codes_registry")
    for d in [3, 5, 7]:
        code = cr.get_code("surface", distance=d)
        _css_ok(code)
        assert code.n > 0
        assert code.num_detectors == code.Hx.shape[0] + code.Hz.shape[0]


def test_repetition_commutes():
    cr = importlib.import_module("codes_registry")
    code = cr.get_code("repetition", distance=5)
    _css_ok(code)
    assert code.n == 5


def test_steane_hardcoded():
    cr = importlib.import_module("codes_registry")
    code = cr.get_code("steane")
    _css_ok(code)
    assert code.n == 7 and code.distance == 3


def test_hgp_builder_css():
    cr = importlib.import_module("codes_registry")
    H1 = np.array([[1, 1, 0, 0], [0, 1, 1, 0]], dtype=np.uint8)
    H2 = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    code = cr.get_code("hgp", H1=H1, H2=H2)
    _css_ok(code)
    assert code.Hx.shape[1] == code.Hz.shape[1]


def test_color_toy_commutes():
    cr = importlib.import_module("codes_registry")
    code = cr.get_code("color", distance=3)
    _css_ok(code)
    assert code.n >= 3
