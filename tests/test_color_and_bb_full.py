import importlib
import numpy as np
import pytest


def _css_ok(code):
    Hx, Hz = code.Hx, code.Hz
    A = (Hx.astype(np.uint8) @ Hz.astype(np.uint8).T) % 2
    assert not (A != 0).any()


def _get_or_skip(family: str, distance: int, **kw):
    cr = importlib.import_module("mghd.codes.registry")
    try:
        return cr.get_code(family, distance=distance, **kw)
    except RuntimeError as exc:
        pytest.skip(f"{family} d={distance} unavailable: {exc}")


def test_color_666_triangle_sizes_and_css():
    for d in [3, 5, 7]:
        code = _get_or_skip("color_666", d)
        _css_ok(code)
        n_expect = (3 * d * d + 1) // 4
        assert code.n == n_expect
        assert code.distance == d
        assert code.k == 1


def test_color_488_triangle_css():
    for d in [3, 5]:
        code = _get_or_skip("color_488", d)
        _css_ok(code)
        assert code.distance == d
        assert code.k == 1
        assert code.Hx.shape[0] > 0 and code.Hz.shape[0] > 0
        assert code.num_detectors == code.Hx.shape[0] + code.Hz.shape[0]


def test_gb_two_block_css_and_shape():
    cr = importlib.import_module("mghd.codes.registry")
    code = cr.get_code("gb", n=31, taps_a=(0, 1, 3), taps_b=(0, 2, 7))
    _css_ok(code)
    assert code.Hx.shape[1] == 2 * 31


def test_bb_bivariate_css_and_shape():
    cr = importlib.import_module("mghd.codes.registry")
    code = cr.get_code(
        "bb",
        n1=11,
        n2=13,
        taps_a_2d=((0, 0), (1, 0), (0, 1)),
        taps_b_2d=((0, 0), (2, 0), (0, 2)),
    )
    _css_ok(code)
    assert code.Hx.shape[1] == 2 * (11 * 13)
