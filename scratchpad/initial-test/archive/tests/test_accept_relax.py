import pytest

pytest.importorskip("torch")

from tools.bench_clustered_sweep_surface import RelaxController, _parse_auto_relax


def test_relax_nullity_then_size():
    ctrl = RelaxController(min_nullity=1, min_size=8, order=_parse_auto_relax("nullity,size,none"))
    assert ctrl.relax() is True
    assert ctrl.min_nullity == 0
    # Size relaxations continue until zero; ensure the controller eventually reports no-relax.
    while ctrl.relax():
        if ctrl.min_nullity is not None:
            assert ctrl.min_nullity >= 0
        if ctrl.min_size is not None:
            assert ctrl.min_size >= 0
        if ctrl.min_nullity == 0 and ctrl.min_size == 0:
            break
    assert ctrl.relax() is False
    assert ctrl.history[-1] == "no-relax"


def test_relax_none_only():
    ctrl = RelaxController(min_nullity=None, min_size=None, order=_parse_auto_relax("none"))
    assert ctrl.relax() is False
    assert ctrl.history[-1] == "no-relax"
