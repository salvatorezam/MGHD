from importlib import import_module
import importlib.metadata as im

from packaging.version import Version


def test_versions():
    pm_version = Version(im.version("PyMatching"))
    assert pm_version >= Version("2.3.0")
    import_module("stim")
    try:
        import_module("cudaq")
    except Exception:
        pass
