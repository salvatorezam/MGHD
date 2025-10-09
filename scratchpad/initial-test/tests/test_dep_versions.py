import importlib.metadata as im

from packaging.version import Version


def test_versions():
    pm = Version(im.version("PyMatching"))
    assert pm >= Version("2.3.0"), pm
    __import__("stim")
    try:
        __import__("cudaq")
    except Exception:
        pass
