import json
from pathlib import Path

import pytest

from mghd.decoders import dem_utils


def test_hash_profile_is_stable():
    profile = {"gate_error": {"a": 0.1}, "other": [1, 2, 3]}
    digest_first = dem_utils._hash_profile(profile)
    digest_second = dem_utils._hash_profile(json.loads(json.dumps(profile)))
    assert digest_first == digest_second
    assert len(digest_first) == 10


def test_dem_cache_path_creates_directory(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    path = dem_utils.dem_cache_path(
        str(cache_dir),
        family="surface",
        distance=3,
        rounds=2,
        profile={"gate_error": {"after_clifford_depolarization": 0.001}},
    )
    assert cache_dir.exists()
    assert path.endswith(".dem")


def test_build_hgp_dem_not_implemented():
    with pytest.raises(NotImplementedError):
        dem_utils.build_hgp_dem()
