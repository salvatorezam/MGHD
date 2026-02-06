import numpy as np

from mghd.codes.registry import get_code
from mghd.decoders.mix import TeacherMix, MixConfig


def test_mix_routes_lsd_and_erasure_surface():
    code = get_code("surface", distance=3)
    mix = TeacherMix(code, code.Hx, code.Hz, mix_cfg=MixConfig(p_mwpf=0.0, p_lsd=1.0, p_mwpm=0.0))
    B = 2
    dets = np.zeros((B, code.Hx.shape[0] + code.Hz.shape[0]), dtype=np.uint8)
    sx = np.zeros((B, code.Hx.shape[0]), dtype=np.uint8)
    sz = np.zeros((B, code.Hz.shape[0]), dtype=np.uint8)
    out = mix.route_batch(dets, sx, sz)
    assert out.get("which") in {"lsd", "mwpm", "mwpm_fallback"}
    # Force erasure route
    mask = np.zeros((B, code.Hx.shape[1]), dtype=np.uint8)
    mask[0, 0] = 1
    out2 = mix.route_batch(dets, sx, sz, erase_data_mask=mask)
    assert out2.get("which") in {"erasure_surface_ml", "erasure_peeling"}


def test_mix_routes_with_weight_overrides():
    code = get_code("surface", distance=3)
    mix = TeacherMix(code, code.Hx, code.Hz, mix_cfg=MixConfig(p_mwpf=0.0, p_lsd=1.0, p_mwpm=0.0))
    dets = np.zeros((1, code.Hx.shape[0] + code.Hz.shape[0]), dtype=np.uint8)
    sx = np.zeros((1, code.Hx.shape[0]), dtype=np.uint8)
    sz = np.zeros((1, code.Hz.shape[0]), dtype=np.uint8)
    weight_overrides = {
        "llr_per_qubit": np.zeros(code.Hx.shape[1], dtype=np.float32),
        "mwpf_scale": {0: 1.0},
        "mwpm_weights": (np.ones(code.Hx.shape[1]), np.ones(code.Hz.shape[1])),
    }
    out = mix.route_batch(dets, sx, sz, weight_overrides=weight_overrides)
    assert out.get("which") in {"lsd", "mwpm", "mwpm_fallback"}
