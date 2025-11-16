from mghd.decoders.dem_metric import (
    manhattan_3d,
    cc_embed_4d,
    cc_distance_4d,
    distance_to_boundary_min_x_or_y,
)


def test_dem_metric_basic_properties():
    a = (0, 0, 0)
    b = (1, 2, 3)
    # Manhattan in 3D
    assert manhattan_3d(a, b) == 6
    assert manhattan_3d(a, a) == 0
    # 4D embedding distance is symmetric and zero on identical points
    ea = cc_embed_4d(*a)
    eb = cc_embed_4d(*b)
    assert len(ea) == len(eb) == 4
    assert cc_distance_4d(a, b) == cc_distance_4d(b, a)
    assert cc_distance_4d(a, a) == 0.0
    # Triangle-ish sanity (not strictly a metric proof, just a smoke check)
    c = (2, 0, 1)
    assert cc_distance_4d(a, c) + cc_distance_4d(c, b) >= cc_distance_4d(a, b) - 1e-9


def test_distance_to_boundary_min_x_or_y_behavior():
    assert distance_to_boundary_min_x_or_y(0, 2, 5) == 0
    assert distance_to_boundary_min_x_or_y(2, 2, 5) == 2
