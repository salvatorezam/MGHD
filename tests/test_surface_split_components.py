import numpy as np

from mghd.qpu.adapters.surface_sampler import split_components_for_side


def test_split_components_returns_expected_keys():
    # Build simple Hx/Hz with a single active check in each to form one component
    Hx = np.array([[1, 0, 1]], dtype=np.uint8)
    Hz = np.array([[0, 1, 1]], dtype=np.uint8)
    synX = np.array([1], dtype=np.uint8)
    synZ = np.array([1], dtype=np.uint8)
    coords_q = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.int32)
    coords_c = np.array([[0.5, -0.5], [0.5, 0.5]], dtype=np.float32)  # Z first, then X
    comps_z = split_components_for_side(
        side="Z", Hx=Hx, Hz=Hz, synZ=synZ, synX=synX, coords_q=coords_q, coords_c=coords_c
    )
    comps_x = split_components_for_side(
        side="X", Hx=Hx, Hz=Hz, synZ=synZ, synX=synX, coords_q=coords_q, coords_c=coords_c
    )
    for comps in (comps_z, comps_x):
        assert isinstance(comps, list)
        if comps:
            c = comps[0]
            for key in (
                "H_sub",
                "xy_qubit",
                "xy_check",
                "synd_bits",
                "bbox_xywh",
                "k",
                "r",
                "kappa_stats",
                "qubit_indices",
                "check_indices",
            ):
                assert key in c
