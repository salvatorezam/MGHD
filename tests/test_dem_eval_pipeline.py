"""Tests for the circuit-level DEM evaluation pipeline."""

import numpy as np
import pytest

from mghd.cli.train import (
    _build_dem_info,
    _teacher_labels_from_matching,
    compute_observable_correction,
    pack_dem_cluster,
)


class TestComputeObservableCorrection:
    """Tests for compute_observable_correction."""

    def test_basic_xor(self):
        """Single observable touching edges 0 and 2: flipping both should cancel."""
        L_obs = np.array([[1, 0, 1, 0]], dtype=np.uint8)  # (1, 4)
        # Flip edges 0 and 2 → correction = (1+1) %2 = 0 (no logical error)
        pred = np.array([1, 0, 1, 0], dtype=np.uint8)
        result = compute_observable_correction(L_obs, pred)
        assert result.shape == (1,)
        assert result[0] == 0

    def test_single_flip(self):
        """Flipping only one edge in a pair gives logical error."""
        L_obs = np.array([[1, 0, 1, 0]], dtype=np.uint8)
        pred = np.array([1, 0, 0, 0], dtype=np.uint8)
        result = compute_observable_correction(L_obs, pred)
        assert result[0] == 1

    def test_identity_L_obs(self):
        """Identity L_obs should pass-through edge predictions."""
        n = 5
        L_obs = np.eye(n, dtype=np.uint8)
        pred = np.array([1, 0, 1, 0, 1], dtype=np.uint8)
        result = compute_observable_correction(L_obs, pred)
        np.testing.assert_array_equal(result, pred)

    def test_multi_observable(self):
        """Two observables with different edge support."""
        L_obs = np.array([
            [1, 1, 0],  # obs 0 touches edges 0,1
            [0, 1, 1],  # obs 1 touches edges 1,2
        ], dtype=np.uint8)
        pred = np.array([1, 1, 0], dtype=np.uint8)
        result = compute_observable_correction(L_obs, pred)
        assert result[0] == 0  # (1+1)%2 = 0
        assert result[1] == 1  # (1+0)%2 = 1


class TestBuildDEMInfoHasLObs:
    """Ensure _build_dem_info returns all expected keys including L_obs."""

    @pytest.mark.parametrize("d", [3, 5])
    def test_keys_present(self, d):
        info = _build_dem_info(d, d, 0.005, "SI1000")
        required_keys = [
            "L_obs", "n_obs", "n_edges", "H_edge", "edge_probs",
            "matching", "sampler", "det_pair_to_edge",
            "det_coords", "edge_coords",
        ]
        for k in required_keys:
            assert k in info, f"Missing key: {k}"

    def test_L_obs_shape(self):
        info = _build_dem_info(3, 3, 0.005, "SI1000")
        L_obs = info["L_obs"]
        assert L_obs.ndim == 2
        assert L_obs.shape[0] == info["n_obs"]
        assert L_obs.shape[1] == info["n_edges"]
        assert L_obs.dtype == np.uint8


class TestTeacherLabelsRoundtrip:
    """Verify teacher labels decode correctly through the pipeline."""

    def test_teacher_ler_low_noise(self):
        """At very low noise, PyMatching teacher should have ~0 LER."""
        d = 3
        p = 0.001
        info = _build_dem_info(d, d, p, "SI1000")
        L_obs = info["L_obs"]
        n_edges = info["n_edges"]

        errors = 0
        n_shots = 200
        det_all, obs_all = info["sampler"].sample(
            shots=n_shots, separate_observables=True
        )
        for i in range(n_shots):
            det_bits = det_all[i].astype(np.uint8)
            true_obs = obs_all[i].astype(np.uint8)

            teacher_edges = _teacher_labels_from_matching(
                det_bits, info["matching"],
                info["det_pair_to_edge"], n_edges,
            )
            teacher_obs = compute_observable_correction(L_obs, teacher_edges)
            if np.any(teacher_obs != true_obs):
                errors += 1

        ler = errors / n_shots
        # At d=3, p=0.001, teacher LER should be very low
        assert ler < 0.15, f"Teacher LER too high: {ler:.3f}"


class TestPackDEMCluster:
    """Basic smoke test for pack_dem_cluster."""

    def test_pack_returns_packed_crop(self):
        info = _build_dem_info(3, 3, 0.005, "SI1000")
        n_edges = info["n_edges"]
        det_all, _ = info["sampler"].sample(shots=1, separate_observables=True)
        det_bits = det_all[0].astype(np.uint8)

        pack = pack_dem_cluster(
            H_edge=info["H_edge"],
            det_coords=info["det_coords"],
            edge_coords=info["edge_coords"],
            det_bits=det_bits,
            y_bits_edge=np.zeros(n_edges, dtype=np.uint8),
            d=3, rounds=3, p=0.005,
            N_max=512, E_max=2048, S_max=32,
        )
        # pack may be None if graph is trivially empty
        if pack is not None:
            assert pack.x_nodes is not None
            assert pack.node_mask is not None
            assert pack.meta is not None
            assert pack.meta.r >= 0  # nQ (edge-nodes)
            assert pack.meta.k >= 0  # nD (detectors)
