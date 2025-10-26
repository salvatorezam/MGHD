import numpy as np
import pytest

pytest.importorskip("torch")

from mghd.codes.pcm_real import rotated_surface_pcm
from mghd.decoders.lsd.clustered import MGHDPrimaryClustered
from mghd.core.core import MGHDDecoderPublic, MGHDv2


@pytest.fixture
def tmp_ckpt(tmp_path):
    import torch

    model = MGHDv2()
    ckpt_path = tmp_path / "mghd_v2_dummy.pt"
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


def test_decoder_reports_invocation(tmp_ckpt):
    decoder = MGHDDecoderPublic(str(tmp_ckpt), device="cpu")
    Hx = rotated_surface_pcm(3, "X")
    Hz = rotated_surface_pcm(3, "Z")
    decoder.bind_code(Hx, Hz)

    clustered = MGHDPrimaryClustered(
        Hz,
        decoder,
        halo=1,
        thresh=0.5,
        temp=1.0,
        r_cap=10,
        batched=True,
        tier0_enable=False,
    )

    syndrome = np.zeros(Hz.shape[0], dtype=np.uint8)
    syndrome[0] = 1
    out = clustered.decode(syndrome, perf_only=False)

    assert out["mghd_invoked"] is not None
    assert isinstance(out["mghd_clusters"], int)
