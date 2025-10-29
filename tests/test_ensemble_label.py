import numpy as np

from mghd.decoders.ensemble import get_teacher_label


class FakePF:
    def __init__(self, bits, w):
        self.bits = bits
        self.w = w

    def decode(self, H_sub, synd_bits, side, dem_meta=None):
        return self.bits, self.w


class FakePM(FakePF):
    pass


def test_get_teacher_label_prefers_valid_and_lower_weight():
    # H_sub with two checks on three qubits
    H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    s = np.array([1, 0], dtype=np.uint8)
    # PF: valid correction with weight 2 -> H @ [0,1,1] = [1,0]
    bits_pf = np.array([0, 1, 1], dtype=np.uint8)
    # PM: valid correction with weight 1 (lower) -> H @ [1,0,0] = [1,0]
    bits_pm = np.array([1, 0, 0], dtype=np.uint8)
    out = get_teacher_label(
        H_sub=H,
        synd_bits=s,
        side="Z",
        mwpf_ctx=FakePF(bits_pf, 2),
        mwpm_ctx=FakePM(bits_pm, 1),
        local_ml_bits=bits_pm,
    )
    assert out.teacher in {"mwpf", "mwpm"}
    assert out.valid is True
    assert out.teacher == "mwpm" and out.weight == 1 and out.matched_local_ml


def test_get_teacher_label_invalid_fallback_and_match_flag():
    H = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    s = np.array([1, 0], dtype=np.uint8)
    # PF: invalid correction (parity mismatch)
    bits_pf = np.array([0, 0], dtype=np.uint8)
    # PM: valid
    bits_pm = np.array([1, 0], dtype=np.uint8)
    out = get_teacher_label(
        H_sub=H,
        synd_bits=s,
        side="X",
        mwpf_ctx=FakePF(bits_pf, 5),
        mwpm_ctx=FakePM(bits_pm, 7),
        local_ml_bits=np.array([1, 0], dtype=np.uint8),
    )
    assert out.teacher == "mwpm" and out.valid
