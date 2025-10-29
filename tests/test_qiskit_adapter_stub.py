class Instr:
    def __init__(self, name, duration=None):
        self.name = name
        self.duration = duration


class Qubit:
    def __init__(self, index):
        self.index = index


class FakeQC:
    def __init__(self):
        self.data = [
            (Instr("h", duration=10), [Qubit(0)], None),
            (Instr("cx", duration=20), [Qubit(0), Qubit(1)], None),
        ]


def test_qiskit_adapter_to_schedule_ir():
    from mghd.qpu.adapters.qiskit_adapter import to_schedule_ir

    qc = FakeQC()
    ir = to_schedule_ir(qc)
    assert len(ir) == 2
    assert ir[0][1] == "h" and ir[1][1] == "cx"
    assert ir[0][2] == (0,) and ir[1][2] == (0, 1)

