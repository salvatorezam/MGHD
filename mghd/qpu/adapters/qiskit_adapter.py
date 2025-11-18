"""Minimal Qiskit adapter producing ScheduleIR tuples."""

from __future__ import annotations

from typing import Any, List, Tuple


def to_schedule_ir(qc: Any) -> List[Tuple[int, str, Tuple[int, ...], Any]]:
    """Return list of (time_slot, gate_name, qubits, duration)."""

    schedule: List[Tuple[int, str, Tuple[int, ...], Any]] = []
    if qc is None:
        return schedule
    data = getattr(qc, "data", None)
    if data is None:
        return schedule
    t = 0
    for instr, qargs, _ in data:
        gate_name = getattr(instr, "name", "gate")
        qubits = tuple(getattr(q, "index", getattr(q, "id", i)) for i, q in enumerate(qargs))
        duration = getattr(instr, "duration", None)
        schedule.append((t, gate_name, qubits, duration))
        t += 1
    return schedule


__all__ = ["to_schedule_ir"]
