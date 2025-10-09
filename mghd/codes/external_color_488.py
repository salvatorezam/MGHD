"""External providers for triangular 4.8.8 color codes."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np


def _faces_to_matrix(faces: Iterable[Iterable[Any]], qubits: Iterable[Any]) -> np.ndarray:
    face_list = [tuple(face) for face in faces]
    qubit_list = list(qubits)
    index = {q: i for i, q in enumerate(qubit_list)}
    H = np.zeros((len(face_list), len(index)), dtype=np.uint8)
    for row, face in enumerate(face_list):
        for q in face:
            H[row, index[q]] = 1
    return H


def build_color_488_panqec(distance: int) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
    import importlib
    import inspect

    codes = importlib.import_module("panqec.codes")
    target_cls = None
    for name, obj in inspect.getmembers(codes):
        lowered = name.lower()
        if inspect.isclass(obj) and "488" in lowered and "color" in lowered:
            target_cls = obj
            break
    if target_cls is None:
        raise ImportError("PanQEC installed but Color 4.8.8 code class not found")

    params = inspect.signature(target_cls).parameters
    if "distance" in params:
        code = target_cls(distance=int(distance))
    elif "size" in params:
        code = target_cls(size=int(distance))
    else:
        code = target_cls(int(distance))
    if hasattr(code, "hx") and hasattr(code, "hz"):
        Hx = np.asarray(code.hx, dtype=np.uint8)
        Hz = np.asarray(code.hz, dtype=np.uint8)
    else:
        qubits = list(getattr(code, "qubits", getattr(code, "data_qubits", [])))
        if not qubits and hasattr(code, "get_qubits"):
            qubits = list(code.get_qubits())
        faces = list(getattr(code, "faces", getattr(code, "stabilizers", [])))
        if not faces and hasattr(code, "get_stabilizers"):
            faces = list(code.get_stabilizers())
        if not faces or not qubits:
            raise RuntimeError("PanQEC Color 4.8.8: could not extract qubits/faces")
        H = _faces_to_matrix(faces, qubits)
        Hx = H.copy()
        Hz = H.copy()
    n = Hx.shape[1]
    layout = {
        "provider": "panqec",
        "tiling": "4.8.8",
        "triangle_side": int(distance),
    }
    return Hx, Hz, n, layout


def build_color_488_pecos(distance: int) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
    import importlib

    pecos = importlib.import_module("pecos")
    Color488 = getattr(pecos.qeccs, "Color488")
    code = Color488(distance=distance)
    if hasattr(code, "hx") and hasattr(code, "hz"):
        Hx = np.asarray(code.hx, dtype=np.uint8)
        Hz = np.asarray(code.hz, dtype=np.uint8)
    else:
        qubits = list(getattr(code, "qubit_set", getattr(code, "data_qudit_set", [])))
        faces = list(getattr(code, "stabilizers", []))
        if not faces:
            raise RuntimeError("PECOS Color 4.8.8: stabilizers unavailable")
        H = _faces_to_matrix(faces, qubits)
        Hx = H.copy()
        Hz = H.copy()
    n = Hx.shape[1]
    layout = {
        "provider": "pecos",
        "tiling": "4.8.8",
        "triangle_side": int(distance),
    }
    return Hx, Hz, n, layout


def build_color_488(distance: int) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
    errors = []
    try:
        return build_color_488_panqec(distance)
    except Exception as exc:
        errors.append(str(exc))
    try:
        return build_color_488_pecos(distance)
    except Exception as exc:
        errors.append(str(exc))
        raise ImportError("No 4.8.8 color code provider available (install panqec or quantum-pecos)") from exc
