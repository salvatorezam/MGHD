from __future__ import annotations

import scipy.sparse as sp
import stim


def dem_to_css_pcm(dem: stim.DetectorErrorModel) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Placeholder DEM→CSS converter.

    The full Stim DEM parsing is non-trivial; until a robust converter is
    integrated we raise NotImplementedError so callers can fall back to other
    builders (e.g., panqec).
    """
    raise NotImplementedError("DEM→CSS parity-check conversion not yet implemented")
