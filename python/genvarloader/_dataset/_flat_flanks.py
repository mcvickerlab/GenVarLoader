"""Reference flank fetch + byte->int tokenization (sub-project C).

Produces flat token buffers from already-gathered variant fields + the reference
genome. No awkward on the hot path.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def build_token_lut(alphabet: bytes, unknown_token: int) -> tuple[NDArray, np.dtype]:
    """Build a 256-entry byte->token lookup table (seqpro-style).

    Every byte value in ``alphabet`` maps to its position; every other byte
    (including ``N`` and padded out-of-bounds positions) maps to ``unknown_token``.
    """
    max_token = max(len(alphabet) - 1, unknown_token)
    dtype = np.uint8 if max_token <= np.iinfo(np.uint8).max else np.int32
    lut = np.full(256, unknown_token, dtype=dtype)
    for i, b in enumerate(alphabet):
        lut[b] = i
    return lut, np.dtype(dtype)
