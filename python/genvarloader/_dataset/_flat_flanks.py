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


def compute_flank_tokens(
    reference,
    v_contigs: NDArray[np.integer],  # (n_var,) contig id per variant
    starts: NDArray[np.integer],  # (n_var,)
    ilens: NDArray[np.integer],  # (n_var,)
    flank_len: int,
    lut: NDArray,
    row_offsets: NDArray[np.int64],  # (b*p + 1,) per-(instance,ploid) variant offsets
) -> tuple[NDArray, NDArray[np.int64]]:
    """Ride-along flank tokens: ``[flank5 | flank3]`` (2*flank_len tokens) per
    variant. Returns ``(token_data, offsets)`` where ``token_data`` is flat
    ``(n_var * 2 * flank_len,)`` and ``offsets == row_offsets`` (one row per variant,
    fixed inner dim 2*flank_len)."""
    starts = np.asarray(starts, np.int32)
    ilens = np.asarray(ilens, np.int32)
    ends = starts - np.minimum(ilens, 0) + 1
    f5 = reference.fetch(v_contigs, starts - flank_len, starts).data.view(np.uint8)
    f3 = reference.fetch(v_contigs, ends, ends + flank_len).data.view(np.uint8)
    n = starts.shape[0]
    f5 = f5.reshape(n, flank_len)
    f3 = f3.reshape(n, flank_len)
    flank_bytes = np.concatenate([f5, f3], axis=1)  # (n, 2L)
    tokens = lut[flank_bytes]  # vectorized 256-LUT gather -> lut.dtype
    return tokens.reshape(-1), np.asarray(row_offsets, np.int64)
