"""Integer indexing into RaggedVariants allele fields is per-variant.

Regression test for #330 / ML4GLand/SeqPro#71: on seqpro <= 0.21.2, indexing a
string-under-axis Ragged concatenated the whole group into one bytes object,
discarding the per-variant boundaries that str_offsets already held. Numeric
fields on the same shared offsets stayed per-variant, so alt/ref silently
disagreed in length with start/ilen.

The multi-byte allele below is load-bearing. With SNPs only every allele is one
byte, the counts coincide, and this test passes on the buggy seqpro too --
gating nothing.
"""

from __future__ import annotations

import numpy as np
from seqpro.rag import Ragged

from genvarloader import RaggedVariants


def _build_rv() -> RaggedVariants:
    """(1, 2, ~v): hap 0 holds 2 variants (alt 'A', 'GG'), hap 1 holds 1 ('TC')."""
    alt_chars = np.frombuffer(b"AGGTC", dtype="S1")
    var_off = np.array([0, 2, 3], dtype=np.int64)  # hap -> variant index
    str_off = np.array([0, 1, 3, 5], dtype=np.int64)  # variant -> byte index

    alt = Ragged.from_offsets(
        alt_chars, (1, 2, None), var_off, str_offsets=str_off
    ).to_strings()
    start = Ragged.from_offsets(
        np.array([10, 20, 30], dtype=np.int32), (1, 2, None), var_off
    )
    ilen = Ragged.from_offsets(
        np.array([0, 1, 1], dtype=np.int32), (1, 2, None), var_off
    )
    return RaggedVariants(alt=alt, start=start, ilen=ilen)


def test_alt_indexes_per_variant_not_concatenated():
    """alt[0][hap] yields one entry per variant, not one concatenated blob."""
    rv = _build_rv()

    assert list(rv.alt[0][0]) == [b"A", b"GG"]
    assert list(rv.alt[0][1]) == [b"TC"]


def test_alt_count_matches_scalar_fields():
    """alt, start and ilen share one offsets object, so their per-hap counts agree."""
    rv = _build_rv()

    for hap, expected_n in ((0, 2), (1, 1)):
        assert len(rv.alt[0][hap]) == expected_n
        assert len(np.asarray(rv.start[0][hap])) == expected_n
        assert len(np.asarray(rv.ilen[0][hap])) == expected_n


def test_alt_aligns_elementwise_with_start():
    """The obvious consumer pattern -- zip(start, alt) -- is correct."""
    rv = _build_rv()

    pairs = list(zip(np.asarray(rv.start[0][0]).tolist(), rv.alt[0][0]))

    assert pairs == [(10, b"A"), (20, b"GG")]
