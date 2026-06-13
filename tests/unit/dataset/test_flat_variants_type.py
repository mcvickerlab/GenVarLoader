from __future__ import annotations

import awkward as ak
import numpy as np
from seqpro.rag import Ragged

from genvarloader import RaggedVariants
from genvarloader._dataset._flat_variants import _FlatAlleles, _FlatVariants


def _alleles(rows, group_off, ploidy):
    """rows: list[bytes] per variant; group_off: per-(b*p)-row variant boundaries.
    shape is (b, p, None) where b = (len(group_off)-1) // ploidy.
    """
    data = np.frombuffer(b"".join(rows), np.uint8).copy()
    seq_off = np.concatenate([[0], np.cumsum([len(r) for r in rows])]).astype(np.int64)
    b = (len(group_off) - 1) // ploidy
    return _FlatAlleles(
        byte_data=data,
        seq_offsets=seq_off,
        var_offsets=np.asarray(group_off, np.int64),
        shape=(b, ploidy, None),
    )


def test_flat_variants_to_ragged_matches_handbuilt():
    # b=2, p=1: group_off has 3 entries (2*1+1)
    # row0 has 2 variants, row1 has 1 variant
    group_off = [0, 2, 3]
    ploidy = 1
    alt = _alleles([b"ACG", b"T", b"GG"], group_off, ploidy)
    ref = _alleles([b"A", b"CC", b"T"], group_off, ploidy)
    from genvarloader._flat import _Flat

    start = _Flat.from_offsets(
        np.array([1, 5, 9], np.int32), (2, None), np.asarray(group_off, np.int64)
    )
    fv = _FlatVariants(fields={"alt": alt, "start": start, "ref": ref})
    rv = fv.to_ragged()

    assert isinstance(rv, RaggedVariants)
    # _build_allele_layout with ploidy=1 produces shape (b, p, ~v, ~l) = (2, 1, ~v, ~l)
    assert ak.to_list(rv["alt"]) == [[[b"ACG", b"T"]], [[b"GG"]]]
    assert ak.to_list(rv["ref"]) == [[[b"A", b"CC"]], [[b"T"]]]
    assert ak.to_list(rv["start"]) == [[1, 5], [9]]
