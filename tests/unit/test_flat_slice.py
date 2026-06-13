"""Instance-axis slicing parity for the flat containers (no torch needed)."""
from __future__ import annotations

import numpy as np
import pytest

import genvarloader as gvl


def _rag_eq(a, b):
    """Element-equality for seqpro Ragged."""
    np.testing.assert_array_equal(np.asarray(a.data), np.asarray(b.data))
    np.testing.assert_array_equal(np.asarray(a.offsets), np.asarray(b.offsets))


def _ak_eq(a, b):
    import awkward as ak

    assert ak.to_list(a) == ak.to_list(b)


@pytest.mark.parametrize("seq_kind", ["reference", "haplotypes", "annotated", "variants"])
@pytest.mark.parametrize("sl", [slice(0, 1), slice(1, 3), slice(2, 4), slice(0, 0)])
def test_flat_slice_matches_direct_index(seq_kind, sl):
    ds = gvl.get_dummy_dataset().with_seqs(seq_kind).with_tracks(False)
    if seq_kind in ("haplotypes", "annotated"):
        ds = ds.with_settings(deterministic=True)

    # A flat batch over the first 4 instances (region 0, samples 0..3 if available,
    # else fall back to a 1-D flat range across the (region, sample) grid).
    n = min(4, int(np.prod(ds.shape)))
    flat_idx = np.arange(n, dtype=np.int64)
    r_idx, s_idx = np.unravel_index(flat_idx, ds.shape)

    flat = ds.with_output_format("flat")[r_idx, s_idx]

    sliced = flat[sl]
    # Direct-index the SAME instance sub-range in ragged mode for the ground truth.
    sub = flat_idx[sl]
    rr, ss = np.unravel_index(sub, ds.shape)
    ragged_sub = ds[rr, ss] if sub.size else None

    # `sliced.to_ragged()` must equal the directly-indexed ragged sub-range.
    got = sliced.to_ragged()
    if seq_kind == "variants":
        if sub.size == 0:
            import awkward as ak

            assert len(got) == 0
        else:
            _ak_eq(got, ragged_sub)
    elif seq_kind == "annotated":
        if sub.size == 0:
            assert got.haps.shape[0] == 0
        else:
            _rag_eq(got.haps, ragged_sub.haps)
            _rag_eq(got.var_idxs, ragged_sub.var_idxs)
            _rag_eq(got.ref_coords, ragged_sub.ref_coords)
    else:
        if sub.size:
            _rag_eq(got, ragged_sub)
    # Length always matches the slice width.
    assert got_len(got) == sub.size


def got_len(x):
    import awkward as ak
    from seqpro.rag import Ragged

    if isinstance(x, ak.Array):
        return len(x)
    if isinstance(x, Ragged):
        return x.shape[0]
    return x.haps.shape[0]  # RaggedAnnotatedHaps
