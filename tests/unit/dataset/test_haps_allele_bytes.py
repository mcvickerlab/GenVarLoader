"""Unit tests for Haps._allele_bytes_sum.

The helper returns the exact total bytes of REF or ALT allele payloads for the
variants selected by each (region, sample, ploid) entry, computed in O(|V|) by
differencing the RaggedAlleles offsets array (no payload read).
"""

import numpy as np
import pytest
import genvarloader as gvl


@pytest.fixture
def ds():
    return gvl.get_dummy_dataset()


def _expected_bytes(haps, idx, kind):
    """Ground-truth: sum byte lengths of allele strings per (instance, ploid).

    Uses get_variants_flat (the canonical decode path) to materialise a
    _FlatVariants record, converts to RaggedVariants via to_ragged(), then sums
    allele byte lengths via str_offsets on the opaque-string Ragged field.

    The field is an opaque-string Ragged of shape ``(b, p, ~v)`` where each
    element is a byte string.  ``field._rl.str_offsets`` gives cumulative byte
    boundaries across all variants; ``np.diff`` gives per-variant byte lengths.
    Segment-summing by ``field.offsets`` (variant-level groups) yields total
    bytes per (b, p) pair.
    """
    import numpy as np

    from genvarloader._dataset._flat_variants import get_variants_flat

    flat_v = get_variants_flat(haps, idx)
    ragv = flat_v.to_ragged()
    field = getattr(ragv, kind)  # opaque-string Ragged, shape (b, p, ~v)
    str_off = np.asarray(field._rl.str_offsets, np.int64)
    per_var_bytes = np.diff(str_off)  # byte length per allele string (flat)
    var_off = np.asarray(field.offsets, np.int64)  # variant-group boundaries (b*p+1,)
    csum = np.concatenate([[0], np.cumsum(per_var_bytes)])
    # Segment-sum per (b, p) group
    per_bp = csum[var_off[1:]] - csum[var_off[:-1]]
    return per_bp.astype(np.int64)


def test_allele_bytes_sum_matches_materialized_alt(ds):
    """Sum of returned int64s must equal len of every alt bytestring of every
    selected variant, summed per (region, sample, ploid)."""
    haps = ds._seqs
    idx = np.arange(np.prod(haps.genotypes.shape[:2]))
    got = haps._allele_bytes_sum(idx, "alt")
    expected = _expected_bytes(haps, idx, "alt")
    np.testing.assert_array_equal(got, expected)


def test_allele_bytes_sum_ref(ds):
    """Same invariant for ref."""
    haps = ds._seqs
    if "ref" not in haps.available_var_fields:
        pytest.skip("dummy dataset does not store ref alleles")
    idx = np.arange(np.prod(haps.genotypes.shape[:2]))
    got = haps._allele_bytes_sum(idx, "ref")
    expected = _expected_bytes(haps, idx, "ref")
    np.testing.assert_array_equal(got, expected)
