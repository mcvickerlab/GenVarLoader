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
    allele byte lengths via awkward's ak.num.

    The field type is ``b * p * var * bytes`` (four dimensions).  Awkward's
    ``bytes`` type is special: ``axis=-1`` does NOT reach into byte content;
    you must use explicit axis indices.  ``ak.num(field, axis=3)`` returns the
    byte length of each allele string (shape ``(b, p, ~v)``); summing on
    ``axis=2`` then gives total bytes per (b, p) instance.
    """
    import awkward as ak

    from genvarloader._dataset._flat_variants import get_variants_flat

    flat_v = get_variants_flat(haps, idx)
    ragv = flat_v.to_ragged()
    field = getattr(ragv, kind)  # type: b * p * ~v * bytes
    # axis=3: byte length of each allele string → shape (b, p, ~v)
    # axis=2: sum across variants per (b, p) → shape (b, p)
    return ak.sum(ak.num(field, axis=3), axis=2).to_numpy().ravel()


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
