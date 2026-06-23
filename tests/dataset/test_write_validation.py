import numpy as np
import numpy.ma as ma
import polars as pl
import pytest

from genvarloader._dataset._write import _reject_unsupported_variants


def _index(alts, refs=None):
    """Build a minimal genoray-style index frame from a list of ALT lists."""
    n = len(alts)
    refs = refs if refs is not None else ["A"] * n
    return pl.DataFrame(
        {
            "CHROM": ["chr1"] * n,
            "POS": list(range(1, n + 1)),
            "REF": refs,
            "ALT": alts,
            "ILEN": [[0]] * n,
        },
        schema_overrides={"ALT": pl.List(pl.Utf8), "ILEN": pl.List(pl.Int32)},
    )


def test_clean_index_passes():
    # all bi-allelic SNPs: no raise
    _reject_unsupported_variants(_index([["T"], ["C"], ["G"]]), "VCF")


def test_symbolic_is_rejected():
    idx = _index([["T"], ["<DEL>"]])
    with pytest.raises(ValueError, match="symbolic"):
        _reject_unsupported_variants(idx, "VCF")


def test_breakend_is_rejected():
    idx = _index([["T"], ["C[chr1:600["]])
    with pytest.raises(ValueError, match="breakend"):
        _reject_unsupported_variants(idx, "PGEN")


def test_multiallelic_is_rejected():
    idx = _index([["T", "C"]])
    with pytest.raises(ValueError, match="multi-allelic"):
        _reject_unsupported_variants(idx, "SVAR")


def test_error_reports_source_and_counts():
    idx = _index([["<DEL>"], ["<INS>"], ["C[chr1:600["]])
    with pytest.raises(ValueError) as exc:
        _reject_unsupported_variants(idx, "SVAR")
    msg = str(exc.value)
    assert "SVAR" in msg
    assert "2 symbolic" in msg
    assert "1 breakend" in msg


# ---------------------------------------------------------------------------
# Unit test for the per-region-max masked reduction in _write._update_max_ends
# This directly mirrors the numpy logic at _write.py lines 1005-1026 so that
# the subtle empty-group / all-empty-region cases are covered without needing
# an svar fixture.
# ---------------------------------------------------------------------------


def _per_region_max(flat_data, offsets, n_groups, v_ends, chromEnds):
    """Mirror the production logic from _write.py (lines 1005-1026).

    flat_data : 1-D int32 array of variant indices (concatenated across all groups)
    offsets   : 1-D array of length n_groups+1 (group start/end positions in flat_data)
    n_groups  : total number of (region × sample × ploidy) groups;
                must equal n_regions × n_samples × n_ploidy when reshaped
    v_ends    : int array mapping variant index → chromEnd of that variant
    chromEnds : int array of length n_regions (fallback end per region)

    Returns: int array of length n_regions with the per-region max end.
    """
    dtype = np.int32
    _flat = np.asarray(flat_data, dtype=dtype)
    _off = np.asarray(offsets, dtype=np.intp)
    _n = n_groups
    _lens = _off[1:] - _off[:-1]
    _empty = _lens == 0
    _per_group = np.full(_n, np.iinfo(dtype).min, dtype=dtype)
    if len(_flat) > 0:
        _grp_ids = np.repeat(np.arange(_n, dtype=np.intp), _lens)
        np.maximum.at(_per_group, _grp_ids, _flat)
    # n_groups == n_regions * n_samples * n_ploidy; here n_samples=n_ploidy=1
    n_regions = len(chromEnds)
    v_idxs = ma.array(_per_group, mask=_empty).reshape(n_regions, 1, 1).max((1, 2))
    result = np.empty(n_regions, dtype=np.int64)
    result[~v_idxs.mask] = np.asarray(v_ends)[v_idxs.data[~v_idxs.mask]]
    result[v_idxs.mask] = np.asarray(chromEnds)[v_idxs.mask]
    return result


def test_per_region_max_mixed_empty_and_nonempty():
    """Region 0 has a variant (idx=2, end=300); region 1 has no variants → fallback."""
    v_ends = [100, 200, 300]  # end positions indexed by variant index
    chromEnds = [150, 250]  # fallback chromEnd per region

    # group layout: 2 regions × 1 sample × 1 ploidy = 2 groups
    # group 0 (region 0): variant idx 2
    # group 1 (region 1): empty
    flat_data = [2]
    offsets = [0, 1, 1]  # group 0 has 1 element, group 1 has 0

    result = _per_region_max(flat_data, offsets, 2, v_ends, chromEnds)
    assert result[0] == 300, "non-empty group: should use v_ends[2]=300"
    assert result[1] == 250, "all-empty region: should fall back to chromEnd=250"


def test_per_region_max_all_empty():
    """Both regions have zero variants → both fall back to chromEnd."""
    v_ends = [100, 200]
    chromEnds = [50, 75]
    flat_data = []
    offsets = [0, 0, 0]  # 2 groups, both empty

    result = _per_region_max(flat_data, offsets, 2, v_ends, chromEnds)
    assert result[0] == 50
    assert result[1] == 75


def test_per_region_max_all_nonempty():
    """Both groups have variants; result is max variant end per region."""
    v_ends = [100, 300, 200]
    chromEnds = [50, 50]
    # group 0: variants 0,1 (ends 100,300) → max=300
    # group 1: variant 2   (end 200)        → max=200
    flat_data = [0, 1, 2]
    offsets = [0, 2, 3]

    result = _per_region_max(flat_data, offsets, 2, v_ends, chromEnds)
    assert result[0] == 300
    assert result[1] == 200
