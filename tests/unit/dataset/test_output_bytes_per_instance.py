"""Per-mode exact-footprint tests.

Invariant: Dataset._output_bytes_per_instance(r, s) == nbytes of the actual
dataset[r, s] output, summed over arrays returned for that instance.
"""

import numpy as np
import pytest
import genvarloader as gvl
from seqpro.rag import Ragged

from genvarloader._dataset._rag_variants import RaggedVariants
from genvarloader._ragged import RaggedAnnotatedHaps

_REASON_242 = (
    "mcvickerlab/GenVarLoader#242 — intervals_to_tracks itv.start<query_start "
    "contract violation; both backends; fix deferred to separate PR"
)


def _materialized_nbytes_per_instance(ds, r_arr, s_arr):
    """Compute actual nbytes by indexing the dataset and measuring."""
    out = ds[r_arr, s_arr]
    # Normalize to tuple
    if not isinstance(out, tuple):
        out = (out,)
    # Each element contributes per-instance bytes.
    n_inst = len(r_arr)
    totals = np.zeros(n_inst, dtype=np.int64)
    for arr in out:
        if isinstance(arr, RaggedVariants):
            # Sum bytes across all fields per instance.
            ploidy = arr.shape[-2] if len(arr.shape) >= 3 else 1
            for fname in arr.fields:
                field = arr[fname]
                if getattr(field, "is_string", False):
                    # Opaque-string allele field (b, p, ~v): sum byte lengths of
                    # all allele strings per instance via str_offsets.
                    str_off = field._rl.str_offsets
                    per_var_bytes = np.diff(np.asarray(str_off, np.int64))
                    var_off = np.asarray(field.offsets, np.int64)  # (b*p+1,)
                    csum = np.concatenate([[0], np.cumsum(per_var_bytes)])
                    per_bp = csum[var_off[1:]] - csum[var_off[:-1]]
                    totals += per_bp.reshape(n_inst, ploidy).sum(-1)
                else:
                    # Numeric field (b, p, ~v): n_variants * itemsize.
                    lens = np.diff(np.asarray(field.offsets)).reshape(n_inst, -1)
                    totals += lens.sum(-1) * field.data.dtype.itemsize
        elif isinstance(arr, RaggedAnnotatedHaps):
            # Sum bytes over all three Ragged fields (haps S1, var_idxs int32, ref_coords int32).
            for ragged_field in (arr.haps, arr.var_idxs, arr.ref_coords):
                lens = np.diff(ragged_field.offsets).reshape(n_inst, -1)
                totals += lens.sum(-1) * ragged_field.data.dtype.itemsize
        elif isinstance(arr, Ragged):
            # Ragged.offsets is (n_inst * ... + 1,); reshape lens to (n_inst, -1)
            lens = np.diff(arr.offsets)
            lens = lens.reshape(n_inst, -1)
            totals += lens.sum(-1) * arr.data.dtype.itemsize
        elif isinstance(arr, np.ndarray):
            per = arr.itemsize * int(np.prod(arr.shape[1:]))
            totals += per
        else:
            raise AssertionError(f"unhandled array type {type(arr)}")
    return totals


def test_reference_mode_exact():
    ds = gvl.get_dummy_dataset().with_seqs("reference").with_tracks(False)
    r = np.arange(ds.full_shape[0])
    s = np.zeros(len(r), dtype=np.int64)
    got = ds._output_bytes_per_instance(r, s)
    expected = _materialized_nbytes_per_instance(ds, r, s)
    np.testing.assert_array_equal(got, expected)


def test_haplotypes_mode_exact():
    ds = (
        gvl.get_dummy_dataset()
        .with_seqs("haplotypes")
        .with_tracks(False)
        .with_settings(deterministic=True)
    )
    r = np.arange(ds.full_shape[0])
    s = np.zeros(len(r), dtype=np.int64)
    got = ds._output_bytes_per_instance(r, s)
    expected = _materialized_nbytes_per_instance(ds, r, s)
    np.testing.assert_array_equal(got, expected)


def test_annotated_mode_exact():
    ds = (
        gvl.get_dummy_dataset()
        .with_seqs("annotated")
        .with_tracks(False)
        .with_settings(deterministic=True)
    )
    r = np.arange(ds.full_shape[0])
    s = np.zeros(len(r), dtype=np.int64)
    got = ds._output_bytes_per_instance(r, s)
    expected = _materialized_nbytes_per_instance(ds, r, s)
    np.testing.assert_array_equal(got, expected)


def test_variants_default_var_fields_exact():
    """Default var_fields = ['alt', 'ilen', 'start']."""
    ds = gvl.get_dummy_dataset().with_seqs("variants").with_tracks(False)
    r = np.arange(ds.full_shape[0])
    s = np.zeros(len(r), dtype=np.int64)
    got = ds._output_bytes_per_instance(r, s)
    expected = _materialized_nbytes_per_instance(ds, r, s)
    np.testing.assert_array_equal(got, expected)


def test_variants_with_ref_exact():
    ds = gvl.get_dummy_dataset().with_seqs("variants").with_tracks(False)
    if "ref" not in ds._seqs.available_var_fields:
        pytest.skip("dummy dataset does not have ref allele")
    ds = ds.with_settings(var_fields=["alt", "ref", "ilen", "start"])
    r = np.arange(ds.full_shape[0])
    s = np.zeros(len(r), dtype=np.int64)
    got = ds._output_bytes_per_instance(r, s)
    expected = _materialized_nbytes_per_instance(ds, r, s)
    np.testing.assert_array_equal(got, expected)


def test_variants_with_info_column_exact():
    ds = gvl.get_dummy_dataset().with_seqs("variants").with_tracks(False)
    info_cols = [
        c
        for c in ds._seqs.available_var_fields
        if c not in {"alt", "ref", "ilen", "start", "dosage"}
    ]
    if not info_cols:
        pytest.skip("dummy dataset has no INFO columns")
    ds = ds.with_settings(var_fields=["alt", "start", "ilen", info_cols[0]])
    r = np.arange(ds.full_shape[0])
    s = np.zeros(len(r), dtype=np.int64)
    got = ds._output_bytes_per_instance(r, s)
    expected = _materialized_nbytes_per_instance(ds, r, s)
    np.testing.assert_array_equal(got, expected)


@pytest.mark.xfail(strict=False, reason=_REASON_242)
def test_haplotypes_plus_tracks_exact():
    ds = (
        gvl.get_dummy_dataset()
        .with_seqs("haplotypes")
        .with_settings(deterministic=True)
    )
    if not ds.active_tracks:
        pytest.skip("dummy dataset has no tracks")
    r = np.arange(ds.full_shape[0])
    s = np.zeros(len(r), dtype=np.int64)
    got = ds._output_bytes_per_instance(r, s)
    expected = _materialized_nbytes_per_instance(ds, r, s)
    np.testing.assert_array_equal(got, expected)


@pytest.mark.xfail(strict=False, reason=_REASON_242)
def test_reference_plus_tracks_exact():
    ds = gvl.get_dummy_dataset().with_seqs("reference")
    if not ds.active_tracks:
        pytest.skip("dummy dataset has no tracks")
    r = np.arange(ds.full_shape[0])
    s = np.zeros(len(r), dtype=np.int64)
    got = ds._output_bytes_per_instance(r, s)
    expected = _materialized_nbytes_per_instance(ds, r, s)
    np.testing.assert_array_equal(got, expected)
