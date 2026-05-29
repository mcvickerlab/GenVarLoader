"""Per-mode exact-footprint tests.

Invariant: Dataset._output_bytes_per_instance(r, s) == nbytes of the actual
dataset[r, s] output, summed over arrays returned for that instance.
"""
import numpy as np
import pytest
import genvarloader as gvl
from seqpro.rag import Ragged


def _materialized_nbytes_per_instance(ds, r_arr, s_arr):
    """Compute actual nbytes by indexing the dataset and measuring."""
    out = ds[r_arr, s_arr]
    # Normalize to tuple
    if not isinstance(out, tuple):
        out = (out,)
    # Each ndarray/Ragged contributes its data nbytes per instance. For Ragged,
    # we sum the per-instance data nbytes via the offsets.
    n_inst = len(r_arr)
    totals = np.zeros(n_inst, dtype=np.int64)
    for arr in out:
        if isinstance(arr, Ragged):
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
