"""The double_buffered slot-fit invariant: the per-instance byte estimate plus the
schema-derived slot overhead must upper-bound the real serialized payload for every
chunk. This is the invariant #315 violated; it must hold across record types and
storage backends."""

import numpy as np
import pytest
import seqpro as sp

import genvarloader as gvl
from genvarloader._shm_layout import HEADER_RESERVED, write_chunk
from genvarloader._slot_overhead import slot_overhead_bytes


def _views(ds):
    DNA = sp.alphabets.DNA
    for ref, alt in [("window", "window"), ("window", "allele"), ("allele", "allele")]:
        if ref == "allele" and getattr(ds._seqs.variants, "ref", None) is None:
            continue
        for uu in (True, False):
            for L in (8, 128):
                opt = gvl.VarWindowOpt(
                    flank_length=L,
                    token_alphabet=DNA,
                    unknown_token=len(DNA),
                    ref=ref,
                    alt=alt,
                )
                yield (
                    ds.with_tracks(False)
                    .with_output_format("flat")
                    .with_seqs("variant-windows", opt)
                    .with_settings(unphased_union=uu, jitter=0)
                )


def _assert_upper_bound(view):
    R, S = view.shape[:2]
    rr, ss = np.meshgrid(np.arange(R), np.arange(S), indexing="ij")
    r, s = rr.reshape(-1), ss.reshape(-1)
    chunk = view[r, s]
    arrays = list(chunk) if isinstance(chunk, tuple) else [chunk]
    buf = memoryview(bytearray(64 * 1024 * 1024))
    real = write_chunk(buf, arrays, n_instances=len(r)) - HEADER_RESERVED
    est = int(
        np.asarray(view._output_bytes_per_instance(r, s, include_offsets=True)).sum()
    )
    assert est + slot_overhead_bytes(view) >= real, (
        f"slot under-sized: est={est} overhead={slot_overhead_bytes(view)} real={real}"
    )


def test_slot_fit_dummy_backend():
    for view in _views(gvl.get_dummy_dataset()):
        _assert_upper_bound(view)


@pytest.mark.parametrize("backend", ["vcf", "pgen", "svar"])
def test_slot_fit_file_backends(backend, request, reference):
    path = request.getfixturevalue(f"phased_{backend}_gvl")
    ds = gvl.Dataset.open(path, reference=reference)
    for view in _views(ds):
        _assert_upper_bound(view)
