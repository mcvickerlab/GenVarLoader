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


def test_slot_fit_svar2_backend(phased_svar2_gvl, svar2_slot_reference):
    """SVAR2 datasets open as Svar2Haps via the released reconstruct path -- the
    coverage gap that let #315 through. The estimate must upper-bound the real
    serialized payload here too.

    Opens with `svar2_slot_reference`, not the `reference` fixture used by
    test_slot_fit_file_backends: the SVAR2 store is built over its own tiny
    reference (see tests/conftest.py::_svar2_slot_src), not the shared
    synthetic_case one -- the two are structurally different reconstructors
    (Svar2Haps vs. Haps), so there is no reason to force a shared reference.
    """
    from genvarloader._dataset._svar2_haps import Svar2Haps

    ds = gvl.Dataset.open(phased_svar2_gvl, reference=svar2_slot_reference)
    assert isinstance(ds._seqs, Svar2Haps), "fixture must open as Svar2Haps"
    for view in _views(ds):
        _assert_upper_bound(view)

    # SCOPE ADD-ON: the sibling "variants" (non-window) output branch reads
    # the same permanently-empty Svar2Haps.genotypes/.variants placeholders
    # (n_vars_total via self.n_variants(), alt bytes via _allele_bytes_sum),
    # so it is presumed similarly under-counting (see "What Task 3 must
    # change" in the Phase-0 findings doc). Locks the sibling defect so
    # Task 3's shared fix must cover it too.
    for uu in (True, False):
        variants_view = (
            ds.with_tracks(False)
            .with_output_format("flat")
            .with_seqs("variants")
            .with_settings(unphased_union=uu, jitter=0)
        )
        _assert_upper_bound(variants_view)
