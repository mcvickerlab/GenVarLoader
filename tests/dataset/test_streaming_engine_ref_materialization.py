"""#307 Finding 1: `build_engine` must materialize reference bytes ONLY for the
contigs some job in the plan actually touches, not for every store contig.

The engine indexes `contig_refs[job.contig_idx]`, so a contig no job references is
never read during reconstruct -- materializing it pulls the whole reference into
Python (and again into the Rust engine), breaking the cohort-independent
bounded-memory story on the reference axis for whole-genome references. Both the
SVAR1 and SVAR2 engines share this shape, so both are covered here.

The check spies on `backend._ref._contig_slice` (the per-contig reference read that
`build_engine` calls) and asserts it fires only for touched contig indices when the
job list is restricted to a strict subset of the store's contigs.
"""

from __future__ import annotations

import numpy as np

import genvarloader as gvl
from genvarloader._dataset._streaming import _Svar1Backend, _Svar2Backend


def _engine_jobs_for_contig(sds, keep_contig: int):
    """Real per-window engine jobs (same shape `_iter_batches` builds for both the
    SVAR1 and SVAR2 engines) restricted to windows on `keep_contig`. Contig index and
    region bounds come from the StreamingDataset's `_regions` map, exactly as both
    `build_engine` call sites derive them."""
    jobs = []
    for r_idx, s_idx in sds._plan():
        contig_idx = int(sds._regions[r_idx[0], 0])
        if contig_idx != keep_contig:
            continue
        jobs.append(
            (
                contig_idx,
                np.ascontiguousarray(sds._regions[r_idx, 1], np.uint32),
                np.ascontiguousarray(sds._regions[r_idx, 2], np.uint32),
                int(s_idx[0]),
                int(s_idx[-1]) + 1,
            )
        )
    return jobs


def _assert_only_touched_sliced(
    backend, jobs, keep_contig: int, monkeypatch, build_engine_extra=()
):
    # `build_engine_extra` carries the record-style backends' extra positional args
    # (`output_length`, `annotated` -- issue #277 Wave A); the SVAR2 backend's
    # `build_engine` takes none, so it passes an empty tuple.
    n_contigs = len(list(backend._contigs))
    assert n_contigs >= 2, "need >1 contig to prove the untouched one is skipped"

    ref = backend._ref
    ref_cls = type(ref)  # `Reference` is a slots dataclass -> patch the class method
    sliced: list[int] = []
    real = ref_cls._contig_slice

    def spy(self, i, *args, **kwargs):
        if self is ref:
            sliced.append(int(i))
        return real(self, i, *args, **kwargs)

    monkeypatch.setattr(ref_cls, "_contig_slice", spy)
    backend.build_engine(jobs, 4, *build_engine_extra)

    assert set(sliced) == {keep_contig}, (
        f"build_engine sliced contigs {sorted(set(sliced))}, expected only "
        f"{{{keep_contig}}} -- untouched contigs must not be materialized (#307)"
    )
    # And the untouched contig(s) were genuinely skipped.
    assert any(i != keep_contig for i in range(n_contigs))


def test_svar2_build_engine_materializes_only_touched_contigs(
    svar2_multicontig_fixture, monkeypatch
):
    fx = svar2_multicontig_fixture
    sds = gvl.StreamingDataset(
        fx.bed, reference=fx.reference_path, variants=fx.svar2_path
    ).with_seqs("haplotypes")
    backend = sds._backend
    assert isinstance(backend, _Svar2Backend)
    keep = 0
    jobs = _engine_jobs_for_contig(sds, keep)
    assert jobs, "fixture must yield at least one window on the kept contig"
    _assert_only_touched_sliced(backend, jobs, keep, monkeypatch)


def test_svar1_build_engine_materializes_only_touched_contigs(
    svar1_multicontig_fixture, monkeypatch
):
    fx = svar1_multicontig_fixture
    sds = gvl.StreamingDataset(
        fx.bed, reference=fx.reference_path, variants=fx.svar_path
    ).with_seqs("haplotypes")
    backend = sds._backend
    assert isinstance(backend, _Svar1Backend)
    keep = 0
    jobs = _engine_jobs_for_contig(sds, keep)
    assert jobs, "fixture must yield at least one window on the kept contig"
    # `_Svar1Backend.build_engine` takes Wave A args (`output_length=-1` ragged,
    # `annotated=False`); the ref-materialization behavior under test is independent
    # of them.
    _assert_only_touched_sliced(
        backend, jobs, keep, monkeypatch, build_engine_extra=(-1, False)
    )
