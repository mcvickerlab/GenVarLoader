"""End-to-end benchmarks: reconstructor via eager Dataset indexing, at the
regression seqlen (16384). Covers haplotype, annotated, variant, track, and the
tracks-only path that REGRESSIONS.md fingered."""

from __future__ import annotations

from tests.benchmarks._indices import batch_indices

SEQLEN = 16384
BATCH = 32


def _bench_indexing(benchmark, ds):
    r, s = batch_indices(ds.shape[0], ds.shape[1], BATCH)
    ds[r, s]  # warmup (JIT link, caches)
    result = benchmark(lambda: ds[r, s])
    assert result is not None


def test_e2e_haplotypes(benchmark, bench_dataset):
    ds = bench_dataset.with_seqs("haplotypes").with_len(SEQLEN)
    _bench_indexing(benchmark, ds)


def test_e2e_annotated(benchmark, bench_dataset):
    ds = bench_dataset.with_seqs("annotated").with_len(SEQLEN)
    _bench_indexing(benchmark, ds)


def test_e2e_variants(benchmark, bench_dataset):
    ds = bench_dataset.with_seqs("variants").with_len(SEQLEN)
    _bench_indexing(benchmark, ds)


def test_e2e_tracks(benchmark, bench_dataset):
    ds = bench_dataset.with_tracks("read-depth").with_len(SEQLEN)
    _bench_indexing(benchmark, ds)


def test_e2e_tracks_only(benchmark, bench_dataset):
    # The exact regression path: tracks only, no sequences.
    ds = bench_dataset.with_seqs(None).with_tracks("read-depth").with_len(SEQLEN)
    _bench_indexing(benchmark, ds)
