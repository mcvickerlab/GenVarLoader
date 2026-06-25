"""End-to-end benchmarks: reconstructor via eager Dataset indexing, at the
regression seqlen (16384). Covers haplotype, annotated, variant, track, and the
tracks-only path that REGRESSIONS.md fingered."""

from __future__ import annotations

import pytest

from tests.benchmarks._indices import batch_indices

SEQLEN = 16384
BATCH = 32

# Fold ITERATIONS calls into each timed sample so per-batch OS-scheduler jitter on
# the shared HPC node averages out. Without this the fast tracks-only path (~1.5 ms)
# is noise-dominated: a single ~0.5 ms scheduler hiccup is ~30% of one call but only
# ~3% of a 10-call sample. pedantic divides the round time by ``iterations``, so the
# reported figure stays per-``ds[r, s]`` (directly comparable across paths/backends).
ROUNDS = 50
ITERATIONS = 10
WARMUP_ROUNDS = 5


def _bench_indexing(benchmark, ds):
    r, s = batch_indices(ds.shape[0], ds.shape[1], BATCH)
    ds[r, s]  # warmup (JIT link, caches) before the timed rounds
    result = benchmark.pedantic(
        lambda: ds[r, s],
        rounds=ROUNDS,
        iterations=ITERATIONS,
        warmup_rounds=WARMUP_ROUNDS,
    )
    assert result is not None


def test_e2e_haplotypes(benchmark, bench_dataset):
    ds = bench_dataset.with_seqs("haplotypes").with_len(SEQLEN)
    _bench_indexing(benchmark, ds)


def test_e2e_annotated(benchmark, bench_dataset):
    ds = bench_dataset.with_seqs("annotated").with_len(SEQLEN)
    _bench_indexing(benchmark, ds)


@pytest.mark.xfail(
    strict=False,
    reason=(
        "pre-existing Phase 2: _FlatVariants has no to_fixed for with_len on variants; "
        "predates Phase 3"
    ),
)
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
