"""svar1-vs-svar2 spliced-haplotype ``Dataset[...]`` benchmark.

Times spliced haplotype reconstruction (``with_seqs("haplotypes")`` +
``splice_info`` + ``var_filter="exonic"``) over the full transcript x sample grid
for both genotype backends. The two datasets are built from the same PGEN, so the
only difference is the backend and the reconstructed haplotypes are byte-identical
(guarded by :func:`test_svar1_svar2_spliced_parity`).

Skips when the full test reference / plink2 are unavailable (see
``data/build_svar_splice.py``), mirroring the committed-dataset skip in the e2e
suite.
"""

from __future__ import annotations

import numpy as np
import pytest

import genvarloader as gvl
from tests.benchmarks.data.build_svar_splice import build

SPLICE_INFO = ("transcript_id", "exon_number")

# Spliced full-batch reconstruction is ~15-25 ms; fold a few iterations per timed
# round so per-call OS jitter on a shared node averages out (see test_e2e).
ROUNDS = 20
ITERATIONS = 3
WARMUP_ROUNDS = 3


@pytest.fixture(scope="session")
def svar_splice_datasets(tmp_path_factory):
    def _open(path, ref):
        return gvl.Dataset.open(
            path, reference=ref, splice_info=SPLICE_INFO, var_filter="exonic"
        ).with_seqs("haplotypes")

    try:
        ds1_path, ds2_path, ref = build(tmp_path_factory.mktemp("svar_splice"))
        return _open(ds1_path, ref), _open(ds2_path, ref)
    except FileNotFoundError as e:
        pytest.skip(f"svar splice benchmark inputs unavailable: {e}")


def _bench_spliced(benchmark, ds):
    ds[:, :]  # warmup (JIT link, memmap fault-in) before the timed rounds
    result = benchmark.pedantic(
        lambda: ds[:, :],
        rounds=ROUNDS,
        iterations=ITERATIONS,
        warmup_rounds=WARMUP_ROUNDS,
    )
    assert result is not None


def test_svar1_spliced_haplotypes(benchmark, svar_splice_datasets):
    ds1, _ = svar_splice_datasets
    _bench_spliced(benchmark, ds1)


def test_svar2_spliced_haplotypes(benchmark, svar_splice_datasets):
    _, ds2 = svar_splice_datasets
    _bench_spliced(benchmark, ds2)


def test_svar1_svar2_spliced_parity(svar_splice_datasets):
    """svar1 and svar2 must produce byte-identical spliced haplotypes — the
    backend comparison (and the svar2 reorder optimization) is only meaningful
    if the outputs agree exactly."""
    ds1, ds2 = svar_splice_datasets
    a = ds1[:, :]
    b = ds2[:, :]
    assert np.array_equal(np.asarray(a.offsets), np.asarray(b.offsets))
    assert np.array_equal(a.data.view("u1"), b.data.view("u1"))
