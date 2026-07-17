"""SVAR2 spliced-*variant* ``Dataset[...]`` benchmark (PR #286 decode path).

Complements ``test_e2e_svar_splice.py``, which times spliced *haplotypes* on the
committed 5-sample PGEN. This times spliced ``with_seqs("variants")`` -- the
regroup path in ``_query._fetch_spliced_variants`` plus the Rayon SVAR2 decode --
on a vcfixture-rs bulk cohort, so cohort size (the dominant axis) is a knob.

Skips when the vcfixture-rs bulk CLI (``VCFIXTURE_BIN`` / ``vcfixture`` on PATH) or
bcftools/samtools are unavailable, mirroring the committed-input skip in the
haplotype benchmark. Kept small so it runs in CI when the CLI is present; the
scaled sweep lives in ``profiling/sweep_svar2_splice_variants.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

import genvarloader as gvl
from tests.benchmarks.data.build_svar2_splice_bulk import build

SPLICE_INFO = ("transcript_id", "exon_number")

N_SAMPLES = 200
N_TRANSCRIPTS = 32
RECORDS = 5_000

# Match the haplotype benchmark's fold-a-few-iterations approach so per-call OS
# jitter on a shared node averages out.
ROUNDS = 20
ITERATIONS = 3
WARMUP_ROUNDS = 3


@pytest.fixture(scope="session")
def svar2_splice_variants_ds(tmp_path_factory):
    try:
        fx = build(
            tmp_path_factory.mktemp("svar2_splice_bulk"),
            n_samples=N_SAMPLES,
            n_transcripts=N_TRANSCRIPTS,
            records=RECORDS,
        )
        return (
            gvl.Dataset.open(fx.gvl_path, reference=fx.reference)
            .with_settings(splice_info=SPLICE_INFO, var_filter="exonic")
            .with_seqs("variants")
        )
    except FileNotFoundError as e:
        pytest.skip(f"svar2 bulk spliced-variant benchmark inputs unavailable: {e}")


def test_svar2_spliced_variants(benchmark, svar2_splice_variants_ds):
    ds = svar2_splice_variants_ds
    ds[:, :]  # warmup (JIT link, memmap fault-in) before timed rounds
    result = benchmark.pedantic(
        lambda: ds[:, :],
        rounds=ROUNDS,
        iterations=ITERATIONS,
        warmup_rounds=WARMUP_ROUNDS,
    )
    assert result is not None


def test_svar2_spliced_variants_deterministic(svar2_splice_variants_ds):
    """The decode must be byte-identical across repeated reads -- the correctness
    oracle the optimization loop is gated against."""
    ds = svar2_splice_variants_ds
    a = ds[:, :]
    b = ds[:, :]
    assert set(a.fields) == set(b.fields)
    for name in a.fields:
        fa, fb = a[name], b[name]
        assert np.array_equal(np.asarray(fa.data), np.asarray(fb.data)), (
            f"{name}: data differ"
        )
        assert np.array_equal(np.asarray(fa.offsets), np.asarray(fb.offsets)), (
            f"{name}: offsets differ"
        )
