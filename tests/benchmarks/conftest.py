"""Session fixtures for the benchmark suite.

- ``bench_dataset``: the committed chr22 GEUVADIS dataset, opened once.
- ``captured_*``: realistic numba-function arguments, recorded once by running
  a real reconstruction batch and capturing the first call (see _capture.py).

All fixtures skip the whole module if the committed dataset is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import genvarloader as gvl
from genvarloader._dataset import _haps, _reconstruct, _tracks
from tests.benchmarks._capture import capture_first_call
from tests.benchmarks._indices import batch_indices

DATA = Path(__file__).resolve().parent / "data"
DS_PATH = DATA / "chr22_geuv.gvl"
REF_PATH = DATA / "chr22.masked.fa.gz"
SEQLEN = 16384
BATCH = 32  # number of (region, sample) pairs to drive per capture


@pytest.fixture(scope="session")
def bench_dataset():
    if not DS_PATH.exists():
        pytest.skip(
            f"Benchmark dataset {DS_PATH} not built. "
            "Run: pixi run -e dev python tests/benchmarks/data/build_realistic.py"
        )
    return gvl.Dataset.open(DS_PATH, REF_PATH)


def _batch_indices(ds, n: int):
    """A flat batch of (region_idx, sample_idx) within the dataset bounds."""
    return batch_indices(ds.shape[0], ds.shape[1], n)


@pytest.fixture(scope="session")
def captured_haplotypes(bench_dataset):
    ds = bench_dataset.with_seqs("haplotypes").with_len(SEQLEN)
    r, s = _batch_indices(ds, BATCH)
    recon = capture_first_call(
        targets=[(_haps, "reconstruct_haplotypes_from_sparse")],
        thunk=lambda: ds[r, s],
    )
    return recon


@pytest.fixture(scope="session")
def captured_diffs(bench_dataset):
    ds = bench_dataset.with_seqs("haplotypes").with_len(SEQLEN)
    r, s = _batch_indices(ds, BATCH)
    return capture_first_call(
        targets=[(_haps, "get_diffs_sparse")],
        thunk=lambda: ds[r, s],
    )


@pytest.fixture(scope="session")
def captured_intervals_to_tracks(bench_dataset):
    ds = bench_dataset.with_seqs(None).with_tracks("read-depth").with_len(SEQLEN)
    r, s = _batch_indices(ds, BATCH)
    return capture_first_call(
        targets=[
            (_reconstruct, "intervals_to_tracks"),
            (_tracks, "intervals_to_tracks"),
        ],
        thunk=lambda: ds[r, s],
    )


@pytest.fixture(scope="session")
def captured_realign_tracks(bench_dataset):
    # shift_and_realign_tracks_sparse only fires on the haplotype+tracks path
    # (_reconstruct.py); the tracks-only path (_tracks.py) never realigns.
    ds = (
        bench_dataset.with_seqs("haplotypes").with_tracks("read-depth").with_len(SEQLEN)
    )
    r, s = _batch_indices(ds, BATCH)
    return capture_first_call(
        targets=[(_reconstruct, "shift_and_realign_tracks_sparse")],
        thunk=lambda: ds[r, s],
    )


# NOTE: a ``captured_germline_ccfs`` fixture was intentionally dropped. The
# committed chr22 GEUVADIS dataset has no CCF field, so ``_infer_germline_ccfs``
# never fires (germline CCF inference is conditional). Task 5 relies on the
# end-to-end variants benchmark for that path instead of a micro-bench.
