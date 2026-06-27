"""Session fixtures for the benchmark suite.

- ``bench_dataset``: the committed chr22 GEUVADIS dataset, opened once.
- ``captured_*``: realistic numba-function arguments, recorded once by running
  a real reconstruction batch and capturing the first call (see _capture.py).

All fixtures skip the whole module if the committed dataset is absent.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import genvarloader as gvl
from genvarloader._dataset import _haps, _reconstruct, _tracks
from tests.benchmarks._capture import CapturedCall, capture_first_call
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
    # Task 13 (Phase 3): the rust default path now calls reconstruct_haplotypes_fused
    # (one FFI crossing) rather than reconstruct_haplotypes_from_sparse.  Force the
    # numba path to capture args that are compatible with the per-kernel benchmark
    # (test_reconstruct_haplotypes_from_sparse benchmarks the raw dispatch entry).
    old_backend = os.environ.get("GVL_BACKEND")
    os.environ["GVL_BACKEND"] = "numba"
    try:
        recon = capture_first_call(
            targets=[(_haps, "reconstruct_haplotypes_from_sparse")],
            thunk=lambda: ds[r, s],
        )
    finally:
        if old_backend is None:
            os.environ.pop("GVL_BACKEND", None)
        else:
            os.environ["GVL_BACKEND"] = old_backend
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
    #
    # Task 14 (Phase 3): the rust default path now calls
    # intervals_and_realign_track_fused (one FFI crossing) rather than the
    # composed numba path, so shift_and_realign_tracks_sparse is no longer a
    # module-level attribute on _reconstruct — capture_first_call's setattr
    # trick cannot intercept the call.  The numba composed path reaches the
    # kernel via _dispatch_get() → _REGISTRY[...]["numba"], which holds a
    # direct function reference that bypasses the module attribute.  We force
    # GVL_BACKEND=numba, then patch the registry entry directly so the recorder
    # wraps the exact callable that _dispatch_get returns (which is also
    # _tracks.shift_and_realign_tracks_sparse — the same object the benchmark
    # replays).
    ds = (
        bench_dataset.with_seqs("haplotypes").with_tracks("read-depth").with_len(SEQLEN)
    )
    r, s = _batch_indices(ds, BATCH)
    original = _reconstruct._shift_and_realign_tracks_sparse_rust_wrapper
    captured: list[CapturedCall] = []

    def recorder(*args, **kwargs):
        if not captured:
            captured.append(CapturedCall(args=args, kwargs=dict(kwargs)))
        return original(*args, **kwargs)

    _reconstruct._shift_and_realign_tracks_sparse_rust_wrapper = recorder
    try:
        ds[r, s]
    finally:
        _reconstruct._shift_and_realign_tracks_sparse_rust_wrapper = original
    if not captured:
        raise RuntimeError(
            "shift_and_realign_tracks_sparse was never called while running the thunk"
        )
    return captured[0]


# NOTE: a ``captured_germline_ccfs`` fixture was intentionally dropped. The
# committed chr22 GEUVADIS dataset has no CCF field, so the then-existing
# ``_infer_germline_ccfs`` kernel never fired. That order-dependent germline-CCF
# inference path has since been removed (#222); the end-to-end variants benchmark
# is now the sole signal for the variant-assembly path.
