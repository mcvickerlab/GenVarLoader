"""Micro-benchmarks: isolated numba hot functions, replayed with realistic
arguments captured from a real reconstruction (see conftest.py)."""

from __future__ import annotations

import numpy as np
import pytest

from genvarloader._dataset._genotypes import (
    get_diffs_sparse,
    reconstruct_haplotypes_from_sparse,
)
from genvarloader._dataset._intervals import intervals_to_tracks
from genvarloader._dataset._tracks import (
    _shift_and_realign_tracks_sparse_rust_wrapper as shift_and_realign_tracks_sparse,
)


def _warm_and_run(benchmark, fn, captured):
    """Warm up JIT once, then benchmark replaying the captured call."""
    args, kwargs = captured.args, captured.kwargs
    fn(*args, **kwargs)  # JIT link / warmup (not timed)
    return benchmark(lambda: fn(*args, **kwargs))


def _out_buffer(captured):
    """Locate the `out` buffer this kernel writes into."""
    out = captured.kwargs.get("out")
    if out is None and captured.args:
        out = captured.args[0]
    return out


def test_get_diffs_sparse(benchmark, captured_diffs):
    # returns a freshly-allocated ndarray (shape (batch, ploidy))
    result = _warm_and_run(benchmark, get_diffs_sparse, captured_diffs)
    assert isinstance(result, np.ndarray)
    assert result.size > 0


@pytest.mark.skip(
    reason="kernel fused into rust (W3/W5); micro-benchmark pending redesign — W6"
)
def test_reconstruct_haplotypes_from_sparse(benchmark, captured_haplotypes):
    # returns None; writes into the preallocated `out` buffer
    _warm_and_run(benchmark, reconstruct_haplotypes_from_sparse, captured_haplotypes)
    out = _out_buffer(captured_haplotypes)
    assert out is not None and np.asarray(out).size > 0


@pytest.mark.skip(
    reason="kernel fused into rust (W3/W5); micro-benchmark pending redesign — W6"
)
def test_intervals_to_tracks(benchmark, captured_intervals_to_tracks):
    # returns None; writes into the preallocated `out` buffer
    _warm_and_run(benchmark, intervals_to_tracks, captured_intervals_to_tracks)
    out = _out_buffer(captured_intervals_to_tracks)
    assert out is not None and np.asarray(out).size > 0


@pytest.mark.skip(
    reason="kernel fused into rust (W3/W5); micro-benchmark pending redesign — W6"
)
def test_shift_and_realign_tracks_sparse(benchmark, captured_realign_tracks):
    # returns None; writes into the preallocated `out` buffer
    _warm_and_run(benchmark, shift_and_realign_tracks_sparse, captured_realign_tracks)
    out = _out_buffer(captured_realign_tracks)
    assert out is not None and np.asarray(out).size > 0
