"""Issue #399 repro attempt #2: abandonment on the MIXED record-backend path.

The non-mixed repro (`test_streaming_abandoned_iter.py`) passes on unfixed
code, so whatever wedges the producer is specific to the mixed
(variants + tracks) drive that #399 was actually observed on.

`max_mem="4k"` forces a genuinely multi-window plan over the fixture's 3x3
grid (the same tuned constant `test_record_mixed_parity_under_forced_windowing`
uses, with the same anti-vacuity assertion), so there IS a live producer with
outstanding work at the moment the consumer walks away.
"""

from __future__ import annotations

import faulthandler
import gc
from contextlib import contextmanager

import pytest

import genvarloader as gvl

BACKENDS = ("vcf", "pgen")
_WATCHDOG_S = 120


@contextmanager
def _deadlock_watchdog(seconds: int = _WATCHDOG_S):
    faulthandler.dump_traceback_later(seconds, exit=True)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


def _mixed_sds(f, **kw):
    return gvl.StreamingDataset(
        f.bed,
        reference=f.reference_path,
        variants=f.variants_path,
        tracks=[f.table, f.bigwigs],
        **kw,
    ).with_seqs("haplotypes")


@pytest.mark.parametrize("backend", BACKENDS)
def test_abandoned_mixed_iter_does_not_wedge_next_consumer(
    streaming_record_tracks_fixture, backend
):
    f = streaming_record_tracks_fixture(backend)

    with _deadlock_watchdog():
        sds = _mixed_sds(f, max_mem="4k")
        assert (
            sds._window_samples < len(f.samples) or sds._window_regions < f.bed.height
        ), "max_mem did not force a multi-window plan; this test is vacuous"

        it = sds.to_iter(batch_size=1)
        next(it)
        del it
        gc.collect()

        n = sum(1 for _ in _mixed_sds(f, max_mem="4k").to_iter(batch_size=1))

    assert n > 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_failing_assert_inside_mixed_loop_does_not_wedge(
    streaming_record_tracks_fixture, backend
):
    """The literal shape #399 describes: an assertion fires inside the loop.

    Reusing the SAME dataset object for the second drive matters -- the mixed
    path caches a second engine on the backend (`_mixed_engine_obj`), so a
    fresh `StreamingDataset` would sidestep exactly the state under suspicion.
    """
    f = streaming_record_tracks_fixture(backend)

    with _deadlock_watchdog():
        sds = _mixed_sds(f, max_mem="4k")

        with pytest.raises(AssertionError):
            for _ in sds.to_iter(batch_size=1):
                raise AssertionError("simulated in-loop test failure")
        gc.collect()

        n = sum(1 for _ in sds.to_iter(batch_size=1))

    assert n > 0
