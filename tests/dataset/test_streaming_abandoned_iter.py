"""Issue #399: abandoning a `to_iter()` generator must not wedge the producer.

A failing assertion inside a `for ... in sds.to_iter()` loop abandons the
generator mid-window. Before the fix, the record-stream (VCF/PGEN) producer
thread was left blocked in `Python::attach` (it needs the GIL to drive the
pgenlib reader) while `EngineState::drop` -- which runs during Python
deallocation, i.e. *holding* the GIL -- blocked in `JoinHandle::join`. That is a
hard GIL deadlock: the next consumer to touch the interpreter hangs forever.

Why this is worth its own test file: the failure mode destroys the signal for
every other streaming test. A real regression inside a `to_iter` loop stops
presenting as a failing assertion with a diff and starts presenting as a CI job
that hangs with no output.

**Watchdog.** A GIL deadlock cannot be interrupted from Python -- a main thread
blocked inside a Rust `join()` never reaches a bytecode boundary, so neither
`SIGALRM` handlers nor `pytest-timeout`'s `signal` method ever run. `faulthandler`
arms a timer in C, so it fires regardless of the GIL; `exit=True` aborts the
process after dumping every thread's stack. That is deliberately harsher than a
test failure: on regression we want a bounded abort *with tracebacks* rather than
an unbounded hang.
"""

from __future__ import annotations

import faulthandler
import gc
import subprocess
from contextlib import contextmanager
from pathlib import Path

import polars as pl
import pytest

# Generous relative to the work (a 40-window sweep over a 2-variant fixture is
# sub-second); small relative to a CI job timeout.
_WATCHDOG_S = 120


@contextmanager
def _deadlock_watchdog(seconds: int = _WATCHDOG_S):
    faulthandler.dump_traceback_later(seconds, exit=True)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


def _many_window_regions(n_regions: int = 40) -> pl.DataFrame:
    """A plan with many windows, so the producer is still mid-`fill` when the
    consumer abandons -- a single-window plan lets the producer finish and exit
    before teardown, which is exactly the case that never deadlocked."""
    return pl.DataFrame(
        {
            "chrom": ["chr1"] * n_regions,
            "chromStart": list(range(n_regions)),
            "chromEnd": [i + 20 for i in range(n_regions)],
        }
    )


@pytest.fixture(scope="module")
def _ref_fasta(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("abandoned_iter_ref")
    fasta = d / "ref.fa"
    fasta.write_text(">chr1\n" + "A" * 100 + "\n")
    subprocess.run(["samtools", "faidx", str(fasta)], check=True)
    return fasta


def _variants_path(source: str) -> Path:
    name = "two_var_two_sample.vcf.gz" if source == "vcf" else "two_var_two_sample.pgen"
    return Path(__file__).parent.parent / "data" / "streaming" / name


@pytest.mark.parametrize("source", ["vcf", "pgen"])
@pytest.mark.parametrize("kind", ["variants", "haplotypes"])
def test_abandoned_iter_does_not_wedge_next_consumer(source, kind, _ref_fasta):
    """Abandon a `to_iter()` mid-window, then drive a fresh one to completion.

    The second drive is the assertion: pre-fix it never returns.
    """
    import genvarloader as gvl

    # `reference` is required at construction for EVERY output kind, variants
    # included -- `StreamingDataset.__init__` raises without it.
    sds = gvl.StreamingDataset(
        _many_window_regions(),
        reference=str(_ref_fasta),
        variants=str(_variants_path(source)),
    ).with_seqs(kind)

    with _deadlock_watchdog():
        # Abandon after the FIRST batch -- what a failing assertion inside the
        # loop does, and the moment most likely to catch the producer mid-`fill`:
        # it has delivered window 0 and taken the second ping-pong slot for
        # window 1, while this thread holds the GIL from here through
        # `gc.collect()`. A producer that needs the GIL to fill (PGEN drives
        # pgenlib through `Python::attach`) is therefore blocked precisely when
        # teardown joins it.
        it = sds.to_iter(batch_size=1)
        next(it)
        del it
        gc.collect()  # force the engine's `__del__` -> Rust `EngineState::drop`

        n = sum(1 for _ in sds.to_iter(batch_size=1))

    assert n > 0


@pytest.mark.parametrize("source", ["vcf", "pgen"])
def test_explicit_generator_close_does_not_wedge(source, _ref_fasta):
    """`close()` is the deterministic path -- teardown must not wait for GC."""
    import genvarloader as gvl

    sds = gvl.StreamingDataset(
        _many_window_regions(),
        reference=str(_ref_fasta),
        variants=str(_variants_path(source)),
    ).with_seqs("variants")

    with _deadlock_watchdog():
        it = sds.to_iter(batch_size=1)
        next(it)
        it.close()

        n = sum(1 for _ in sds.to_iter(batch_size=1))

    assert n > 0
