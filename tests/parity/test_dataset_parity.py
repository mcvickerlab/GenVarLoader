"""Dataset-level backstop: _write_track_legacy round-trip must be identical across backends.

Exercises ``splits_sum_le_value`` through the real track-writing path
(``_write_track_legacy``) with a multi-sample BigWigs track.  BigWigs
implements the ``IntervalTrack`` protocol so ``_write_track_legacy`` is
valid for it; note that the top-level ``gvl.write`` routes BigWigs to
a separate fully-Rust path that skips the dispatch, so we call
``_write_track_legacy`` directly here to target the kernel under test.

Why subprocess?
  ``GVL_BACKEND`` is read at import time via the dispatch registry.
  Running each backend in a fresh process prevents numba-cache leakage
  and ensures the env var is consumed before any kernel is JIT-compiled.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.parity

# Repo root is needed so `tests.*` is importable inside the subprocess.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# The snippet is passed to `python -c`.  It:
#   1. Puts the repo root on sys.path (mirrors pytest's pythonpath=["."] config).
#   2. Reads GVL_BACKEND *before* any genvarloader import (dispatch reads env at
#      call time, but numba JIT caches triggers on first import — clean env wins).
#   3. Calls _write_track_legacy directly so splits_sum_le_value is dispatched
#      through the backend selected by GVL_BACKEND.
_WRITE_SNIPPET = r"""
import sys, os
sys.path.insert(0, sys.argv[3])           # repo root → tests.* importable

from pathlib import Path
from tests.parity._fixtures import build_write_inputs, MAX_MEM
from genvarloader._dataset._write import _write_track_legacy

out_dir = Path(sys.argv[1])
work_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

bed, track = build_write_inputs(work_dir)
_write_track_legacy(out_dir, bed, track, track.samples, MAX_MEM)
"""


def _run_write(tmp_path: Path, backend: str, shared: Path) -> Path:
    """Run _write_track_legacy in a subprocess with GVL_BACKEND=backend."""
    out = tmp_path / f"out_{backend}"
    env = {**os.environ, "GVL_BACKEND": backend}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _WRITE_SNIPPET,
            str(out),  # argv[1]: output dir
            str(shared),  # argv[2]: shared fixture dir (bigwigs + regions)
            str(_REPO_ROOT),  # argv[3]: repo root for sys.path
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.stderr:
        # Surface any warnings/numba output for debugging, but don't fail.
        print(result.stderr, file=sys.stderr)
    return out


def test_write_round_trip_identical_across_backends(tmp_path):
    """Both backends must produce byte-identical intervals and offsets arrays."""
    # Build fixtures once in a shared directory; both subprocesses use the same
    # bigWig files so the only variable is GVL_BACKEND.
    shared = tmp_path / "inputs"

    numba_out = _run_write(tmp_path, "numba", shared)
    rust_out = _run_write(tmp_path, "rust", shared)

    # The written layout is: out_dir/intervals.npy and out_dir/offsets.npy.
    # These are written as raw binary via np.memmap (no numpy array header),
    # so they cannot be loaded with np.load.  Use np.fromfile with the known
    # dtypes (INTERVAL_DTYPE for intervals, int64 for offsets).
    from genvarloader._ragged import INTERVAL_DTYPE

    for rel, dtype in [("intervals.npy", INTERVAL_DTYPE), ("offsets.npy", np.int64)]:
        a = np.fromfile(numba_out / rel, dtype=dtype)
        b = np.fromfile(rust_out / rel, dtype=dtype)
        np.testing.assert_array_equal(
            a,
            b,
            err_msg=f"{rel}: numba vs rust mismatch",
        )
