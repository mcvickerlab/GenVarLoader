"""Cgroup-aware numba thread cap + a per-thread dispatch predicate.

numba.get_num_threads() reports host logical CPUs, not the cgroup allocation
(e.g. 208 reported vs. 52 allocated). Forking the misdetected count makes
parallel=True regions pay a flat ~37 ms fork-join for trivial work. We cap the
worker count down to the real allocation once at import, and route copy kernels
to a serial variant unless there is enough work to amortize the fork-join.
"""

from __future__ import annotations

import os

import numba

# Parallel only pays off when each worker gets at least this many bytes to copy.
# Below `num_threads * _MIN_BYTES_PER_THREAD` total, the serial kernel wins.
_MIN_BYTES_PER_THREAD = 1 << 20  # 1 MiB


def _resolve_num_threads() -> int:
    hard_max = numba.get_num_threads()
    env = os.environ.get("GVL_NUM_THREADS")
    if env:
        return max(1, min(int(env), hard_max))
    try:
        real = len(os.sched_getaffinity(0))  # respects cgroup cpuset (Linux)
    except AttributeError:
        real = os.cpu_count() or 1  # non-Linux fallback
    return max(1, min(real, hard_max))


def cap_numba_threads() -> int:
    """Cap numba's parallel worker count to the resolved value. Idempotent."""
    n = _resolve_num_threads()
    numba.set_num_threads(n)
    return n


def should_parallelize(total_bytes: int) -> bool:
    """True iff a copy of `total_bytes` is large enough to justify fork-join."""
    return total_bytes >= numba.get_num_threads() * _MIN_BYTES_PER_THREAD
