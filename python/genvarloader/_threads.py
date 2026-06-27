"""Cgroup-aware thread-count resolver + rayon pool initializer.

Resolves the effective worker count from GVL_NUM_THREADS or the
cgroup cpuset (Linux sched_getaffinity), capped by the number of CPUs
available (or numba's thread pool size if numba is installed).
Seeds RAYON_NUM_THREADS so rayon's global pool picks it up on first
use.  Must run before the first rust parallel call (rayon reads the
env var at global-pool init time). Idempotent.
"""

from __future__ import annotations

import os

_MIN_BYTES_PER_THREAD = 1 << 20  # 1 MiB
_NUM_THREADS: int | None = None


def _detect_cpus() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))  # respects cgroup cpuset (Linux)
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def _max_threads() -> int:
    """Upper bound on usable threads: CPU count, or numba's pool size if available."""
    try:
        import numba  # noqa: F401 (optional; still in venv during migration)

        return max(1, numba.get_num_threads())
    except Exception:
        return _detect_cpus()


def _resolve_num_threads() -> int:
    env = os.environ.get("GVL_NUM_THREADS")
    if env:
        try:
            n = int(env)
            # Cap to available CPUs / numba pool so users can't over-subscribe.
            return max(1, min(n, _max_threads()))
        except ValueError:
            # A malformed override (e.g. "auto") must not break `import
            # genvarloader`; fall through to detection instead.
            pass
    return _detect_cpus()


def cap_threads() -> int:
    """Resolve worker count once and pin rayon's pool via RAYON_NUM_THREADS.

    Must run before the first rust parallel call (rayon reads RAYON_NUM_THREADS
    at global-pool init). Idempotent.
    """
    global _NUM_THREADS
    if _NUM_THREADS is None:
        _NUM_THREADS = _resolve_num_threads()
        os.environ.setdefault("RAYON_NUM_THREADS", str(_NUM_THREADS))
    return _NUM_THREADS


def num_threads() -> int:
    return cap_threads()


def should_parallelize(total_bytes: int) -> bool:
    """True iff a copy of `total_bytes` is large enough to justify fork-join."""
    n = _max_threads()
    return total_bytes >= n * _MIN_BYTES_PER_THREAD
