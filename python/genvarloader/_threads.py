"""Cgroup-aware thread-count resolver + rayon pool initializer.

Resolves the effective worker count from GVL_NUM_THREADS or the
cgroup cpuset (Linux sched_getaffinity). Seeds RAYON_NUM_THREADS so
rayon's global pool picks it up on first use. Must run before the
first rust parallel call (rayon reads the env var at global-pool init
time). Idempotent.
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


def _resolve_num_threads() -> int:
    env = os.environ.get("GVL_NUM_THREADS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
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
    return total_bytes >= num_threads() * _MIN_BYTES_PER_THREAD
