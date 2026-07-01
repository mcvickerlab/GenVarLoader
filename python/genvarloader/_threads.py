"""Cgroup-aware thread-count resolver + rayon pool initializer.

Resolves the effective worker count from GVL_NUM_THREADS or the
cgroup cpuset (Linux sched_getaffinity). Seeds RAYON_NUM_THREADS so
rayon's global pool picks it up on first use. Must run before the
first rust parallel call (rayon reads the env var at global-pool init
time). Idempotent.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

_MIN_BYTES_PER_THREAD = 1 << 20  # 1 MiB
_NUM_THREADS: int | None = None

_TRUTHY = frozenset({"1", "true", "yes", "on"})

# cgroup CPU-quota files (module-level so tests can repoint them).
_CGROUP_V2_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")
_CGROUP_V1_QUOTA = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
_CGROUP_V1_PERIOD = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _cgroup_cpu_quota() -> int | None:
    """Effective CPU count implied by a CFS quota, or None if unlimited/unreadable.

    A CFS *quota* (cpu.max / cpu.cfs_quota_us) is invisible to
    sched_getaffinity, so a 15.3-core container still reports 16 cores by
    affinity. See issue #263.
    """
    # cgroup v2: "<quota> <period>" or "max <period>".
    try:
        raw = _CGROUP_V2_CPU_MAX.read_text().split()
    except OSError:
        raw = None
    if raw and len(raw) == 2:
        quota_s, period_s = raw
        if quota_s != "max":
            try:
                quota, period = int(quota_s), int(period_s)
            except ValueError:
                quota = period = 0
            if quota > 0 and period > 0:
                return max(1, math.ceil(quota / period))
        else:
            return None  # explicitly unlimited

    # cgroup v1 fallback.
    quota = _read_int(_CGROUP_V1_QUOTA)
    period = _read_int(_CGROUP_V1_PERIOD)
    if quota is not None and quota > 0 and period:
        return max(1, math.ceil(quota / period))
    return None


def _force_parallel() -> bool:
    """True iff GVL_FORCE_PARALLEL is set to a truthy value (read live)."""
    return os.environ.get("GVL_FORCE_PARALLEL", "").strip().lower() in _TRUTHY


def _detect_cpus() -> int:
    try:
        affinity = max(
            1, len(os.sched_getaffinity(0))
        )  # respects cgroup cpuset (Linux)
    except AttributeError:
        affinity = max(1, os.cpu_count() or 1)
    quota = _cgroup_cpu_quota()
    if quota is not None:
        return max(1, min(affinity, quota))
    return affinity


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
    # GVL_FORCE_PARALLEL bypasses the size gate so the multithreaded paths run
    # on small inputs (tests, repro harnesses). See issue #263.
    if _force_parallel():
        return True
    return total_bytes >= num_threads() * _MIN_BYTES_PER_THREAD
