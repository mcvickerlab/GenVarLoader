import os

import genvarloader._threads as th


def _constrain_detected_cpus(monkeypatch, n: int) -> None:
    """Make CPU detection report exactly `n`, regardless of platform.

    `_resolve_num_threads` prefers `os.sched_getaffinity` (Linux, cgroup-aware)
    and falls back to `os.cpu_count` elsewhere. macOS has no `sched_getaffinity`,
    so patch it where it exists and otherwise patch the fallback.
    """
    try:
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(n)))
    except AttributeError:
        monkeypatch.setattr(os, "cpu_count", lambda: n)


def test_resolve_honors_env_override(monkeypatch):
    monkeypatch.setenv("GVL_NUM_THREADS", "7")
    assert th._resolve_num_threads() == 7


def test_resolve_env_not_clamped(monkeypatch):
    # New behavior: env is NOT clamped to any numba limit; user is responsible.
    monkeypatch.setenv("GVL_NUM_THREADS", "9999")
    assert th._resolve_num_threads() == 9999


def test_resolve_uses_cgroup_affinity(monkeypatch):
    monkeypatch.delenv("GVL_NUM_THREADS", raising=False)
    _constrain_detected_cpus(monkeypatch, 52)
    assert th._resolve_num_threads() == 52


def test_resolve_malformed_env_falls_back_to_affinity(monkeypatch):
    # a non-integer override must not break import; fall through to detection
    monkeypatch.setenv("GVL_NUM_THREADS", "auto")
    _constrain_detected_cpus(monkeypatch, 52)
    assert th._resolve_num_threads() == 52


def test_should_parallelize_threshold(monkeypatch):
    # Reset cached thread count so monkeypatch takes effect.
    monkeypatch.setattr(th, "_NUM_THREADS", None)
    monkeypatch.delenv("GVL_NUM_THREADS", raising=False)
    _constrain_detected_cpus(monkeypatch, 4)
    thresh = 4 * th._MIN_BYTES_PER_THREAD
    assert th.should_parallelize(thresh - 1) is False
    assert th.should_parallelize(thresh) is True
