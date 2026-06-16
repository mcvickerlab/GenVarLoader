import os

import numba

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
    # env wins, clamped to >= 1 and <= numba hard max
    monkeypatch.setattr(numba, "get_num_threads", lambda: 64)
    assert th._resolve_num_threads() == 7


def test_resolve_env_clamped_to_numba_max(monkeypatch):
    monkeypatch.setenv("GVL_NUM_THREADS", "9999")
    monkeypatch.setattr(numba, "get_num_threads", lambda: 64)
    assert th._resolve_num_threads() == 64


def test_resolve_uses_cgroup_affinity(monkeypatch):
    monkeypatch.delenv("GVL_NUM_THREADS", raising=False)
    # host reports 208 logical CPUs, cgroup allows 52 -> min wins
    monkeypatch.setattr(numba, "get_num_threads", lambda: 208)
    _constrain_detected_cpus(monkeypatch, 52)
    assert th._resolve_num_threads() == 52


def test_resolve_malformed_env_falls_back_to_affinity(monkeypatch):
    # a non-integer override must not break import; fall through to detection
    monkeypatch.setenv("GVL_NUM_THREADS", "auto")
    monkeypatch.setattr(numba, "get_num_threads", lambda: 208)
    _constrain_detected_cpus(monkeypatch, 52)
    assert th._resolve_num_threads() == 52


def test_should_parallelize_threshold(monkeypatch):
    monkeypatch.setattr(numba, "get_num_threads", lambda: 4)
    thresh = 4 * th._MIN_BYTES_PER_THREAD
    assert th.should_parallelize(thresh - 1) is False
    assert th.should_parallelize(thresh) is True
