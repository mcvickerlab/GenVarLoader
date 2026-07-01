import math
import os

import pytest

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


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on", "On"])
def test_force_parallel_truthy(monkeypatch, val):
    monkeypatch.setenv("GVL_FORCE_PARALLEL", val)
    # Below the byte threshold, but forced on → parallel.
    assert th.should_parallelize(0) is True


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "", "banana"])
def test_force_parallel_falsy_falls_back_to_threshold(monkeypatch, val):
    monkeypatch.setattr(th, "_NUM_THREADS", None)
    monkeypatch.delenv("GVL_NUM_THREADS", raising=False)
    monkeypatch.setenv("GVL_FORCE_PARALLEL", val)
    _constrain_detected_cpus(monkeypatch, 4)
    # Not forced → normal size gate applies.
    assert th.should_parallelize(0) is False
    assert th.should_parallelize(4 * th._MIN_BYTES_PER_THREAD) is True


def test_force_parallel_unset(monkeypatch):
    monkeypatch.delenv("GVL_FORCE_PARALLEL", raising=False)
    monkeypatch.setattr(th, "_NUM_THREADS", None)
    monkeypatch.delenv("GVL_NUM_THREADS", raising=False)
    _constrain_detected_cpus(monkeypatch, 4)
    assert th.should_parallelize(0) is False


def test_cgroup_quota_v2_parses_cpu_max(monkeypatch, tmp_path):
    f = tmp_path / "cpu.max"
    f.write_text("1530000 100000\n")  # 15.3 cores
    monkeypatch.setattr(th, "_CGROUP_V2_CPU_MAX", f)
    monkeypatch.setattr(th, "_CGROUP_V1_QUOTA", tmp_path / "nope_quota")
    monkeypatch.setattr(th, "_CGROUP_V1_PERIOD", tmp_path / "nope_period")
    assert th._cgroup_cpu_quota() == math.ceil(1530000 / 100000)  # 16


def test_cgroup_quota_v2_max_is_none(monkeypatch, tmp_path):
    f = tmp_path / "cpu.max"
    f.write_text("max 100000\n")  # unlimited
    monkeypatch.setattr(th, "_CGROUP_V2_CPU_MAX", f)
    monkeypatch.setattr(th, "_CGROUP_V1_QUOTA", tmp_path / "nope_quota")
    monkeypatch.setattr(th, "_CGROUP_V1_PERIOD", tmp_path / "nope_period")
    assert th._cgroup_cpu_quota() is None


def test_cgroup_quota_v1_fallback(monkeypatch, tmp_path):
    q = tmp_path / "cfs_quota_us"
    p = tmp_path / "cfs_period_us"
    q.write_text("800000\n")
    p.write_text("100000\n")
    monkeypatch.setattr(th, "_CGROUP_V2_CPU_MAX", tmp_path / "absent")
    monkeypatch.setattr(th, "_CGROUP_V1_QUOTA", q)
    monkeypatch.setattr(th, "_CGROUP_V1_PERIOD", p)
    assert th._cgroup_cpu_quota() == 8


def test_cgroup_quota_none_when_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(th, "_CGROUP_V2_CPU_MAX", tmp_path / "absent1")
    monkeypatch.setattr(th, "_CGROUP_V1_QUOTA", tmp_path / "absent2")
    monkeypatch.setattr(th, "_CGROUP_V1_PERIOD", tmp_path / "absent3")
    assert th._cgroup_cpu_quota() is None


def test_detect_cpus_takes_min_of_affinity_and_quota(monkeypatch, tmp_path):
    _constrain_detected_cpus(monkeypatch, 16)  # affinity reports 16
    monkeypatch.setattr(th, "_cgroup_cpu_quota", lambda: 15)
    assert th._detect_cpus() == 15


def test_detect_cpus_ignores_quota_when_none(monkeypatch):
    _constrain_detected_cpus(monkeypatch, 16)
    monkeypatch.setattr(th, "_cgroup_cpu_quota", lambda: None)
    assert th._detect_cpus() == 16


def test_cap_threads_overwrites_ambient_rayon(monkeypatch):
    # An ambient RAYON_NUM_THREADS (base image) must NOT win over GVL's count.
    monkeypatch.setenv("RAYON_NUM_THREADS", "16")
    monkeypatch.setattr(th, "_NUM_THREADS", None)
    monkeypatch.setenv("GVL_NUM_THREADS", "4")
    n = th.cap_threads()
    assert n == 4
    assert os.environ["RAYON_NUM_THREADS"] == "4"


def test_cap_threads_sets_when_unset(monkeypatch):
    monkeypatch.delenv("RAYON_NUM_THREADS", raising=False)
    monkeypatch.setattr(th, "_NUM_THREADS", None)
    monkeypatch.setenv("GVL_NUM_THREADS", "3")
    th.cap_threads()
    assert os.environ["RAYON_NUM_THREADS"] == "3"
