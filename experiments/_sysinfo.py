"""Shared CPU / system / microarch capture for experiment benchmarks.

Benchmark results are only reproducible if the hardware they ran on is
recorded. ``system_info()`` returns a best-effort dict (cross-platform: macOS
+ linux-64); ``write_and_log()`` prints a short summary and writes the full
dict to ``<out_dir>/system_info.json`` next to a bench's results.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path

_SIMD_FLAGS = {
    # x86
    "sse4_2",
    "avx",
    "avx2",
    "fma",
    "bmi2",
    "avx512f",
    "avx512bw",
    "avx512vl",
    "avx512dq",
    "avx512cd",
    # arm
    "neon",
    "asimd",
    "sve",
    "sve2",
}


def _sysctl(key: str) -> str | None:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", key], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _proc_cpuinfo_field(field: str) -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith(field.lower()):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def _cpu_brand_fallback() -> str | None:
    if platform.system() == "Darwin":
        return _sysctl("machdep.cpu.brand_string")
    if platform.system() == "Linux":
        return _proc_cpuinfo_field("model name")
    return platform.processor() or None


def _physical_cpus() -> int | None:
    if platform.system() == "Darwin":
        v = _sysctl("hw.physicalcpu")
        return int(v) if v and v.isdigit() else None
    if platform.system() == "Linux":
        try:
            pairs = set()
            phys = core = None
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("physical id"):
                    phys = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":", 1)[1].strip()
                elif not line.strip() and phys is not None and core is not None:
                    pairs.add((phys, core))
                    phys = core = None
            if pairs:
                return len(pairs)
        except Exception:
            pass
    return None


def _ram_gib() -> float | None:
    if platform.system() == "Darwin":
        v = _sysctl("hw.memsize")
        if v and v.isdigit():
            return round(int(v) / 1024**3, 1)
    if platform.system() == "Linux":
        kb = _proc_cpuinfo_field  # not used; parse meminfo directly
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) / 1024**2, 1)
        except Exception:
            pass
        del kb
    return None


def _package_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out: dict = {}
    for pkg in (
        "genvarloader",
        "genoray",
        "seqpro",
        "numpy",
        "polars",
        "awkward",
        "torch",
        "py-cpuinfo",
    ):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
        except Exception:
            pass
    return out


def system_info() -> dict:
    """Best-effort hardware + environment snapshot for a bench run."""
    info: dict = {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "logical_cpus": os.cpu_count(),
        "physical_cpus": _physical_cpus(),
        "ram_gib": _ram_gib(),
    }

    cpu_brand = None
    cpu_microarch = None
    try:
        import cpuinfo  # py-cpuinfo

        ci = cpuinfo.get_cpu_info()
        cpu_brand = ci.get("brand_raw") or None
        info["cpu_hz_advertised"] = ci.get("hz_advertised_friendly")
        info["cpu_count_cpuinfo"] = ci.get("count")
        fam, mod, step = ci.get("family"), ci.get("model"), ci.get("stepping")
        if fam is not None:
            # x86 microarch identity (codename derivable from family/model).
            cpu_microarch = f"{ci.get('arch')} family={fam} model={mod} stepping={step}"
        flags = ci.get("flags") or []
        info["cpu_simd"] = sorted(f for f in flags if f in _SIMD_FLAGS)
    except Exception as e:  # noqa: BLE001 - cpuinfo is best-effort
        info["cpu_info_error"] = repr(e)

    info["cpu_brand"] = cpu_brand or _cpu_brand_fallback()
    # On Apple Silicon / ARM there is no x86 family/model; the brand string (e.g.
    # "Apple M4 Pro") plus arch is the microarch identity.
    info["cpu_microarch"] = (
        cpu_microarch or f"{info['cpu_brand']} ({platform.machine()})"
    )
    info["packages"] = _package_versions()
    return info


def write_and_log(out_dir, extra: dict | None = None) -> dict:
    """Compute system_info, print a short summary, write system_info.json.

    Returns the full info dict (callers may copy fields into result rows).
    """
    info = system_info()
    if extra:
        info.update(extra)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "system_info.json"
    out_path.write_text(json.dumps(info, indent=2, default=str))

    print("System info:")
    for k in (
        "cpu_brand",
        "cpu_microarch",
        "logical_cpus",
        "physical_cpus",
        "ram_gib",
        "platform",
        "python",
    ):
        print(f"  {k}: {info.get(k)}")
    print(f"  full -> {out_path}")
    return info


if __name__ == "__main__":
    print(json.dumps(system_info(), indent=2, default=str))
