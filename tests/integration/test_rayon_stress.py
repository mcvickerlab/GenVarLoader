"""Stress reproducer for issue #263: concurrent spawn workers iterating a
Dataset under forced-parallel + oversubscribed rayon must not deadlock.

A hang (futex-parked workers, per #263) surfaces as future.result(timeout=...)
raising TimeoutError → test failure. Clean completion across repeated launches
is evidence the cause was oversubscription (fixed by the cap_threads changes).
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FTimeoutError
from pathlib import Path

import numpy as np
import pyBigWig
import pytest
from genoray import VCF

import genvarloader as gvl

pytestmark = pytest.mark.slow

# Tuning: if a slow CI box trips the timeout WITHOUT a real hang, lower
# ITERS_PER_WORKER/LAUNCHES or raise PER_LAUNCH_TIMEOUT_S — never remove the
# timeout (it is the deadlock detector). Keep N_WORKERS*RAYON_NUM_THREADS > cores.
N_WORKERS = 5
ITERS_PER_WORKER = 40
LAUNCHES = 4
PER_LAUNCH_TIMEOUT_S = 120


def _iterate_dataset(ds_path: str, reference_path: str, iters: int) -> int:
    """Worker body (must be importable/picklable for spawn). Returns bytes touched."""
    # Force the parallel path and oversubscribe: many rayon threads per worker.
    os.environ["GVL_FORCE_PARALLEL"] = "1"
    os.environ["RAYON_NUM_THREADS"] = "8"
    ds = gvl.Dataset.open(Path(ds_path), reference=Path(reference_path)).with_seqs(
        "haplotypes"
    )
    total = 0
    n = len(ds)
    for _ in range(iters):
        out = ds[:n, :]
        first = out[0] if isinstance(out, tuple) else out
        total += int(np.asarray(getattr(first, "data", first)).size)
    return total


@pytest.fixture()
def stress_dataset(source_bed, vcf_dir, reference, tmp_path: Path) -> tuple[Path, Path]:
    contig_sizes = [("chr1", 2_000_000), ("chr2", 2_000_000)]
    bw_paths: dict[str, str] = {}
    for i, sample in enumerate(["s0", "s1", "s2"]):
        bw_path = tmp_path / f"{sample}.bw"
        with pyBigWig.open(str(bw_path), "w") as bw:
            bw.addHeader(contig_sizes, maxZooms=0)
            v = float(i + 1)
            bw.addEntries(
                ["chr1", "chr2"],
                [499_990, 17_320],
                ends=[500_030, 17_340],
                values=[v, v],
            )
        bw_paths[sample] = str(bw_path)
    out = tmp_path / "stress.gvl"
    gvl.write(
        path=out,
        bed=source_bed,
        variants=VCF(vcf_dir / "filtered_source.vcf.gz"),
        tracks=gvl.BigWigs("5ss", bw_paths),
        max_jitter=2,
    )
    return out, reference.path


def test_concurrent_spawn_workers_do_not_deadlock(stress_dataset):
    ds_path, ref_path = stress_dataset
    ctx = mp.get_context("spawn")
    for launch in range(LAUNCHES):
        with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx) as ex:
            futs = [
                ex.submit(
                    _iterate_dataset, str(ds_path), str(ref_path), ITERS_PER_WORKER
                )
                for _ in range(N_WORKERS)
            ]
            try:
                results = [f.result(timeout=PER_LAUNCH_TIMEOUT_S) for f in futs]
            except FTimeoutError:
                pytest.fail(
                    f"launch {launch}: worker did not finish within "
                    f"{PER_LAUNCH_TIMEOUT_S}s — likely the #263 rayon deadlock."
                )
                raise  # unreachable (pytest.fail raises); marks branch NoReturn for pyrefly
            assert all(r > 0 for r in results)
