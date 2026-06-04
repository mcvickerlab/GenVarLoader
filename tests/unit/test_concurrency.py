"""Concurrency regression tests for atomic cache + dataset creation (issue #21).

These tests prove that N processes racing to build the same .gvlfa cache or
write to the same dataset path produce exactly one valid artifact with no
orphan .tmp.* / .old.* directories left behind.
"""

import multiprocessing as mp
import shutil
from pathlib import Path

import numpy as np
import pytest

import genvarloader._fasta_cache as fc

# Use spawn so workers re-import cleanly regardless of host start method.
_CTX = mp.get_context("spawn")


# ---------------------------------------------------------------------------
# Worker functions (must be module-level for pickle under spawn)
# ---------------------------------------------------------------------------


def _build_cache_worker(src_str):
    import genvarloader._fasta_cache as _fc

    _fc.ensure_cache(Path(src_str))


def _write_worker(path_str, vcf_str, bed_dict):
    import polars as pl

    import genvarloader as gvl

    bed = pl.DataFrame(bed_dict)
    gvl.write(path=Path(path_str), bed=bed, variants=vcf_str, overwrite=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_concurrent_ensure_cache_no_corruption(tmp_path, ref_fasta):
    """N concurrent processes building the same .gvlfa cache produce a valid,
    byte-identical result with no orphan .tmp.* / .old.* directories."""
    src = tmp_path / "ref.fa.bgz"
    shutil.copy(ref_fasta, src)
    shutil.copy(str(ref_fasta) + ".fai", str(src) + ".fai")
    if Path(str(ref_fasta) + ".gzi").exists():
        shutil.copy(str(ref_fasta) + ".gzi", str(src) + ".gzi")

    # Single-process reference build in an isolated sub-directory
    ref_dir = tmp_path / "single"
    ref_dir.mkdir()
    ref_copy = ref_dir / "ref.fa.bgz"
    shutil.copy(src, ref_copy)
    shutil.copy(str(src) + ".fai", str(ref_copy) + ".fai")
    if Path(str(src) + ".gzi").exists():
        shutil.copy(str(src) + ".gzi", str(ref_copy) + ".gzi")
    _meta, single_data = fc.ensure_cache(ref_copy)
    expected = np.array(np.memmap(single_data, np.uint8, "r"))

    # N concurrent builders against the same source path
    procs = [
        _CTX.Process(target=_build_cache_worker, args=(str(src),)) for _ in range(6)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0, f"worker exited with code {p.exitcode}"

    _meta2, data = fc.ensure_cache(src)
    got = np.array(np.memmap(data, np.uint8, "r"))
    np.testing.assert_array_equal(got, expected)

    # No orphan temp / old dirs published beside the cache
    assert list(tmp_path.glob("ref.fa.bgz.gvlfa.tmp.*")) == []
    assert list(tmp_path.glob("ref.fa.bgz.gvlfa.old.*")) == []


@pytest.mark.slow
def test_concurrent_gvl_write_one_valid_dataset(tmp_path, synthetic_case, reference):
    """N concurrent processes writing to the same dataset path (overwrite=True)
    leave exactly one valid, openable dataset with no orphan dirs."""
    import polars as pl

    import genvarloader as gvl

    vcf = str(synthetic_case.vcf_path)

    # Build the bed dict using the same column names gvl.write expects,
    # mirroring the pattern in tests/unit/dataset/test_write_atomic.py.
    bed = synthetic_case.regions.select(
        chrom=pl.col("chrom"),
        chromStart=pl.col("start"),
        chromEnd=pl.col("end"),
    ).head(2)

    bed_dict = bed.to_dict(as_series=False)

    dest = tmp_path / "shared.gvl"

    procs = [
        _CTX.Process(target=_write_worker, args=(str(dest), vcf, bed_dict))
        for _ in range(4)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=180)
        assert p.exitcode == 0, f"worker exited with code {p.exitcode}"

    # Exactly one valid dataset published; no orphan temp / old dirs
    assert dest.is_dir(), "destination dataset directory does not exist"
    assert list(tmp_path.glob("shared.gvl.tmp.*")) == []
    assert list(tmp_path.glob("shared.gvl.old.*")) == []

    ds = gvl.Dataset.open(dest, reference=reference)
    assert len(ds) > 0
