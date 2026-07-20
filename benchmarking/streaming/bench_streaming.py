"""Cohort-size sweep for the streaming VCF/PGEN backends (issue #276 Task 13).

For each requested sample count `--n`, drives a `StreamingDataset(...).to_iter(
batch_size=...)` full sweep over a `vcfixture bulk`-generated cohort (see
`gen_fixtures.sh` / the `gen-bench-vcf`/`gen-bench-pgen` pixi tasks) and compares
the **engine** strategy (the shipped `RecordStreamEngine` producer/consumer --
window N+1 decodes on a background thread while window N's batches are
generated) against a **forced-synchronous** baseline that rebuilds a
single-window engine per plan step, so no window's decode can ever overlap the
previous window's generation. This is the same "A vs. a no-overlap baseline"
shape the SVAR1 effort used (`cold_cache_overlap.py`, issue #283/#296) --
there is no separate "readahead" design for VCF/PGEN (`_VcfBackend`/
`_PgenBackend` support only the "engine" prefetch strategy; see
`_dataset/_streaming.py`'s backend docstrings), so "synchronous" here is
constructed at the harness level by NOT reusing one engine across windows,
rather than by an internal toggle.

METHODOLOGY -- READ BEFORE INTERPRETING NUMBERS
-------------------------------------------------
This runs on a shared, noisy node (see the project's perf-gate convention:
CLAUDE.md / docs/roadmaps/streaming-dataset.md / the
`gvl-rust-perf-gate-shared-node-noise` memory). Wall-clock timings
(windows/s, items/s) are printed as SECONDARY color, best/median-of-N
(`--repeats`), never a pass/fail gate. The load-bearing signals are the
DETERMINISTIC counters printed alongside them:
  - n_windows      -- read-window count the plan drives (region/sample-chunk
                       count from `StreamingDataset._plan`)
  - n_batches      -- `to_iter` batches yielded
  - n_rows         -- (region, sample) cells processed; must equal
                       shape[0]*shape[1] for both strategies (a harness
                       correctness check, not just "didn't crash")
  - bytes_emitted  -- total haplotype bytes yielded (deterministic given a
                       fixed seed/input; a wrong batch-generation change
                       would change this even if the byte-diff never crashes)
  - peak_rss_kb    -- `resource.getrusage(RUSAGE_SELF).ru_maxrss`, the
                       process-wide high-water mark (monotonic; read once at
                       the end of a run)
Same-session before/after comparisons (engine vs. sync, or this run vs. a
prior run of THIS script in the SAME session) are the only trustworthy
comparison on this node -- do not compare an absolute number here against a
number from a different session or a different, quieter machine.

PGEN DECODE VOLUME (Task 4 narrowing landed -- measured via `pgen_variants_decoded`)
-------------------------------------------------------------------------------------
`PgenWindowFiller` (`src/record_stream/pgen.rs`) used to re-scan the WHOLE
contig's variant prefix from record 0 on every window's fill (the old
"coarse `var_start`" behavior). That has since been narrowed: each window now
binary-searches a per-contig POS index built once (in `PgenWindowFiller::new`)
to compute a tight `[var_start, var_end)` range, padded by the contig's max
REF length so a spanning deletion upstream of the window is never dropped --
see that module's "Narrowed `var_start`/`var_end`" doc section for the full
correctness argument (over-inclusion is safe via `OverlapMode::Variant`;
under-inclusion would be a bug, hence the pad).

The deterministic gate for this is the Rust-side `pgen_variants_decoded`
counter (`genvarloader.genvarloader.pgen_variants_decoded()`, reset via
`pgen_variants_decoded_reset()`), exercised in
`tests/dataset/test_streaming_scale.py`. This harness does not itself read
that counter (it would need to import the extension module directly, out of
scope for a benchmark harness) -- for reference, at N=10000
(`--n-regions 200 --region-len 200`, 4 windows) it dropped from 400,032
(= 4 x 100,008 pvar variants, the old coarse-prefix-per-window total, exactly
`n_windows * pvar_variants`) to 98,583 (~1x total variant count -- each
window now decodes close to only its own range). See
`docs/roadmaps/streaming-optimization-baseline.md` (before) and
`docs/roadmaps/streaming-dataset.md`'s "Optimization pass (#276) results"
section (after) for the full measurement. The `n_windows`/`pvar_variants`
print below is now a coarse cross-check only, not evidence of a whole-prefix
re-decode.

COMPARE-DATASET (optional, `--compare-dataset`)
-------------------------------------------------
`gvl.write` rejects symbolic/breakend variants that `StreamingDataset` reads
without complaint, and the bench fixture's `germline-1kgp` profile emits
symbolic SVs. So `--compare-dataset` runs ALL drivers (engine, sync, AND the
written-Dataset sweep) on a bcftools/plink2-filtered, symbolic-SV-free
variant set for the whole `bench_one` invocation -- the numbers under
`--compare-dataset` are on this common denominator, not the raw fixture, and
the `bytes_emitted` cross-check at the end of `bench_one` is only meaningful
because every driver saw the same input.

COLD CACHE (optional, `--cold`)
---------------------------------
Like `cold_cache_overlap.py`: unprivileged, so "cold" means "never-faulted by
THIS process", not "guaranteed absent from the page cache" -- `--cold` copies
the fixture files into a fresh `tempfile.mkdtemp()` directory (never read by
this process before) ahead of each timed run, so producer-thread I/O overlap
isn't conflated with an already-warm page cache from a prior run/repeat.

Run:
    pixi run -e dev python benchmarking/streaming/bench_streaming.py --n 1000
    pixi run -e dev python benchmarking/streaming/bench_streaming.py --help
"""

from __future__ import annotations

import argparse
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import polars as pl

import genvarloader as gvl

_FIXTURE_DIR = Path(__file__).parent
_CONTIG_RE = re.compile(r"##contig=<ID=([^,>]+),length=(\d+)>")


# --------------------------------------------------------------------------- #
# Fixture / reference plumbing
# --------------------------------------------------------------------------- #


def _contig_lengths(vcf_or_bcf: Path) -> dict[str, int]:
    """Parse `##contig=<ID=...,length=...>` header lines. `vcfixture bulk`
    sets `length` to the contig's POPULATED SPAN (max POS actually written),
    not a real chromosome length (see vcfixture-rs `src/bulk/mod.rs`) -- so
    this is the exact span a synthetic reference needs to cover, not a guess.
    """
    out = subprocess.run(
        ["bcftools", "view", "-h", str(vcf_or_bcf)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    lens = {m.group(1): int(m.group(2)) for m in _CONTIG_RE.finditer(out)}
    if not lens:
        raise ValueError(f"no ##contig header lines found in {vcf_or_bcf}")
    return lens


def _build_reference(
    contig_lengths: dict[str, int], out_fasta: Path, seed: int
) -> Path:
    """Synthesize a random-sequence FASTA covering each contig's populated
    span (+ a small pad past the last variant, for indel/pad-char reach at
    the contig end). Deterministic given `seed`; skipped if already built
    (fixture reuse across repeats/strategies)."""
    if (
        out_fasta.exists()
        and (out_fasta.with_suffix(out_fasta.suffix + ".fai")).exists()
    ):
        return out_fasta
    rng = np.random.default_rng(seed)
    with out_fasta.open("w") as fh:
        for contig, length in contig_lengths.items():
            pad = 64
            seq = "".join(rng.choice(list("ACGT"), length + pad))
            fh.write(f">{contig}\n{seq}\n")
    subprocess.run(
        ["samtools", "faidx", str(out_fasta)], check=True, capture_output=True
    )
    return out_fasta


def _make_bed(
    contig_lengths: dict[str, int], n_regions: int, region_len: int
) -> pl.DataFrame:
    """`n_regions` non-overlapping regions per contig, spread across each
    contig's populated span -- enough regions to force multiple read windows
    (`StreamingDataset`'s default `_window_regions=64`) at any of the sweep's
    `--n` sample counts."""
    chroms: list[str] = []
    starts: list[int] = []
    for contig, length in contig_lengths.items():
        if n_regions * region_len > length:
            n = max(1, length // region_len)
        else:
            n = n_regions
        s = np.unique(np.linspace(0, max(length - region_len, 0), n).astype(np.int64))
        chroms.extend([contig] * len(s))
        starts.extend(s.tolist())
    starts_arr = np.asarray(starts, dtype=np.int64)
    return pl.DataFrame(
        {
            "chrom": chroms,
            "chromStart": starts_arr,
            "chromEnd": starts_arr + region_len,
        }
    )


def _count_pvar_variants(pgen_path: Path) -> int:
    """Non-header (`^#`) line count in the sibling `.pvar` -- the size of the
    prefix `PgenWindowFiller` re-scans on every window (see module docstring).
    """
    pvar = pgen_path.with_suffix(".pvar")
    out = subprocess.run(
        ["grep", "-vc", "^#", str(pvar)], capture_output=True, text=True
    )
    # grep exits 1 if there are zero matching lines; treat that as 0, not an error.
    return int(out.stdout.strip() or "0")


def _filtered_variants_for_dataset(kind: str, fixture_dir: Path, n: int) -> Path:
    """`gvl.write` (via `_write.py`'s `_reject_unsupported_variants`) requires
    bi-allelic, non-symbolic, non-breakend variants; `StreamingDataset` has no
    such check and reads whatever the fixture contains. `vcfixture bulk`'s
    `germline-1kgp` profile emits symbolic SVs (`<DEL>`/`<INS>`), so a raw
    `gvl.write` call on the bench fixture raises `ValueError`. Derive a
    filtered variant source ONCE (cached under `fixture_dir`, like
    `_build_reference`) and use it for EVERY driver in the `--compare-dataset`
    arm -- engine, sync, AND dataset -- so all sweeps see the IDENTICAL
    variant set -- byte-identical parity (and the `bytes_emitted` cross-check)
    requires the same input, not just the same (region, sample) cells."""
    filt_bcf = fixture_dir / f"bench_{n}.filtered.bcf"
    if not filt_bcf.exists():
        src_bcf = fixture_dir / f"bench_{n}.bcf"
        subprocess.run(
            [
                "bcftools",
                "view",
                "-m2",
                "-M2",
                "-v",
                "snps,indels",
                "-Ob",
                "-o",
                str(filt_bcf),
                str(src_bcf),
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["bcftools", "index", str(filt_bcf)], check=True, capture_output=True
        )
    if kind == "vcf":
        return filt_bcf

    filt_pgen = fixture_dir / f"bench_{n}.filtered.pgen"
    if not filt_pgen.exists():
        subprocess.run(
            [
                "plink2",
                "--bcf",
                str(filt_bcf),
                "--make-pgen",
                "--allow-extra-chr",
                "--output-chr",
                "chrM",
                "--out",
                str(fixture_dir / f"bench_{n}.filtered"),
            ],
            check=True,
            capture_output=True,
        )
    return filt_pgen


# --------------------------------------------------------------------------- #
# Sweep drivers
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class RunResult:
    elapsed_s: float
    n_windows: int
    n_batches: int
    n_rows: int
    bytes_emitted: int
    peak_rss_kb: int


def _drive_engine(sds: "gvl.StreamingDataset", batch_size: int) -> RunResult:
    """The shipped path: one `to_iter()` sweep, `_backend.build_engine` called
    ONCE for the whole plan -- the producer thread decodes window N+1 while
    the consumer generates window N's batches (`_streaming.py`'s "engine"
    `_iter_batches` branch)."""
    n_windows = sum(1 for _ in sds._plan())
    t0 = time.perf_counter()
    n_batches = 0
    n_rows = 0
    n_bytes = 0
    for data, r_idx, _s_idx in sds.to_iter(batch_size=batch_size):
        n_batches += 1
        n_rows += len(r_idx)
        n_bytes += data.data.nbytes
    elapsed = time.perf_counter() - t0
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return RunResult(elapsed, n_windows, n_batches, n_rows, n_bytes, peak_rss)


def _drive_sync(sds: "gvl.StreamingDataset", batch_size: int) -> RunResult:
    """Forced-synchronous baseline: rebuild a ONE-JOB engine per plan window
    and fully drain it before starting the next window's engine. Because a
    fresh producer thread can only start once `build_engine` is called (and
    the prior engine object is dropped once exhausted), no window's decode
    can overlap the previous window's consumption -- the cross-window
    pipelining `_drive_engine` gets from a single whole-plan engine is
    structurally absent here, without needing a Rust-side toggle (VCF/PGEN
    have no `read_window`/`generate_batch` split to drive by hand the way
    SVAR1's Design C does)."""
    backend = sds._backend
    assert backend is not None
    ploidy = backend.ploidy

    plan_jobs: list[tuple[int, np.ndarray, int, int]] = []
    for r_idx, s_idx in sds._plan():
        contig_idx = int(sds._regions[r_idx[0], 0])
        plan_jobs.append((contig_idx, r_idx, int(s_idx[0]), int(s_idx[-1]) + 1))

    t0 = time.perf_counter()
    n_batches = 0
    n_rows = 0
    n_bytes = 0
    for contig_idx, r_idx, s_lo, s_hi in plan_jobs:
        job = [
            (
                contig_idx,
                np.ascontiguousarray(sds._regions[r_idx, 1], np.uint32),
                np.ascontiguousarray(sds._regions[r_idx, 2], np.uint32),
                s_lo,
                s_hi,
            )
        ]
        engine = backend.build_engine(job, batch_size)
        while True:
            nxt = engine.next_batch()
            if nxt is None:
                break
            n_batches += 1
            data, offsets = nxt
            n_rows += (len(offsets) - 1) // ploidy
            n_bytes += np.asarray(data).nbytes
        del engine  # joins the producer thread before the next window starts
    elapsed = time.perf_counter() - t0
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return RunResult(elapsed, len(plan_jobs), n_batches, n_rows, n_bytes, peak_rss)


def _drive_dataset(
    sds: "gvl.StreamingDataset", fp: "FixturePaths", batch_size: int
) -> tuple[RunResult, float, int]:
    """Write a gvl.Dataset once from the SAME variants+reference+bed, then iterate
    it over the identical region-major (region, sample) plan order the streaming
    run uses (haplotypes-only, jitter=0). Returns (sweep RunResult, write_time_s,
    dataset_bytes). Write cost is preprocessing, reported separately."""
    del batch_size  # Dataset[r, s] access is not batched the way to_iter() is
    ds_dir = Path(tempfile.mkdtemp(prefix="gvl_bench_ds_"))
    try:
        t_w = time.perf_counter()
        gvl.write(
            path=ds_dir / "ds",
            bed=fp.bed,
            variants=fp.variants,
            max_jitter=0,
            overwrite=True,
        )
        write_time = time.perf_counter() - t_w
        ds_bytes = sum(
            f.stat().st_size for f in (ds_dir / "ds").rglob("*") if f.is_file()
        )

        ds = gvl.Dataset.open(ds_dir / "ds", reference=fp.reference).with_seqs(
            "haplotypes"
        )

        t0 = time.perf_counter()
        n_batches = n_rows = n_bytes = 0
        for r_idx, s_idx in sds._plan():
            # `ds[r_idx, s_idx]` with two equal-shaped integer arrays is numpy
            # "advanced" (fancy) indexing -- it pairs elements (zip-style) and
            # raises a broadcast error whenever len(r_idx) != len(s_idx), which
            # a real window plan always has (r_idx is the window's region
            # chunk, s_idx the full sample axis). `np.ix_` builds the outer
            # product instead, matching the FULL region-major (region, sample)
            # block the plan cell represents (verified empirically: shape
            # comes back as (len(r_idx), len(s_idx), ploidy, None)).
            haps = ds[
                np.ix_(r_idx, s_idx)
            ]  # region-major outer product, same plan cells
            n_batches += 1
            n_rows += len(r_idx) * len(s_idx)
            n_bytes += np.asarray(haps.data).nbytes
        elapsed = time.perf_counter() - t0
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return (
            RunResult(elapsed, n_batches, n_batches, n_rows, n_bytes, peak_rss),
            write_time,
            ds_bytes,
        )
    finally:
        shutil.rmtree(ds_dir, ignore_errors=True)


_DRIVERS = {"engine": _drive_engine, "sync": _drive_sync}


def _fmt(vals: list[float]) -> str:
    return f"best={min(vals):.3f}s median={sorted(vals)[len(vals) // 2]:.3f}s"


# --------------------------------------------------------------------------- #
# Per-N, per-backend bench
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class FixturePaths:
    variants: Path
    reference: Path
    bed: pl.DataFrame
    workdir_to_cleanup: Path | None = field(default=None)


def _prepare_fixture(
    kind: str,
    n: int,
    fixture_dir: Path,
    n_regions: int,
    region_len: int,
    ref_seed: int,
    cold: bool,
) -> FixturePaths:
    bcf_path = fixture_dir / f"bench_{n}.bcf"
    pgen_path = fixture_dir / f"bench_{n}.pgen"
    src_variants = bcf_path if kind == "vcf" else pgen_path
    if not src_variants.exists():
        gen_hint = "gen-bench-vcf" if kind == "vcf" else "gen-bench-pgen"
        raise FileNotFoundError(
            f"{src_variants} does not exist. Generate it first, e.g.:\n"
            f"  N={n} SIZE=10MB CONTIGS=chr1 pixi run {gen_hint}\n"
            "(gen-bench-pgen requires gen-bench-vcf's bench_<n>.bcf to already exist)."
        )

    contig_lengths = _contig_lengths(bcf_path)
    ref_path = fixture_dir / f"bench_{n}.ref.fa"
    _build_reference(contig_lengths, ref_path, seed=ref_seed)
    bed = _make_bed(contig_lengths, n_regions, region_len)

    if not cold:
        return FixturePaths(src_variants, ref_path, bed)

    # Cold-cache mode: copy this run's fixture files into a directory this
    # process has never read from before (see module docstring's caveat --
    # unprivileged, so "cold" means "never-faulted", not "cache-evicted").
    tmp_dir = Path(tempfile.mkdtemp(prefix="gvl_bench_streaming_cold_"))
    if kind == "vcf":
        cold_variants = tmp_dir / bcf_path.name
        shutil.copy2(bcf_path, cold_variants)
        csi_path = bcf_path.with_suffix(bcf_path.suffix + ".csi")
        if csi_path.exists():
            shutil.copy2(csi_path, tmp_dir / csi_path.name)
    else:
        cold_variants = tmp_dir / pgen_path.name
        for suffix in (".pgen", ".pvar", ".psam"):
            shutil.copy2(
                pgen_path.with_suffix(suffix), tmp_dir / (pgen_path.stem + suffix)
            )
    cold_ref = tmp_dir / ref_path.name
    shutil.copy2(ref_path, cold_ref)
    shutil.copy2(
        ref_path.with_suffix(ref_path.suffix + ".fai"),
        cold_ref.with_suffix(cold_ref.suffix + ".fai"),
    )
    return FixturePaths(cold_variants, cold_ref, bed, workdir_to_cleanup=tmp_dir)


def bench_one(
    kind: str,
    n: int,
    fixture_dir: Path,
    batch_size: int,
    repeats: int,
    strategies: list[str],
    n_regions: int,
    region_len: int,
    ref_seed: int,
    cold: bool,
    compare_dataset: bool,
) -> None:
    print(f"\n{'=' * 70}\n{kind.upper()} backend, N={n} samples\n{'=' * 70}")

    # `--compare-dataset` needs EVERY driver (engine, sync, AND dataset) to
    # consume the identical variant set, or the final bytes_emitted
    # cross-check compares a filtered-input driver against unfiltered-input
    # drivers -- a fairness violation that either spuriously raises or masks
    # a genuine divergence depending on whether symbolic SVs happen to land
    # in the sampled windows. Compute the filtered set ONCE, up front, before
    # any driver runs, and substitute it into every driver's fixture below.
    filtered_variants: Path | None = None
    if compare_dataset:
        filtered_variants = _filtered_variants_for_dataset(kind, fixture_dir, n)
        print(
            f"[compare-dataset] engine, sync, AND dataset will all read the "
            f"symbolic-SV-filtered variant set ({filtered_variants.name}) -- "
            "gvl.write rejects symbolic/breakend variants that StreamingDataset "
            "reads without complaint, so this is the common denominator that "
            "keeps the bytes_emitted cross-check apples-to-apples."
        )

    timings: dict[str, list[float]] = {s: [] for s in strategies}
    last_result: dict[str, RunResult] = {}
    for rep in range(repeats):
        for strategy in strategies:
            fp = _prepare_fixture(
                kind, n, fixture_dir, n_regions, region_len, ref_seed, cold
            )
            if filtered_variants is not None:
                fp = replace(fp, variants=filtered_variants)
            try:
                sds = gvl.StreamingDataset(
                    fp.bed, reference=fp.reference, variants=fp.variants
                ).with_seqs("haplotypes")
                result = _DRIVERS[strategy](sds, batch_size)
            finally:
                if fp.workdir_to_cleanup is not None:
                    shutil.rmtree(fp.workdir_to_cleanup, ignore_errors=True)

            expected_rows = sds.shape[0] * sds.shape[1]
            if result.n_rows != expected_rows:
                raise RuntimeError(
                    f"{kind}/{strategy} rep {rep + 1}: yielded {result.n_rows} rows, "
                    f"expected {expected_rows} (shape[0]*shape[1]) -- harness is not "
                    "exercising the full dataset"
                )

            timings[strategy].append(result.elapsed_s)
            last_result[strategy] = result
            print(
                f"[rep {rep + 1}/{repeats}] {strategy:>6s}: {result.elapsed_s:.3f}s  "
                f"windows={result.n_windows} batches={result.n_batches} "
                f"rows={result.n_rows} bytes={result.bytes_emitted} "
                f"peak_rss_kb={result.peak_rss_kb}"
            )

    if compare_dataset:
        assert filtered_variants is not None
        fp = _prepare_fixture(
            kind, n, fixture_dir, n_regions, region_len, ref_seed, cold
        )
        fp_ds = replace(fp, variants=filtered_variants)
        try:
            sds = gvl.StreamingDataset(
                fp_ds.bed, reference=fp_ds.reference, variants=fp_ds.variants
            ).with_seqs("haplotypes")
            ds_result, write_time, ds_bytes = _drive_dataset(sds, fp_ds, batch_size)
        finally:
            if fp.workdir_to_cleanup is not None:
                shutil.rmtree(fp.workdir_to_cleanup, ignore_errors=True)
        last_result["dataset"] = ds_result
        print(
            f"[dataset] sweep {ds_result.elapsed_s:.3f}s  rows={ds_result.n_rows} "
            f"bytes={ds_result.bytes_emitted} peak_rss_kb={ds_result.peak_rss_kb}  "
            f"| preprocessing: write={write_time:.3f}s disk={ds_bytes} bytes"
        )

    print("-" * 70)
    print(
        f"{'strategy':>8s}  {'windows/s (best)':>18s}  {'items/s (best)':>16s}  runs (s)"
    )
    for strategy in strategies:
        vals = timings[strategy]
        r = last_result[strategy]
        best = min(vals)
        windows_per_s = r.n_windows / best if best > 0 else float("nan")
        items_per_s = r.n_rows / best if best > 0 else float("nan")
        runs_str = " ".join(f"{v:.3f}" for v in vals)
        print(
            f"{strategy:>8s}  {windows_per_s:18.1f}  {items_per_s:16.1f}  "
            f"[{runs_str}]  ({_fmt(vals)})"
        )

    emitted = {k: r.bytes_emitted for k, r in last_result.items()}
    if len(set(emitted.values())) > 1:
        raise RuntimeError(
            f"{kind}: bytes_emitted diverged across drivers {emitted} — streaming "
            "and written Dataset must be byte-identical; a driver is not exercising "
            "the same cells"
        )

    if "engine" in timings and "sync" in timings:
        best_engine = min(timings["engine"])
        best_sync = min(timings["sync"])
        ratio = best_sync / best_engine if best_engine > 0 else float("nan")
        print(
            f"same-session ratio: sync/engine = {ratio:.3f} "
            "(>1 means the engine's cross-window overlap wins on THIS run; "
            "shared-node noise -- not a pass/fail gate, see module docstring)"
        )

    if kind == "pgen":
        pgen_path = fixture_dir / f"bench_{n}.pgen"
        pvar_variants = _count_pvar_variants(pgen_path)
        any_result = next(iter(last_result.values()))
        print(
            f"PGEN window plan: n_windows={any_result.n_windows}, "
            f"pvar_variants={pvar_variants} (coarse worst case would be "
            f"{any_result.n_windows * pvar_variants} decode-units; the shipped "
            "narrowed var_start/var_end -- see src/record_stream/pgen.rs's "
            "'Narrowed var_start/var_end' doc section and this script's module "
            "docstring -- keeps actual decode close to pvar_variants, not the "
            "product; genvarloader.genvarloader.pgen_variants_decoded() has the "
            "exact measured count, not read by this harness)."
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Cohort-size sweep for StreamingDataset's VCF/PGEN backends "
            "(engine vs. forced-synchronous baseline), issue #276 Task 13. "
            "See the module docstring for methodology, the PGEN var_start "
            "narrowing this pass shipped, and the --compare-dataset arm."
        ),
    )
    p.add_argument(
        "--n",
        type=int,
        action="append",
        required=True,
        help="sample count for a sweep point; repeat for a multi-point sweep "
        "(e.g. --n 1000 --n 5000 --n 50000). Fixtures must already exist "
        "(bench_<n>.bcf / bench_<n>.pgen under --fixture-dir), generated via "
        "the gen-bench-vcf/gen-bench-pgen pixi tasks.",
    )
    p.add_argument(
        "--fixture-dir",
        type=Path,
        default=_FIXTURE_DIR,
        help="directory containing bench_<n>.bcf / bench_<n>.pgen (default: "
        "this script's directory, matching gen_fixtures.sh's OUT_DIR default)",
    )
    p.add_argument(
        "--backend",
        choices=["vcf", "pgen", "both"],
        default="both",
        help="which backend(s) to sweep (measured in SEPARATE loops -- VCF's "
        "tabix-seekable decode and PGEN's non-GIL-free, window-narrowed decode "
        "have different overlap characteristics; never averaged together)",
    )
    p.add_argument(
        "--strategy",
        choices=["engine", "sync", "both"],
        default="both",
        help="engine (shipped, cross-window overlap) vs. sync (forced, no "
        "overlap) -- see module docstring",
    )
    p.add_argument(
        "--compare-dataset",
        action="store_true",
        help="also write a gvl.Dataset once (write time + on-disk size reported "
        "SEPARATELY as preprocessing cost) and time a full region-major "
        "Dataset[r,s] sweep over the identical (region, sample) cells, for a "
        "streaming-vs-written throughput comparison. gvl.write rejects symbolic/"
        "breakend variants, so in this mode ALL drivers (engine, sync, dataset) "
        "read a bcftools/plink2-filtered variant set (bi-allelic snps/indels "
        "only) instead of the raw fixture, keeping the bytes_emitted cross-check "
        "apples-to-apples",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="timed runs per strategy (best/median-of-N)",
    )
    p.add_argument(
        "--n-regions",
        type=int,
        default=200,
        help="BED regions per contig (spread across each contig's populated "
        "span) -- enough to force multiple read windows",
    )
    p.add_argument(
        "--region-len", type=int, default=200, help="length of each BED region (bp)"
    )
    p.add_argument(
        "--ref-seed", type=int, default=0, help="synthetic-reference RNG seed"
    )
    p.add_argument(
        "--cold",
        action="store_true",
        help="copy fixture files into a fresh never-read tmp dir before each "
        "timed run, so producer-thread I/O overlap isn't conflated with an "
        "already-warm page cache (see module docstring's cold-cache caveat)",
    )
    args = p.parse_args()

    strategies = ["engine", "sync"] if args.strategy == "both" else [args.strategy]
    backends = ["vcf", "pgen"] if args.backend == "both" else [args.backend]

    print(
        f"bench_streaming: n={args.n} backends={backends} strategies={strategies} "
        f"batch_size={args.batch_size} repeats={args.repeats} cold={args.cold}"
    )
    print(
        "NOTE: shared/noisy node -- wall-clock (windows/s, items/s) is secondary "
        "color, best/median-of-N; the deterministic counters (n_windows, "
        "n_batches, n_rows, bytes_emitted, peak_rss_kb) are the load-bearing "
        "signal. See module docstring.\n"
    )

    for n in args.n:
        for kind in backends:
            bench_one(
                kind,
                n,
                args.fixture_dir,
                args.batch_size,
                args.repeats,
                strategies,
                args.n_regions,
                args.region_len,
                args.ref_seed,
                args.cold,
                args.compare_dataset,
            )


if __name__ == "__main__":
    sys.exit(main())
