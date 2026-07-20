"""Cold-cache A-vs-C measurement harness (issue #283).

Compares Design A (the landed producer-thread `Svar1StreamEngine`, the default
``_prefetch_strategy="engine"``) against Design C (single-thread read-ahead-one-
window, ``_prefetch_strategy="readahead"``) for a full
``list(sds.to_iter(batch_size=...))`` sweep over a SVAR1 store. This harness only
MEASURES -- it does not pick a winner; PR 2 ships whichever design a cold-cache
measurement favors (the controller reads this harness's output and makes that call
in a separate task, per ``.superpowers/sdd/task-9-design-note.md``).

COLD CACHE, NO ROOT
--------------------
This harness does NOT (and, unprivileged, cannot) drop the kernel page cache via
``echo 3 > /proc/sys/vm/drop_caches``. Instead every timed run gets its OWN
freshly-built ``.svar`` store (new inode, under a fresh ``tempfile.mkdtemp()``
directory, never read by this process before the timed sweep touches it) --
"cold" here means "never-faulted", not "guaranteed absent from the page cache".
A store's own build (writing the VCF/BCF/``.svar`` files) does populate the page
cache with the pages it just wrote, so a store that is too small can still read
back partially warm. To make the timed read dominated by genuinely cold I/O
rather than a warm hit off the build's own write-back cache, size the store
(``--n-variants`` x ``--n-samples`` x ``--contig-len``) so a full sweep's variant
footprint is large relative to typical page-cache residency -- the CLI defaults
aim for "thousands of variants, hundreds of samples, a contig long enough to
spread many read windows across it", per the design note; bump the flags up for a
more convincing cold-cache measurement on a quieter box.

NODE IS SHARED / NOISY
-----------------------
This runs on a shared, noisy node (see the project's perf-gate convention in
CLAUDE.md / docs/roadmaps/streaming-dataset.md). Report best-of-N (N>=3,
``--repeats``) per strategy, not a single sample, and treat the reported ratio as
secondary color, NOT a pass/fail gate.

Run:
    pixi run -e dev python benchmarking/streaming/cold_cache_overlap.py
    pixi run -e dev python benchmarking/streaming/cold_cache_overlap.py --help
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl

import genvarloader as gvl

_CONTIG = "chr1"


def _make_vcf(
    path: Path, n_variants: int, n_samples: int, contig_len: int, seed: int
) -> None:
    """A SNP-only, biallelic VCF -- same shape as `test_streaming_scale.py`'s
    fixture, just parameterized. SNP-only keeps haplotype length independent of
    genotype, which isn't needed here (this harness doesn't check parity) but
    keeps generation simple and fast."""
    if n_variants > contig_len - 4:
        raise ValueError(
            f"n_variants ({n_variants}) must be < contig_len - 4 ({contig_len - 4})"
        )
    rng = np.random.default_rng(seed)
    header = [
        "##fileformat=VCFv4.2",
        f"##contig=<ID={_CONTIG},length={contig_len}>",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        + "\t".join(f"S{i}" for i in range(n_samples)),
    ]
    positions = np.sort(
        rng.choice(np.arange(2, contig_len - 2), n_variants, replace=False)
    )
    # Vectorized genotype generation: (n_variants, n_samples, 2) alleles -> "a|b"
    # strings, joined per-row. A per-cell Python loop over n_variants*n_samples
    # (which can be in the hundreds of thousands to millions at realistic sizes)
    # is slow enough to matter here.
    alleles = rng.integers(0, 2, size=(n_variants, n_samples, 2))
    gt_rows = np.char.add(
        np.char.add(alleles[:, :, 0].astype("U1"), "|"), alleles[:, :, 1].astype("U1")
    )
    lines = header
    for i, pos in enumerate(positions):
        lines.append(
            f"{_CONTIG}\t{pos}\t.\tA\tG\t.\t.\t.\tGT\t" + "\t".join(gt_rows[i])
        )
    path.write_text("\n".join(lines) + "\n")


def _build_store(
    tmp_dir: Path, n_variants: int, n_samples: int, contig_len: int, seed: int
) -> tuple[Path, Path]:
    """Build a FRESH reference FASTA + SparseVar (.svar) store under `tmp_dir`.
    `tmp_dir` must be a directory this process has never read from before
    (a fresh `tempfile.mkdtemp()` per timed run) -- see the module docstring."""
    from genoray import VCF, SparseVar

    ref = tmp_dir / "ref.fa"
    rng = np.random.default_rng(seed + 1)
    seq = "".join(rng.choice(list("ACGT"), contig_len))
    ref.write_text(f">{_CONTIG}\n{seq}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True, capture_output=True)

    vcf = tmp_dir / "in.vcf"
    _make_vcf(vcf, n_variants, n_samples, contig_len, seed)
    bcf = tmp_dir / "in.bcf"
    subprocess.run(
        ["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)],
        check=True,
        capture_output=True,
    )
    subprocess.run(["bcftools", "index", str(bcf)], check=True, capture_output=True)

    svar = tmp_dir / "store.svar"
    SparseVar.from_vcf(
        svar,
        VCF(bcf),
        max_mem="1g",
        samples=[f"S{i}" for i in range(n_samples)],
        overwrite=True,
    )
    return svar, ref


def _make_bed(n_regions: int, region_len: int, contig_len: int) -> pl.DataFrame:
    """`n_regions` non-overlapping regions spread across the contig, so a full
    sweep spans many read windows (the window plan chunks both regions and
    samples -- see `StreamingDataset._plan`)."""
    if n_regions * region_len > contig_len:
        raise ValueError(
            f"n_regions * region_len ({n_regions * region_len}) exceeds "
            f"contig_len ({contig_len}); reduce --n-regions or --region-len, or "
            "raise --contig-len"
        )
    starts = np.linspace(0, contig_len - region_len, n_regions).astype(np.int64)
    starts = np.unique(starts)
    return pl.DataFrame(
        {
            "chrom": [_CONTIG] * len(starts),
            "chromStart": starts,
            "chromEnd": starts + region_len,
        }
    )


def _time_sweep(
    bed: pl.DataFrame,
    ref: Path,
    svar: Path,
    batch_size: int,
    max_mem: str,
    strategy: str,
) -> float:
    """Time one full `list(sds.to_iter(...))` sweep under `strategy`. Construction
    (which reads the store's static variant table) happens OUTSIDE the timed
    region -- only the sweep itself (window reads + prefetch + generation) is
    timed, matching what a training loop actually pays per epoch."""
    sds = gvl.StreamingDataset(
        bed, reference=ref, variants=svar, max_mem=max_mem
    ).with_seqs("haplotypes")
    object.__setattr__(sds, "_prefetch_strategy", strategy)

    t0 = time.perf_counter()
    n_rows = 0
    for _data, r_idx, _s_idx in sds.to_iter(batch_size=batch_size):
        n_rows += len(r_idx)
    elapsed = time.perf_counter() - t0
    assert n_rows == sds.shape[0] * sds.shape[1], (
        f"sweep yielded {n_rows} rows, expected {sds.shape[0] * sds.shape[1]} "
        "(shape[0]*shape[1]) -- harness is not exercising the full dataset"
    )
    return elapsed


def _time_readahead_with_overlap_proxy(
    bed: pl.DataFrame, ref: Path, svar: Path, batch_size: int, max_mem: str
) -> tuple[float, float, float]:
    """Re-drives the SAME Design C loop `_iter_batches` runs (read_window /
    svar1_prefetch_runs / generate_batch), but with timers around the prefetch and
    generate calls separately, to give a CRUDE proxy for how much of the total
    time is prefetch vs. generation. This is NOT a measurement of cross-thread
    overlap (Design C is single-threaded) -- it can only show whether the kernel's
    own readahead had time to act between the prefetch fold and the following
    generate() call touching the same pages; treat it as color, not a hard number.
    Returns (total_wall, prefetch_time_sum, generate_time_sum).
    """
    from genvarloader.genvarloader import svar1_prefetch_runs

    sds = gvl.StreamingDataset(
        bed, reference=ref, variants=svar, max_mem=max_mem
    ).with_seqs("haplotypes")
    backend = sds._backend
    assert backend is not None

    plan = list(sds._plan())
    t_prefetch = 0.0
    t_generate = 0.0
    t0 = time.perf_counter()
    if plan:
        cur = backend.read_window(*plan[0])
        for i, (r_idx, s_idx) in enumerate(plan):
            if i + 1 < len(plan):
                nxt = backend.read_window(*plan[i + 1])
                t_pf0 = time.perf_counter()
                svar1_prefetch_runs(backend._store, nxt[0], nxt[1])
                t_prefetch += time.perf_counter() - t_pf0
            else:
                nxt = None
            n_rows = len(r_idx) * len(s_idx)
            for lo in range(0, n_rows, batch_size):
                hi = min(lo + batch_size, n_rows)
                t_g0 = time.perf_counter()
                backend.generate_batch(r_idx, s_idx, cur[0], cur[1], lo, hi, -1)
                t_generate += time.perf_counter() - t_g0
            cur = nxt
    total = time.perf_counter() - t0
    return total, t_prefetch, t_generate


def _fmt(vals: list[float]) -> str:
    return f"best={min(vals):.3f}s median={sorted(vals)[len(vals) // 2]:.3f}s"


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Cold-cache A-vs-C measurement for StreamingDataset's SVAR1 prefetch "
            "strategies (engine vs. readahead), issue #283. See the module "
            "docstring for the cold-cache-without-root methodology and its caveats."
        ),
    )
    p.add_argument("--n-variants", type=int, default=4000, help="variants in the store")
    p.add_argument("--n-samples", type=int, default=300, help="samples in the store")
    p.add_argument(
        "--contig-len", type=int, default=1_000_000, help="contig length (bp)"
    )
    p.add_argument(
        "--n-regions", type=int, default=128, help="BED regions swept per run"
    )
    p.add_argument(
        "--region-len", type=int, default=500, help="length of each BED region (bp)"
    )
    p.add_argument("--batch-size", type=int, default=64, help="to_iter batch_size")
    p.add_argument(
        "--max-mem", type=str, default="64MB", help="StreamingDataset max_mem"
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="timed runs per strategy (best-of-N reported); each run builds its "
        "OWN fresh store, so N>=1 always costs N store builds per strategy",
    )
    p.add_argument("--seed", type=int, default=0, help="base RNG seed")
    p.add_argument(
        "--overlap-proxy",
        action="store_true",
        help="also report the crude prefetch-vs-generate timing breakdown for "
        "the readahead strategy (re-drives the sweep once more per repeat)",
    )
    p.add_argument(
        "--keep-tmp",
        action="store_true",
        help="keep the per-run tmp dirs (stores) instead of deleting them after "
        "each timed run; useful for inspecting store size / debugging",
    )
    args = p.parse_args()

    print(
        f"cold_cache_overlap: n_variants={args.n_variants} n_samples={args.n_samples} "
        f"contig_len={args.contig_len} n_regions={args.n_regions} "
        f"region_len={args.region_len} batch_size={args.batch_size} "
        f"max_mem={args.max_mem} repeats={args.repeats}"
    )
    print(
        "NOTE: this node is shared/noisy -- best-of-N per strategy is reported; "
        "treat the ratio as secondary color, not a pass/fail gate.\n"
    )

    strategies = ["engine", "readahead"]
    # FIXED per-strategy seed offset (Minor 2): `hash(strategy)` is randomized per
    # process by PYTHONHASHSEED, so it made engine vs. readahead see differently-seeded
    # stores run-to-run. A stable offset keeps each strategy's per-rep store
    # deterministic while still differing between strategies.
    strategy_seed_offset = {s: i * 50 for i, s in enumerate(strategies)}
    timings: dict[str, list[float]] = {s: [] for s in strategies}
    proxy_rows: list[tuple[float, float, float]] = []

    kept_dirs: list[Path] = []
    for rep in range(args.repeats):
        for strategy in strategies:
            tmp_dir = Path(tempfile.mkdtemp(prefix="gvl_cold_cache_"))
            try:
                svar, ref = _build_store(
                    tmp_dir,
                    args.n_variants,
                    args.n_samples,
                    args.contig_len,
                    seed=args.seed + rep * 1000 + strategy_seed_offset[strategy],
                )
                bed = _make_bed(args.n_regions, args.region_len, args.contig_len)

                if strategy == "readahead" and args.overlap_proxy:
                    elapsed, t_pf, t_gen = _time_readahead_with_overlap_proxy(
                        bed, ref, svar, args.batch_size, args.max_mem
                    )
                    proxy_rows.append((elapsed, t_pf, t_gen))
                else:
                    elapsed = _time_sweep(
                        bed, ref, svar, args.batch_size, args.max_mem, strategy
                    )
            finally:
                if args.keep_tmp:
                    kept_dirs.append(tmp_dir)
                else:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

            timings[strategy].append(elapsed)
            print(f"[rep {rep + 1}/{args.repeats}] {strategy:>10s}: {elapsed:.3f}s")

    print()
    print("=" * 60)
    print(f"{'strategy':>10s}  {'runs (s)':<40s}")
    print("-" * 60)
    for strategy in strategies:
        vals = timings[strategy]
        runs_str = " ".join(f"{v:.3f}" for v in vals)
        print(f"{strategy:>10s}  [{runs_str}]  ({_fmt(vals)})")
    print("-" * 60)

    best_engine = min(timings["engine"])
    best_readahead = min(timings["readahead"])
    ratio_e_over_c = (
        best_engine / best_readahead if best_readahead > 0 else float("nan")
    )
    ratio_c_over_e = best_readahead / best_engine if best_engine > 0 else float("nan")
    print(
        f"best-of-{args.repeats}: engine={best_engine:.3f}s readahead={best_readahead:.3f}s"
    )
    print(
        f"ratio: engine/readahead={ratio_e_over_c:.3f}  "
        f"readahead/engine={ratio_c_over_e:.3f}"
    )
    print("=" * 60)

    if proxy_rows:
        print()
        print(
            "readahead crude overlap proxy (prefetch-call time vs. generate-call "
            "time; single-threaded, see docstring caveat):"
        )
        for i, (total, t_pf, t_gen) in enumerate(proxy_rows):
            other = total - t_pf - t_gen
            print(
                f"  rep {i + 1}: total={total:.3f}s prefetch={t_pf:.3f}s "
                f"generate={t_gen:.3f}s other/overhead={other:.3f}s"
            )

    if kept_dirs:
        print()
        print("kept tmp dirs (--keep-tmp):")
        for d in kept_dirs:
            print(f"  {d}")


if __name__ == "__main__":
    sys.exit(main())
