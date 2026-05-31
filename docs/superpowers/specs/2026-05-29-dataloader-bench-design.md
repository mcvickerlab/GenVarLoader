# DataLoader Bench — Design

**Status:** draft, awaiting user review
**Date:** 2026-05-29

## Motivation

The prefetching dataloader design (`2026-05-28-prefetching-dataloader-design.md`)
introduces two new `to_dataloader` modes — `"buffered"` and `"double_buffered"` —
alongside the existing `mode=None` baseline. We need a benchmark that:

1. **Compares the three modes head-to-head** under realistic-but-bounded
   conditions, so we can claim the new modes are wins and identify *where*
   each wins.
2. **Maps how throughput scales** with the four user-facing knobs (threads,
   region length, batch size, buffer bytes) per output mode, providing the
   data needed to inform a future auto-tuning heuristic.

The bench targets variant-data-only workloads (no BigWig/Table tracks). Total
wall time budget: **20 minutes** on a developer workstation.

## Goals & non-goals

In scope:

- Output modes: `haplotypes`, `annotated`, `variants` (drops `reference`)
- Modes: `None`, `"buffered"`, `"double_buffered"`
- Axes: threads, region length, batch size, buffer bytes
- 1KG chr21+chr22 sparse-var (.svar) dataset
- CSV results + single matplotlib summary plot

Out of scope:

- Tracks (BigWig, Table) — variant-only
- Spliced datasets
- VCF and PGEN backends (svar only, the fastest path)
- Multi-GPU / DDP
- Zero-copy (`copy=False`) variants — keep default `copy=True`
- Real GPU step / `.to(device)` — measure loader throughput only
- `pin_memory=True` — would mask `copy` semantics

## Directory layout

```
experiments/dataloader/
├── _common.py        # shared configs, dataset prep, replication helper
├── bench.py          # main entrypoint: configs → measurements → CSV
├── plot_results.py   # CSV → results_plot.png
├── results.csv       # output (committed only on intentional refresh)
└── results_plot.png  # output
```

Mirrors the layout of `experiments/bench_table_overlap/`.

## Data

**Source:** `tests/data/1kg/filtered.svar` (5 samples, chr21+chr22, ~287k
variants, ploidy 2). The committed `phased_1kg.svar.gvl` fixture is *not*
reused directly because its regions are baked at write time and the bench
varies region length.

**Region BED generation:** for each unique `region_length` value the bench
generates a BED by taking the 100 regions in `tests/data/1kg/regions.bed`,
recentering them on their midpoint, and resizing to the target length. The
resulting BED is written to a temp directory, then `gvl.write` produces a
fresh `.gvl` dataset at `tmp/dataset_rL{length}.gvl`.

**Dataset writes are amortized once at bench startup**, keyed by region
length. With 5 unique lengths (`{1_000, 2_500, 5_000, 10_000, 25_000}`),
this is 5 `gvl.write` calls; expected total ≈ tens of seconds.

**Per-cell epoch sizing:** the 1KG fixture has 5 samples × 100 regions =
500 (region, sample) instances per epoch. To amortize buffer fills the
bench iterates **≥3 epochs or ≥1.5 s wall time, whichever first**, after
one discarded warmup epoch.

## Parameter axes

| Axis | Factorial values | Fan values (added around midpoint) |
|---|---|---|
| `mode` | `None`, `"buffered"`, `"double_buffered"` | — |
| `with_seqs` | `"haplotypes"`, `"annotated"`, `"variants"` | — |
| `threads` | `{1, 8}` | `{2, 4, 16}` |
| `region_length` | `{1_000, 10_000}` | `{2_500, 5_000, 25_000}` |
| `batch_size` | `{16, 128}` | `{32, 64, 256}` |
| `buffer_bytes` | `{256 MiB, 2 GiB}` | `{512 MiB, 1 GiB, 4 GiB}` |

**Midpoints used during 1-axis fans:** `threads=4`, `region_length=5_000`,
`batch_size=64`, `buffer_bytes=1 GiB`. When fanning along axis A, the other
three axes are held at their midpoint.

`mode=None` does not accept `buffer_bytes`; baseline cells use `{1, 8}`
threads × `{16, 128}` batch × `{1_000, 10_000}` region only.

## Cell count

- New modes: 2 modes × 3 outputs × (2⁴ factorial + 4 axes × 3 fan values)
  = 2 × 3 × 28 = **168 cells**
- Baseline: 3 outputs × 2 threads × 2 batch × 2 region = **24 cells**
- **Total: 192 cells.**

At ~4 s/cell (warmup + ≥1.5 s measurement + loader teardown) this is
~13 min of measurement on top of ~1–3 min of dataset writes, fitting the
20 min budget.

## Measurement protocol

For each cell:

1. Construct `Dataset.open(rL_dataset_path).with_seqs(output_mode)`,
   apply `.with_settings(deterministic=True)` when needed by the spec
   (`haplotypes` / `annotated`).
2. Build the loader: `dataset.to_dataloader(batch_size=..., mode=...,
   buffer_bytes=..., shuffle=False, num_workers=0)`.
3. **Warmup:** iterate one full epoch, discard timing.
4. **Measure:** until `elapsed ≥ 1.5 s` *and* `epochs ≥ 3`:
   - `t0 = time.perf_counter()`
   - iterate one epoch, summing `instances` and `bytes`
     - `bytes` = `sum(_output_bytes_per_instance[r, s])` over the epoch's
       (r, s) pairs (cheap; the table is computed once per cell)
   - record per-epoch wall time
5. Record a single row aggregating across measured epochs.
6. Tear down loader; `gc.collect()` to clip RSS between cells.

**Hard cap:** 10 s per cell to prevent a pathological config from blowing
the budget; if hit, mark `timed_out=True` and continue.

### CSV columns

```
mode, with_seqs, threads, region_length, batch_size, buffer_bytes,
n_epochs, instances, bytes, wall_s, instances_per_s, MiB_per_s,
peak_rss_MiB, timed_out, git_sha, host, started_at
```

`peak_rss_MiB`: `resource.getrusage(RUSAGE_SELF).ru_maxrss` delta over the
measurement window (monotonic on Linux; we snapshot before and after).
Used mostly to spot `buffer_bytes` overshoot.

## Thread isolation via subprocess re-exec

`RAYON_NUM_THREADS` and the BLAS thread pools are pinned at process start;
they cannot be safely mutated after the rayon pool initializes. The bench
therefore dispatches one child process per `threads` value:

```python
# bench.py
if "--child" in sys.argv:
    run_cells_for_threads(int(os.environ["BENCH_THREADS"]))
else:
    write_datasets_per_region_length()
    init_csv()
    for n_threads in [1, 2, 4, 8, 16]:
        env = {
            **os.environ,
            "BENCH_THREADS": str(n_threads),
            "RAYON_NUM_THREADS": str(n_threads),
            "POLARS_MAX_THREADS": str(n_threads),
            "OMP_NUM_THREADS": str(n_threads),
            "MKL_NUM_THREADS": str(n_threads),
            "OPENBLAS_NUM_THREADS": str(n_threads),
        }
        subprocess.run(
            [sys.executable, __file__, "--child"], env=env, check=True
        )
```

The parent writes the CSV header once and child runs append. Children only
execute cells whose `threads` matches their pinned value.

## Plotting

`plot_results.py` reads `results.csv` and emits a single `results_plot.png`
with a 3×4 grid of small multiples:

- **Rows:** output mode (`haplotypes`, `annotated`, `variants`).
- **Cols:** axis (`threads`, `region_length`, `batch_size`, `buffer_bytes`).
- **X:** the column's axis values; **Y:** `instances_per_s` (linear).
- **Series:** 3 lines per panel, one per `mode` (None / buffered /
  double_buffered).
- Each panel pulls fan-axis rows where the other three axes sit at their
  midpoint. The `mode=None` series is absent from the `buffer_bytes`
  column.

A second optional plot (commented out by default) does `MiB_per_s` instead
of `instances_per_s` to highlight that larger output modes saturate
bandwidth at smaller batch counts.

## File-level change summary

New:

- `experiments/dataloader/_common.py`
- `experiments/dataloader/bench.py`
- `experiments/dataloader/plot_results.py`

Generated (gitignored where appropriate):

- `experiments/dataloader/results.csv`
- `experiments/dataloader/results_plot.png`
- temp `.gvl` datasets under a bench-managed `tmp/` directory (cleaned on
  exit; `tmp/` is added to `.gitignore` if not already covered)

No changes to `python/genvarloader/`. The bench is a pure consumer of the
public API delivered by the prefetching dataloader plan.

## Open questions deferred to implementation

- Exact memory cost of the dataset-write step at `region_length=25_000`
  (the largest variant). If it pushes setup past ~3 min, reduce the
  region_length fan to 4 values.
- Whether `_output_bytes_per_instance` is cheap enough to call once per
  loader, or whether to cache it in `_common.py` keyed by `(dataset_path,
  with_seqs)`. Default: compute once per cell; revisit if profiling shows
  it dominates.
