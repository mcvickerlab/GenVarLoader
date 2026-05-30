# Performance & memory regressions: GVL 0.6.1 → 0.24.x

**Status:** observed, reproduced under a fully-controlled parity test, not yet root-caused upstream.
**Discovered:** 2026-05-29, while regenerating the GenVarLoader manuscript throughput
benchmarks (`gvl-paper`).
**Affected versions:** regression is present in the current release line (measured on
**0.24.1**) relative to **0.6.1**, the version the manuscript benchmarks were originally
collected on.

This document exists to (a) record the evidence so the manuscript can pin to 0.6.1
without silently shipping ~18–20× worse numbers, and (b) open the door for upstream
optimization. Some of the slowdown may be the unavoidable cost of correctness fixes
added since 0.6.1, but the magnitude strongly suggests there is also low-hanging
optimization work.

---

## TL;DR

Track dataloading is **~18–20× slower** on 0.24.1 than 0.6.1, confirmed by a fully
controlled parity test (same machine, same inputs, same regions/samples, single numba
thread, tracks-only on both sides). It also uses **dramatically more RAM**.

### Controlled parity test (decisive)

Both datasets built from the *identical* 300-region BED, the same `merged.norm.bcf` and
the same BigWig table (TCGA_ATAC, 61 samples, seqlen 16384, ploidy 2). Benched single
numba thread, tracks-only (0.6.1 `return_sequences=False`; 0.24.1 `with_seqs(None)`),
150 batches × 3 reps. Same byte accounting (`Σ track.numel() × element_size`).

| batch | GVL 0.6.1 | GVL 0.24.1 | slowdown |
|---|---|---|---|
| 8  | 3630 MiB/s | 184 MiB/s | **19.7×** |
| 32 | 8604 MiB/s | 495 MiB/s | **17.4×** |

0.6.1 on this machine reproduces the original manuscript numbers
(`results/track_results.csv`: 2931 / 5851 MiB/s on the full dataset), so there is no
machine/era confound.

### Supporting (earlier, less-controlled)

| Modality | Dataset | Cell | 0.6.1 | 0.24.1 | Slowdown |
|---|---|---|---|---|---|
| Tracks (existing 0.24.1 ds, tracks+seqs+variants) | TCGA_ATAC 16384 | 1 thr, batch 32 | 5851 | 472 | 12.4× |
| Haplotypes (smoke grid)¹ | 1KGP 16384 | batch 32 | 1696–3729 | 163–170 | 10–22× |

¹ Smoke grid (5 cold batches/cell), noisy; superseded by the parity test above for tracks.

Memory: peak RSS for track dataloading scales steeply with seqlen and is far higher than
0.6.1. On 0.24.1, TCGA_ATAC tracks peak ~123 GB at seqlen 16384 (top of grid) and exceed
**256 GB** at seqlen 131072 — cells that completed within a ≤256 GB SLURM allocation on
0.6.1.

---

## What it is NOT (ruled out)

These were checked and are **not** the (primary) cause:

- **Not the benchmark harness.** Pure tracks-only output (`with_seqs(None)`, batch = a
  single track tensor) is still ~18× slow. The manuscript harness additionally leaves
  sequence/variant reconstruction on (datasets are written "annotated [variants]", so a
  "tracks" batch returns `[RaggedVariants, track_tensor]`), but disabling it only recovers
  **~19%** (461 → 549 MiB/s at 1 thread, batch 32). Real, but minor.
- **Not the numba threading layer (tbb).** 0.6.1 *required* `tbb`; 0.24.1 made it optional
  and the bench env didn't install it, so numba fell back to `omp`. But on a 64-core node,
  omp vs tbb tracks throughput is nearly identical (omp 559→365, tbb 557→446 across
  1→64 threads) — both ~10× below 0.6.1. tbb is worth restoring for scaling headroom, but
  it does not explain the regression.
- **Not machine / dataset / byte-accounting.** Controlled away by the parity build above.

## Secondary finding: broken multi-thread scaling

On 0.24.1, track-dataloading throughput **decreases** with more numba threads instead of
increasing (TCGA_ATAC 16384, batch 32, on a 64-core node):

| threads | omp | tbb |
|---|---|---|
| 1 | 559 | 557 |
| 8 | 576 | 500 |
| 32 | 488 | 511 |
| 64 | 365 | 446 |

Throughput peaks at ~1–8 threads and regresses thereafter under both layers — i.e. the
per-batch bottleneck is serial, and adding threads only adds contention. This is
consistent with the ~18× serial regression being the dominant cost.

---

## Methodology (apples-to-apples)

The comparison reuses the *same* benchmark loop for both versions, so only the GVL version
differs. Throughput is a pure rate (`bytes_delivered / wall_seconds`), independent of how
many batches are measured, so it is directly comparable across runs.

- **Measurement loop:** `gvl-paper@c46a489:throughput/bin/benchmark_tracks.py` /
  `benchmark_haps.py`. `burn_in=5`, then time N steady-state batches; throughput =
  `Σ batch.numel() × element_size / seconds / 2²⁰` (MiB/s).
- **0.6.1 numbers:** the committed manuscript results,
  `gvl-paper/results/{track_results,hap_results}.csv` (collected 2024 on 0.6.1).
- **0.24.1 numbers:** re-run on the same machine (carter-cn class node), same datasets
  where possible.
  - Tracks: real on-disk dataset `seqlen_16384_native.gvl` (TCGA_ATAC, 61 samples,
    2402 regions, ploidy 2, written by 0.24.1), `with_tracks("read-depth").with_len(16384)`,
    `NUMBA_NUM_THREADS=1`, 200 batches × 3 reps.
  - Haplotypes: `--test` grid via the Nextflow pipeline.

The parity test (controlled build, single thread, tracks-only on both sides) removes the
earlier confounds — both sides use the same machine, regions, samples, byte accounting,
and thread count.

---

## Memory regression detail

`gvl-paper/hap_track_throughput/results/tracks_memory/TCGA_ATAC_16384_native.csv`
(peak RSS vs batch size, 0.24.1, threads=64):

| seqlen | peak RSS plateau (top of grid) |
|---|---|
| 2048 | ~21.5 GB |
| 16384 | ~123 GB |

Peak scales ~5–6× per 8× seqlen at the npb cap (`max_npb = 2³³`), so seqlen 131072 lands in
the several-hundred-GB range and 1048576 well beyond a single node. On 0.6.1 the same grid
cells fit within a ≤256 GB allocation. The OOM-kills observed in the new pipeline (SLURM
`--mem` cgroup limit, exit 137) are a *symptom* of this regression, not a benchmark bug.

---

## Suggested upstream investigation

1. **Profile the serial tracks-only hot path** on 0.24.1 (`py-spy`/`memray`) with
   `with_seqs(None).with_tracks("read-depth")`, single thread — this is where the ~18×
   lives. The variant-annotation path (~19%) and the threading layer (tbb) are secondary;
   start with the core interval/track reconstruction.
2. **Bisect 0.6.1 → 0.24.1** on the controlled parity probe (build a small TCGA_ATAC tracks
   dataset, bench tracks-only single thread at batch 8/32). Cheap (~1 min/version),
   low-variance, and now has a clean 0.6.1 baseline to compare against.
3. **Fix multi-thread scaling** — throughput currently *decreases* with thread count; find
   the serial section / lock / oversubscription. Restoring `tbb` as a (default?) dep gives
   scaling headroom but won't fix the negative scaling on its own.
4. **Memory:** identify what now scales with seqlen at fixed nucleotides-per-batch; on 0.6.1
   the same npb fit in far less RAM.

## Reproduce the tracks-only probe (0.24.1)

```python
import genvarloader as gvl
from time import perf_counter
# tracks-only: with_seqs(None) -> batch is a single track tensor (no RaggedVariants)
ds = gvl.Dataset.open(DS_PATH, FASTA).with_seqs(None).with_tracks("read-depth","tracks").with_len(16384)
for bs in (8, 32):
    dl = ds.to_dataloader(batch_size=bs, shuffle=False)
    ny=nnuc=0; burn=5; nb=150; t0=perf_counter(); esz=4; done=False
    while not done:
        for b in dl:
            trk = b[1] if isinstance(b,(list,tuple)) else b   # track tensor
            if ny==burn: t0=perf_counter()
            if ny>=burn: nnuc+=trk.numel(); esz=trk.element_size()
            ny+=1
            if ny>=nb: done=True; break
    print(bs, nnuc/(perf_counter()-t0)/2**20*esz, "MiB/s")
```
Run with `NUMBA_NUM_THREADS=1` for the single-thread numbers. The 0.6.1 side uses the
equivalent old API (`gvl.Dataset.open(ds, fasta, return_sequences=False)`; restored in
`gvl-paper/hap_track_throughput/bin_gvl061/benchmark_tracks.py`). Build both datasets from
the *same* BED + variants + BigWig table to keep regions/samples identical.

### Check the numba threading layer

```python
import numpy as np, numba as nb
@nb.njit(parallel=True)
def f(x):
    s=0.0
    for i in nb.prange(x.size): s+=x[i]
    return s
f(np.ones(1000)); print(nb.threading_layer())   # 'tbb' if installed, else 'omp'
```
0.6.1 pulled `tbb` transitively; 0.24.1 made it optional, so fresh installs report `omp`.

---

## Profiling results (local, chr22 GEUVADIS slice)

Profiled on the committed `tests/benchmarks/data/chr22_geuv.gvl` slice (5 samples,
chr22, GEUVADIS read-depth tracks, real 1kGP indels, 165 regions), `NUMBA_NUM_THREADS=1`,
seqlen 16384, batch 32 × 200 batches, via `pixi run -e dev profile-{haps,tracks,variants}`
(py-spy) and `memray-{haps,tracks,variants}`. This localizes *where* time and memory go on
the current release line; absolute parity numbers vs 0.6.1 still require the cluster-scale
dataset. Note: on this slice the `tracks` mode is `with_seqs(None)` (reference-coordinate
tracks via `intervals_to_tracks`, no indel re-alignment); haplotype-coordinate track
re-alignment (`shift_and_realign_tracks_sparse`) only runs on the combined haplotype+tracks
path — see the end-to-end benchmarks.

Because this slice is tiny, total py-spy wall time is dominated by one-time module import +
numba JIT compile (`<frozen importlib>` / `llvmlite` frames, ~50% of all samples). The
numbers below are therefore reported **restricted to the steady-state `getitem` hot path**
(stacks under `_dataset/_query.py:getitem`), with self-time percentages relative to that
hot-path subset. Hot-path sample counts: haps 760/1482, tracks 237/959, variants 798/1520.
memray peak heap is whole-process (it includes the JIT'd code), so all three modes plateau
at the same ~3.8 GB; the discriminating memray signal is cumulative allocation churn and the
top *gvl-attributable* allocators, reported below.

### Haplotypes (heavily-used path)
- Top self-time frames (within `getitem`, ~7.6 s hot): `awkward.contents.numpyarray._carry`
  (19.1%), `awkward._kernels.__call__` (15.8%), `awkward._nplikes.array_module.concat`
  (8.9%), then gvl's numba dispatch `_reconstruct.py:189/201 __call__` (2.5% + 2.4%) and
  `_haps.py:764 _reconstruct_haplotypes` (2.0%). Dominant inclusive frames:
  `_query.py:153 _getitem_unspliced` (4.7%) → `reverse_complement_ragged` (`_query.py:337-339`,
  ~4.7%) and `awkward ak_to_packed.to_packed` (1.7%).
- Peak heap (memray): 3.821 GB; total allocated 18.96 GB over 1.43 M allocations. Top
  gvl-path allocators: `awkward array_module.empty` (5.375 GB cumulative), `awkward concat`
  (2.150 GB), `awkward numpyarray._carry` (1.935 GB). (The `<frozen importlib>` 3.25 GB /
  `get_data` 2.62 GB entries are one-time numba/JIT import allocations, constant across modes.)
- Maps to hypothesis: **#1 serial bottleneck** (the per-batch cost is awkward/ragged
  assembly + reverse-complement, all serial Python/awkward) and **#4 memory** (the awkward
  `empty`/`concat`/`_carry` churn is the bulk of the per-batch allocation).

### Tracks (REGRESSIONS.md headline)
- Top self-time frames (within `getitem`, ~2.4 s hot): `awkward numpyarray._carry` (19.4%),
  `awkward._kernels.__call__` (8.0%), then gvl's `_tracks.py:608 _call_float32` — the
  `intervals_to_tracks` numba call (6.8%) — and `awkward concat` (3.8%). Dominant inclusive
  frames: `_query.py:153 _getitem_unspliced` (4.9%) → `reverse_complement_ragged`
  (`_query.py:339`, 4.9%) and awkward `ak_where.where` (1.5%).
- Peak heap (memray): 3.805 GB; total allocated 11.63 GB over 1.38 M allocations (lowest
  churn of the three modes). Top gvl-path allocators: `awkward array_module.empty` (1.720 GB),
  `awkward concat` (859.8 MB), `awkward numpyarray._carry` (859.8 MB).
- Maps to hypothesis: **#1 serial bottleneck** — even with `with_seqs(None)`, the steady-state
  cost is awkward ragged carry/concat plus the serial `intervals_to_tracks` numba kernel and a
  reverse-complement pass; nothing here parallelizes across the batch. Secondary **#4 memory**
  via the same awkward `empty`/`concat`/`_carry` churn (lower magnitude than haps/variants).

### Variants
- Top self-time frames (within `getitem`, ~8.0 s hot): `awkward numpyarray._carry` (16.4%),
  `awkward concat` (8.8%), `awkward._kernels.__call__` (7.0%), then gvl's numba dispatch
  `_reconstruct.py:189/201 __call__` (3.6% + 2.3%) and `_haps.py:398 _haplotype_ilens`
  (0.8%). Dominant inclusive frames: `_query.py:153 _getitem_unspliced` (4.1%) →
  `reverse_complement_ragged` (3.0%), awkward `ak_where.where` (1.4%), `to_packed` (1.4%).
- Peak heap (memray): 3.826 GB; total allocated 16.01 GB over 1.44 M allocations. Top
  gvl-path allocators: `awkward array_module.empty` (3.461 GB), `awkward numpyarray._carry`
  (1.721 GB), `awkward concat` (1.720 GB).

### Takeaway
Across all three modes the steady-state per-batch cost is dominated by **awkward-array ragged
assembly** — `numpyarray._carry`, `_kernels.__call__`, and `concat` together account for
~25–45% of hot-path self-time everywhere — driven by `_query.py:_getitem_unspliced` and its
`reverse_complement_ragged` step, not by gvl's numba kernels. The haplotype and track hot
paths therefore **share the same bottleneck**: the awkward/ragged packing+carry+concat
pipeline, with gvl's own numba work (`intervals_to_tracks` for tracks, `_reconstruct_haplotypes`
for haps) a distinct but smaller second tier. The single most promising optimization target is
the ragged-assembly / reverse-complement path in `_query.py` (reduce per-batch awkward
`empty`/`concat`/`_carry` churn — also the top memory allocator in every mode), which would
help all three output modes at once. Caveat: this is the **0.24.x side only** — no 0.6.1
profile was run locally, so these results localize *where* current cost lives but do not by
themselves attribute the ~18–20× tracks regression to a specific frame; that still requires the
controlled 0.6.1↔0.24.x bisect on the cluster-scale dataset.
