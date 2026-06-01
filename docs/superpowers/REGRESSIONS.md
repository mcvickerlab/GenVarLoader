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

## Optimization plan: flat-buffer ragged transforms (not "remove Ragged")

Code-read of the reconstruction path (2026-05-30) refines the takeaway above. The
**reconstruction itself is already flat-numpy + numba and is NOT the bottleneck**: every
reconstructor (`_reconstruct.py:228`, `_haps.py:759/819`, `_tracks.py:622`, `_reference.py:64`)
writes a pre-allocated flat buffer with a buffer-writer numba kernel and wraps it with the
*cheap* `Ragged.from_offsets` (O(1), no awkward). The awkward `_carry`/`_kernels`/`concat`/`empty`
churn comes entirely from **two post-reconstruction transforms that are implemented in awkward**:

1. **`reverse_complement_ragged` (`_query.py:337`)** — the dominant one:
   ```python
   Ragged(ak.to_packed(ak.where(to_rc, reverse_complement(rag), rag)))
   ```
   This is doubly wasteful: (a) `reverse_complement(rag)` (`_ragged.py:303`) is evaluated for
   the **entire batch** before `ak.where` selects — every row is RC'd even though only
   negative-strand rows need it; and (b) `reverse_complement` is itself 3 awkward passes
   (`ak.to_packed` → `ak.transform` complement → `ak_str.reverse`), then `ak.where` builds a
   merged array and the outer `ak.to_packed` is yet another full copy ≈ **4–5 passes/batch**.
   Same pattern in `RaggedVariants.rc_` (`_rag_variants.py:218`).
2. **Output materialization `to_padded`/`to_numpy` (`_ragged.py:271`)** — `ak_str.rpad` /
   `ak.pad_none`+`fill_none`+`ak.to_numpy`; the `awkward array_module.empty` top allocator.

**Conclusion:** the lever is *not* "stop using `Ragged` internally" (recon already doesn't go
through awkward) and *not* "micro-optimize awkward wrapping". It is: **do the per-batch
transforms (RC, padding) on the flat `(data, offsets)` buffers with numba**, leaving offsets
untouched (RC and fixed-length pad are length-preserving / shape-deterministic). The generic
DNA primitives belong **upstream in `seqpro.rag`** (seqpro owns the `Ragged` type and dense
`sp.reverse_complement`); gvl keeps the composite-type orchestration (`RaggedAnnotatedHaps`,
`RaggedVariants`, `RaggedIntervals`).

### Prototype + microbench (SeqPro branch `feat/ragged-rc-flat`)

A flat-buffer `reverse_complement(rag, comp_lut, mask=, copy=)` was prototyped in
`seqpro/rag/_ops.py`: a single-pass `@nb.njit` kernel that reverse-complements only the
`mask`-selected rows in place over the flat buffer, reusing the input offsets. Microbench
(`scratch_bench_rc.py`, batch=32, ~16 kb rows, 18/32 rows masked, `NUMBA_NUM_THREADS=1`),
output verified byte-identical to the awkward path:

| impl | ms/call | speedup | peak MB/call |
|---|---|---|---|
| awkward (gvl current) | 0.919 | 1.0× | 5.78 |
| flat numba (`copy=True`, returns new) | 0.155 | **5.9×** | 0.53 |
| flat numba kernel (in-place, owned buffer) | 0.097 | **9.5×** | ~0 |

The 5.78 MB allocated to RC a 0.52 MB batch is the ~11× awkward churn the profile flagged as
the top allocator; the flat kernel cuts it to ~1× (copy) or 0 (in-place). Note: at batch=32
the kernel is *faster single-threaded* (0.097 ms) than multi-threaded (0.124 ms) — thread
dispatch overhead exceeds the benefit for so few small rows, so the real impl should not
parallelize at typical batch sizes.

**Caveats (unchanged):** profile is **0.24.x only** — this is *a* speedup, not yet shown to be
*the* fix for the ~18–20× tracks regression vs 0.6.1 (confirm whether 0.6.1 RC'd differently
before claiming so). Padding (`to_padded`) is the obvious next flat-buffer target after RC.

### Shipped + integrated (2026-05-31)

The flat-buffer kernel was released as **seqpro 0.12.1** —
`seqpro.rag.reverse_complement(rag, comp_lut, *, mask=None, copy=True)` — and **parallelized**
across rows (gvl runs very large and/or buffered-loader batches, where thread dispatch is
amortized; the prototype's "don't parallelize" note applied only to the batch=32 microbench).

Integrated into gvl on this branch via a thin wrapper `reverse_complement_masked(rag, mask)`
in `_ragged.py`:
- Calls seqpro with `copy=False` (mutates in place) — safe because every call site operates on
  a freshly reconstructed, caller-owned batch.
- Reuses the existing `_COMP` LUT (`bytes.maketrans(b"ACGT", b"TGCA")`), so output is
  **byte-identical** to the old awkward `reverse_complement`.
- Replicates the per-region mask across inner fixed axes (`np.repeat(mask, n_rows // mask.size)`):
  seqpro's flat kernel wants one mask entry per *flattened* ragged row, whereas awkward's
  `ak.where` used to left-align-broadcast the per-region mask across the ploidy axis.

All **3** masked `ak.where(to_rc, rc(x), x)` sites were replaced: `_query.py` haps,
`_reference.py` unspliced and spliced reference. The old awkward `reverse_complement`
(`_ragged.py:303`) is retained **only** for `RaggedVariants.rc_` (variant allele strings, not
a flat S1 batch). Dependency floor raised: seqpro 0.12.x requires **genoray ≥ 2.7.1**
(genoray 2.4.0 capped seqpro `<0.12`); both bumped in `pixi.toml` + `pyproject.toml`.

End-to-end bench (1024 rows × 8 kb, 268/512 masked) vs the old awkward idiom:

| impl | ms/call | speedup | peak MB/call |
|---|---|---|---|
| awkward (old) | 13.26 | 1.0× | 92.3 |
| seqpro 0.12.1 flat (in-place, parallel) | 0.38 | **34.7×** | ~0 |

The parallelism pays off far more at buffered-loader scale (34.7×) than the single-thread
prototype's 5.9–9.5× on 32 small rows. Verification: 200 randomized trials byte-identical to
the awkward path; 319 dataset tests pass (the 3 `test_ds_haps_1kg` errors are the missing
`gen-1kg` fixture, not regressions).

### Shipped + integrated — flat-buffer `to_padded` (2026-05-31)

The second flat-buffer transform landed: densify-and-right-pad, the top awkward
`array_module.empty` allocator in the profile. Released as **seqpro 0.13.0** —
`seqpro.rag.to_padded(rag, pad_value, *, length=None)` — a single-pass, parallel byte-copy
kernel over the flat `(data, offsets)` buffer (one dtype-agnostic `uint8`-view copy into a
pre-filled output; `length=None` pads to batch max, explicit `length` pads/truncates;
ragged-axis-last, non-record only). Shipped via SeqPro PR #37 with 17 tests + a microbench.

Integrated into gvl on this branch: `to_padded` in `_ragged.py` is now a **thin pass-through**
to `seqpro.rag.to_padded` (the 13-line awkward `ak_str.rpad` / `ak.pad_none`+`fill_none`+
`to_numpy` body collapses to one delegate call). The `(rag, pad_value)` positional contract is
unchanged, so all call sites port untouched (`_query.py` haps/tracks padded output;
`_reference.py` reference padded output; `RaggedVariants`/`RaggedAnnotatedHaps.to_padded`).
Dependency floor raised: seqpro 0.13 requires **genoray ≥ 2.7.2** (genoray 2.7.1 capped
seqpro `<0.13`); both bumped in `pixi.toml` + `pyproject.toml`.

seqpro-side microbench (1024 rows × ~4 kb S1, pad to batch max) vs the old awkward idiom:

| impl | ms/call | speedup | peak MB/call |
|---|---|---|---|
| awkward (`ak_str.rpad` → `to_numpy`) | ~1.36 | 1.0× | +4.20 |
| seqpro 0.13.0 flat (parallel) | ~0.41 | **~3.3×** | +4.19 |

Unlike RC, peak allocation is **comparable** — both paths materialize the same dense output
array; the win is eliminating the awkward intermediate `empty`/`concat`/`pad_none` churn
(the cumulative-allocation hotspot), not the output allocation itself. Verification: byte-identical
to the old awkward path across every gvl dtype/pad pattern (S1+`b"N"`, int32+`-1`,
int32+`iinfo.max`, float32+`0.0`, `(batch, ploidy, None)` leading dims, sliced non-contiguous);
248 gvl unit tests pass, ruff clean. (Pyrefly reports `missing-module-attribute` on the
`seqpro.rag` imports — a pre-existing resolver artifact that also affects the `reverse_complement`
import; the symbols resolve at runtime and tests pass.)

### Re-profile after RC + `to_padded` (2026-05-31, local slice; memray + py-spy)

Re-profiled (memray allocation + `sudo py-spy` self-time) on the **regenerated, correct**
`chr22_geuv.gvl` slice (option 2: `extend_to_length=False` + exact-window variants, fixed after
merging main's #197 PGEN fix — regions ~16,384 bp, `intervals.npy` 10.4 MB, `variants.arrow`
6.5 MB), `NUMBA_NUM_THREADS=1`, seqlen 16384, batch 32 × 200 batches.

> **⚠️ The earlier "Profiling results" baseline is NOT a clean comparator.** It was run on the
> **broken** committed dataset (regions up to 3.1 Mb, `intervals.npy` 36 MB). This re-profile is
> on the **fixed** dataset. So before/after deltas conflate the flat-buffer transforms with the
> dataset fix — they do **not** isolate the transform effect. Treat the numbers below as a
> **current-state localization** on the correct dataset, not as a measured speedup. A clean
> isolation needs a same-(fixed-)dataset A/B (revert the flat transforms, re-profile) or the
> cluster-scale throughput bisect. Also: this slice is tiny and noisy — the `getitem` hot path is
> only 28% (tracks) / 52% (haps) / 61% (variants) of wall; the rest is one-time JIT/import.

**Current-state self-time within the `getitem` hot path (py-spy, fixed dataset, flat RC+to_padded):**
- **Tracks:** awkward `_carry` 23.5%, awkward `_kernels.__call__` 16.2%, `intervals_to_tracks`
  (`_tracks.py` `_call_float32`) 10.3%, `reverse_complement_ragged` self **2.9%**.
- **Haps:** awkward `_kernels.__call__` 29.3%, `_carry` 14.6%, `_reconstruct_haplotypes` 5.7%,
  `concat` 4.9%, `_getitem_unspliced` 4.1%, `reverse_complement_ragged` self 3.3%.
- **Variants:** `_carry` 15.8%, `_kernels.__call__` 15.2%, `concat` 7.1%, `_reconstruct` dispatch 8.6%.
- **Awkward ragged-assembly self-time (`_carry`+`_kernels`+`concat`) = 39–53% of the hot path**
  (tracks 43%, haps 53%, variants 39%) — the single largest bucket in every mode.

**Two clean takeaways (independent of the broken-baseline confound):**
1. **The dominant cost is awkward ragged assembly**, not the transforms we flattened: `_carry` +
   `_kernels.__call__` + `concat` (per-region `ak.concatenate` + the `_carry` of fancy-indexing
   `dataset[regions,samples]`) is ~40–53% of hot-path self-time; gvl's own numba kernels
   (`intervals_to_tracks` 10% tracks, `_reconstruct_haplotypes` 6% haps) are a clear second tier.
2. **The flat RC kernel itself is cheap, but its awkward *glue* isn't fully gone.**
   `reverse_complement_ragged` is ~23% of wall inclusive (tracks), yet its kernel self-time is only
   ~3% — ~half of that 23% is awkward `_carry`/`_kernels` + seqpro `Ragged.__getitem__`/`__init__`/
   `ak`-dispatch *around* the flat kernel (contiguity/packing + Ragged (re)construction). So even a
   flat transform still pays awkward wrapping on each call.

**Still open — 0.6.1 throughput parity.** RC + `to_padded` are flat and the dataset is correct, but
the transform *effect* is not yet cleanly measured and the dominant *assembly* cost remains.
Remaining gap-closers: (a) a **same-dataset A/B** (revert the flat transforms on the fixed slice,
re-profile) to actually isolate the RC+to_padded effect — local and cheap, but the slice is noisy;
(b) flat-buffer / de-awkward the **ragged-assembly path** in `_getitem` (the per-region
`ak.concatenate` + fancy-index `_carry`, now the #1 self-time bucket) and the awkward glue around
the flat kernels — the biggest remaining lever, harder than the standalone transforms; (c) the
controlled 0.6.1↔0.24.x bisect on the **cluster-scale** dataset — the only real proof of the
~18–20× throughput regression and its fix. Update this section as each lands until the gap closes.

---

### Task 10 (2026-06-01): Variants path spike — decision NOT to flatten numeric fields

**Decision: (b) — RaggedVariants stays awkward-native. No code change.**

**Profiling run:** `memray run -fo variants.t10.memray.bin profile.py --mode variants`
(2000 batches × 32, chr22_geuv.gvl, NUMBA_NUM_THREADS=1). Results:

| metric | value |
|---|---|
| Total allocated | 18.205 GB |
| Peak RSS | 3.589 GB |
| Total allocations | 2,382,364 |

**Top allocators by size:**
1. `_reconstruct.py:152` → **8.410 GB** (track output buffer, `np.empty` for out/tracks arrays)
2. `_reconstruct.py:183` → **4.206 GB** (per-track `_tracks` array)
3. importlib bootstrap → 3.240 GB (one-time JIT/import)
4. llvmlite/ffi → 733 MB (numba compilation)
5. `awkward/_nplikes/array_module.py:153` (awkward `empty`) → **209 MB**

**Top allocators by count:**
1. llvmlite/ffi → 1,535,233 (numba)
2. pyarrow.compute → 54,135
3. awkward `array_module.empty` → 52,132
4. `intervals_to_tracks` → 48,120
5. numpy `_wrapreduction` → 42,130

**Why (b) — key observations:**

1. **The variants-specific assembly is not in the hot list.** `RaggedVariants.__init__` (`ak.zip` at
   `_rag_variants.py:60`) and `_get_alleles` (`_haps.py:730`) are absent from both top-5 lists.
   The awkward `empty` at 209 MB / 52K calls is dominated by track work (`_reconstruct.py`), not
   variant assembly.

2. **Numeric fields are already flat on the assembly side.** `start`, `ilen`, and `dosage` are
   assembled via plain numpy fancy-indexing (`self.variants.start[v_idxs]`) + `Ragged.from_offsets`
   in `_haps.py:_get_variants`. There is no awkward intermediate for these scalars — the only
   awkward call in the assembly hot path is `ak.zip` at the very end (to build the `RaggedVariants`
   container) and the layout manipulation in `_get_alleles` for the allele strings.

3. **The `rc_` path's awkward cost is in the allele strings, not the numerics.** `rc_` calls
   `ak.where(to_rc, reverse_complement(self["alt"]), self["alt"])` — this is variable-length-of-
   variable-length (`alt`/`ref` are `(batch, ploidy, ~variants, ~allele_len)`). The numeric fields
   (`start`/`ilen`/`dosage`) are NOT touched by `rc_`. Flattening the numeric `rc_` would be a
   no-op — those fields are plain `Ragged` already and `rc_` never reverses them.

4. **Prior py-spy data corroborates.** Variants had the lowest awkward share (~39%) vs tracks 43%,
   haps 53% of the `getitem` hot path. The profiling pattern here — tracks dominating even in
   "variants mode" (which co-fetches tracks) — is consistent with that earlier finding.

**Principled limit of the flat-buffer initiative:** `RaggedVariants` is the natural terminus of
Tasks 5–10. Its allele strings (`alt`/`ref`) are genuinely variable-length-of-variable-length and
therefore cannot be densified into a flat array without either padding (changing the data contract)
or building a two-level offset table (essentially re-implementing awkward). The numeric fields are
already assembled with no awkward intermediate. The `ak.zip` constructor call and `_get_alleles`
layout manipulation are the minimal irreducible awkward overhead for this data shape. Any further
optimization here would require replacing the awkward container at the public API boundary
(`RaggedVariants` is user-facing), which is a much larger breaking change than the internal
transform flattening done in Tasks 5–9.

**No files changed. Gate results:** `pytest tests/dataset -q -m "not slow"` → 44 passed;
`pytest tests/dataset/test_flat_getitem_snapshot.py -q` → 9 passed.

---

### Task 11 (2026-06-01): Flat-buffer getitem refactor — final A/B write-up

#### Summary of changes (Tasks 1–11)

All non-variant reconstructors (`Ref`, `Haps`/annotated, `Tracks` float32,
`RefTracks`, `HapsTracks`) now return `_Flat`/`_FlatAnnotatedHaps` — pure-numpy
`(data, offsets, shape)` containers that route through flat numba kernels for
RC and padding without any awkward dispatch. `RaggedVariants` stays awkward-native
by deliberate design (Task 10: allele strings are genuinely VLVL; numeric fields
have no awkward intermediate; flattening the public API boundary is out-of-scope).

The legacy `isinstance(rag, Ragged)` and `isinstance(rag, RaggedAnnotatedHaps)` branches
in `reverse_complement_ragged` and `pad` that handled pre-flat reconstructor output were
removed in Task 11 (grep-verified unreachable: no `Ragged.from_offsets` / `RaggedAnnotatedHaps`
call in any reconstructor's `__call__`; all return `_Flat` cast as `Ragged` via type hints).
The `else r.to_numpy()` fallback in the int-densify step was likewise removed. An awkward
guard test (`tests/dataset/test_no_awkward_in_hotpath.py`) now asserts 0 awkward dispatches
across `tracks_fixed`, `haps_fixed`, `ref_fixed`, `haps_tracks_fixed`, and `haps_ragged`.

#### Flat-buffer getitem refactor A/B: Task 0 baseline vs final (memray, N_BATCHES=2000, batch=32)

Dataset: `chr22_geuv.gvl` (5 samples, chr22, GEUVADIS, 165 regions, `extend_to_length=False`).
`NUMBA_NUM_THREADS=1`. Both baseline and final use the same fixed dataset (committed at `638dcf6`).
Baseline = pre-refactor commit; final = current branch tip (Task 11).

**Tracks mode** (`with_seqs(None).with_tracks(...).with_len(16384)`):

| metric | baseline | final | delta |
|---|---|---|---|
| Total allocated | 41.953 GB | 8.248 GB | **−80.3%** |
| Total allocations | 584,855 | 544,653 | −6.9% |
| Peak RSS | 3.570 GB | 3.569 GB | ~0 |

Top gvl-attributable allocators by size:

| frame | baseline | final |
|---|---|---|
| `awkward array_module.empty` | 16.819 GB | (not in top 5) |
| `awkward array_module.concat` | 8.410 GB | (not in top 5) |
| `awkward numpyarray._carry` | 8.410 GB | (not in top 5) |
| `_call_float32` (track output buffer) | 4.205 GB | 4.205 GB (unchanged) |

The three awkward `empty`/`concat`/`_carry` frames that dominated tracks-baseline
cumulative allocation are absent from the final top-5; the remaining dominant allocator
is the unavoidable track output `np.empty` buffer.

**Haplotypes mode** (`with_seqs("haplotypes").with_tracks(...).with_len(16384)`):

| metric | baseline | final | delta |
|---|---|---|---|
| Total allocated | 88.162 GB | 18.774 GB | **−78.7%** |
| Total allocations | 843,357 | 781,335 | −7.4% |
| Peak RSS | 3.588 GB | 3.569 GB | ~0 |

Top gvl-attributable allocators by size:

| frame | baseline | final |
|---|---|---|
| `awkward array_module.empty` | 35.747 GB | (not in top 5) |
| `awkward array_module.concat` | 16.819 GB | (not in top 5) |
| `awkward numpyarray._carry` | 16.819 GB | (not in top 5) |
| `_reconstruct.py __call__` (track out) | 8.410 GB | 8.410 GB (unchanged) |
| `_reconstruct_haplotypes` (hap buffer) | (not in top 5) | 2.102 GB |

The awkward frames vanish. `_reconstruct_haplotypes` (the flat numba buffer allocation)
surfaces in the final top-5 as the haps-specific allocation, which is expected.

**Variants mode** (`with_seqs("variants").with_tracks(...).with_len(16384)`):

| metric | baseline | final | delta |
|---|---|---|---|
| Total allocated | 84.586 GB | 17.265 GB | **−79.6%** |
| Total allocations | 1,066,033 | 1,003,429 | −5.9% |
| Peak RSS | 3.587 GB | 3.569 GB | ~0 |

Top gvl-attributable allocators by size:

| frame | baseline | final |
|---|---|---|
| `awkward array_module.empty` | 33.854 GB | 209 MB (residual, RaggedVariants-native) |
| `awkward numpyarray._carry` | 16.832 GB | (not in top 5) |
| `awkward array_module.concat` | 16.824 GB | (not in top 5) |
| `_reconstruct.py __call__` (track out) | 8.410 GB | 8.410 GB (unchanged) |

Variants retain a small residual awkward `empty` footprint (209 MB, ~52 K calls)
from `RaggedVariants` construction — this is the irreducible minimum for the awkward-native
variants path (Task 10 deliberate decision). The bulk (33.6 GB → 209 MB, **−99.4%** on the
`empty` frame alone) is eliminated.

#### Takeaway

Across all three modes, cumulative memory allocation drops **~79–80%** (total allocated;
peak RSS is flat — the dominant `np.empty` for track output and hap reconstruction buffers
is unchanged). The eliminated allocations are the awkward `empty`/`concat`/`_carry` frames
that performed per-batch ragged assembly in the old path.

Peak RSS is unchanged because the peak-RSS bottleneck is the track output buffer
(`_reconstruct.py` lines 152/183 — `np.empty` for the pre-allocated flat out array and
per-track scratch, together 4.2 + 8.4 GB cumulative = peak in the steady state), which
is retained in both baseline and final.

#### py-spy CPU self-time A/B (2026-06-01, `sudo py-spy`, fixed dataset, N_BATCHES=2000)

Same-dataset A/B from the preserved baseline (`*.baseline.speedscope.json`, Task 0 pre-refactor)
vs the final speedscope captured on the merged refactor. Metric: leaf self-time aggregated from
the speedscope samples (weights = seconds of sampled CPU time, single-thread). "getitem hot path"
= samples whose stack passes through `_dataset/_query.py` getitem. "Awkward self-time" = leaf
self-time in awkward frames (`numpyarray._carry`, `_kernels.__call__`, `concatenate`, `to_packed`,
`array_module.empty`, list/offset layout ops) — the per-batch ragged-assembly + glue this refactor
targeted.

| mode | getitem hot-path self-time (s) | awkward self-time within getitem | total sampled wall (s) |
|---|---|---|---|
| **tracks**   | 5.70 → **0.57** (−90%) | 78.8% (4.49 s) → **0.0% (0 s)** | 7.00 → **2.19** (3.2×) |
| **haps**     | 13.63 → **3.18** (−77%) | 69.8% (9.51 s) → **9.4% (0.30 s)** | 14.85 → **4.43** (3.4×) |
| **variants** | 17.22 → **7.64** (−56%) | 74.7% (12.87 s) → **55.1% (4.21 s)** | 18.59 → **9.02** (2.1×) |

Whole-profile awkward leaf self-time: tracks 64.1% → **0.9%**, haps 64.0% → **7.0%**,
variants 69.2% → **46.8%**.

**Reading:** the awkward ragged-assembly/glue churn is effectively eliminated from the tracks
hot path (4.49 s → 0 s) and the haps hot path (9.51 s → 0.30 s). For variants, awkward self-time
drops ~67% (12.87 s → 4.21 s) but remains the largest bucket — expected and by design, since
`RaggedVariants` is awkward-native (Task 10): only the assembly/RC *glue* around it was removed,
not the awkward allele-string container itself. This is the **throughput** counterpart to the
memray allocation A/B above (which the earlier profiling caveat flagged as not-yet-demonstrated):
total sampled CPU time falls ~2–3.4× per mode, and CPU self-time *within getitem* falls 56–90%.

**Caveats:** the chr22 GEUVADIS slice is tiny, so total wall is partly one-time JIT/import (why
getitem-inclusive samples are <100% of each profile, and the final tracks profile is only ~2.2 s
total); single-thread; py-spy sampling is statistical (~few-percent frames are noisy). The
direction and magnitude (awkward churn → ~0 for tracks/haps) are unambiguous; treat the exact
percentages as approximate. Residual dominant gvl-side costs are now the numba kernels
(`intervals_to_tracks`, `_reconstruct_haplotypes`) and, for variants, the awkward `RaggedVariants`
assembly.
