# Performance & memory regressions: GVL 0.6.1 → 0.24.x

**Status:** observed, reproduced, not yet root-caused upstream.
**Discovered:** 2026-05-29, while regenerating the GenVarLoader manuscript throughput
benchmarks (`gvl-paper`).
**Affected versions:** regression is present in the current release line (measured on
**0.24.1**) relative to **0.6.1**, the version the manuscript benchmarks were originally
collected on.

This document exists to (a) record the evidence so the manuscript can pin to 0.6.1
without silently shipping ~10–30× worse numbers, and (b) open the door for upstream
optimization. Some of the slowdown may be the unavoidable cost of correctness fixes
added since 0.6.1, but the magnitude strongly suggests there is also low-hanging
optimization work.

---

## TL;DR

Haplotype and track dataloading are **roughly an order of magnitude slower** and use
**dramatically more RAM** on 0.24.1 than on 0.6.1, for identical datasets and grid cells.

| Modality | Dataset | Cell | 0.6.1 | 0.24.1 | Slowdown |
|---|---|---|---|---|---|
| Tracks (matched, rigorous) | TCGA_ATAC, seqlen 16384 | 1 thread, batch 8 | 2931 MiB/s | 177 MiB/s | **16.6×** |
| Tracks (matched, rigorous) | TCGA_ATAC, seqlen 16384 | 1 thread, batch 32 | 5851 MiB/s | 472 MiB/s | **12.4×** |
| Haplotypes (smoke grid)¹ | 1KGP, seqlen 16384 | batch 32 | 1696–3729 MiB/s | 163–170 MiB/s | 10–22× |
| Haplotypes (smoke grid)¹ | 1KGP, seqlen 2048 | batch 32 | 617–654 MiB/s | 29 MiB/s | 21–22× |

¹ The haplotype numbers come from a `--test` grid (only 5 measured batches per cell, cold)
and are noisier / biased slow, especially at `batch_size=1`. The **conservative floor**
from the haplotype data is ~5–6× (large seqlen, batch 32). The tracks numbers are the
rigorous ones: 200 measured batches, 3 replicates, <5% spread.

Memory: peak RSS for track dataloading scales steeply with seqlen and is far higher than
0.6.1. On 0.24.1, TCGA_ATAC tracks peak ~123 GB at seqlen 16384 (top of grid) and exceed
**256 GB** at seqlen 131072 — cells that completed within a ≤256 GB SLURM allocation on
0.6.1.

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

### Caveats / confounds (for honest upstream triage)

1. **Haplotype data is a smoke grid** (5 batches, cold). Treat the ~5–6× large-batch floor
   as the trustworthy lower bound; the tracks 12–17× is the rigorous figure.
2. **Thread control differs slightly.** 0.6.1 set numba threads explicitly; the 0.24.1
   re-run used `NUMBA_NUM_THREADS=1`. DataLoader worker processes may add parallelism not
   captured by that env var. This would, if anything, make 0.24.1 look *faster* than a true
   single-thread comparison — i.e. it biases against finding a regression.
3. **Hardware/time gap.** 0.6.1 numbers are from 2024; 0.24.1 from 2026. Same cluster, but
   not bit-for-bit the same node.

---

## Secondary finding: per-batch `RaggedVariants` reconstruction

On 0.24.1, opening a dataset written in "annotated [variants]" mode and requesting tracks
yields a **list per batch**: `[RaggedVariants(shape=(B, ploidy, None)), track_tensor]`.
The benchmark only consumes the track tensor, yet GVL reconstructs the `RaggedVariants`
object every batch. The old tracks-only path delivered a single tensor.

This per-batch variant annotation is plausibly a meaningful chunk of the track-dataloading
slowdown (work performed and immediately discarded). Worth checking whether annotation can
be made lazy / opt-out when the consumer doesn't read it, or whether dataset write defaults
changed to always annotate.

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

1. **Bisect 0.6.1 → 0.24.1** on the rigorous tracks probe (TCGA_ATAC seqlen-16384, batch
   8/32, single thread) to localize the commit(s) that introduced the slowdown and the
   memory growth. The probe is cheap (~1 min/version) and low-variance.
2. **Profile the track dataloading hot path** on 0.24.1 (e.g. `py-spy`/`memray`) — separate
   the cost of variant annotation (`RaggedVariants`) from track reconstruction itself.
3. **Check write-time defaults** — did datasets start being written with variant annotation
   always-on, forcing per-batch reconstruction even for track-only consumers?
4. **Memory:** identify what now scales with seqlen at fixed nucleotides-per-batch; on 0.6.1
   the same npb fit in far less RAM.

## Reproduce the rigorous tracks probe

```python
import genvarloader as gvl
from time import perf_counter
ds = gvl.Dataset.open(DS_PATH, FASTA).with_tracks("read-depth","tracks").with_len(16384)
for bs in (8, 32):
    dl = ds.to_dataloader(batch_size=bs, shuffle=False)
    ny=nnuc=0; burn=5; nb=200; t0=perf_counter(); esz=4; done=False
    while not done:
        for b in dl:
            trk = b[1] if isinstance(b,(list,tuple)) else b   # track tensor
            if ny==burn: t0=perf_counter()
            if ny>=burn: nnuc+=trk.numel(); esz=trk.element_size()
            ny+=1
            if ny>=nb: done=True; break
    print(bs, nnuc/(perf_counter()-t0)/2**20*esz, "MiB/s")
```
Run with `NUMBA_NUM_THREADS=1` for the single-thread numbers above.
