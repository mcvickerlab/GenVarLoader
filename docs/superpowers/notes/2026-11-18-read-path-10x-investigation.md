# Read path: is a 10× speedup available? (investigation handoff, 2026-11-18)

Status: **analysis done, nothing implemented.** Step 1 measured 2026-11-18: calla-asca is GPU-bound (see "Step 1 result").
Repos involved: GenVarLoader (`main` @ 3c68aa9c), genoray (6.0.2 @ 9b56ed4), calla (55d22f2), calla-asca (`design-spec` @ 141a684), calla-methyl, aster, gvf-geuvadis.

Tags: **[code]** read in source, **[measured-repo]** number written in a repo doc/commit, **[measured-11-18]** microbench run during this investigation (M4 Pro, `tests/data/1kg` SVAR1, 100 regions × 5 samples × 10 kb), **[inferred]**.

## Resume here

1. On the GPU box: `git switch perf/profile-train-step` in calla-asca (branch pushed, one commit on `design-spec`), then
   `./px python scripts/profile_train_step.py --steps 50 --warmup 10` and again with `--max-items 16384`.
   It prints real-loop ms/step, fetch (per calla source: Variants / RefSeq / Expr) vs H2D vs GPU fwd/bwd, a cProfile of the fetch grouped by package, and a HOST-BOUND / GPU-BOUND verdict.
2. Decide from the output:
   - `Variants` source dominates fetch → gvl SVAR2 decode + variant-window build is the target → prototype "variant keys out, windows built on GPU" (Proposal 1).
   - `RefSeq` / calla Python dominates → fix in calla (suspect: `RefSeq._call_one_to_many` `np.unique` dedup then per-item Python gather, calla `data/sources/tokens.py:679-722`).
   - GPU-bound → calla-asca isn't loader-limited; drop it as the driver.
3. Quick independent win (not gvl): gvf-geuvadis p-SAGE-net baseline one-hots to float32 on CPU (`bench/accuracy/psagenet_data.py:73-102`), measured loader-bound at 36.9 items/s on A30 (`psagenet_train.py:31-32,338`). Ship uint8 to GPU and one-hot there. ~30 min.

## Step 1 result (2026-11-18, GPU box): calla-asca is GPU-bound

Ran calla-asca `perf/profile-train-step` @ b613404 on 1× RTX PRO 6000 Blackwell (cu130), fp32, `num_workers=0`, full `peaks_F1000.gvl` (149k regions × 148 samples), 50 steps after 10 warmup.

| `--max-items` | items / regions per batch | real loop (A) | fetch | ↳ AscaVariants | ↳ RefSeq | H2D | GPU fwd/bwd/step | Verdict |
|---|---|---|---|---|---|---|---|---|
| 4096 (train default) | 472 / 4 | 83.1 ms (12.0 steps/s) | 13.9 ms | 6.7 ms | 3.1 ms | 1.2 ms | 37.8 ms | **GPU-bound** |
| 16384 | 2006 / 17 | 91.0 ms (11.0 steps/s) | 27.0 ms | 12.0 ms | 8.8 ms | 2.4 ms | 48.4 ms | **GPU-bound** |

- Fetch is ~37% (4096) / ~56% (16384) of GPU time. gvl is the largest single fetch cost (`decode_variants_from_svar2_readbound` 5.7–6.8 ms/step), but making it free would not move the step time: the GPU is the bottleneck.
- The real loop is ~30 ms/step slower than fetch + H2D + GPU summed (83 vs 53 ms; 91 vs 78 ms). Not explained yet: pass A is a mean over different batches, pass B reports medians. Possible causes are Python/launch overhead hidden by B's syncs or batch variance. Worth one look before trusting the absolute numbers.
- The earlier 296–390 ms/step estimate is stale: it predates calla 55d22f2 (Helion kernels, fewer host syncs).
- Per the decision rule in step 2: **calla-asca is not loader-limited — drop it as the driver for gvl read-path work.** The largest available calla-asca win is overlapping the fetch with the GPU (`num_workers>0` or a prefetch thread): ~83 → ~38 ms/step at most, with no gvl change.
- Raw logs were in the session scratchpad only; rerun with `CALLA_ASCA_DATA_DIR=<design-spec worktree>/data ./px python scripts/profile_train_step.py ...` (a fresh worktree has no `data/`).

## Follow-ups (2026-11-18, same GPU box; host load avg ~190, so absolute CPU times are inflated, ratios within a run are fair)

- **calla-asca prefetch thread** (calla-asca `perf/profile-train-step` @ 3848f10, pass P): real loop 160.1 → 77.9 ms/step at 4096 items (2.05×), 136.3 → 93.7 ms at 16384 (1.45×). gvl's `to_dataloader(mode="buffered"|"double_buffered")` does **not** fit calla-asca: fixed `batch_size` slicing vs `PackingSampler`, one gvl source vs calla's multi-source batch, and `RefDataset.to_dataloader` has no `mode`. A plain thread works because the SVAR2 decode releases the GIL (`src/ffi/mod.rs:1964`, `py.detach`).
- **gvf-geuvadis p-SAGE-net** (Resume step 3): GPU one-hot on `perf/psagenet-gpu-onehot` @ 7bdbfe92. Bit-identical outputs; steady-state ~43 → ~75 items/s (1.74×) on RTX PRO 6000.
- **calla-methyl** (`perf/profile-train-step` @ 4b5d4d7, base `varnorm`): **host-bound**. One worker fetch ≈ 340–450 ms/batch (182 items); production 6 workers ≈ 75 ms/step, matching fetch/6. Fetch split: **figwig label read ~200–230 ms (50–58%)** — one `FigWigReader.read` per unique sample, `calla_methyl/labels.py:61-79` via `data.py:281`; calla RefSeq call+tensorize ~88 ms (~25%; gvl `RefDataset` only 7–11 ms of it, rest `tokens.py:679` + `rag_to_nested`); **gvl variant windows 45–94 ms (13–20%)**; `pin_memory` 45–55 ms. gvl is third — fix figwig batching first.
- **Net for gvl read-path work:** no surveyed workload is currently gvl-bound. Proposals 1–3 stay parked until one is.

## Headline conclusions

- **Current SVAR1 haplotype kernel is already memcpy-bound** [measured-11-18]: b=1024 × 10 kb (20.5 MB) in 1.22 ms serial ≈ 17 GB/s. Kernel tuning cannot give 10× on CPU; only writing fewer bytes can.
- **Rayon gives no speedup in any mode** [measured-11-18 on M4 Pro; measured-repo on Xeon, `docs/roadmaps/phase-5-w6-perf-rebaseline.md:137-157`]. Untested hypothesis: first-touch page faults on freshly allocated output buffers dominate. Test: rerun the kernel into a reused, pre-faulted output buffer (~30 min). Could be 2–4× on CPU with no format change.
- **Real downstream workloads (survey below) almost all use flat tokenized `variant-windows` + reference-only `RefDataset`.** Nobody uses tracks, splicing, jitter, insertion fill, in-gvl RC, AF filtering, or FORMAT fields. `haplotypes` is used only by baselines. So haplotype/track kernel work is low priority for real users.
- **Fully random (region, sample) sampling is a hard requirement** (user has seen within-epoch mini-overfitting with region-grouped batches). Designs must benefit from within-batch region sharing when it exists, but must not depend on it.
- **SVAR2 read path has ~10× of plain implementation slack** [code + measured-repo]: ~21 µs/cell, 171 ms single-thread for an 8192-cell spliced batch (issue #349 note, `_svar2_haps.py:1514`).

## Proposals, ranked for the real workloads

### 1. Variant keys out, windows built on GPU (top candidate; pending step 1 above)
- gvl emits per batch: a **unique-variant table** (position + SVAR2 u32 key, ~8 B each) and a per-item CSR / presence bitmask into it. A torch/Triton op builds variant windows (and `RefDataset` peaks/tiles, and haplotypes for baselines) against a **GPU-resident 2-bit reference** (hg38 ≈ 775 MB).
- Bytes per variant [inferred]: flank 16 → ~66 B of tokens today vs ~8 B (~8×); flank 128 (aster) → ~514 B vs 8 B (~60×). Plus dedup: a common variant in a one-gene × 290-sample batch is shipped (and could be embedded) once.
- Removes CPU work that is ~61% of variant-windows time (`tokenize` 28%, `slice_flanks` 20%, `assemble_alt_window` 13%; chr22_geuv profile, `docs/roadmaps/phase-5-w4-final-ab.md` / round-3 profile).
- Degrades gracefully under random sampling (no sharing → still per-variant key instead of tokens).
- Mirrors SVAR2's dense layout (variant table + presence bits), so SVAR2 keys can be decoded on GPU directly.

### 2. Fused SVAR2 read path (no format change), ~1–2 weeks across genoray + gvl
Waste found in the current chain [code]:
- genoray `gather_haps_readbound_impl` (`genoray/src/query/gather.rs:869-1090`) is **serial** and **re-decodes the dense window per (region, sample)**; dense indel presence read bit-by-bit (~:1045).
- `reader.lut_arrays()` **copies the whole `long_alleles.bin` on every FFI call** (`genoray/src/nrvk.rs:171-177`; called from gvl `src/ffi/mod.rs:1112,1308,1773,1982`). Should be mmap'd.
- gvl `split_to_flat` (`src/svar2/mod.rs:201`) re-copies everything and re-packs presence bit-by-bit.
- gvl `merge_hap` (`src/svar2/mod.rs:45-66`) allocates a Vec per haplotype and `sort_by_key`s already-sorted inputs; run twice (diffs + reconstruct).
- `decode_alt` (`src/svar2/mod.rs:17-29`) returns `Cow::Owned(Vec)` per inline allele per haplotype.
- numpy range-cache lookup ~3.4 ms/batch at genome scale (`_svar2_ranges.py:203,259`), 93% numpy-pass-bound.
- aster's mut2vec profile: ~97% of loader CPU is gvl (`decode_variants_from_svar2_readbound` 60.7% + memmap 34.1%) — aster `docs/roadmap.md:289-298` — though aster still has 8.6× loader headroom.
Target: one Rust pass walking mmap'd streams straight into output, parallel over queries, dense window decoded once per region per batch.

### 3. SVAR3 format (only if, after 1+2, the CPU side is still the bottleneck), ~3–4 weeks
- **Fixed genomic tiles (~1 Mb) × sample blocks (~1024 haps)**, each (tile, block) one contiguous chunk. Tiles are fixed at write time and independent of BED regions — changing regions never requires a rewrite. Adding samples appends blocks.
- Inside a chunk, hap-major: per-tile site table (pos as in-tile offset, key, INFO), per-hap sparse site-index list for rare variants + short local bit row for common ones (keep SVAR2's per-variant sparse/dense routing, but no contig-wide bit stride → ~1 contiguous read per (region, hap) instead of ~12–20 scattered 4 KB pages).
- FORMAT fields parallel to each hap's calls (SVAR2 dense FORMAT is `n_samples`-strided: one random read per dense variant per hap).
- **Drop `max_del` lookback**: duplicate tile-spanning deletions into the next tile (flagged) and store cumulative ilen at each (tile, hap) start.
- Index: u16 call counts per (tile, hap) + u64 base per (tile, block); ~40 MB for 1kG, ~6 GB at 1M haps. 16 kb sub-index inside a tile.
- Codec: frame-of-reference bit-packing decodable on GPU and CPU at GB/s; zstd only as optional cold layer.
- Rejected: PBWT/GBZ-style compression — random single-sample access needs checkpoint replay, conflicts with random sampling.

### Demoted / dropped
- GPU haplotype reconstruction as a standalone goal: only baselines use haplotypes (Proposal 1's op covers them anyway).
- Track kernel rewrite: no users. (Measured slack is real: realign ~0.35 ns/position vs ~0.06 for haplotypes; per-track repeat of the variant walk, zero-filled scratch, element-indexed copies in `src/tracks/mod.rs:241,323,371`, `src/intervals.rs:19`.) The only track-like need is calla-methyl's per-haplotype ref-coordinate uint16 methylation counts, currently served by figwig — a missing feature, not a speed issue.
- Region-grouped batching as a required trick: violates random sampling.

## GVL read path facts (HEAD 3c68aa9c)

- Fixed Python cost ~40–70 µs/batch [measured-11-18]; not the bottleneck at large batches.
- SVAR1 haplotypes: 2 FFI calls; **diffs computed twice** (Python pre-pass + again inside `reconstruct_haplotypes_fused`, `src/ffi/mod.rs:776`) — 0.15–0.5 ms of a 1.2–8 ms batch; serial RC post-pass `src/reverse.rs:56` (9.3% in round-3 profile). Reference copied per haplotype, nothing shared across S·P.
- Annotated haplotypes: 9 B/position, 8.4 ms at b=1024 L=10 k, ~bandwidth-bound [measured-11-18].
- **Legacy VCF/PGEN-written datasets copy the full R·S·P offsets array 2–3× per batch** via `_as_starts_stops` (`_dataset/_genotypes.py:13`, `np.stack`) [code, copy verified]. SVAR-linked datasets don't. Cheap fix — candidate for a standalone issue.
- Variants (SVAR1): ~6.5 ns/variant + fixed cost [measured-11-18]; non-i32/f32 FORMAT dtypes fall back to a Python per-row loop (`_flat_variants.py:545`).
- Dataloaders: single producer process; one extra full copy into shm. No downstream project uses `to_dataloader` in production (aster tried: 2 GiB buffered stalls >2 min on full corpus, no provenance — aster `bench_dataloader.py:49-61`, `dataset.py:713-719`).

### Possible correctness bug — needs a fresh repro
Multi-track output appears written **track-major** (`out[track_ofst*n_per_track:...]`, `_dataset/_reconstruct.py:237`, `_dataset/_tracks.py:401`) while offsets/shape are built `(b, t, …)`-major (`repeat "b p -> b t p"`). On a synthetic 2-track tracks-only fixed-length dataset (b=4), `both[:, 0]` / `both[:, 1]` did not equal the single-track outputs; flattened rows matched a track-major pattern except rows 1 and 4 (unexplained). The repro script was lost with the session scratchpad. To re-repro: write a 2-bigwig dataset, compare `ds.with_tracks(["a","b"])[r, s]` against each single-track read. No downstream project uses tracks, so impact today is low.

## Downstream workload survey

All variant sources are SVAR/SVAR2. Batches are packed to an element budget (calla `PackingSampler`), mostly `num_workers=0`. Projects work around gvl #363 (zero SVAR2 variant counts; recomputed via `SparseVar2.region_counts`) and read private attrs (`_seqs.reference`, `_seqs.n_variants`, `_idxer`, `_FlatVariantWindows`, `Reference.from_path(in_memory=False)`) — these should become public API (issue not yet filed).

| Project / config | gvl mode | Region | Batch / sampling | Cohort | Loader-bound? |
|---|---|---|---|---|---|
| aster mut2vec | variant-windows flank 128, `unphased_union`, ploidy 1, INFO VAF/PURPLE | 5 Mb tiles | 4096 variants; 8 regions × 64 samples blocks + 65k shuffle buffer | GDC 14.6k / Hartwig 7k, somatic SVAR2 | **No** — 235k var/s vs 27.4k GPU, 8.6× headroom (aster `reports/mut2vec/README.md:443-450`) |
| aster tum2vec | `variants`, whole tumor | all tiles | 8 tumors, 12 workers | GDC | No after fixes; cost was aster's `_id83` |
| calla-asca train | variant-windows flank 16 + RefDataset peak | ragged, peak ±1 kb (≤5.3 kb) | 4096 elements (~350 pairs); few peaks × many individuals | 1kGP SVAR2, ~125 train indiv., 149k peaks | **Probably** [inferred]: GPU ~105 ms of 296–390 ms step (commit 7cc3261, README:73); `num_workers=0` |
| calla-methyl train | variant-windows flank 16/32 + RefDataset tiles + CpG intervals | fixed 4096 bp, 669k windows | 32k elements (~180 pairs), **uniform random** | 1kGP SVAR2, 498 ONT samples | Roughly balanced at 6 workers, ~20 GB/worker (commit a86d484) |
| gvf-geuvadis GVF | variant-windows flank 16 + RefDataset promoter/cCRE | 40 kb / 262 kb | 2^16 elements; one gene × ~290 samples | Geuvadis SVAR, 451 samples | **No** — 88–100% GPU (`plans/2026-08-19-ddp-bench-path.md:13`), epoch cached on GPU |
| gvf-geuvadis p-SAGE-net | haplotypes + reference, 40 kb | 40 kb | 16, random | same | **Yes** — CPU one-hot float32 (see Resume step 3) |

Within-batch reference sharing: yes in calla-asca train, gvf-geuvadis, calla-methyl eval; no in calla-methyl train and aster.

## SVAR2 format summary (genoray 6.0.2)
Per contig; each variant routed to `var_key` (sparse, sample-major CSR: u32 positions + 2-bit SNP / u32 indel keys) or `dense` (shared site table + 1-bit hap-major matrix, bit `hap*V_dense + v` with contig-wide `V_dense`) by `cost_model.rs` (SNP dense at ≥~190 carrier haps of 6404). No compression (bit-packing only, mmap). `max_del` bounds leftward search for indels. Long indel alleles in a per-contig LUT. gvl caches per-(region, hap) index ranges at `gvl.write` time (`genotypes/svar2_ranges/`, 28 B per non-empty cell) so reads do no search. Benchmarks: `docs/superpowers/notes/2026-07-03-svar2-mvp-benchmark.md` (old adapter path; germline SVAR2 5.1× slower than SVAR1, 5.67× smaller).
