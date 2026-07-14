# SVAR2 gvl MVP — validate & benchmark first, wire second

**Date:** 2026-07-03 · **Epic:** SVAR 2.0 · **Repo:** GenVarLoader
**Worktree:** `.claude/worktrees/svar2-m6b-kernel` (branch `svar2-m6b-kernel`)
**Follows:** `tmp/handoffs/2026-07-03-svar2-m6-plan3-mvp-benchmark-handoff.md`

## Summary

Plan 3 delivered the SVAR2 two-source reconstruction kernels and a validated
`SparseVar2Source` adapter (`python/genvarloader/_dataset/_svar2_source.py`). This
work finishes the MVP by **proving the SVAR2 value proposition on real chr21 data
before committing to invasive Dataset integration**. The session runs:

- **A** — clean up merged genoray worktrees (mechanical)
- **C** — end-to-end test for the `realign_tracks` adapter path
- **D** — validate both backends on real chr21 germline + somatic data
- **E** — benchmark SVAR1-backed vs SVAR2-backed gvl (latency + store size)
- **B** — Dataset dispatch wiring — **design sketch only**, finalized in a later
  brainstorm once E's numbers are in

### Guiding decision

Task D (validation) and Task E (benchmark) **do not depend on Task B** (full Dataset
dispatch): both run against `SparseVar2Source` directly. Task B is invasive to the
central reconstructor and its cost/shape depend on facts E will surface (notably the
all-samples-per-batch decode cost). So we **prove value first (A→C→D→E), then design
and wire B** with the benchmark in hand. This is measure-first and YAGNI: we do not
build invasive core integration before the payoff is confirmed.

## Background: why SVAR2 doesn't fit the SVAR1 reconstructor

The SVAR1 path (`Haps`, `_haps.py`) materializes per-region genotype offsets plus a
variant table on disk and memmaps them; `Haps` holds `genotypes` (a Ragged of
`v_idxs`), a `_Variants` table, and cached `ffi_static` arrays.

The SVAR2 path is a **live-query** model: `SparseVar2Source` calls genoray
`SparseVar2.overlap_batch(contig, regions)` at request time and decodes
`var_key ⋈ dense` inline through the Plan-3 kernels — there is no materialized
genotype table. The two models share none of that state, which is why B is a new
reconstructor rather than a flag on `Haps` (see Task B).

Three frictions make B genuinely invasive (deferred, but recorded here so the
validation/benchmark methodology accounts for them):

1. **Query-granularity mismatch.** `overlap_batch` is *per-contig, all-samples,
   all-ploidy* — it has no sample-subset argument (confirmed in genoray
   `python/genoray/_svar2_batch.py`). gvl's `Reconstructor.__call__` receives an
   arbitrary `idx` of `(region, sample)` pairs, possibly across multiple contigs.
2. **No write/detection path.** `_write.py` only handles `SparseVar`/VCF/PGEN; nothing
   writes an SVAR2-backed gvl dataset, and `_build_seqs` (`_open.py`) has no SVAR2
   branch.
3. **Reconstructor shape.** A branch inside `Haps` would leave `genotypes`/`_Variants`/
   `ffi_static` dead for SVAR2.

## Environment (from the handoff — do not re-derive)

- gvl worktree build: `pixi run -e default maturin develop` after Rust edits; the
  compiled module is `genvarloader.genvarloader` (abi3).
- Rust tests: `pixi run -e default cargo test --no-default-features [FILTER]`.
- Commits: prek hooks intentionally not installed here — use `git commit --no-verify`.
- genoray is the pre-built **2.15.0 wheel** in gvl's `default` env (not editable).
  Rebuild via `cd /carter/users/dlaub/projects/genoray && pixi run -e py310 maturin
  build --release` then re-point `pixi.toml` + `pixi install -e default`.

## Task A — Worktree cleanup (mechanical)

Confirm `git -C <wt> status` is clean, then remove the three merged/stale genoray
worktrees:

```
git -C /carter/users/dlaub/projects/genoray worktree remove .claude/worktrees/svar-2-m6b
git -C /carter/users/dlaub/projects/genoray worktree remove .claude/worktrees/svar-2-m6c
git -C /carter/users/dlaub/projects/genoray worktree remove .claude/worktrees/svar-2-m6-core
```

No design surface. Abort a removal if that worktree reports uncommitted work.

## Task C — `realign_tracks` e2e test

`SparseVar2Source.realign_tracks` (the region→R·S track expansion) has no end-to-end
test; only the Rust driver unit test `svar2_track_realign_del` exists.

**Oracle: gvl's SVAR1 _pure-Python_ `shift_and_realign_track_sparse`
(`_dataset/_tracks.py:708`), fed the variants materialized from genoray
`SparseVar2.decode`.** This is the strongest low-cost oracle and resolves the handoff's
(a)/(b) tension:
- It reuses *trusted* SVAR1 realign logic — including the DEL-anchor branch already
  present at `_tracks.py:755` — so we don't hand-roll a second track-shift
  implementation (the weakness of a fresh pure-Python consensus).
- It is a **pure-Python** implementation, genuinely independent of the SVAR2 **Rust**
  kernel under test (`shift_and_realign_tracks_from_svar2`, a closure refactor of the
  *Rust* SVAR1 kernel — a different implementation from this Python fallback).
- Setup is trivial: build a synthetic single-hap genotype layout from decode records —
  `geno_v_idxs = arange(n_h)`, `geno_offsets = [0, n_h]`, `v_starts = pos_h`,
  `ilens = ilen_h` — mirroring how `test_svar2_reconstruct.py` feeds decode records to
  its oracle.

**Construction:**
- Reuse the store builder + `decode_batch` extraction from
  `tests/test_svar2_reconstruct.py`.
- Use **DEL-only variants** so insertion-fill is bypassed and the strategy choice is
  irrelevant (any valid `strategy_id`/`params` works — track realign is ilen-only for
  DELs).
- Define a per-region reference track (f32); realign through the SVAR2 adapter path
  (`SparseVar2Source.realign_tracks`) and, per hap, through the pure-Python oracle fed
  that hap's decoded `(pos, ilen)` with `shift=0` and `out` sized to the hap length.
- **Assert per-`(r,s,p)`** float equality between the two (`np.testing.assert_allclose`).

Place alongside `tests/test_svar2_reconstruct.py`.

## Task D — Real-data validation (exploratory, not pytest)

Prove the whole thing on real chr21 in `/carter/users/dlaub/repos/for_loukik/`.

**Inputs**
- Germline (1000G): `chr21.bcf` (177 MB, 3202 samples) — **no `.csi`**, must index.
- Somatic (GDC): `gdc.chr21.bcf` (1.1 GB, 16007 samples), `.csi` present.
- **Reference:** `/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa` (GDC GRCh38;
  `.fai` and a gvl-preprocessed `.gvlfa` already present).

**Steps**
1. Verify contig naming (`chr21` vs `21`) between the FASTA and each BCF; if the
   germline BCF uses `21`, `bcftools annotate --rename-chrs` (or query with the BCF's
   own naming).
2. `bcftools index --csi chr21.bcf` (germline).
3. Normalize+atomize both BCFs to biallelic: `bcftools norm -m -any --atomize -Ob`,
   reusing the `genoray_pipeline.py` / `run_conversion_pipeline` recipe (samples via
   `bcftools query -l`, chroms via the canonical `##contig` regex,
   `run_conversion_pipeline(..., chunk_size=25_000, ploidy=2, ...)`).
4. Build **both** stores per source: `.svar` (SVAR1, `SparseVar.from_vcf`) and `.svar2`
   (SVAR2, `run_conversion_pipeline`) — genoray 2.15.0 wheel.
5. **Validate** gvl returns haplotypes + variants through both backends:
   - SVAR2 haplotypes: `SparseVar2Source(sv2).reconstruct("chr21", regions,
     ref_bytes, ref_offsets, pad_char)`.
   - SVAR2 variants: genoray `SparseVar2.decode`.
   - SVAR1: a gvl `Dataset` over the `.svar` store (haplotype + variants modes).
   - Spot-check a few regions agree in spirit (both from the same source variants).

**Scope:** small — a handful of regions × a few samples for the correctness spot-check.
Correctness is already proven by the Rust + adapter tests; D proves the **real-data
plumbing**: that genoray converts real multi-thousand-sample BCFs and that the adapter
works on a non-synthetic store.

**Compute:** germline may run interactively; the somatic conversion (16k samples,
1.1 GB) is heavy — run via `sbatch -p carter-compute`.

**Success:** both stores build from the real BCFs, and gvl returns non-empty, sane
haplotypes + variants through both backends, with spot-checks agreeing.

## Task E — Benchmark SVAR1-backed vs SVAR2-backed gvl

Once D produces both stores, measure and tabulate.

**Fairness rule (the crux):** query the **same workload** on both backends — **all
samples for a fixed region set**. This matches genoray's per-contig/all-samples query
granularity and is a realistic population-scale workload; a `(region, sample)`-subset
workload would unfairly tax the SVAR2 adapter, which always decodes all samples.

**Method:** warm caches, N repeats, report **median**. Same regions/sample set for both
backends.

**Measurements** (germline **and** somatic):
- **Hap latency** — SVAR1 gvl `Dataset` vs SVAR2 `SparseVar2Source`.
- **Variant latency** — SVAR1 gvl variants mode vs genoray `SparseVar2.decode`.
- **Store size** — `du` of the `.svar` (SVAR1) vs `.svar2` (SVAR2) directory.

**Output:** a 2×3 table — germline/somatic × {hap latency, variant latency, store
size}.

**Recorded caveat:** this compares adapter-vs-Dataset, not Dataset-vs-Dataset (B is not
wired). SVAR2 latency therefore excludes gvl's batching/collation overhead. Somatic
(dense somatic mutations) is where SVAR2's two-channel `var_key ⋈ dense` layout should
win on store size.

## Task B — Dataset dispatch (design sketch; finalize after E)

Deferred by the session's sequencing decision. Direction for the later brainstorm:

- A **new `SparseVar2Reconstructor`** implementing the `Reconstructor` protocol
  (`_protocol.py`) — *not* a branch inside `Haps` (SVAR2 shares none of `Haps`'s
  `genotypes`/`_Variants`/`ffi_static` state).
- `__call__` adapts gvl's batch model to genoray's: group `idx`→`(r,s)` **by contig**,
  decode all-samples-per-region via `SparseVar2Source`, **select** the requested
  `(r,s,p)` rows, thread jitter `shifts` and contig `ref` bytes from `Reference`
  (`_reference.py`), and return the identical `_Flat`/`Ragged` output contract.
- A new **write branch** (record an svar2 link + a `backend:"svar2"` flag in
  `metadata.json`, no genotype materialization) plus **open-path** detection in
  `_build_seqs` (`_open.py`).
- **Open question E will inform:** the all-samples-per-batch decode cost — whether to
  push sample-subsetting down into genoray (`overlap_batch`) or accept
  decode-all-then-select for the MVP.
- **Additive guarantee:** the SVAR 1.0 path must stay byte-identical; existing SVAR1
  haplotype/track tests must remain green.

## Carried-over gotchas (from the handoff)

- Subagents default to the **main repo, not the worktree** — every dispatch must `cd`
  the gvl worktree and guard with `git rev-parse --show-toplevel`. The gvl worktree is
  a different repo from genoray.
- `cargo test` needs `--no-default-features`; `maturin develop` after Rust edits.
- genoray `SparseVar2.decode`/`overlap_batch` return **empty ALT for pure DELs** —
  reconstruction injects the anchor `ref[pos]`. Any SVAR2 consumer must honor this.
- genoray positions are **0-based**; hap order is region-major `h=(r·S+s)·P+p`.
