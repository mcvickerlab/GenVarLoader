# SVAR2 read-bound `Dataset.__getitem__` optimization — consolidated results (2026-07-06)

Plan: `docs/superpowers/plans/2026-07-05-svar2-readbound-getitem-perf.md`
Branches: gvl `svar2-m6b-kernel` (PR #266) + genoray `svar-2`.

This report consolidates Phase A (baselining) and Phase B (B1-B4 optimization)
of the SVAR2 read-bound `getitem` perf effort. It supersedes any per-task
summary for headline numbers; per-task detail lives in
`.superpowers/sdd/task-{A2,A3,B1,B2,B3,B4-step1,B4a,B4b,B4c}-report.md` and the
condensed ledger in `.superpowers/sdd/progress.md` (section "SDD Progress —
SVAR2 read-bound getitem perf").

## 1. Correction to the A3 baseline note (dense_union)

The original A3 native-baseline commentary is sometimes paraphrased as "the
union oracle (`dense_union()`) is never invoked on the read-bound path." That
overstates what was found and is corrected here:

- `dense_union()` **is** called on the read-bound path (genoray
  `src/query.rs:771`) — it is not absent.
- What A3 actually established (verified against the real call graph, not
  grep-only) is that it's **cheap and below the sampling threshold**: it never
  shows up with measurable self-time in any of the 4 `perf` captures, and the
  disqualifying whole-cohort entry points (`overlap_batch`/`overlap_sample`)
  are genuinely absent from the read-bound call chain
  (`SparseVar2.find_ranges` → `gather_haps_readbound`/`gather_ranges_readbound`).
  `genoray_core::search::SearchTree::build` (1.54% haplotypes_germline,
  0.55% haplotypes_somatic) is the benign per-region `find_ranges` search
  phase, not the whole-cohort union oracle.
- **Correct statement for future reference:** *`dense_union()` is called on
  the read-bound path but is cheap (below sampling threshold); the
  whole-cohort union/oracle path (`overlap_batch`/`overlap_sample`) is what's
  absent, not `dense_union()` itself.*

## 2. Baseline reference numbers (Phase A)

`perf stat -e instructions,cycles` on the whole profiled process (one
`gvl.write` + `Dataset.open` + K warm `ds[:, :]` calls), frame-pointer build,
committed in `tmp/svar2_mvp/prof_out/readbound/native_after_b1b3.md` — this
file is misleadingly named; **it holds the PRE-B1-B3 (original A3) capture**,
not a post-B1-B3 one (naming preserved as-is per the task record; do not
invert this when reading raw files).

| combo | K | instructions | cycles | insn/cycle |
|---|---|---|---|---|
| haplotypes_germline | 178 | 136,108,783,319 | 63,090,572,307 | 2.16 |
| variants_germline | 6547 | 434,768,134,979 | 142,981,599,031 | 3.04 |
| haplotypes_somatic | 38 | 211,440,945,021 | 78,077,607,534 | 2.71 |
| variants_somatic | 1922 | 470,732,727,589 | 138,279,404,076 | 3.40 |

A3 also profiled the Python layer (A2) and native layer (A3) to rank hot
functions. Python-layer hot path (both modes): pure-Python `_ragged_arange_gather`
(and `_2level`) issuing repeated small `numpy.arange`/`ndarray.repeat` calls —
the clearest vectorize/push-to-Rust candidate. Native-layer hot path: haplotypes
dominated by numpy int64 add/sub kernels + kernel/mmap page-fault time (~30%
each), with gvl/genoray Rust only ~8% of self-time; variants dominated by
`gather_haps_readbound` (12.85% germline) + `decode_variants_from_split` +
`split_to_flat` (11.64% combined, germline) + numpy `PyArray_Repeat` (9.13%).

`tmp/svar2_mvp/prof_out/readbound/native_baseline.md` (also misleadingly
named) holds a **second, POST-B1-B3 re-profile** captured in Task B4 Step 1 to
enumerate the cargo-asm work-list (not a matched-K comparison against the
table above — K differs run-to-run because the harness doesn't pin sample
count):

| combo | K | instructions | cycles | insn/cycle |
|---|---|---|---|---|
| haplotypes_germline | 191 | 135,051,821,403 | 64,154,539,866 | 2.11 |
| variants_germline | 7143 | 456,803,347,416 | 165,527,068,344 | 2.76 |
| haplotypes_somatic | 37 | 203,147,989,079 | 76,904,197,639 | 2.64 |
| variants_somatic | 1792 | 435,990,647,839 | 139,669,338,811 | 3.12 |

**Because K is not matched between these two captures (different runs,
different cohort sizes drawn each time), they are not diffed directly as a
before/after number.** The reliable before/after deltas are the per-task,
matched-K, same-session measurements below.

## 3. Optimizations applied — per-task matched-K deltas

All changes are **byte-identical**: the svar2 pytest suite
(`pytest tests/dataset -k svar2`, 32 tests reading the real `svar2_mvp`
stores — haplotypes, variants, and tracks parity vs the SVAR1 oracle) stayed
32/32 green through every task, with zero new failures/errors introduced in
the full tree at any step (see §5, Parity).

Each row below is its own same-session, same-K, git-stash-based before/after
measurement (not one cumulative run) — presented per-task as instructed,
since no single cumulative baseline-vs-final run was captured.

| Task | Repo | Change | Mode/cohort (K) | Instructions before → after | Δ instructions | Status |
|---|---|---|---|---|---|---|
| **B1** | gvl | Skip redundant pre-reconstruct diffs gather for deterministic haplotype reads (`need_hap_lengths` inverted-default) | haplotypes_germline (K=178) | 136,108,783,319 → 127,529,646,381 | **−6.3%** (cycles −4.7%) | byte-identical |
| **B2** | gvl | Pre-size `split_to_flat` + `decode_variants_from_split` allocations | variants_somatic (K=300) | 160,637,370,802 → 161,125,582,889 | **+0.30% (noise)** | byte-identical, kept anyway |
| **B2** | gvl | (same change, haplotypes) | haplotypes_somatic (K=300) | 895,401,688,408 → 895,815,473,126 | **+0.05% (noise)** | byte-identical, kept anyway |
| **B3** | gvl | De-dup twin ragged reorder-index computation in `_reconstruct_variants` | variants_germline (K=500) | 59,472,976,343 → 58,383,771,799 | **−1.83%** | byte-identical |
| **B4a** | genoray (`svar-2`) | `gather_haps_readbound` asm fix: skip-prefix→slice, bounds-check hoist, inline 2-pointer merge (replaces per-hap `merge_keys` allocation) | variants_germline (K=500) | 58.36e9 → 53.41e9 (avg of 3 runs/side) | **≈−8.5%** (−4.95B instr) | byte-identical (proven + tie-break test) |
| **B4b** | gvl | `decode_variants_from_split` asm fix: hoist per-iter `q = h/ploidy` division; `get_unchecked` on presence bit (proven in-bounds, `debug_assert`-guarded) | variants_germline (K=500) | ≈53,464.9M → ≈53,317.4M (avg of 2 runs/side) | **≈−0.28%** | byte-identical (proven) |
| **B4c** | gvl | `split_to_flat` asm fix: hoist `q = h/ploidy` division out of both hot loops (4 div → 0), no `unsafe` | variants_germline (K=500) | ≈53,296.4M → ≈53,212.6M (avg of 3 runs/side) | **≈−0.15%** | byte-identical |

**Headline result: B4a (genoray `gather_haps_readbound` asm fix) is the
single largest byte-identical win, ~8.5% instructions on the
variants_germline (K=500) workload** — larger than B1's 6.3% haplotypes win
and an order of magnitude larger than B4b/B4c. B2 is a null-delta-but-kept
scalability improvement (see reasoning below); B1 and B3 are Python-layer
structural/DRY wins with real, smaller, matched-K deltas.

### Why B2 shows no measurable win here

`decode_variants_from_split`/`split_to_flat` pre-sizing is a genuine
allocation-count reduction (proven via cargo unit tests and code review), but
the harness's fixed `REGIONS` (3 windows, ~500-1000bp each) keep
`dense_total`/`total_bits`/`cap` tiny per call, so few `Vec` reallocations are
actually eliminated per call. The eliminated cost is a rounding error against
the multi-billion-instruction total dominated by write+open setup and
Python/numpy glue at this harness's scale. The optimization is kept as a
zero-risk scalability improvement: its payoff should scale with region
size/variant density, which this benchmark doesn't exercise, and both targets
independently rank high in the A3/B4-Step-1 native profiles (`split_to_flat`
5.05-5.21% self-time, `decode_variants_from_split` 6.59-5.43% self-time).

## 4. Profiled-but-deferred candidates

Task B4 Step 1 re-profiled the native layer after B1-B3 and enumerated every
gvl/genoray Rust symbol with ≥1.5% self-time in any mode (6 functions). The
user approved a scoped **sequential** asm pass on the top 3 by ROI (B4a/b/c
above, all done). The remaining 3 were profiled and explicitly **deferred**,
not forgotten, as a controller-approved scope decision (do only the top-3 asm
targets this round; leave the rest for a follow-up pass):

| Symbol | Repo | Max self-time | Reason deferred |
|---|---|---|---|
| `svar2_codec::decode_key` | genoray `svar2-codec/src/lib.rs:237` | 2.33% (variants_germline) | Below the top-3 ROI cutoff; scope decision to do only the top-3 sequentially this round |
| `genoray_core::spine::merge_keys` | genoray `src/spine.rs:63` | 2.31% (variants_somatic) | Same; note B4a's inline merge in `gather_haps_readbound` already eliminated one call site of this pattern, but the standalone `merge_keys` function itself (used elsewhere) was not asm-optimized |
| `genoray_core::search::SearchTree::build` | genoray `src/search.rs:93` | 1.54% (haplotypes_germline) | Same; also note this is the **benign** per-region `find_ranges` search phase, not the whole-cohort union oracle (see §1) — deferring it is a pure perf scope call, not a correctness concern |

## 5. Parity

- **svar2 suite: 32/32 byte-identical, held through every task** (B1, B2, B3,
  B4a, B4b, B4c) — covers haplotypes, variants, and tracks paths against the
  SVAR1 oracle (`pytest tests/dataset -k svar2`).
- **Full tree: NOT fully green, but no regressions.** `pixi run -e dev gen`
  (ground-truth fixture generation) is **broken pre-existing to this plan**:
  `VcfBuilder.__init__() got an unexpected keyword argument 'fileformat'`
  (`tests/_builders/case.py:324`), a vcfixture (unpinned, `>=0.5.0`) API-drift
  issue unrelated to svar2 or this perf work. This produces a fixed ~428
  errors / 46 failed baseline in `pytest tests -q` that held constant,
  unchanged, through every task in this plan. The passed count held at 559
  (+1 from B1's added micro-test) with **zero new failures or errors**
  introduced at any step — confirmed by identical failing-test-ID sets with
  each change stashed vs applied.
- **This is a known issue / CI blocker requiring a separate fix**
  (pin vcfixture or update the `VcfBuilder` call site at
  `tests/_builders/case.py:324`), tracked here for visibility. It is **out of
  scope** for this perf plan (dependency/test-infra issue, not a code
  regression) and should be filed/fixed separately before branch CI can go
  green.

## 6. Deferred features (out of scope this round, by design)

- **Tracks**: out of scope for this round's profiling and optimization.
  `Svar2Haps.get_haps_and_shifts`'s tracks caller still runs the diffs kernel
  via the `need_hap_lengths=True` default (B1's inverted default preserves
  this — only the pure-haplotypes `__call__` entry point opts out with
  `need_hap_lengths=False`). The B1 double-gather is therefore **unaddressed
  for tracks by design**; tracks parity (`test_svar2_tracks_match_svar1*`)
  was verified unaffected/green throughout, but no tracks-path perf work was
  done.
- **Variant-windows**: guarded with `NotImplementedError` in `Svar2Haps` —
  cannot be profiled or optimized until implemented. Deferred until that
  feature lands.

## 7. API / format / docs reconciliation (B5 Step 2)

**Read-path internals only; no API/format/doc-surface change.**

- `need_hap_lengths` (B1) is an internal parameter on
  `Svar2Haps.get_haps_and_shifts`, not a public API — it is not exported and
  is not part of `genvarloader.__all__`.
- All B2/B3/B4a/B4b/B4c changes are Rust/Python read-path internals
  (allocation pre-sizing, asm-level instruction elimination, a de-duplicated
  index computation) with no change to any public symbol, function signature
  exposed to users, on-disk dataset format, or CLI/bcftools/plink2
  preprocessing requirement.
- No genoray kernel FFI signature changed (`gather_haps_readbound`,
  `decode_variants_from_split`, `split_to_flat` all keep their existing
  signatures — only their bodies changed).
- Per the "Maintaining the `genvarloader` skill" and "Docs audit" rules in
  `CLAUDE.md`: since nothing in `__all__`, `gvl.write`, `Dataset.open`, or any
  `Dataset.with_*` signature/default changed, **no update to
  `skills/genvarloader/SKILL.md`, `docs/source/api.md`, or any other prose doc
  is required**, and none was made. No genoray docs (`docs/roadmap/svar-2.md`)
  needed updating either, since no genoray kernel signature changed — the
  genoray commits below are pure asm/allocation fixes on existing functions.

## 8. Two-repo commit list

**gvl** (`svar2-m6b-kernel`, PR #266), in order:
- `56c7b36` — B1: skip redundant pre-reconstruct gather for deterministic haplotype reads
- `a297d24` — B2: pre-size split_to_flat + decode_variants_from_split allocations
- `ecfc057` — B3: compute the pos/ilen ragged reorder index once in variants decode
- `2bdee38` — B4 Step 1: re-profile native layer after B1-B3; enumerate cargo-asm work-list
- `1e894d2` — B4b: decode_variants_from_split asm fix (byte-identical)
- `85a8925` — B4b follow-up: debug_assert guard for the get_unchecked invariant
- `e99a0d9` — B4c: split_to_flat asm fix — hoist q=h/ploidy division out of hot loops (byte-identical)

**genoray** (`svar-2`):
- `a7c32b3` — B4a: gather_haps_readbound asm fix — kill skip/take waste, elide bounds checks, inline the per-hap merge (byte-identical)
- `69b3c97` — B4a follow-up: test covering gather_haps_readbound same-position SNP+indel tie; cross-ref merge_keys

## 9. Sources

Per-task detail, TDD evidence, and full profiler output: `.superpowers/sdd/task-{A2,A3,B1,B2,B3,B4-step1,B4a,B4b,B4c}-report.md`.
Condensed ledger: `.superpowers/sdd/progress.md` (section "SDD Progress — SVAR2 read-bound getitem perf").
Raw captures: `tmp/svar2_mvp/prof_out/readbound/{python_baseline.md,native_after_b1b3.md,native_baseline.md,asm_targets.md}`.
