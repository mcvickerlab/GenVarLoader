# Design: `StreamingDataset` — variants-output surface (Wave B) + global variant ids

**Date:** 2026-07-20
**Status:** design (pending spec review)
**Roadmap:** `docs/roadmaps/streaming-dataset.md` (Plan 5)
**Issues:** [#304](https://github.com/mcvickerlab/GenVarLoader/issues/304) — Wave B (variants-output surface) ·
[#305](https://github.com/mcvickerlab/GenVarLoader/issues/305) — record-backend `var_base` for narrowed/multi-contig windows
**Folds in:** [#202](https://github.com/mcvickerlab/GenVarLoader/issues/202) — `with_seqs("variants")` not clipped to the region window ·
[#231](https://github.com/mcvickerlab/GenVarLoader/issues/231) — custom per-call FORMAT/INFO `var_fields`
**Follows:** Wave A [#277](https://github.com/mcvickerlab/GenVarLoader/issues/277) — `with_len` +
jitter + `with_seqs("annotated")` — **merged into `streaming`** via PR
[#308](https://github.com/mcvickerlab/GenVarLoader/pull/308) (2026-07-20)
**Target branch:** `streaming` (Wave A already merged; every Wave B PR targets `streaming` directly)
**Backends:** SVAR1 + VCF + PGEN (the backends merged into `streaming`). SVAR2 ([#298](https://github.com/mcvickerlab/GenVarLoader/issues/298)) inherits the generic code when it merges.

## Summary

Bring the write-free `StreamingDataset` up to the written `Dataset`'s **variant-level** output
surface — the "expensive half" of output-mode breadth that Wave A deliberately deferred. Unlike
Wave A (which reused the existing window buffer and the fused reconstruct kernels), Wave B needs
**new per-variant channels decoded into the window buffer** and a **new consumer path** that
bypasses reconstruction entirely.

What ships:

- **`sds.with_seqs("variants")`** → `RaggedVariants` batches, byte-identical to `gvl.write()` +
  `Dataset[r, s]` under the matching config.
- **`sds.with_seqs("variant-windows")`** → `_FlatVariantWindows` (flanked tokenized ref/alt
  windows).
- **`min_af` / `max_af`** filtering (variants output only — matches the written path, which raises
  `NotImplementedError` for haplotype/annotated AF filtering, `_haps.py:692-695`).
- **`var_fields`** — custom per-call FORMAT/INFO columns (default `["alt", "ilen", "start"]`,
  issue #231).

Two decisions taken during design (see [Decisions](#decisions)):

1. **#202 (region clipping): fix both paths.** Streaming clips variants to the region window from
   the start; the written path is corrected to match (its current `regions=None` gather returns an
   unclipped set). Parity is asserted against the **corrected** oracle.
2. **Scope: full Wave B, delivered as a stack of small PRs** (below), docs/skill/api folded into
   each PR that adds a knob.

### Why this is the expensive half

The written variants path (`get_variants_flat`, `python/genvarloader/_dataset/_flat_variants.py:775`)
does **not** use a reconstruct kernel. It gathers per-`(region, sample, ploid)` sparse variant
indices, applies AF filtering, and assembles ragged buffers via `assemble_variant_buffers_u8/i32`
(`src/ffi/mod.rs:463-576`) plus `gather_*`/`compact_keep_*`/`fill_empty_*` helpers
(`src/variants/mod.rs`). It consumes per-variant data the streaming window buffer
(`src/record_stream/transpose.rs::DecodedWindow`) does **not** carry today:

- **REF allele bytes** — for `RaggedVariants.ref`/`.end` and `variant-windows(ref="allele")`.
- **the `AF` INFO field** — for `min_af`/`max_af`.
- **dosages / per-call FORMAT fields** — for `var_fields` (#231).
- **dataset-global variant ids** — streaming's `geno_v_idxs` are window-**local** column indices;
  the written `v_idxs` are global. This is exactly the gap #305 closes.

`fill_decoded_window` throws all of this away today; Wave B extracts what each knob needs, per
backend.

## Background: what already exists (Wave A base)

`StreamingDataset` (`python/genvarloader/_dataset/_streaming.py`) is a frozen-dataclass lazy view,
**iteration-only** (`to_iter`; `__getitem__` returns `None`). It **always requires a reference**
(`_streaming.py:179-181`), so every backend already holds the full reference contig bytes
(`ContigRef.ref_bytes`, `src/record_stream/engine.rs:69`). Wave A established:

- `with_seqs(kind)` dispatch (`_streaming.py:712`) — currently `"haplotypes"`/`"annotated"` only;
  `"variants"`/`"variant-windows"` raise `NotImplementedError` pointing at this issue.
- The `DecodedWindow` static table (`v_starts`/`ilens`/`alt_alleles`/`alt_offsets`) + per-hap CSR
  (`geno_offsets`/`geno_v_idxs`) + **`var_base`** (`src/record_stream/transpose.rs:29-46`).
- The `WindowFiller` trait (`fill(job, contig, slot)`, `engine.rs:44`) with `VcfWindowFiller`
  (`vcf.rs`) and `PgenWindowFiller` (`pgen.rs`); the producer/consumer engine recycles the
  `DecodedWindow` slot between windows (every filler MUST set `var_base` on every call).
- `generate_batch_core` (`src/ffi/mod.rs:994`) adds `var_base` to every non-negative
  `annot_v_idxs` entry after reconstruction (the annotated path). **PR-A retires this scalar
  `var_base` in favor of a per-variant `global_v_idxs` array — see the PR-A correctness note.**

The written variants oracle (branch `main`):

- `RaggedVariants` (`_dataset/_rag_variants.py:201`) — shape `(batch, ploidy, ~variants)`;
  requires `alt` + `start` + **one of** `{ref, ilen}`; `ilen`/`end` derivable from allele lengths;
  extra numeric FORMAT/INFO fields ride along.
- REF bytes / AF / dosage / custom FORMAT are all read from **on-disk tables** in the written
  dataset dir (`variants.arrow`, `dosages.npy`, `var_field_data`), loaded into memory at open —
  **not** re-read from source. Streaming has no dataset dir, so it must produce these from the
  **live** source (genoray `DenseChunk` for VCF/PGEN; the SVAR store).

## Design

### PR-A (#305) — per-variant dataset-global variant ids for the record backends

> **Design corrected during plan-writing** (2026-07-20). The original approach — a scalar
> `var_base` added to window-local ids (`var_base + local`) — is **fundamentally insufficient**,
> and #305's body (a `var_start`-based base) is doubly wrong. A cross-repo genoray change replaces
> it. See the correctness note below.

**The real defect: the kept set is non-contiguous.** `OverlapMode::Variant` filters on variant
**extent** (POS + ref_consumed), applied per record in genoray's `vcf_reader.rs:535-554`
(`extent_overlaps`). A window can therefore **exclude an interior variant behind a spanning
deletion** — e.g. regions `[(110,115)]`, variants DEL@100 (extent→120, kept), SNP@105 (extent 106,
**dropped**), SNP@112 (extent 113, kept): the survivors are global ids `{0, 2}`, not `{0, 1}`.
Same-POS mixed-extent ties (cf. the SVAR1 max-ends tie bug) are a second source of interior
exclusion. genoray's own SVAR2 query path proves this: `spine.rs` test
`test_gather_keys_excludes_interior_non_overlap_behind_spanning_deletion` asserts exactly
`[kr(100,0), kr(112,2)]`.

The record-stream `DenseChunk` (`genoray types.rs:137-179`) then **renumbers survivors to dense
local indices `0..n` and carries no per-variant global id** — the gap is destroyed inside genoray.
A scalar `var_base` can only reconstruct a **contiguous** range `{base, base+1, …}`, which
mismatches the written oracle's gapped `{base, base+2, …}` (the write path stores true global ids
with gaps: `_var_ranges.py` `with_row_index` over the atomized per-contig table + the same
extent-overlap join). **No value of `var_base` fixes this** — the fix must carry per-variant global
ids. (SVAR1 and the SVAR2 query path already do, via `pack_vk_src`; only the VCF/PGEN record-stream
path is affected.)

**This is a latent bug in already-merged Wave A.** `with_seqs("annotated")` on VCF/PGEN emits
`annot_v_idxs` via `var_base + local` (`generate_batch_core`, `ffi/mod.rs:994`), so it is silently
**wrong for any window containing a spanning deletion or same-POS tie** — even single-contig at
window 0, not just the multi-contig/narrowed case the code comment ("KNOWN GAP", `vcf.rs:153-170`)
acknowledges. PR-A fixes annotated too.

**Fix (decision: genoray emits global ids).** Mirror the SVAR2 `pack_vk_src` pattern:

- **genoray PR** — add a per-variant global-id column to `DenseChunk` (e.g. `global_idx: Vec<i32>`,
  one per surviving variant, in chunk order), populated by the record source through
  atomization/exclusion so each survivor carries its **true dataset-global** id:
  - **PGEN** is index-addressed (`PgenRecordSource` scans a global `[var_start, var_end)` range),
    so a survivor's id is `var_start + scan_offset` — cheap and tie-safe, no position matching.
  - **VCF** is tabix position-addressed; the source needs a **per-contig variant base** (the
    cumulative atom count before the fetch region). genoray adds a per-contig variant index built
    once at source construction (it has no cheap `contig_ranges`-equivalent today) so the running
    within-contig atom index + that base gives the global id.
  - Merge to genoray `main`; bump gvl's git `rev` (genoray is a first-party dep reached by rev
    bump — CLAUDE.md → Development Notes; the roadmap's other genoray prerequisites set precedent).
- **gvl PR-A** — bump the rev; add `global_v_idxs: Vec<i32>` to `DecodedWindow`; populate it in
  `fill_decoded_window` from `chunk.global_idx`; **retire the scalar `var_base`** — `generate`
  emits `annot_v_idxs` (and, in PR-B1, variants `v_idxs`) directly from `global_v_idxs[local]`,
  not `var_base + local`. The `generate_batch_core` `var_base` add-back is removed/zeroed.

**Tests.** Parity fixtures that specifically exercise **interior exclusion** (a spanning deletion
that drops an interior variant; a same-POS SNP+DEL mixed-extent tie) plus **multi-contig** and
**narrowed/partial-prefix** windows, on **both VCF and PGEN**, asserting `annot_v_idxs` are the
exact gapped global ids the written `Dataset` produces. Mirrors `svar1_multicontig_fixture` and adds
the interior-exclusion case (which SVAR1 already handles correctly and the record path currently
does not). Closes #305; provides the global-id foundation PR-B1 needs.

### PR-B0 (#202) — clip variants to the region window (both paths)

On the written path, `get_variants_flat` gathers `v_idxs` straight from the per-cell sparse
genotype set and uses `regions` only for flank/window token assembly — never to filter by region
extent (the `regions=None` call site the issue names). So `RaggedVariants` can include variants
outside the queried window (boundary-overlapping indels; for PGEN, variants elsewhere on the
contig). The haplotype/annotated path does clip correctly (`regions=req.regions, keep=req.keep`).

**Fix.** Thread a region-extent-overlap filter into the variants `v_idxs` gather so
`with_seqs("variants")` is windowed the same way haplotype output is. Streaming's variants consumer
(PR-B1) applies the identical clip natively (its haplotype path already clips per-row via the `rb`
region bounds in `generate_batch_core`). Closes #202.

**Branch/ordering.** The written-path fix targets **`streaming`** with the rest of Wave B (decided:
keep all Wave B on one branch to avoid cross-branch merge-timing coupling with the parity tests).
It lands before PR-B1 so PR-B1's variants-parity is asserted against the corrected oracle. The fix
flows to `main` at the streaming milestone merge, closing #202 for all users then; it also stands
alone, so it can be cherry-picked to `main` earlier if desired.

### PR-B1 — streaming variants output (`RaggedVariants`), default fields, region-clipped

The default `var_fields = ["alt", "ilen", "start"]` needs **no REF and no AF** — all three come
from channels `DecodedWindow` already carries (`v_starts`/`ilens`/`alt`/`alt_offsets`) plus the
per-hap CSR and the per-variant `global_v_idxs` (PR-A). So the first variants PR ships without
touching the decode.

- **New consumer path** in `RecordBackend` (and the SVAR1 backend): for the `[row_lo, row_hi)`
  slice, gather each `(region, sample, ploid)`'s window-local `v_idxs` from the per-hap CSR
  (`geno_offsets`/`geno_v_idxs`), **clip to the region window** (PR-B0 semantics), map local→global
  via `global_v_idxs[local]` (PR-A), and feed the **existing** `assemble_variant_buffers_u8/i32` kernel plus the
  `gather_*`/`fill_empty_*` helpers. This is the DRY win: the written and streaming variants paths
  share one Rust assembly kernel; the only difference is where the per-hap variant lists come from
  (streaming's window-local CSR vs the written per-cell sparse set).
- **Python**: `StreamingDataset.with_seqs("variants")` maps `_seq_kind → RaggedVariants`; `to_iter`
  pulls a variants batch (a new engine `next_batch_variants`) and yields `RaggedVariants`.
- **Parity**: byte-identical vs the corrected written `Dataset.with_seqs("variants")` on SVAR1 +
  VCF + PGEN.

### PR-B2 — `min_af` / `max_af`

- **VCF/PGEN**: request the `AF` INFO column from genoray (`DenseChunk.info_staged`,
  `StagedColumn::Float`) by adding an `AF` `FieldSpec` to the chunk reader the filler constructs.
- **SVAR1**: read AF from its store.
- Add an `afs` channel to `DecodedWindow`; the variants consumer builds
  `keep = (af >= min_af) & (af <= max_af)` and compacts before assembly — mirroring
  `get_variants_flat`'s AF filter (`_flat_variants.py:810-833`).
- **Guard parity**: the written path raises if `min_af`/`max_af` is set without a cached `AF`
  column (`_haps.py:319-325`); streaming reproduces the same error surface.

### PR-B3 — `var_fields` (dosage + custom FORMAT/INFO, #231)

- **dosage / FORMAT**: request the columns from genoray. `DenseChunk` exposes two encodings —
  dense `format_staged` (multi-sample VCF, PGEN) or carrier-sparse `format_by_carrier` (the k-way
  merge over single-sample VCFs); the consumer handles both.
- **numeric INFO** columns (beyond AF): `info_staged`.
- **SVAR**: read from store (`_svar_format_fields`).
- Per-sample gather into ride-along ragged fields, compacted by the same `keep` mask, mirroring
  `get_variants_flat:822-883`. `available_var_fields` on the streaming side is computed from what
  the live source exposes (analogous to `Haps.__post_init__:310-317`).

### PR-B4 — REF bytes + `variant-windows` (`_FlatVariantWindows`)

- **REF bytes** (VCF/PGEN): genoray's `DenseChunk` does not expose REF (`// pub refe` commented
  out in `types.rs`). Since `StreamingDataset` always has a reference, slice
  `ref_bytes = reference[start : start + ref_len]` with `ref_len = alt_len − ilen` (`alt_len` from
  `alt_offsets`) — exact for the normalized, left-aligned biallelic variants gvl requires. **No
  genoray change.** SVAR reads REF from its store. This enables the explicit `ref` var_field.
  - *Risk / validated by parity*: this assumes VCF `REF` == reference at `[POS, POS+len(REF))`,
    which holds for variants normalized against the dataset's reference. The parity test is the
    gate; if an edge case diverges, the fallback is to expose REF from genoray's `DenseChunk`.
- **variant-windows** (`_FlatVariantWindows`, `_flat_variants.py:267`): tokenized flanked ref/alt
  windows. The token LUT is built in Python `with_seqs` exactly as the written path does
  (`_impl.py:742-757`, `VarWindowOpt` with `ref`/`alt ∈ {"window","allele"}`, `flank_length`,
  `token_alphabet`, `unknown_token`). `ref="window"` flanks the reference genome (streaming has
  it); `ref="allele"` uses REF bytes (above). Reuse the written window-assembly kernels.

## Decisions

- **#202 → fix both paths** (chosen). Streaming clips natively; the written path is corrected so
  parity is against a correct oracle rather than a preserved bug. Consistent with the migration
  convention (a buggy oracle is fixed in its own PR, not reproduced) and keeps `with_seqs("variants")`
  meaningfully windowed for all users.
- **Scope → full Wave B, staged PRs** (chosen). One design, delivered as PR-A → PR-B0 → PR-B1 →
  PR-B2 → PR-B3 → PR-B4, so each lands small and independently parity-tested.
- **Reuse `assemble_variant_buffers_*`** rather than a streaming-specific kernel — the streaming
  CSR + static table already match the kernel's inputs.
- **REF via reference-slice**, not a genoray `DenseChunk` change — avoids a cross-repo dependency;
  parity is the gate.
- **Global ids → genoray emits them** (chosen, corrected 2026-07-20). Interior exclusion makes the
  kept set non-contiguous, so the scalar `var_base` is unfixable; genoray's `DenseChunk` gains a
  per-variant global-id column (SVAR2 `pack_vk_src` pattern), consumed via `DecodedWindow.global_v_idxs`.
  This is a cross-repo genoray PR + rev bump — the one place Wave B deliberately touches genoray
  (REF stays in-repo via reference-slice). It also fixes the latent Wave A annotated bug.
- **Branch (resolved)**: Wave A is merged into `streaming` (PR #308), so **every Wave B PR —
  including PR-A and the PR-B0 #202 fix — targets `streaming` directly.** No stacking on the (now
  deleted) `spec/277-output-mode-wave-a` branch, and no `main`-vs-`streaming` split for #202.

## Testing / parity plan

- **Parity oracle**: byte-identical vs `gvl.write()` + `Dataset[r, s]` under the matching
  `with_seqs(...)` / `min_af` / `max_af` / `var_fields`, on the **corrected** (#202-clipped) written
  path. One parity fixture per knob, on **SVAR1 + VCF + PGEN**.
- **Interior exclusion (the #305 crux)**: a window with a spanning deletion that drops an interior
  variant, and a same-POS SNP+DEL mixed-extent tie, on **both VCF and PGEN**, asserting
  `annot_v_idxs` (and PR-B1 `v_idxs`) are the exact **gapped** global ids the written `Dataset`
  produces — the case a scalar `var_base` cannot express.
- **Multi-contig / narrowed windows**: a fixture where the window does not start at global variant 0,
  asserting global ids are exact. Mirrors `svar1_multicontig_fixture`.
- **AF-missing guard**: setting `min_af`/`max_af` without a cached `AF` column raises the same error
  as the written path.
- **Empty groups**: `(region, sample, ploid)` cells with no in-window variant produce the written
  path's dummy fills (`DummyVariant`, `_flat_variants.py:37`).
- **genoray-side tests**: the `DenseChunk.global_idx` column is correct under interior exclusion for
  both the index-addressed (PGEN) and position-addressed (VCF) record sources (genoray PR's own
  suite, mirroring the existing `spine.rs` interior-exclusion test).
- **gvl Rust unit tests**: `fill_decoded_window` copies `global_idx → global_v_idxs`; `generate`
  emits `global_v_idxs[local]` (not `var_base + local`); the variants consumer's region clip.
- **Rebuild note** (CLAUDE.md): `maturin develop --release` before pytest after any `src/` change;
  run the full tree before pushing symbol/shared-code changes.

## Decomposition (PR stack)

| PR | Scope | Depends on | Parallel? |
|----|-------|-----------|-----------|
| **genoray PR** | `DenseChunk.global_idx` per-variant global-id column (PGEN index-addressed; VCF per-contig base) + interior-exclusion tests | — | ⛔ cross-repo prerequisite for PR-A |
| **PR-A** | #305: bump rev; `DecodedWindow.global_v_idxs`; retire scalar `var_base`; re-point annotated; interior-exclusion + multi-contig parity fixtures (fixes latent Wave A annotated bug) | genoray PR, Wave A (merged, #308) | — foundation |
| **PR-B0** | #202 written-path region clip (targets `streaming`) | — | ✅ independent of PR-A |
| **PR-B1** | streaming `with_seqs("variants")`, default fields, region-clipped, reuse assemble kernel | PR-A, PR-B0 | serial base for B2–B4 |
| **PR-B2** | `min_af`/`max_af` (AF channel) | PR-B1 | ✅ vs B3 |
| **PR-B3** | `var_fields` dosage + custom FORMAT/INFO (#231) | PR-B1 | ✅ vs B2 |
| **PR-B4** | REF bytes (reference-slice) + `variant-windows` | PR-B1 (+B3 for `ref` field) | last |

Docs / `skills/genvarloader/SKILL.md` / `docs/source/*.md` / `api.md` updates fold into each PR
that adds a public knob (CLAUDE.md public-API + docs-audit gates). Update
`docs/roadmaps/streaming-dataset.md` (Plan 5) status markers and the StreamingDataset project board
as each PR lands.

## Risks

- **REF reference-slice divergence** — mitigated by parity; genoray REF exposure is the fallback
  (PR-B4).
- **Cross-repo genoray change** — PR-A is gated on a genoray PR (`DenseChunk.global_idx`) merging to
  genoray `main` + a gvl rev bump. Sequence it like the roadmap's other genoray prerequisites; do
  the genoray PR first. The VCF per-contig base inside genoray is the non-trivial part (position-
  addressed, no existing `contig_ranges`); PGEN is cheap (index-addressed).
- **Latent Wave A annotated bug** — merged `with_seqs("annotated")` is wrong under interior
  exclusion today. PR-A fixes it, but until it lands the merged annotated output is unsafe for any
  VCF/PGEN cohort with spanning deletions; call this out on the board and in #305.
- **FORMAT dense-vs-carrier encodings** (#231) — the consumer must handle both `format_staged` and
  `format_by_carrier`; covered by per-source parity fixtures.
- **SVAR2** — not in this wave; it inherits the generic consumer when #298 merges, but its store
  read of REF/AF/FORMAT must be wired then.

## References

- Streaming roadmap: `docs/roadmaps/streaming-dataset.md` (Plan 5).
- Wave A spec: `docs/superpowers/specs/2026-07-19-streaming-output-mode-breadth-wave-a-design.md`.
- Written variants path: `python/genvarloader/_dataset/_flat_variants.py` (`get_variants_flat:775`),
  `_dataset/_rag_variants.py` (`RaggedVariants:201`), `_dataset/_haps.py`
  (`get_variants_flat` dispatch `:588-606`; AF `NotImplementedError` `:692-695`; provenance
  `_Variants.from_table:104-165`), `_dataset/_impl.py` (`with_seqs:642`, `var_fields:338-381`).
- Streaming engine: `src/record_stream/{engine.rs,transpose.rs,vcf.rs,pgen.rs}`,
  `src/ffi/{mod.rs,stream_engine.rs}`, `python/genvarloader/_dataset/_streaming.py`.
- Assembly kernels: `src/ffi/mod.rs:463-576` (`assemble_variant_buffers_*`), `src/variants/mod.rs`.
- genoray `DenseChunk` (pinned rev `1d756ad`): `info_staged`/`format_staged`/`format_by_carrier`,
  REF commented out.
