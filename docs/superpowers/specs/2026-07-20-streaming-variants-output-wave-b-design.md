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
  `annot_v_idxs` entry after reconstruction (the annotated path).

The written variants oracle (branch `main`):

- `RaggedVariants` (`_dataset/_rag_variants.py:201`) — shape `(batch, ploidy, ~variants)`;
  requires `alt` + `start` + **one of** `{ref, ilen}`; `ilen`/`end` derivable from allele lengths;
  extra numeric FORMAT/INFO fields ride along.
- REF bytes / AF / dosage / custom FORMAT are all read from **on-disk tables** in the written
  dataset dir (`variants.arrow`, `dosages.npy`, `var_field_data`), loaded into memory at open —
  **not** re-read from source. Streaming has no dataset dir, so it must produce these from the
  **live** source (genoray `DenseChunk` for VCF/PGEN; the SVAR store).

## Design

### PR-A (#305) — correct dataset-global variant ids for the record backends

**Problem.** `RecordBackend::generate` emits global ids as `var_base + local`. Both
`VcfWindowFiller::fill` and `PgenWindowFiller::fill` currently hard-code `slot.var_base = 0`
(`vcf.rs:170`, `pgen.rs:614`). #305's body is **stale**: it describes PGEN as already correct from
`var_start`, but Wave A commit `0b98174b` *reverted* PGEN to `0` ("was buggy PGEN var_start"),
so #305 covers **both** backends.

**Why `var_start` alone is wrong.** `var_start` (PGEN) / the window's first candidate (VCF) is a
**padded, over-inclusive** search lower bound. The `PgenRecordSource`/`VcfRecordSource` then run a
precise anchor-trimmed `extent_overlaps` filter that can drop a **leading prefix** of padded-in
candidates whose trimmed extent doesn't reach `win_start`. When it drops `skip_count` of them, the
window's local column 0 is global id `var_start + skip_count`, not `var_start`.

**Why `var_base + local` is still valid.** The dropped candidates are always a *contiguous leading
prefix* of the candidate range: candidates are position-sorted, and the window is a suffix-overlap,
so anything not overlapping `[win_start, …)` sorts before it (even a long spanning DEL is anchored
at a POS before the window). No interior candidate is dropped, so the kept set is a **contiguous
global range** starting at `var_start + skip_count`, and a single additive base recovers every id.

**Fix.**

- The record source reports `skip_count` (the number of leading candidates its precise filter
  dropped) — a gvl-internal addition to `Pgen/VcfRecordSource`, **no genoray change**.
- `PgenWindowFiller::fill`: `slot.var_base = var_start + skip_count` (`var_start` is already global
  via the `.pvar` prescan's `contig_ranges`/`contig_pos`, `pgen.rs:407-410`).
- `VcfWindowFiller`: it has no per-contig index today. Add a **per-contig cumulative
  variant-count table** built once at `VcfWindowFiller::new` (a lightweight header/index pass,
  mirroring PGEN's `.pvar` prescan and how `sample_indices` is already resolved once at
  construction rather than per-window). This yields `contig_global_lo`; then
  `slot.var_base = contig_global_lo + within_contig_offset + skip_count`.

**Tests.** A multi-contig VCF **annotated-output** parity fixture (mirrors
`svar1_multicontig_fixture`, which already proves SVAR1's `var_base = 0` global-ids path is correct
for multi-contig) plus the analogous PGEN case. This unblocks correct multi-contig / narrowed
`annot_v_idxs` for the **annotated** output already shipped in Wave A, and provides the global-id
foundation the variants surface (PR-B1) needs. Closes #305.

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
per-hap CSR and `var_base` (PR-A). So the first variants PR ships without touching the decode.

- **New consumer path** in `RecordBackend` (and the SVAR1 backend): for the `[row_lo, row_hi)`
  slice, gather each `(region, sample, ploid)`'s window-local `v_idxs` from the per-hap CSR
  (`geno_offsets`/`geno_v_idxs`), **clip to the region window** (PR-B0 semantics), map local→global
  via `var_base`, and feed the **existing** `assemble_variant_buffers_u8/i32` kernel plus the
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
- **Branch (resolved)**: Wave A is merged into `streaming` (PR #308), so **every Wave B PR —
  including PR-A and the PR-B0 #202 fix — targets `streaming` directly.** No stacking on the (now
  deleted) `spec/277-output-mode-wave-a` branch, and no `main`-vs-`streaming` split for #202.

## Testing / parity plan

- **Parity oracle**: byte-identical vs `gvl.write()` + `Dataset[r, s]` under the matching
  `with_seqs(...)` / `min_af` / `max_af` / `var_fields`, on the **corrected** (#202-clipped) written
  path. One parity fixture per knob, on **SVAR1 + VCF + PGEN**.
- **Multi-contig / narrowed windows** (the #305 gap): a multi-contig VCF (and PGEN) fixture where
  the window does not start at global variant 0, asserting global `v_idxs` / `annot_v_idxs` are
  exact. Mirrors `svar1_multicontig_fixture`.
- **AF-missing guard**: setting `min_af`/`max_af` without a cached `AF` column raises the same error
  as the written path.
- **Empty groups**: `(region, sample, ploid)` cells with no in-window variant produce the written
  path's dummy fills (`DummyVariant`, `_flat_variants.py:37`).
- **Rust unit tests**: `var_base = var_start + skip_count` under a synthetic leading-skip window;
  the per-contig VCF count table; the variants consumer's local→global mapping and region clip.
- **Rebuild note** (CLAUDE.md): `maturin develop --release` before pytest after any `src/` change;
  run the full tree before pushing symbol/shared-code changes.

## Decomposition (PR stack)

| PR | Scope | Depends on | Parallel? |
|----|-------|-----------|-----------|
| **PR-A** | #305 `var_base` (VCF+PGEN) + `skip_count` + VCF per-contig table + multi-contig fixtures | Wave A (merged, #308) | — foundation |
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
- **`skip_count` reporting** — requires a small `Pgen/VcfRecordSource` API addition; if the source
  can't cheaply report it, the filler can recompute the leading-skip from the window's precise
  overlap predicate. Either way gvl-internal.
- **VCF per-contig count pass cost** — one construction-time pass over the index/header; acceptable
  (same amortization class as the `.pvar` prescan). Watch that it doesn't re-scan per window.
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
