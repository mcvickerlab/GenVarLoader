# Design: fail-fast VCF + `annotated`; defer VCF global variant ids

**Date:** 2026-07-21
**Branch:** `streaming-vcf-annotated-failfast` → merges into `streaming`
**Relates to:** #305 (record-backend global variant ids), #311 (annotated `var_idxs` wrong under
interior exclusion), #304 (Wave B variants output). genoray #134 (PGEN global ids, merged).

## Problem

`with_seqs("annotated")` emits per-position **dataset-global** variant ids
(`AnnotatedHaps.var_idxs`). SVAR1 carries these for free; PGEN derives them from the `.pvar` absolute
row index (genoray #134). A **VCF** source has no cheap per-record global id — genoray leaves
`RawRecord.global_idx = -1` (`vcf_reader.rs`), so gvl's local→global gather is skipped and the emitted
ids are **silently wrong** for any window that is multi-contig, mid-contig (narrowed), or drops an
interior variant behind a spanning deletion (#311) — i.e. essentially every real training window. The
merged Wave A parity test passed only because its fixtures are single-contig / whole-contig / no
interior exclusion.

Making VCF global ids correct requires a **one-time full-source record scan** (VCF has no
`.pvar`-equivalent), because narrowed windows demand the exact within-contig record rank. That is an
expensive price for one output mode.

## Key finding: variants output does **not** need the global id

`RaggedVariants`, `_FlatVariants`, and `_FlatVariantWindows` expose field **values** —
`alt`/`ref`/`start`/`ilen`/`dosage`/`var_fields` — and **no** `v_idxs` field. The written path's
`v_idxs = genos.data` is an internal **gather index** into the on-disk variant table
(`self.variants.info["AF"][v_idxs]`, `_gather_alleles(v_idxs, …)`). Streaming reads those fields
straight off each record (genoray's `ChunkAssembler` already decodes them into `DenseChunk`) and never
gathers from a table, so **variants / variant-windows are self-contained from the records**. Ordering
and AF filtering fall out of reading position-sorted records and each record's `INFO/AF`.

So the global id is needed **only** for `annotated` — a per-nucleotide provenance field with no
substitute. Dropping annotated for VCF removes the *entire* reason to build the VCF scan.

## Decision

Do **not** build the genoray VCF global-id machinery now. Instead:

**(a) Fail-fast guard (this PR).** In `StreamingDataset._materialize` (`_streaming.py`), where the
SVAR2 output-mode guard already lives, raise `NotImplementedError` for
`isinstance(self._backend, _VcfBackend) and _annotated`. The guard fires before `build_engine`, so no
Rust/FFI is involved. This converts the #311 silent-wrongness into an honest error pointing users to a
PGEN/SVAR source. `with_seqs("annotated")` docstring updated to state the limitation.

- Tests: drop `"vcf"` from `test_streaming_annotated_parity.py::BACKENDS` (SVAR1 + PGEN still prove the
  local→global gather); add `test_vcf_annotated_fails_fast` asserting the guard raises at
  materialization.
- The Rust `-1`/`remap_annot_local_to_global` fallback stays in place (still correct for the
  now-unreachable VCF path; shared with PGEN/SVAR1).

**(b) Record the finding.** Correct the Wave B spec
(`docs/superpowers/specs/2026-07-20-streaming-variants-output-wave-b-design.md`) with dated notes:
variants output is self-contained from records for VCF (no `global_v_idxs` dependency); the genoray
VCF scan is deferred and needed only for annotated. Annotate #305 / #311 with the same, cross-linking
seqlab#199 (CSI/TBI metadata pseudo-bins) as the eventual cheap cross-contig base.

## Scope / non-goals

- **No genoray change.** VCF `RawRecord.global_idx` stays `-1`.
- **No change to SVAR1/PGEN annotated**, plain VCF haplotypes, `with_len`, or jitter.
- Revisit the VCF scan only when a concrete VCF-annotated streaming use case appears.

## Verification

- `test_vcf_annotated_fails_fast` — VCF + annotated raises `NotImplementedError`.
- `test_streaming_annotated_parity.py` SVAR1 + PGEN cases still pass (behavior unchanged).
- `ruff check` / `ruff format` clean on touched files.
