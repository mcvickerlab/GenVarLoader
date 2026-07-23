# Streaming variants output, Wave B PR-B3 + PR-B4 — design

**Issue:** [#304](https://github.com/mcvickerlab/GenVarLoader/issues/304) (Wave B of
[#277](https://github.com/mcvickerlab/GenVarLoader/issues/277)) ·
**Branch target:** `streaming` ·
**Parent design:** `docs/superpowers/specs/2026-07-20-streaming-variants-output-wave-b-design.md`

Completes the streaming variants-output surface: `var_fields` (PR-B3) and
`with_seqs("variant-windows")` (PR-B4). PR-B0/B1 (`with_seqs("variants")`) and PR-B2
(`min_af`/`max_af`) have landed; these are the last two items in Wave B.

## Corrections to the parent design

Three findings from the landed code change what the parent spec assumed. They are the
reason this design exists as its own document rather than a plan against the parent.

**1. `format_by_carrier` is unreachable on gvl's streaming path.** The parent says the
consumer must handle both dense `format_staged` and carrier-sparse `format_by_carrier`.
gvl's VCF filler constructs `VcfRecordSource` (`src/record_stream/vcf.rs:66`), a natively
dense multi-sample source, and PGEN is dense as well. genoray produces
`carriers`/`format_by_carrier` **only** for the k-way merge over single-sample VCFs, which
gvl never constructs. Dense-only, with a hard error if `format_by_carrier.is_some()` as a
tripwire against a future genoray source change.

**2. FORMAT/dosage has no written parity oracle on VCF/PGEN.** `gvl.write` for a VCF or
PGEN source links genoray's index straight to `variants.arrow`
(`python/genvarloader/_dataset/_write.py:741`, `:913`). `Haps.from_path`'s `dosages` and
`var_field_data` loading lives entirely inside the `svar_meta.json` branch. So a written
VCF/PGEN dataset's `available_var_fields` is structurally
`{alt, ilen, start} ∪ numeric-index-columns ∪ {ref}` — **dosage and custom FORMAT are
SVAR-only on the written path.** Building them for streaming VCF/PGEN would create a
capability with no parity gate, contradicting the effort's byte-identical-parity contract.

**3. The `variant-windows` kernel already exists and is Python-free.**
`src/variants/windows.rs::assemble_windows_mode` *is* the written path's kernel (reached
via the `assemble_variant_buffers_u8`/`_i32` FFI) and takes plain `ndarray` views, so the
streaming producer thread can call it directly. PR-B4 is plumbing plus a REF-bytes input,
not a new kernel.

**Consequence: `ref` moves into PR-B3.** It is a `var_field`, not a windows-only input.
SVAR1 already reads REF at `_Svar1Backend` construction and discards it
(`python/genvarloader/_dataset/_streaming.py:1281`); VCF/PGEN use the reference-slice. This
removes the parent's "PR-B4 depends on PR-B3 for the `ref` field" coupling and leaves PR-B4
purely about `variant-windows`.

## Decomposition

| PR | Scope | Parity oracle |
|----|-------|---------------|
| **B3a** | Public `var_fields` knob + `available_var_fields`/`active_var_fields`; N staged INFO columns on `DecodedWindow` (generalizing PR-B2's single hardcoded `AF` column); `ref` bytes. All three backends. | ✅ svar1 + vcf + pgen |
| **B3b** | Per-call FORMAT fields (`dosage` + custom `Number=G`), **SVAR1 only**; VCF/PGEN raise. | ✅ svar1; guard-parity vcf/pgen |
| **B4** | `with_seqs("variant-windows")`; consumes B3a's REF for `ref="allele"`. | ✅ svar1 + vcf + pgen |

Rejected alternatives:

- **B3a + B3b as one PR.** The two halves sit on different data axes (per-variant vs.
  per-`(variant, sample, ploid)`), cover different backends, and have different parity
  coverage. Bundling them lets the un-oracled VCF FORMAT path ride in on the SVAR1 gate.
- **Teach `gvl.write` to persist FORMAT columns for VCF sources** so an oracle exists.
  That is a write-path feature (#231 territory), not streaming, and it would balloon
  Wave B well past its remaining scope.

## PR-B3a — per-variant `var_fields`

### Public surface

`StreamingDataset.with_settings(var_fields=[...])`, plus `available_var_fields` and
`active_var_fields` properties. This mirrors the written path, where `var_fields` lives on
`Dataset.open(...)` and `Dataset.with_settings(...)` — **not** on `with_seqs`. Validation
mirrors `_impl.py:338-345`: a field outside `available_var_fields` raises `ValueError`
naming both the missing fields and the available set. The default stays
`["alt", "ilen", "start"]`, so every existing caller is byte-unchanged.

### `available_var_fields` is backend-derived

The one genuine asymmetry vs. the written path: streaming has no on-disk artifact to
introspect, so each backend computes the set once at construction from the live source.

- **SVAR1** — `index.arrow` schema, filtered exactly as `_Variants.available_info_fields`
  does (`_haps.py:191-194`: numeric columns minus `POS`/`ILEN`), plus `ref` (the store
  always carries REF; `_streaming.py:1281` already asserts it).
- **VCF/BCF** — declared `INFO` header fields with `Type=Integer|Float`, via the
  `_declared_info_fields` helper PR-B2 added, plus `ref`.
- **PGEN** — `ref` only. No INFO path, consistent with PR-B2's AF guard.

### Rust data flow

`DecodedWindow.afs: Vec<f32>` (`src/record_stream/transpose.rs:49`) is joined by
`info_cols: Vec<InfoCol>`, where `InfoCol` is `{ name: String, values: InfoVals }` and
`InfoVals` is `I32(Vec<i32>) | F32(Vec<f32>)`. genoray's `StagedColumn` has exactly those
two variants, so "arbitrary dtype" collapses to two.

`afs` stays a distinct field rather than becoming one of `info_cols`: the AF filter runs
*before* assembly and is not a ride-along output, so keeping it separate avoids a
lookup-by-name on the hot keep path.

`transpose.rs:80`'s `info_staged.first()` pattern-match generalizes to a zip over the
requested `FieldSpec` list. genoray guarantees `info_staged` is in the same relative order
as the `fields` slice passed to `VcfChunkReader::new`, filtered by category
(`genoray/src/types.rs:151-158`), so the zip is well-defined.

The consumer gathers each column by the same window-local `v_idxs` that
`assemble_variants_window` already walks, then compacts with the same `keep` mask — the
AF and region-clip compaction stays a single pass. `VariantsBatch`
(`src/variants/mod.rs:13`) grows an `info_out: Vec<(String, InfoVals)>`, and
`next_batch_variants` marshals each into the returned dict under its own key.

### REF bytes

- **SVAR1** stops discarding REF at `_streaming.py:1281`, keeping it as a global
  `(ref_alleles, ref_offsets)` pair alongside `alt_alleles`/`alt_offsets`.
  `assemble_variants_window` gains an optional REF gather identical to its ALT one.
- **VCF/PGEN** slice from `ContigRef.ref_bytes` (already resident in the engine,
  `src/record_stream/engine.rs:73`) at `[start, start + alt_len - ilen)`, with `alt_len`
  taken from `alt_offsets`. The engine already holds both the bytes and `pad_char` for
  out-of-bounds padding.

This is exact for the normalized, left-aligned biallelic variants gvl requires. Parity is
the gate; if an edge case diverges, the fallback is exposing REF from genoray's
`DenseChunk` (currently commented out at `genoray/src/types.rs:142`).

### Named risk — INFO dtype divergence

The written path's INFO column dtype is whatever genoray wrote into the index (possibly
`i8`/`u16`/…), while streaming stages `i32`/`f32`. Byte-identical parity requires these
agree.

**Resolution:** streaming resolves each requested field's `FieldSpec` from the VCF header
`Type=` using the same rule genoray's index builder uses, so the dtypes agree by
construction — the same approach PR-B2 took for `AF` (`HtslibType::Float`,
`StorageDtype::F32`). Non-numeric INFO types (`Flag`, `String`, `Character`) are excluded
from `available_var_fields`, matching `available_info_fields`'s numeric-only filter.

**The parity suite must include a `Type=Integer` INFO field**, not only `AF`, precisely to
catch an `i32`-vs-narrower mismatch. If the dtypes diverge anyway, the fallback is a
documented dtype normalization on the streaming side (values byte-identical, dtype
widened) — a decision the test makes, not one assumed here.

## PR-B3b — per-call FORMAT fields (SVAR1 only)

A different data axis: per `(variant, sample, ploid)`, parallel to `geno_v_idxs`, not to
the window's static variant table.

**SVAR1 is nearly free.** `_Svar1Backend` already borrows `variant_idxs.npy` as a
zero-copy mmap, and `dosages.npy` / `<field>.npy` are stored on *the same hap-major CSR
offsets*. The backend memmaps each requested field once at construction, and the window
fill slices the identical `[start, stop)` ranges `find_ranges` already produced — no new
search, no new offsets, and the same `keep` compaction applies. Field discovery reuses
`_svar_format_fields(svar_path)` (`_haps.py:242`) verbatim; `dosage` is available iff
`dosages.npy` exists, matching `_has_dosage_file_on_disk`.

**VCF/PGEN raise.** Requesting `dosage` or a FORMAT field on those backends fails at
`with_settings` with a message stating the field is not available from this source. That is
exactly what a written dataset from the same source reports, since `gvl.write` never
persists them there — guard-parity, not a hidden capability gap.

`available_var_fields` therefore gains its FORMAT/dosage entries only on SVAR1, keeping the
property honest per backend.

## PR-B4 — `with_seqs("variant-windows")`

### Output type

`_FlatVariantWindows.to_ragged()` returns a `dict[str, Ragged]`
(`_flat_variants.py:294`), which *is* the written `Dataset`'s ragged-mode output for this
kind. Streaming has no flat/ragged switch — it always emits ragged — so streaming yields
that dict directly and parity is dict-vs-dict.

### Public surface

`StreamingDataset.with_seqs("variant-windows", opt: VarWindowOpt)`, validated the way
`Dataset.with_seqs`'s `variant-windows` branch is (`_impl.py`). The `ref="allele"` branch
requires REF alleles, which B3a makes available on all three backends.

### Token LUT

Built Python-side via `build_token_lut(opt.token_alphabet, opt.unknown_token)`, exactly as
`_impl.py:741-747`, and passed into `build_engine` with its dtype. Rust monomorphizes over
`Tok ∈ {u8, i32}`, mirroring the existing `assemble_variant_buffers_u8`/`_i32` split.

### Rust

A new `EngineBackend::generate_variant_windows` sits beside `generate_variants` and shares
its per-row CSR walk, region clip, and AF/keep compaction, then calls the **existing**
`windows::assemble_windows_mode::<Tok>` with:

- `v_contigs` = zeros — streaming windows are single-contig by construction;
- `reference` = `ContigRef.ref_bytes`, `ref_offsets = [0, len]` — identical to what
  `generate_batch_core` already passes (`engine.rs:211-212`). Because `build_engine`
  materializes each touched contig **whole** (per #307/#309), `fetch_windows` receives
  absolute contig coordinates and needs no rebasing;
- `ref_global`/`ref_off_global` = B3a's REF table, supplied only when `opt.ref == "allele"`.

`next_batch_variant_windows` marshals each present buffer as
`(data, seq_offsets, row_offsets)` alongside the B3a scalar fields; Python packs the dict.

### Out of scope, fail-fast

`unphased_union` and `dummy_variant`. Neither exists on streaming's `with_settings`, and
both default to off/`None` on the written path, so defaults are byte-identical and nothing
diverges silently. Reverse-complement is already unsupported for this type by design
(`_flat_variants.py:274`).

## Guards and error handling

- **Unknown `var_fields`** → `ValueError` naming missing + available, mirroring
  `_impl.py:338-345`.
- **FORMAT/dosage on VCF/PGEN** → error stating the field is not available from this
  source, and that a written dataset from the same source does not expose it either.
- **`var_fields` with non-variants output** → `NotImplementedError`, matching the exception
  type PR-B2 uses for `min_af`/`max_af` with non-variants output. This is **intentionally
  stricter than the written path**, which silently ignores it. Streaming's established
  convention
  (`_streaming.py:485-495`, PR-B2) is to fail rather than accept a setting it will drop;
  the field is inert on both paths, so this costs no byte-parity. Documented as a
  deliberate divergence.
- **`format_by_carrier.is_some()`** → hard error rather than silent mis-decode.
  Unreachable today (dense sources only); a tripwire for a future genoray source change.
- **SVAR2** stays haplotypes-only; `with_seqs("variant-windows")` there keeps raising the
  existing `NotImplementedError`.

## Testing and parity plan

Oracle remains `gvl.write()` + `Dataset[r, s]` at `jitter=0`, byte-identical, extending
`tests/dataset/test_streaming_variants_parity.py`.

- **B3a** — per-backend parity for a `Type=Float` INFO field, a `Type=Integer` INFO field
  (the dtype canary above), and `ref`. `available_var_fields`/`active_var_fields` asserted
  equal to the written `Dataset`'s for the same source. Reuse PR-A's interior-exclusion and
  narrowed-window fixtures: a ride-along column compacted by the wrong mask surfaces as
  misaligned values rather than a crash, so those fixtures are the real gate. Keep the
  established `total_variants > 0` non-vacuous guard.
- **B3b** — SVAR1 `dosage` + custom FORMAT parity; VCF/PGEN guard tests asserting the
  error surface.
- **B4** — all four `(ref, alt) ∈ {window, allele}²` combinations × both token dtypes ×
  three backends, plus an empty-group case (a region/sample with no in-window variant) at
  `dummy_variant=None`.
- **Rust unit tests** — staged-column zip order vs. the requested `FieldSpec` list; REF
  reference-slice length `== alt_len - ilen`; `assemble_windows_mode` under zero
  `v_contigs` matching the written FFI call on identical inputs.
- **Gates** — `pixi run -e dev maturin develop --release` before pytest (CLAUDE.md); full
  tree before pushing.

## Docs

`var_fields` and `variant-windows` are new public streaming knobs, so each PR that adds one
updates `docs/source/dataset.md`, `docs/source/faq.md`, and
`skills/genvarloader/SKILL.md` (CLAUDE.md public-API + docs-audit gates). No new `__all__`
symbol is expected — `VarWindowOpt` and `FlatVariantWindows` are already exported — so
`docs/source/api.md` should be unchanged; verify with the `api.md` sync check in CLAUDE.md.

Update `docs/roadmaps/streaming-dataset.md` and the StreamingDataset project board as each
PR lands.

## Risks

- **INFO dtype divergence** — see PR-B3a. Mitigated by resolving `FieldSpec` from the
  header with genoray's own rule, gated by a `Type=Integer` parity case.
- **REF reference-slice divergence** — assumes VCF `REF` equals the reference at
  `[POS, POS+len(REF))`, which holds for variants normalized against the dataset's
  reference. Parity is the gate; exposing REF from genoray's `DenseChunk` is the fallback.
- **`available_var_fields` drift** — streaming derives it from the live source while the
  written path derives it from the written artifact. The equality assertion in the B3a
  test suite is what keeps the two definitions from drifting apart.
