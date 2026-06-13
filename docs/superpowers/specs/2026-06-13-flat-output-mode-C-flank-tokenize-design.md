# Flat output mode — sub-project C: flank fetch + tokenization + window mode

**Date:** 2026-06-13
**Repo:** `mcvickerlab/GenVarLoader` (gvl)
**Driven by:** gvl issue #214 (super-batch / flat-buffer output mode), genvarformer
`_read_flank_seq` + `_tokenizer` hot-path, `gvf-germ-som` variant-encoder (varenc) training run.
**Status:** approved design; ready for `writing-plans`.

**Depends on:** sub-project **A** (flat variant decode) from
`2026-06-13-flat-output-mode-design.md`. C consumes A's flat buffers (`v_idxs`,
per-`(instance, ploid)` offsets, `FlatAlleles`, `variants.start` / `variants.ilen`).

**Scope note:** this spec covers parent sub-project **C** *and* the window part of parent
sub-project **D**. The two were split in the umbrella doc, but the window mode is just
flank-fetch + allele assembly composed over the same decode, so they land together.

---

## 0. Problem

genvarformer fetches reference flanks around every variant and tokenizes them on the hot
path, per `(region, sample)`:

- `_read_flank_seq` (`tokens.py:53`) reads `flank5 = [start−L, start)` and
  `flank3 = [end, end+L)` from the reference for every variant, returning
  `(n_var, 2·flank_len)` bytes.
- `_tokenizer` (`_helpers.py:17`, `tokens.py:549`) maps those bytes → ints via
  `seqpro.tokenize` with a hardcoded `{A:0, C:1, G:2, T:3, N→4}` LUT.

In the issue #214 profile `seqpro.tokenize` costs **3.55 s**. The reference reads are
sample-invariant — the same variant appearing in N samples within a super-batch yields N
identical flank reads — yet nothing dedups them.

The varenc model also wants per-variant **windows**: `flank5·ref·flank3` and
`flank5·alt·flank3`. Today genvarformer assembles these downstream from the flank bytes +
allele bytes.

This sub-project moves the flank fetch + tokenization (and the window assembly) into gvl as
pure-numpy/numba flat token output, eliminating the awkward round-trip and the redundant
per-sample work.

---

## 1. Goal

On top of A's flat variants output, gvl produces flat **int-token** flank / window buffers
with zero awkward on the hot path, byte-identical to genvarformer's current
`_read_flank_seq` + `_tokenizer` (and downstream window assembly) output:

- **Ride-along flanks:** `FlatVariants.flank_tokens` — `[flank5 | flank3]` tokens per variant.
- **Window mode:** `FlatVariantWindows` with `ref_window` / `alt_window` token buffers; raw
  alleles are dropped (they are folded into the windows).

Tokenization uses a **caller-supplied seqpro-style LUT** (ordered alphabet + unknown token).

---

## 2. Public API

The dataset convention: `with_*` methods are **type-system state transitions**; everything
else is configuration on `with_settings()`. This spec follows that split.

### 2.1 Settings (no type change) — `Dataset.with_settings(...)`

New keyword arguments, mirroring the existing flat-kwargs style:

- **`flank_length: int | None`** — number of reference bases on each side. `0`/`None`
  disables flanks. Must be `≥ 0`.
- **`token_alphabet: bytes | None`** — ordered byte alphabet, e.g. `b"ACGT"`.
- **`unknown_token: int | None`** — token id for any byte not in `token_alphabet`
  (i.e. `N` / out-of-bounds pad). Mirrors `seqpro.tokenize`'s `unknown_token`.

gvl builds a 256-entry `uint` LUT internally from `(token_alphabet, unknown_token)`.

**Token dtype:** `uint8` when `max(token ids) ≤ 255` (the default for a DNA alphabet), else
`int32`. Consumers cast to `int32` / `long` for torch; offsets are `int64`. Documented.

These settings are shared by both ride-along and window modes.

### 2.2 Type transition — `Dataset.with_seqs(kind)`

- **`"variants"`** (unchanged) → `FlatVariants`. When `flank_length` is set, the optional
  `flank_tokens` field is populated (ride-along); when unset it is `None`. Same type either
  way — a conditionally-present field, exactly like how `var_fields` conditionally adds
  fields. No new type parameter.
- **`"variant-windows"`** (new kind) → **`FlatVariantWindows`**. A genuinely different output
  type, hence a real `with_seqs` transition. Requires `flank_length` + token settings to be
  set (else raises). Raw allele byte fields are **not** present in this kind.

Both kinds require genotypes (storage is `Haps`), like the existing `"variants"` kind.

### 2.3 New public types (`genvarloader/__init__.py` `__all__`)

- **`FlatVariantWindows`** (new): the variant **scalar** fields (`start` / `ilen` / `dosage` /
  `info[...]` as `FlatRagged`) **plus** `ref_window` and `alt_window`. Each window is ragged in
  **both** the variant axis *and* window length (`(b, p, ~v, ~win)`), so it uses the **two-level
  flat layout of `FlatAlleles`** (token `data`, per-variant `seq_offsets`, per-`(instance,ploid)`
  `var_offsets`, `shape`) rather than single-ragged `FlatRagged`. The window `data` holds tokens
  (the configured dtype) instead of allele bytes. Implements `.to_ragged()`, `reshape`, `squeeze`
  delegating to each field.

`FlatVariants` (defined in A) gains an optional field:

- **`flank_tokens: FlatRagged | None`** — shape `(b, p, ~v, 2·L)`, `[flank5 | flank3]` tokens
  per variant (ragged in the variant axis, fixed inner dim `2·L`). Matches genvarformer's
  `v_flank` Ragged layout field-for-field; `.to_ragged()` reproduces it.

### 2.4 Skill update

`skills/genvarloader/SKILL.md` must document the new `with_settings` keywords
(`flank_length`, `token_alphabet`, `unknown_token`), the new `"variant-windows"` kind, the
`FlatVariants.flank_tokens` field, and `FlatVariantWindows` (CLAUDE.md mandates this for any
public-API change).

---

## 3. Decode flow (inside A's flat variant decode)

Inputs already produced by A: flat `v_idxs`, per-`(instance, ploid)` output offsets,
`variants.start[v_idxs]`, `variants.ilen[v_idxs]`, `FlatAlleles` (alt/ref bytes + offsets),
and the per-group contig ids (as genvarformer derives via `repeat_by_counts`).

1. **Window/flank coordinates** (numba over the gathered fields):
   - `flank5 = [start − L, start)`
   - `end = start − min(ilen, 0) + 1`  (extends across the deleted reference span;
     matches `_read_flank_seq`'s `ends = start − ilen.clip(max=0) + 1`)
   - `flank3 = [end, end + L)`
   - `ref_window = [start − L, end + L)` — a single contiguous reference read.
2. **(Optional, benchmarked) dedup** — see §4. Default path is correct without it.
3. **Reference reads** via the existing `_fetch_impl` / `padded_slice` kernel
   (`_reference.py:117,150`), which pads out-of-bounds positions with `pad_char` (`N`).
4. **Tokenize** the fetched bytes through the 256-entry LUT in numba (1 op/byte).
5. **Emit:**
   - **flanks:** `2·L` tokens per variant → `FlatVariants.flank_tokens`
     (`FlatRagged.from_offsets`, shape `(b, p, ~v, 2·L)`, reusing the variant offsets).
   - **windows:**
     - `ref_window` = the single contiguous `[start − L, end + L)` read, tokenized
       (variable length per variant: `(end − start) + 2·L`).
     - `alt_window` = `flank5_tok · alt_tok · flank3_tok`, assembled from the tokenized
       flank reads + the tokenized `FlatAlleles.alt` bytes (variable length).
     - Both wrapped as `FlatRagged` with two-level offsets, assembled into
       `FlatVariantWindows` alongside the scalar fields.
6. **Scatter** (only if deduped): write tokens back to per-occurrence positions. Output is
   identical to the non-deduped path.

In ragged output mode the boundary calls `.to_ragged()` on each field; in flat mode the flat
wrappers are returned raw (per A's boundary change).

---

## 4. Dedup — benchmarked optimization, not a prerequisite

Reference reads are RAM-resident and cache-friendly, so the reads themselves are already
cheap; a naïve global dedup (hash set, or `np.unique(return_inverse=True)`'s O(n log n) sort)
can cost more than it saves. Because we chose **materialized output** (§1, decided in
brainstorming), the result is **byte-identical with or without dedup** — so dedup is a
transparent internal optimization that can land and be measured separately.

**Design constraint:** the dedup tally must be cheaper than the reads + tokenization it
eliminates. Exploit that svar genotypes give v_idxs **sorted within each
`(instance, ploid)` group**: a parallel linear scan / k-way merge over the sorted runs yields
unique variants without a global sort. Baseline to beat in the benchmark:
`np.unique(v_idxs, return_inverse=True)`.

**Where the win is:** primarily the **tokenization** (and, for windows, the **assembly**),
which dedup performs once per unique variant rather than once per occurrence. For ride-along
flanks (a flat `2·L` LUT gather) the win may be marginal versus the scatter-copy cost; for
window mode (variable-length concat assembly) it is larger. The benchmark decides whether
dedup is enabled per mode.

**Plan:** implement the correct per-occurrence path first (parallel read + tokenize +
assemble); add the sorted-run dedup as a measured optimization gated on beating the baseline.

---

## 5. Edge cases & constraints

- **Out-of-bounds flanks** (contig ends, negative starts): `padded_slice` fills `N`, which the
  LUT maps to `unknown_token`. Tested at contig boundaries.
- **Empty regions** (zero variants): emit empty windows / flanks (ragged-empty). The
  **all-N dummy window** is **deferred to sub-project B** — once B's dummy-variant insertion
  lands and its wiring is known, a dummy variant's window resolves naturally to an all-`N`
  (all-`unknown_token`) window. Noted here, not implemented here.
- **Reverse-complement (`rc_neg`):** flanks/windows stay in **reference orientation, not
  reverse-complemented**, matching genvarformer's `_read_flank_seq` (which reads raw
  reference). Flagged as an edge to verify against current consumer behaviour; if rc is needed
  later it would complement in token space.
- **Composition:** the flank/token settings compose with `subset_to`, `min_af`/`max_af`,
  `var_filter="exonic"`, and `with_output_format`. Combinations already unsupported for
  variants (e.g. splicing + variants) keep raising the same errors.
- `"variant-windows"` without `flank_length` + token settings raises a clear error.

---

## 6. Equivalence & acceptance (primary gate)

1. **Byte-identity.** Across an index matrix — scalar, list, 2-D `(region, sample)`, empty
   regions, SNP / insertion / deletion, multi-ploidy, OOB flanks at contig ends — assert:
   - `FlatVariants.flank_tokens` (re-wrapped via `.to_ragged()`) is **element-identical** to
     the current genvarformer `_read_flank_seq` + `_tokenizer` output for the same batch.
   - `FlatVariantWindows.ref_window` / `alt_window` are element-identical to the windows
     genvarformer's varenc assembles today.
2. **Dedup invariance.** Dedup-on and dedup-off produce identical output.
3. **No awkward on the hot path.** A test / micro-bench asserts `awkward.highlevel.__getitem__`
   is absent from the flank/window decode call stack in flat mode.
4. **Consumer parity.** genvarformer's existing batch-equality guardrail passes with the thin
   wrapper (§7) swapped in.

---

## 7. genvarformer consumer thinning (validation, separate repo)

Lands alongside this gvl change to prove the win and exercise the API:

- **Ride-along:** delete `_read_flank_seq` (`tokens.py:53`) and the `_tokenizer(v_flank)` call
  (`tokens.py:549`); consume `FlatVariants.flank_tokens` directly (cast offsets to int32 for
  `Nested`).
- **Window mode:** replace varenc's downstream flank+allele concatenation with gvl's
  `ref_window` / `alt_window`.
- Validate against the existing batch-equality guardrail.

---

## 8. Testing strategy

- gvl: extend the flat snapshot / equality tests with `flank_length` + `"variant-windows"`
  parametrizations across the §6 index matrix; add the dedup-invariance test and the
  awkward-absence guard. Add a micro-bench for the dedup decision (§4).
- Run the existing suite (`pixi run -e dev test`) to confirm no regression in the ragged or
  `"variants"` paths.
- genvarformer: run the batch-equality guardrail with the thinned consumer.

---

## 9. Scope boundary

**In scope:** ride-along `flank_tokens` on `FlatVariants`; the `"variant-windows"` kind and
`FlatVariantWindows` (ref/alt windows = flanks + alleles, raw alleles dropped); the
seqpro-style LUT settings on `with_settings`; internal sorted-run dedup as a benchmarked,
output-invariant optimization; the genvarformer consumer thinning; byte-identity +
dedup-invariance + awkward-absence tests.

**Out of scope (own specs / later):** the all-`N` dummy window for empty regions (sub-project
B); tracks / intervals; reverse-complement of flanks/windows; the general reference-sequence
(`RefSeq`) tokenization path beyond what A0 already provides; the `double_buffered` /
`__getitems__` super-batch entry (sub-project E).
