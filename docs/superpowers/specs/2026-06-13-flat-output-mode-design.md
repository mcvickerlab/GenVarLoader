# Flat output mode — design (A0 + A)

**Date:** 2026-06-13
**Repo:** `mcvickerlab/GenVarLoader` (gvl)
**Driven by:** gvl issue #214 (super-batch / flat-buffer output mode), genvarformer PR #19
(numba tokenizer hot-path), `gvf-germ-som` trans gene-regulation training run.
**Status:** approved design; ready for `writing-plans`.

Related prior gvl designs: `2026-05-31-flat-buffer-getitem-pipeline-design.md`,
`2026-06-01-flat-buffer-getitem-followups-design.md`,
`2026-06-07-ragged-variants-pack-lazy-views-design.md`.

---

## 0. Problem (why this exists)

A genvarformer training loop indexes a gvl dataset with a list of flat `(region, sample)`
indices (`dataset[flat_idx]`). In a CPU profile of the production data path the decode is
**97 % of wall (~1017 ms/batch)** and **`awkward.highlevel.__getitem__` alone is 62 % of
wall**. The GPU sits idle waiting on it (see issue #214 for the full profile).

gvl already reconstructs into pure-numpy flat `(data, offsets, shape)` buffers internally
(`_Flat`, numba kernels) and only wraps them in awkward `Ragged` / `RaggedVariants` at the
public boundary (`_dataset/_query.py:107`, `o.to_ragged()`). The consumer then does a
*second* awkward pass before dropping back to numpy for torch. The awkward round-trip
across the boundary — and for the **variants** output, gvl's own awkward indexing inside
`_get_variants` — is the cost.

---

## 1. Decomposition of issue #214 (umbrella)

This spec implements **A0 + A**. The remaining sub-projects each get their own
spec → plan cycle.

| ID | Sub-project | Depends on | Notes |
|----|-------------|-----------|-------|
| **A0** | Flat passthrough for non-variant outputs (seqs / haps / annotated-haps / reference) | — | Trivial: `_Flat` already exists; stop calling `to_ragged()`. |
| **A** | Flat **variant** decode kernel — numba reimplementation of `_get_variants` over sparse genotypes, no awkward | A0 (shared boundary/API) | The real foundation; removes gvl's internal 62 % awkward in the variants path. |
| **B** | Absorb variant assembly into gvl: phased+unphased ploidy merge, within-group sort, dummy-variant / empty-region padding | A | Much of this already exists as gvf PR #19 `_ragged_numba.py` kernels to upstream + generalize. |
| **C** | Flank fetch + tokenization in gvl: sample-invariant reference dedup, `seqpro.tokenize` via a **caller-supplied byte→int LUT**, flat int-token output | A | Moves `seqpro.tokenize` (3.55 s) down; dedups per unique region. |
| **D** | Window allele output mode: `ref_window = flank5·ref·flank3`, `alt_window = flank5·alt·flank3` as flat token buffers | C | Single contiguous reference read for `ref_window`; assembly for `alt_window`; dummy = all-`N` window. |
| **E** | `double_buffered` transport composition + `__getitems__` super-batch entry | A | Transport already decomposes `RaggedVariants` to flat buffers in `_shm_layout.py` (kind=2); largely "don't re-wrap on read" + relax file-backed / splice / `insertion_fill` constraints. |

**Decision (boundary):** tokenization lives in gvl *optionally*, driven by a caller-supplied
byte→int LUT (sub-project C). So B/C/D are genuinely gvl features, and `seqpro.tokenize`
moves down with dedup applied.

**Recommended order:** A0 → **A** → (B, C — independent, both need A) → D (needs C).
E needs only A's payload shape but delivers most once B/C shrink the payload, so sequence it
after B/C unless the training run needs prefetch overlap sooner.

Each gvl sub-project lands with the matching **genvarformer consumer thinning** (strip awkward
from `_fetch`, `_read_flank_seq`, `_add_dummy_variant`, `rag_to_nested`), validated against
genvarformer's existing batch-equality guardrail.

---

## 2. Goal (this spec: A0 + A)

`dataset.with_output_format("flat")[idx]` returns pure-numpy `(data, offsets, shape)`
containers with **zero awkward on the hot path**, **byte-identical when re-wrapped** to
today's output, for the seqs / haps / annotated-haps / reference outputs (A0) and the
variants output (A).

---

## 3. Public API

- **`Dataset.with_output_format(fmt: Literal["ragged", "flat"]) -> Dataset`** — returns a
  frozen-dataclass copy with a new `output_format` field (default `"ragged"`), mirroring the
  existing `with_*` lazy-view methods. Threads into `QueryView` as a `flat_output` flag.
  Orthogonal to `output_length` and to subsetting; composes with them. The `__getitems__`
  plural / super-batch entry is **deferred to E** — `with_output_format` is independent of it.

- **New public types** (added to `genvarloader/__init__.py` `__all__`). Promote the existing
  internals to documented public names; keep the underscored aliases so no internal call site
  breaks:
  - **`FlatRagged`** (= `_Flat`): `.data`, `.offsets`, `.shape`, `.to_ragged()`,
    `.to_fixed(length)`, `.to_padded(pad)`, `.reshape`, `.squeeze`.
  - **`FlatAnnotatedHaps`** (= `_FlatAnnotatedHaps`): `.haps`, `.var_idxs`, `.ref_coords`,
    plus `.to_ragged()` / `.to_fixed()` / `.to_padded()`.
  - **`FlatVariants`** (new): mirrors `RaggedVariants` field-for-field —
    `start` / `ilen` / `dosage` / `info[...]` as `FlatRagged`; `alt` / `ref` as
    **`FlatAlleles`**. Implements `.to_ragged() -> RaggedVariants`, plus `reshape` / `squeeze`
    delegating to each field.
  - **`FlatAlleles`** (new): two-level flat bytestring — `byte_data: uint8`, `seq_offsets`
    (per-variant byte offsets), `var_offsets` (per-`(instance, ploid)` variant offsets),
    `shape` carrying ploidy. Layout matches genvarformer's `_bpvl` / `_decompose_bytestring`
    exactly (inner-before-outer offsets).

- **Skill update:** `skills/genvarloader/SKILL.md` must document the new `with_output_format`
  method, the flat output mode, and the new public types (CLAUDE.md mandates this for any
  public-API change).

---

## 4. Boundary change (`_dataset/_query.py`)

Today `getitem` (line ~107) unconditionally maps `_Flat` / `_FlatAnnotatedHaps → to_ragged()`,
after the `output_length` pad / `to_fixed` massaging (lines ~92–103).

In **flat mode** (`view.flat_output`):
- **Skip** `to_ragged()` and **skip** the `output_length` pad / `to_fixed` massaging — return
  the flat wrappers raw. The consumer densifies via `.to_fixed` / `.to_padded` when it wants
  dense output. (Flat mode is orthogonal to `output_length`; the buffers are valid regardless,
  and densification is the consumer's choice.)
- **`reshape` / `squeeze`** (lines ~112–117) still apply. `FlatRagged` already has both;
  `FlatVariants` / `FlatAlleles` gain `reshape` / `squeeze` delegating to each field.
- **`rc_neg`** (negative-strand reverse-complement, lines ~158–160) routes through
  `_Flat.reverse_masked(comp=_COMP)` (already implemented) instead of
  `reverse_complement_ragged`.

**Edge to verify (not blocking):** reverse-complement semantics for the **variants** record
output (`FlatVariants`) — confirm whether variant records are reverse-complemented at all in
the current ragged path, and match that behaviour exactly.

---

## 5. A — flat variant decode (replaces awkward `_get_variants`)

Numba over the sparse buffers already in hand: `geno_offset_idx`
(from `_get_geno_offset_idx`, computed via `ravel_multi_index`), `genotypes.offsets`,
`genotypes.data` (the v_idxs store), `variants.start`, `variants.ilen`, `variants.alt`,
`variants.ref`, `variants.info[...]`, `dosages`. This reuses the exact pattern
`reconstruct_haplotypes_from_sparse` already uses (it consumes `geno_offset_idx` +
`geno_offsets` + `geno_v_idxs` in numba with no awkward) — A emits the variant *fields*
instead of reconstructed sequences.

1. **Gather selected v_idxs** — numba over `geno_offset_idx` → flat `v_idxs` buffer +
   per-`(instance, ploid)` output offsets `(b*p + 1)`. Replaces
   `self.genotypes[r, s].to_packed()`.
2. **Scalar fields** (`start`, `ilen`, `info[...]`) — dense fancy-index `arr[v_idxs]` →
   `FlatRagged.from_offsets(data, shape, offsets)`. **`dosage`** is parallel to genotypes
   (not gathered via `v_idxs`); gather its payload by the same genotype offset ranges,
   matching today's `self.dosages[r, s]`.
3. **Alleles** (`alt`, `ref`) — segmented numba gather. Preallocate exact bytes with the
   `_allele_bytes_sum` approach (already computes per-instance totals without touching
   payload), copy each selected variant's allele bytes into a flat buffer, and build the
   two-level offsets → `FlatAlleles`. Replaces `getattr(self.variants, kind)[v_idxs]` +
   `.to_packed()` + `_build_allele_layout`.
4. **Filters** —
   - `min_af` / `max_af`: compute `info["AF"][v_idxs]`, build the keep mask, compact `v_idxs`
     and recompute offsets in numba. Replaces the awkward `genos[_keep]` /
     `ak.to_regular(...)` path.
   - `exonic`: `choose_exonic_variants` is already numba and yields `keep` / `keep_offsets`;
     apply as a flat compaction.
5. **Assemble** `FlatVariants(**fields)`. In ragged mode the boundary calls
   `.to_ragged()` → an element-identical `RaggedVariants`.

---

## 6. A0 — flat passthrough (non-variant outputs)

The seqs / haps / annotated-haps / reference reconstructors already return `_Flat` /
`_FlatAnnotatedHaps`. A0 is the boundary returning them public-wrapped (`FlatRagged` /
`FlatAnnotatedHaps`) instead of `to_ragged()`, with `reshape` / `squeeze` / `rc_neg`
preserved via the existing `_Flat` methods.

**Deferred:** tracks / intervals (`RaggedIntervals`) — `_call_intervals` is its own awkward
path and not yet a `_Flat`. Out of A0 scope; handled in B or E.

---

## 7. Equivalence & acceptance (primary gate)

1. **Byte-identity.** `FlatVariants.to_ragged()` rebuilds `RaggedVariants` via gvl's existing
   `_build_allele_layout` (same boxing as gvf's `_bpv` / `_bpvl`). Across an index matrix —
   scalar, list, 2-D `(region, sample)`, empty regions, AF-filtered, exonic-filtered,
   multi-ploidy — assert `flat.to_ragged()` is **element-identical** to the current
   `dataset[idx]`. Extend `tests/dataset/test_flat_getitem_snapshot.py` and
   `tests/dataset/test_flat_variants.py`.
2. **No awkward on the hot path.** A test / micro-bench asserts `awkward.highlevel.__getitem__`
   is absent from the variant-decode call stack in flat mode.
3. **Consumer parity.** genvarformer's existing batch-equality guardrail passes with the thin
   wrapper (§8) swapped in.

---

## 8. genvarformer consumer thinning (validation, separate repo)

Lands alongside this gvl change to prove the win and exercise the API:
- `RefSeq._call_one_to_many` + `rag_to_nested` (`_helpers.py`): consume `FlatRagged` directly →
  `Nested` (`torch.from_numpy(data)`, offsets cast to int32), dropping the
  `np.asarray(rag.data)` round-trip.
- `Variants._fetch` step 1: receives `FlatVariants` — PR #19's `_decompose_bytestring` boxing
  on the *inputs* disappears (already flat). The merge / sort / dummy kernels stay in gvf for
  now (that is sub-project B) but now take flat inputs.
- Validate against the existing batch-equality guardrail.

---

## 9. Error handling & constraints

- `with_output_format("flat")` composes with `subset_to` and `output_length`. Combinations
  already unsupported in the ragged path (e.g. spliced + variants) keep raising the same
  errors.
- Flat types are views into freshly allocated buffers — safe to wrap with `torch.from_numpy`
  without a copy. Offsets are int64; consumers wanting torch `Nested` cast to int32. Document
  this.
- Promoting `_Flat` etc. to public names must keep the underscored aliases working so no
  internal call site breaks.

---

## 10. Testing strategy

- gvl: extend the flat snapshot / equality tests with a `flat`-mode parametrization across the
  index matrix in §7; add the awkward-absence guard.
- Run the existing suite (`pixi run -e dev test`) to confirm no regression in the ragged path.
- genvarformer: run the batch-equality guardrail with the thin wrapper.

---

## 11. Scope boundary

**In scope:** A0 (seqs / haps / annotated-haps / reference) + A (variants) flat decode; the
public types; `with_output_format`; byte-identity + awkward-absence tests; the genvarformer
thin-wrapper validation.

**Out of scope (own specs):** B (assembly absorb), C (flank + tokenize), D (windows),
E (`double_buffered` + `__getitems__` super-batch entry), and the tracks / intervals flat path.
