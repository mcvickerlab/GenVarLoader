# Design: Migrate GenVarLoader off awkward onto seqpro `_core.Ragged`

**Date:** 2026-06-22
**Status:** Approved design — ready for implementation planning.
**Branch (GVL):** `rust-ragged-audit` · **Repo (seqpro):** `~/projects/SeqPro` (new PR)
**Supersedes/implements:** `docs/handoffs/2026-06-22-migrate-off-awkward-to-seqpro-rust-ragged.md`

## 1. Goal

Remove `awkward` from GenVarLoader's **production data structures** so that seqpro can
delete its legacy awkward backend (`_array.py`). Today GVL passes against the Rust
`_core.Ragged` backend only because seqpro still ships `_array.py` and registers awkward
behaviors for the `"Ragged"` type. GVL still uses awkward internally (~98 `ak.*` sites
across 14 files, plus a `RaggedVariants(ak.Array)` subclass).

This is a **two-repo, coordinated effort**: GVL goes awkward-free, and seqpro gains the
upstream support GVL needs (a ragged concat kernel + a `to_ak` record fix) and then drops
the awkward backend. The pixi pins already point at local editable `../SeqPro` and
`../genoray`, so development is integrated; pins are repointed to released versions before
merge.

Awkward remains a **test-only** dependency (oracles). The goal is zero `ak.*` in
production code, not in test oracles.

## 2. Target representation (Phase 0 — resolved)

`RaggedVariants` becomes a **thin wrapper class around a single record `_core.Ragged`**
held as `self._rag`, with shape `(batch, ploidy, ~variants)`.

The crux that made handoff option *(a)* (a single record `Ragged`) look infeasible was the
heterogeneous nesting depth of the fields: `start`/`ilen`/`dosage` are `(b, p, ~v)` (one
ragged axis) while `alt`/`ref` are `(b, p, ~v, ~l)` (two ragged axes), so they cannot
share offsets and cannot form one record. **The fix is to store `alt`/`ref` as opaque
`'S'`-layout strings**: the `~length` axis is absorbed into the string layout, collapsing
the field shape to `(b, p, ~v)`. Now every field shares the same variant-level offsets and
forms one record. This was proven feasible end-to-end (see §7).

Field inventory of `self._rag`:

| Field | dtype/layout | shape | source |
|---|---|---|---|
| `alt` | opaque string `'S'` | `(b, p, ~v)` | required |
| `start` | `POS_TYPE` (int) | `(b, p, ~v)` | required |
| `ref` | opaque string `'S'` | `(b, p, ~v)` | optional (one of `ref`/`ilen` required) |
| `ilen` | int32 | `(b, p, ~v)` | optional |
| `dosage` | `DOSAGE_TYPE` | `(b, p, ~v)` | optional |
| `**extra` | numeric | `(b, p, ~v)` | optional |

All fields are constructed sharing the **identical** variant-level offsets object (a
hard requirement of `Ragged.from_fields`).

### Why a wrapper class (not a bare `Ragged`, not a subclass)

- **Wrapper (chosen):** preserves the documented method/property public API
  (`.alt`, `.ref`, `.start`, `.ilen`, `.end`, `.shape`, `.fields`, `__getitem__`,
  `__len__`, `reshape`, `squeeze`, `to_packed`, `rc_`, `pad`, `to_nested_tensor_batch`),
  while the record `Ragged` does the heavy lifting (indexing, slicing, field access, len,
  equality). `RaggedVariants` stays in `__all__`.
- **Bare record `Ragged`:** rejected — would force `variants["alt"]` and free functions,
  breaking the documented `.alt` / `.rc_()` API for downstream users (e.g. genvarformer).
- **Subclass `_core.Ragged`:** rejected — `_core` has no behavior-registration, so
  slicing/`from_fields`/ufuncs return a base `Ragged` and silently drop the subtype.

This also aligns `RaggedVariants` with `RaggedIntervals`/`RaggedAnnotatedHaps`, already
`Ragged`-backed dataclasses.

### Deletions in `_rag_variants.py`

`ak.behavior` registration, `from_ak`, the `RaggedVariant(ak.Record)` class, and the
awkward layout-munging helpers (`_pack_alleles`, `_decompose_alleles`,
`_is_canonical_alleles`, `_build_allele_layout`/`_alt_layout_parts` consumers,
`_alleles_to_nested_tensor`'s awkward walk). The file shrinks substantially.

## 3. Per-operation mappings (the non-obvious ones)

| Operation | awkward today | `_core` replacement |
|---|---|---|
| Construct | `ak.zip` + `__list__:"Ragged"` tagging | build char-R2 ragged per allele field → `.to_strings()`; `Ragged.from_fields({...})` sharing one offsets object |
| Field access | `super().__getitem__("alt")` (ak.Array) | `self._rag["alt"]` (zero-copy field Ragged) |
| `.ilen` (derived) | `ak.str.length(alt) - ak.str.length(ref)` | char-view `.lengths` difference |
| `.end` | `start + ak.num(ref, -1)` / clipped ilen | char-view `.lengths` / clipped `ilen` |
| `rc_` | `ak.where(mask, rc(a), a)` over a union | flat **allele-level R=1** view + `seqpro.rag.reverse_complement(view, comp_lut, mask=, copy=False)` in place (already prototyped as `reverse_complement_masked`) |
| `to_packed` | `ak.to_packed` + canonical/non-canonical munging | `self._rag.to_packed()` directly (record `to_packed`, made obligatory upstream — §4.C) |
| `to_padded` | `ak_str.rpad` / `pad_none` + `fill_none` | `Ragged.to_padded` on the char view → dense `(b, p, v, l)` |
| `pad` (≥1 variant/group) | `ak.pad_none(1)` + `fill_none` | min-length-1 ragged-axis insert with per-field/string sentinels (see §4, Decision A) |
| nested-tensor batch | awkward layout walk | char-view `.data` + `.offsets` directly into `nt_jag` |
| shm (de)serialize | `from_ak` / `ak.zip` | `.data` (dict) / `.offsets` / `str_offsets` / `.shape` buffers |

## 4. SeqPro upstream changes (new PR in `~/projects/SeqPro`)

### A. Ragged-axis concatenation — Rust kernel
Add `seqpro.rag.concatenate` for concatenating ragged arrays along the ragged axis.
Implement the kernel in **`src/ragged.rs`** (Rust + PyO3), matching the existing
`_ragged_pack` / `_ragged_nested_pack` pattern, with the Python wrapper in `_ops.py`.
Pure offset-arithmetic + buffered copy (rayon-parallel); no Python loops.

Consumers:
- `RaggedIntervals.prepend_pad_itv` (GVL `_ragged.py`) — today `ak.concatenate(..., axis=2)`
  prepending a `(b, t, 1)` regular pad to a `(b, t, ~v)` ragged. Direct use of the new op.
- `RaggedVariants.pad` — "ensure ≥1 variant per group, filling new entries with per-field
  sentinels (incl. string fields)." Evaluate during planning whether to (i) express via
  the generic `concatenate` + an empty-group mask, or (ii) add a small record-aware
  `pad_ragged(rag, min_length, fill)` helper to seqpro. Default lean: build on the generic
  `concatenate` to keep the seqpro surface minimal; fall back to a dedicated helper only if
  the record/string sentinel handling is awkward in GVL.

Update `SeqPro/skills/seqpro/SKILL.md` for the new public op (required by seqpro CLAUDE.md).

### B. `to_ak()` record bug — fix
`_ingest.to_ak` (`_ingest.py:153`) does `ak.zip({f: to_ak(rag[f]) for f in rag.fields},
depth_limit=1)`, which fails on **multi-leading-axis records** (`ValueError: cannot
broadcast RegularArray of size 2 with size 4`). Fix so whole-record `to_ak()` works for
`(b, p, ~v)` records. This restores the whole-record test oracle (per-field oracles already
work and are the fallback). Add a regression test in seqpro.

### C. Record / opaque-string `to_packed` ("Spec D") — required
Currently raises `NotImplementedError`. Implement `to_packed` for record and
opaque-string-under-axis Rageds in seqpro so GVL's `RaggedVariants.to_packed` packs the
record directly (no `to_chars()` dance). Add seqpro tests. This lifts the §6.1 constraint;
update §3's `to_packed` mapping to pack the record in one call.

### D. Delete the awkward backend (the payoff)
Once GVL is awkward-free: delete `python/seqpro/rag/_array.py` + `_gufuncs.py`, relocate the
awkward *interop helpers* still imported by `_ingest.py`/`__init__.py` (`unbox`, `RagParts`,
`_parts_to_content`, `DTYPE_co`, `RDTYPE_co`) into a small `_ak_interop.py`, drop the dead
`_array.Ragged` fallback in `_ops.to_packed` and the `_ArrayRagged` branch in
`_core.is_rag_dtype`. (This relocation was prototyped during the audit and works.) Lands in
the same seqpro PR as A–C.

**Deletion is green-lit — not gated on the GPU run.** The audit ledger flagged
genvarformer's GPU/CUDA test subset (9 files, not runnable on osx-arm64) as a pre-deletion
check; we are explicitly **not** blocking on it. Any genvarformer GPU fallout is handled by
its maintainers (also us) in a follow-up. Run the GPU subset opportunistically when a
Linux+CUDA host is available.

## 5. GVL files & phasing (one branch, follows handoff Phases 1–3)

**Phase 1 — core container.**
- `_rag_variants.py` (27 `ak.`): rewrite as the wrapper around a record `Ragged`; delete the
  awkward helpers/registration listed in §2.
- `_shm_layout.py` (7): (de)serialize record buffers (`.data` dict / `.offsets` /
  `str_offsets` / `.shape`) instead of `from_ak` / `ak.zip`.

**Phase 2 — ragged ops.**
- `_ragged.py` (14): `prepend_pad_itv`'s `ak.concatenate(axis=2)` → `seqpro.rag.concatenate`
  (§4.A); drop the now-unused awkward RC helpers (`reverse_complement`,
  `_ak_comp_dna_helper`) if nothing else uses them. Audit remaining `.to_ak()` calls.

**Phase 3 — consumers.** Replace `.to_ak()` round-trips and `ak.*` aggregations with `_core`
native ops (`.to_numpy`, `.to_padded`, `.lengths`, `.data`, `.offsets`, record field access)
or numpy on flat buffers:
`_haps.py` (7, incl. the AF-filter `_core→to_ak→ak.to_regular→Ragged→to_packed` round-trip),
`_indexing.py` (9), `_splice.py` (8), `_chunked.py` (6), `_write.py` (5),
`_flat_variants.py` (4), `_impl.py` (3), `_torch.py` (3), `_tracks.py` (2),
`_reference.py` (2), `_flat.py` (1).
`_chunked.py` and `_indexing.py` rely on `RaggedVariants` being an `ak.Array` subclass
(`len`, slicing, indexing); these must move to the wrapper's `__len__`/`__getitem__`, whose
semantics must match today's exactly (multi-leading-axis indexing has edge cases — §6.4).

**Phase 4 — verify & unblock.**
`grep -rn -E "ak\.|import awkward" python/genvarloader --include='*.py'` → zero in production
(only deliberate test/interop spots remain). Then execute §4.D in seqpro (delete `_array.py`)
and run all consumer suites against the `_array`-free seqpro.

## 6. Constraints / gotchas (surfaced by probing — must hold in implementation)

1. **`to_packed` on records / string-under-axis fields** raised `NotImplementedError` at probe
   time ("Spec C"); **§4.C makes it obligatory upstream**, so GVL packs the record directly.
   (The `to_chars()`-first view is the fallback only until §4.C lands.)
2. **Whole-record `to_ak()` is buggy** on multi-leading-axis records (fixed by §4.B). Until
   that lands, tests compare **per-field** (`rv["alt"].to_ak()`), which works.
3. **`reverse_complement` requires a single ragged axis** (rejects the R=2 char view:
   `expected 2 offsets arrays, got 1`). RC must use the flat **allele-level R=1** view, as
   `reverse_complement_masked` already does.
4. **Multi-leading-axis record indexing has subtleties** (a fancy-index probe behaved
   unexpectedly). The wrapper's `__getitem__` must preserve today's `_indexing.py` semantics;
   test against the awkward oracle.

## 7. Feasibility evidence (probes run against seqpro 0.16.0 `_core`)

- Build R=2 S1 `(b, p, ~v, ~l)` → `.to_strings()` → opaque `(b, p, ~v)`; round-trips via
  per-field `to_ak`. ✅
- `Ragged.from_fields({"alt": opaque_str, "start": numeric})` sharing one offsets object →
  record `Ragged`, dtype `[('alt','S1'),('start','<i4')]`, shape `(b, p, ~v)`. ✅
- `len`, slicing, fancy-index, `rv[i,j]["alt"]` field access on the record. ✅
- `to_padded` on the char view → dense `(b, p, v, l)`; opaque-string `.to_padded` → dense
  string array. ✅
- char-view `.lengths` → per-variant allele lengths (for `ilen`/`end`). ✅
- `to_packed`: fails on record/opaque (NotImplementedError) but **succeeds on the char
  view** → confirms the `to_chars()`-first strategy. ✅
- Whole-record `to_ak()` fails on `(b, p, ~v)` (the §4.B bug). ⚠️ (per-field works)

## 8. Verification

- **GVL:** `pixi run -e dev pytest tests -q` stays at **800 passed, 0 failed** throughout.
  `pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/`
  (pre-push hook enforces both). Run the full tree before pushing (renames/shared code).
  Awkward retained as test-only oracle (per-field comparisons; whole-record once §4.B lands).
- **seqpro:** own suite stays green (`pixi run -e dev pytest tests/` → 544 passed); add tests
  for `concatenate` (§4.A), the `to_ak` record fix (§4.B), and record `to_packed` (§4.C).
- **Final gate (with §4.D):** after deleting `_array.py`, GVL/genoray/genvarformer CPU suites
  all green against the `_array`-free seqpro. The genvarformer GPU/CUDA subset is run
  opportunistically on a Linux+CUDA host (not a blocker — §4.D).
- No new differential harness: we're swapping one already-Rust-backed impl for awkward, and
  the existing green suite already encodes byte-identical parity (it passed on both backends).

## 9. PR strategy, merge sequencing, doc updates

- **PR strategy:** one bundled GVL PR on `rust-ragged-audit`; one seqpro PR for §4. Solo
  maintainer — keep-branch after the plan (see [[feedback_pr_strategy]]).
- **Merge order:** seqpro PR (A–C) first or together; repoint GVL/genoray/genvarformer
  `pixi.toml` from local editable `{ path = "../SeqPro" }` to the released/merged seqpro
  version before merging (handoff caveat). §4.D (awkward deletion) lands in the same seqpro
  PR as A–C.
- **Docs:** update `SeqPro/skills/seqpro/SKILL.md` for `concatenate`; update GVL
  `skills/genvarloader/SKILL.md` only if `RaggedVariants`'s public behavior changes (method
  API is preserved; note any equality/RC semantics change). Add a note to
  `docs/roadmaps/rust-migration.md` that Python-level awkward removal via `_core` is a
  precursor to Phases 1–2 (and tick the relevant "remove awkward" bullets if appropriate).

## 10. Out of scope

- Rust-crate migration (`rust-migration.md` roadmap) — separate initiative.
- Removing awkward from test oracles.
- Public API method changes to `RaggedVariants` (`.alt`/`.rc_()` etc. preserved).

## 11. Risks

- **Indexing parity** (§6.4) is the highest-risk area; mitigate with oracle-based tests in
  `_indexing.py`/`_chunked.py` paths.
- **genvarformer GPU/CUDA paths** exercise `_core.Ragged` through flash-attn/Nested and
  aren't run locally; deletion proceeds anyway (§4.D) — gvf maintainers handle any fallout.
- **Cross-repo merge coupling** (pixi repoint) — follow the handoff caveat checklist.
