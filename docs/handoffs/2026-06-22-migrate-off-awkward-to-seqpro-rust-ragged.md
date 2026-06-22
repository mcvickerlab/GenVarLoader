# Handoff: Migrate GenVarLoader off awkward-array onto seqpro's Rust-backed `Ragged`

**Date:** 2026-06-22
**Status:** Not started. This is the remaining (largest) piece of the Rust-Ragged migration.
**Audience:** GenVarLoader maintainers.

## TL;DR

seqpro 0.16 flipped its public `seqpro.rag.Ragged` from the awkward backend (`_array.Ragged`) to the
Rust backend (`_core.Ragged`). GenVarLoader was audited and is **green against the Rust backend**
(`pixi run -e dev pytest tests` → 800 passed) — but **only because seqpro still ships the legacy awkward
`_array.py` module**. GVL still uses awkward internally (~98 `ak.*` call sites across 14 files, plus its own
`RaggedVariants(ak.Array)` subclass) and depends on seqpro registering awkward behaviors for the `"Ragged"`
type. **seqpro cannot delete `_array.py` (retire awkward) until GVL stops using awkward.** This doc is the
plan to get GVL fully onto Rust-backed `Ragged` so the awkward backend can be deleted upstream.

## Background / what's already done

The "Rust-Ragged Consumer Audit" (see `SeqPro/docs/superpowers/specs/2026-06-21-rust-ragged-audit-ledger.md`)
flipped seqpro to the Rust backend and fixed all three consumers to pass against it:
- **seqpro**: public `Ragged` is now `_core.Ragged`; own suite 544 passed. Draft PR `ML4GLand/SeqPro#56`.
- **genoray**: green (456). Draft PR `d-laub/genoray#66`.
- **GenVarLoader**: green (800) **on the `_core` backend, with `_array.py` still present**. Draft PR
  `mcvickerlab/GenVarLoader#238` (branch `rust-ragged-audit`). The audit added `.to_ak()` conversions and
  swapped record construction off bare `ak.zip`, but did **not** remove awkward from GVL.
- **genvarformer**: CPU subset green (371). `genome_emb.py` was migrated off `seqpro.rag._array.unbox` →
  `seqpro.rag.Ragged(...)` interop (already done). Draft PR `standardmodelbio/genvarformer#38`.

⚠️ **Caveat in all consumer PRs:** their `pixi.toml` pins seqpro/genoray/gvl to **local editable paths**
(`{ path = "../SeqPro" }`). Repoint to the released/merged versions before merging.

## Why this migration is needed (the upstream blocker)

We attempted to delete seqpro `_array.py` outright. It broke **30 GVL tests**. Root causes (these ARE the
coupling points you must remove):

1. **Lost ufunc behavior registration.** `_array.py` runs, at import time:
   `ak.behavior[np.ufunc, "Ragged"] = apply_ufunc`. GVL tags its awkward arrays with `__list__: "Ragged"` and
   relies on this so that `arr1 == arr2` / `ak.all(...)` dispatch correctly. Without it:
   `TypeError: no numpy.equal overloads for custom types: Ragged, Ragged`.
2. **Lost high-level wrapper class.** `_array.py` registers `ak.behavior["*", "Ragged"] = _array.Ragged`.
   When you access a record field (e.g. `rec["alt"]`) on a `"Ragged"`-tagged awkward array, awkward wraps the
   result in `_array.Ragged`, which exposes `.data`, `.lengths`, `.offsets`, `.shape`. GVL reads those
   attributes off field arrays. Without the class, the field is a plain `ak.Array` and `.data`/`.lengths` raise.
3. **`ak.where` union types.** GVL does the equivalent of `Ragged(ak.where(mask, reverse_complement(a), a))`.
   With the awkward behavior gone, the two branches no longer share the `"Ragged"` type, so `ak.where`
   produces `union[bytes, var * Ragged]`, and `seqpro.rag.Ragged(union_array)` fails with
   `ValueError: Expected 1 ragged dimension, got 0` (the awkward→`_core` ingest path can't parse a UnionArray).

Restoring just (1) in a slim shim fixed 19/30; (2) and (3) require GVL-side changes. The lesson:
**re-implementing the awkward behavior in seqpro just to keep GVL working defeats the purpose.** GVL must
stop depending on awkward-Ragged behavior. Then seqpro deletes `_array.py` cleanly.

## The awkward footprint in GVL (what to migrate)

~98 `ak.*` usages across 14 `python/genvarloader/` files (run
`grep -rn -E "ak\.|import awkward|RaggedVariants" python/genvarloader --include='*.py'` to refresh):

| file | ~ak. uses | role / notes |
|------|-----------|--------------|
| `_dataset/_rag_variants.py` | 27 | **`RaggedVariants` — GVL's own `ak.Array` subclass.** The core variant container. Biggest item. |
| `_ragged.py` | 14 | `RaggedIntervals`/`RaggedTracks`/`RaggedAnnotatedHaps` utils; `ak.concatenate(..., axis=2)`, `.to_ak()`, the `ak.where(mask, reverse_complement(rag), rag)` pattern (see line ~353). |
| `_dataset/_indexing.py` | 9 | indexing over ragged structures |
| `_dataset/_splice.py` | 8 | spliced-haplotype assembly; `.to_ak()` |
| `_shm_layout.py` | 7 | shared-memory (de)serialization of `RaggedVariants` (`from_ak`, `ak.zip`) |
| `_dataset/_haps.py` | 7 | haplotype reconstruction; AF-filter path does `_core→to_ak→ak.to_regular→Ragged→to_packed` round-trip |
| `_chunked.py` | 6 | chunk iteration; treats `RaggedVariants` as `ak.Array` subclass (`len()`, slicing) |
| `_dataset/_write.py` | 5 | write pipeline; `ak.concatenate`, `ak.max(...).to_numpy()` |
| `_dataset/_flat_variants.py` | 4 | flat-mode variant output |
| `_torch.py` | 3 | `to_nested_tensor(rag: Ragged | ak.Array)` — torch interop |
| `_dataset/_impl.py` | 3 | Dataset glue |
| `_dataset/_tracks.py` | 2 | track stacking (already partly de-awkwarded in the audit: `_ragged_stack_tracks` was vectorized) |
| `_dataset/_reference.py` | 2 | reference sequence handling |
| `_flat.py` | 1 | (docstring notes flat `Ragged` never wraps awkward) |

### Tests that break when awkward behavior is removed (your migration target list)
From the deletion attempt, these fail without the awkward `"Ragged"` behavior and must be made to pass on
pure `_core`:
`tests/dataset/test_flat_variants.py`, `tests/integration/dataset/test_query_filters.py`,
`tests/dataset/test_flat_mode_equivalence.py`, `tests/unit/ragged/test_ragged_rc_packing.py`,
`tests/unit/ragged/test_rag_variants.py`, `tests/unit/dataset/test_output_bytes_per_instance.py`,
`tests/dataset/test_flat_getitem_snapshot.py`.
Note: some tests use awkward arrays as **oracles** (`ak.to_list(got["alt"]) == ak.to_list(exp["alt"])`) — those
oracle comparisons are fine to keep awkward-based (awkward stays a *test/interop* dependency); the goal is to
remove awkward from **production data structures**, not necessarily from test oracles.

## Migration plan (suggested phases)

**Phase 0 — decide the target representation.** `RaggedVariants` is the crux: it's an `ak.Array` subclass
carrying parallel variant fields. Decide its replacement:
- (a) A seqpro **record `Ragged`** (`_core.Ragged` with fields via `seqpro.rag.zip`/`from_fields`) — fields
  accessed via `rag["alt"]`, dense via `.to_numpy()`/`.to_padded()`. This is the most "rust-ragged-native" target.
- (b) A small plain dataclass of per-field `_core.Ragged`s if record-Ragged ergonomics don't fit.
  Map every current `RaggedVariants` operation (construction, field access, `len`, slicing, `==`, shm
  round-trip, `ak.where`-based RC) to the chosen form before touching code.

**Phase 1 — `_rag_variants.py` + `_shm_layout.py`.** Replace the `RaggedVariants(ak.Array)` subclass with the
Phase-0 target. Replace `from_ak`/`ak.zip` construction and the shm serialization with `_core.Ragged`
data/offsets buffers (`.data`, `.offsets`, `.shape`, and `seqpro.rag.zip`/`from_fields` for records). Kill the
"`__list__: Ragged`" tagging entirely — that's the awkward-behavior hook.

**Phase 2 — `_ragged.py` ops.** Replace:
- `ak.concatenate([...], axis=2)` (RaggedIntervals padding) → a `_core`-native concat. seqpro has no public
  multi-axis ragged concat today (see Upstream below); interim option is offset-arithmetic on `.data`/`.offsets`
  like the already-vectorized `_ragged_stack_tracks` in `_dataset/_tracks.py` (good reference pattern).
- The `Ragged(ak.to_packed(ak.where(mask, reverse_complement(rag), rag)))` RC idiom → use
  `seqpro.rag.reverse_complement(rag, comp_lut, mask=mask)` (already `_core`-native and backend-agnostic) instead
  of `ak.where` over a union. This removes the union-type failure (root cause 3).

**Phase 3 — the consumers of the above** (`_haps.py`, `_splice.py`, `_indexing.py`, `_flat_variants.py`,
`_write.py`, `_impl.py`, `_chunked.py`, `_reference.py`, `_torch.py`). Replace `.to_ak()` round-trips and
`ak.*` aggregations with `_core.Ragged` methods (`.to_numpy`, `.to_padded`, `.lengths`, `.data`, `.offsets`,
record `to_numpy` → dict) or numpy on the flat buffers. `_chunked.py`'s "RaggedVariants is an ak.Array
subclass" assumptions (len/slicing) need updating to the Phase-0 target's API.

**Phase 4 — verify & unblock upstream.** `grep -rn -E "ak\.|import awkward" python/genvarloader --include='*.py'`
should only match deliberate interop/oracle spots (ideally zero in production). Then confirm seqpro can delete
`_array.py`: in a seqpro checkout, delete `python/seqpro/rag/_array.py` + `_gufuncs.py`, relocate the awkward
*interop helpers* (`unbox`, `RagParts`, `_parts_to_content`, `DTYPE_co`, `RDTYPE_co`) that `_ingest.py` and
`__init__.py` import into a small `_ak_interop.py`, drop the dead `_array.Ragged` fallback in `_ops.to_packed`
and the `_ArrayRagged` branch in `_core.is_rag_dtype`, and run all suites. (That relocation was prototyped in
the deletion attempt and works; it's the *consumer* awkward-behavior dependence that blocked it.)

## Optional upstream (seqpro) work — only if GVL needs it

Add to `_core.Ragged` only what the migration actually requires (keep it minimal, front-load `validate=`):
- **Multi-axis / nested ragged concatenation** (for `_ragged.py`'s `ak.concatenate(..., axis=2)` on
  `RaggedIntervals`). seqpro currently has no public ragged-concat; either add one to seqpro `_core`/`_ops`
  (Numba/offset-arithmetic, no Python loops per seqpro's hot-path rule) or implement it in GVL on flat buffers.
- **UnionArray-free RC**: none needed if Phase 2 uses `seqpro.rag.reverse_complement` instead of `ak.where`.
- **Record-field ergonomics**: if `RaggedVariants` becomes a record `Ragged`, confirm `rag["field"]`,
  `rag.to_numpy()` (returns `{field: ndarray}`), and `to_padded` cover the access patterns; file an upstream
  issue if a field operation is missing rather than reaching back into awkward.
Coordinate any upstream change against seqpro PR `ML4GLand/SeqPro#56`.

## Verification

```bash
# GVL must stay green throughout (data already generated once via: pixi run -e dev gen)
pixi run -e dev pytest tests -q          # target: 800 passed, 0 failed (same as today)
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/  # pre-push hook enforces this
# Final gate (in a seqpro checkout, after GVL is awkward-free):
#   delete _array.py + _gufuncs.py, relocate interop helpers, run: pixi run -e dev pytest tests/  → 544 passed
#   then: GVL/genoray/genvarformer suites all still green against the _array-free seqpro
```

## Pointers

- seqpro Rust Ragged API: `SeqPro/python/seqpro/rag/_core.py`; skill `SeqPro/skills/seqpro/SKILL.md` (updated for `_core`).
- Audit ledger (full breakage history + fix loci): `SeqPro/docs/superpowers/specs/2026-06-21-rust-ragged-audit-ledger.md`.
- Good reference for de-awkwarding a hot path: `_dataset/_tracks.py::_ragged_stack_tracks` (vectorized offset gather).
- The `genome_emb.py` migration in genvarformer (`unbox` → `Ragged(...)`) is a worked example of the awkward→`_core` swap.
