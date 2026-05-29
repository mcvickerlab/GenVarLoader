# Issue #191 — var_fields loading + dosage correctness fix

**Status:** Spec ready
**Date:** 2026-05-24
**Issue:** [#191](https://github.com/mcvickerlab/GenVarLoader/issues/191) — Dosage field attached to Variants source even when not requested
**Scope:** `python/genvarloader/_dataset/_haps.py`, no public API changes

## Problem

Opening a `Dataset` with a `RaggedVariants` output and a SVAR backend that has dosages causes a downstream `ak.broadcast_records` crash:

```
ValueError: cannot broadcast records because fields don't match:
    alt, ref, start
    alt, dosage, ref, start
```

Root cause: in `Haps._get_variants`, `dosage` is added to the output record unconditionally whenever `dosages.npy` exists on disk — independent of the user's `var_fields`. Every other field (`ref`, `ilen`, info columns) is correctly gated by `var_fields`; dosage is the lone exception.

A secondary problem: `_Variants.from_table` eagerly loads every numeric column from the variants table into `self.info` via `to_numpy()`, regardless of what the user asked for. Not a correctness bug, but wasteful on large SVAR files where info columns can be GB-scale.

## Goals

1. **Correctness:** the output `RaggedVariants` record schema matches the user's `var_fields` exactly. No phantom dosage field.
2. **Discoverability:** `Dataset.available_var_fields` lists every field the on-disk dataset *could* provide (including `dosage` when present), not just what was loaded.
3. **Lazy loading:** info columns and dosages are only loaded when `var_fields` requests them. Schema discovery does not require loading data.
4. **`with_settings(var_fields=...)` honors lazy loading:** expanding the field set after open lazily loads the newly requested columns, not the full set.

## Non-goals

- No public API additions (no new methods, no new args beyond what already exists on `Dataset.open` and `Dataset.with_settings`).
- No `__all__` change.
- No on-disk format change.
- No rename of `available_var_fields` / `active_var_fields`.
- Default `var_fields` stays `["alt", "ilen", "start"]` — the minimum useful set.

## Design

### Phase 1 — correctness (the visible bug)

In `_dataset/_haps.py`:

1. **Gate dosage output by `var_fields`.** In `_get_variants` (~L597):
   ```python
   if self.dosages is not None and "dosage" in self.var_fields:
       ...
       fields["dosage"] = Ragged(ak.to_packed(dosages))
   ```

2. **List dosage as available.** In `Haps.__post_init__`, when `self.dosages is not None`, add `"dosage"` to `available_var_fields`.

That's the minimum to fix the user's crash.

### Phase 2 — lazy loading

In `_dataset/_haps.py`:

1. **`_Variants.from_table` accepts an `info_fields` filter:**
   ```python
   @classmethod
   def from_table(
       cls,
       path: str | Path,
       one_based: bool = True,
       info_fields: set[str] | None = None,  # None = load all (back-compat)
   ): ...
   ```
   When `info_fields` is set, only those numeric columns are `.to_numpy()`'d into `self.info`. POS/ILEN/ALT always load; REF loads if present in schema.

2. **Schema-peek helper:**
   ```python
   @staticmethod
   def available_info_fields(path: str | Path) -> list[str]:
       """Return numeric column names without loading any data."""
       schema = pl.scan_ipc(path).collect_schema()
       return [k for k, v in schema.items() if v.is_numeric() and k not in {"POS", "ILEN"}]
   ```
   Used by `Haps` to compute `available_var_fields` from the file's schema rather than from `self.variants.info.keys()`.

3. **Lazy info loading on extension:**
   ```python
   def load_info(self, fields: Iterable[str]) -> None:
       """Add missing numeric fields to self.info by re-reading from path."""
       missing = [f for f in fields if f not in self.info]
       if not missing:
           return
       # re-read just the missing columns
       df = pl.read_ipc(self.path, columns=missing, memory_map=False)
       for f in missing:
           self.info[f] = df[f].to_numpy()
   ```
   Called by `with_settings` when the user expands `var_fields`.

4. **`Haps.available_var_fields`** computed from schema peek + reference presence + dosage presence — not from `self.variants.info.keys()`. Done in `__post_init__` with an extra path argument or by re-peeking inside.

5. **`Haps.from_path(..., var_fields=None)`** new parameter:
   - When provided, filters which info columns load and gates the dosages memmap on `"dosage" in var_fields`.
   - When `None`, current behavior: load nothing extra (`var_fields` defaults to the minimum set `["alt", "ilen", "start"]`).
   - Important: `var_fields` is propagated to both branches of `from_path` (SVAR and legacy `variants.arrow`).

6. **`Dataset.open` plumbs `var_fields` to `Haps.from_path`** rather than relying on the post-construction `replace(haps, var_fields=...)`. The schema-peek `available_var_fields` is computed inside `Haps.__post_init__` so validation in `_impl.py:288` continues to work — but now reflects what the file truly offers, not what got loaded.

7. **`Dataset.with_settings(var_fields=...)`:** if the new set adds fields not yet loaded, call `haps.variants.load_info(new_fields)` and, if `"dosage"` was added, lazily memmap dosages. Then `replace(haps, var_fields=new_var_fields)` as today.

### Default-field semantics

- `Dataset.open()` without `var_fields=` → `["alt", "ilen", "start"]` (minimum useful set). No dosage. No optional info columns. No ref unless the user asks.
- `Dataset.available_var_fields` lists everything the file could provide.
- `Dataset.active_var_fields` lists what is currently configured.

This matches the current implicit contract — only the *loading* changes from eager to lazy.

## File-by-file changes

- `python/genvarloader/_dataset/_haps.py`
  - `_Variants.from_table`: add `info_fields` parameter.
  - `_Variants`: add `available_info_fields` staticmethod and `load_info` method.
  - `Haps.__post_init__`: compute `available_var_fields` from schema peek + dosage presence.
  - `Haps.from_path`: add `var_fields` parameter; plumb to `from_table` and dosage memmap.
  - `Haps._get_variants`: gate `fields["dosage"]` on `"dosage" in self.var_fields`.

- `python/genvarloader/_dataset/_impl.py`
  - `Dataset.open`: pass `var_fields` (the user's, or the default) into `Haps.from_path` rather than handling via post-construction `replace`.
  - `Dataset.with_settings`: when `var_fields` expands the active set, call `load_info` (and memmap dosages if needed) before `replace`.

- `python/genvarloader/_dataset/_open.py`
  - Update the `OpenRequest` stage that builds `Haps` to forward `var_fields` to `from_path`.

## Testing

In `tests/dataset/`:

1. **Dosage gating (the bug):**
   - Open a dataset with SVAR-dosages. Set `with_settings(var_fields=["alt", "ref", "start"])`. Fetch a batch. Assert `"dosage"` not in `batch.fields`.
   - Same dataset, `with_settings(var_fields=["alt", "ref", "start", "dosage"])`. Assert `"dosage"` is in `batch.fields`.

2. **`available_var_fields` discovery:**
   - SVAR-with-dosages: assert `"dosage"` in `dataset.available_var_fields`.
   - SVAR-without-dosages: assert `"dosage"` not in `dataset.available_var_fields`.

3. **Lazy loading:**
   - Open a dataset, assert `dataset._seqs.variants.info` does not contain numeric columns that weren't in `var_fields`. (Internal-state test; OK because the file is internal.)
   - `with_settings(var_fields=[..., "AF"])` on a dataset that didn't initially request AF — assert `"AF"` is now in `dataset._seqs.variants.info`.

4. **Default-field shape:**
   - `Dataset.open(...)` without `var_fields` — assert `dataset.active_var_fields == ["alt", "ilen", "start"]`.

5. **Existing test suite stays green** (parity check; the bulk of dataset tests exercises every code path).

## Risk

- Low for Phase 1 (3-line correctness fix + 1-line discoverability fix).
- Medium for Phase 2 (touches `_Variants`, `Haps.from_path`, `Dataset.open`, `Dataset.with_settings`). Mitigated by: (a) `info_fields=None` keeps legacy behavior for any caller that doesn't pass it; (b) schema-peek is read-only and trivially testable; (c) the existing test suite exercises the load paths.

## Verification

Per project conventions (CLAUDE.md):

- `pixi run -e dev test` — pytest + cargo green.
- `pixi run -e dev ruff check python/` clean.
- `pixi run -e dev typecheck` (pyrefly) — baseline preserved, no new errors.
- `skills/genvarloader/SKILL.md` review: this PR does NOT change public API surface, so no skill update needed. (Confirm during implementation.)

## Out of scope (captured for later)

- Renaming `available_var_fields` / `active_var_fields`.
- Unified "available_fields" property covering both variants and tracks.
- Lazy loading for tracks data (separate concern).
- Caching schema peeks across multiple `Dataset.open` calls on the same path.
