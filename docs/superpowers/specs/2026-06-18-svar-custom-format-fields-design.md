# Surface genoray SparseVar custom FORMAT fields in Dataset/RaggedVariants

**Issue:** #231
**Date:** 2026-06-18
**Status:** Approved design

## Problem

A `.svar` written by **genoray** can carry arbitrary per-call (`Number=G`) FORMAT
fields. Each is registered in `<svar>/metadata.json["fields"]` (name → numpy
dtype string) and stored as `<svar>/<name>.npy`, sharing the genotype `offsets`.
genoray exposes them fully (`SparseVar.available_fields`,
`SparseVar.with_fields([...])`).

GenVarLoader cannot surface them: `Dataset.open(..., var_fields=["<name>"])`
raises *"missing from `available_var_fields`"*. Only the hardcoded `dosage`
per-call field is supported. Consumers storing a per-call annotation on the
`.svar` (e.g. a de-clustering subtype code) must currently bypass `gvl.Dataset`
and re-implement a memmap-over-offsets reader.

Versions observed in the issue: genvarloader 0.33.0, genoray 2.12.0. The feature
is already present in the installed genoray 2.9.2 (`available_fields` is an
instance attribute set in `SparseVar.__init__`; on-disk format is `<name>.npy`
+ `metadata.json["fields"]`, sharing `offsets.npy`).

## Key facts established during design

- Custom FORMAT fields are **per-call** (`Number=G`): one value per genotype
  call, sharing the genotype `offsets`. They follow the existing **`dosage`**
  code path (gathered by genotype offset ranges via `_gather_rows`), **not** the
  per-variant INFO path (`haps.variants.info[k]` gathered by `v_idxs`).
- The hot path was refactored into `_flat_variants.py:_get_flat_variants`
  (the issue's cited `_haps.py`/`_rag_variants.py` line numbers predate this).
  All variant output flows through a `fields: dict[str, Any]` built there.
- `_FlatVariants.to_ragged()` passes **every** field through as `**kwargs` to
  `RaggedVariants`, whose `__init__` already accepts `**kwargs: Ragged[np.number]`.
  Arbitrary custom field names flow through with no constructor change.
- The variant-windows path builds `wfields = {k: v for k, v in fields.items()
  if k not in ("alt", "ref")}` — custom fields ride along automatically.
- `DummyVariant.scalar_for(name, dtype)` already has a generic fallback
  (`int → 0`, `float → NaN`) for unknown names, so dummy empty-group fill works
  for custom fields without change.
- On disk, gvl reads the gvl-dataset offsets (`<gvl>/genotypes/offsets.npy`)
  for genotypes **and** dosages; custom fields use the same offsets — consistent
  with the existing dosage path.

## Scope

In scope: generalize the hardcoded `dosage` handling so **any** genoray custom
per-call FORMAT field can be requested via `Dataset.open(..., var_fields=[...])`
and surfaces in the **variants** (RaggedVariants), **variant-windows**, and
**flat** output modes. `dosage` stays a named special-case (unchanged) for
back-compat.

Out of scope: per-variant INFO columns (already handled), writing custom fields
(genoray's job), AnnotatedHaps / sequence output modes (custom fields are
variant metadata, not sequence).

## Design

### 1. Discovery — `_haps.py`

New module-level helper, reading the registry directly (no `SparseVar`
instantiation — mirrors the direct-memmap dosage approach and avoids loading
genoray's string index):

```python
def _svar_format_fields(svar_dir: Path) -> dict[str, np.dtype]:
    """genoray custom per-call FORMAT fields: name -> dtype, from <svar>/metadata.json."""
    meta = svar_dir / "metadata.json"
    if not meta.is_file():
        return {}
    fields = json.loads(meta.read_text()).get("fields", {})
    return {name: np.dtype(dt) for name, dt in fields.items()}
```

In `Haps.__post_init__` (`_haps.py:293`), append the custom field names to
`available_var_fields`, de-duplicated so a name shared with an INFO column
appears once:

```python
custom_fmt = _svar_format_fields(self.variants.path.parent)
base = (
    ["alt", "ilen", "start"]
    + schema_info_fields
    + (["ref"] if self.variants.ref is not None else [])
    + (["dosage"] if has_dosage_file else [])
)
self.available_var_fields = base + [f for f in custom_fmt if f not in base]
```

This makes the `_impl.py:331` missing-field guard pass for requested custom
fields.

`self.variants.path` is `<svar>/index.arrow` for SVAR datasets, so
`.parent` is the svar dir. For non-SVAR / synthetic in-memory datasets
`metadata.json` won't exist and the helper returns `{}` (no behavior change).

### 2. Load — `_haps.py:from_path`

- New `Haps` field: `var_field_data: dict[str, Ragged] = field(default_factory=dict)`,
  parallel to `dosages`.
- Compute `custom_fmt = _svar_format_fields(svar_path)` inside the
  `svar_meta_path.exists()` branch.
- Exclude custom fields from the INFO lookup (they are **not** columns in
  `index.arrow`):
  ```python
  info_fields = {f for f in var_fields if f not in builtin and f not in custom_fmt}
  ```
- For each requested name in `custom_fmt`, memmap and build a `Ragged` on the
  same gvl genotype offsets used for genotypes/dosages:
  ```python
  var_field_data: dict[str, Ragged] = {}
  for name in var_fields:
      if name in custom_fmt:
          mm = np.memmap(svar_path / f"{name}.npy", dtype=custom_fmt[name], mode="r")
          var_field_data[name] = Ragged.from_offsets(mm, rag_shape, offsets.reshape(2, -1))
  ```
- Pass `var_field_data=var_field_data` to the `cls(...)` constructor. For the
  legacy (non-SVAR) branch it stays empty.

### 3. Hot path — `_flat_variants.py:_get_flat_variants`

Custom fields follow the dosage gather pattern (lines 699–713):

- Gather each requested custom field via
  `_gather_rows(geno_offset_idx, field_offsets, field_data)` against the
  UNFILTERED offsets.
- Apply the same AF `_compact_keep(field_data, unfiltered_row_offsets, keep)`
  when `keep is not None`.
- After the dosage block (~line 755), add each to the `fields` dict:
  ```python
  for name, rag in haps.var_field_data.items():
      if name not in haps.var_fields:
          continue
      cf_data = np.asarray(rag.data)
      cf_off = np.asarray(rag.offsets, np.int64)
      gathered, _ = _gather_rows(geno_offset_idx, cf_off, cf_data)
      if keep is not None:
          gathered, _ = _compact_keep(gathered, unfiltered_row_offsets, keep)
      fields[name] = _Flat.from_offsets(gathered, shape, row_offsets)
  ```
  (Exact placement/sharing with the dosage gather to be settled in
  implementation; the dosage block is the template.)
- The "other info fields" loop (line 758) must skip names handled here — add
  them to its skip set (they are not in `haps.variants.info`, so this avoids a
  `KeyError`).

Because `to_ragged()`, the variant-windows `wfields` filter, and
`DummyVariant.scalar_for` all iterate `fields` generically, **all three output
modes and dummy empty-group fill work with no further changes**.

## Decisions / edge cases

- **Name collision** (a name in both `metadata.json["fields"]` and an INFO
  column): the per-call FORMAT field **wins** — excluded from `info_fields`,
  loaded as a custom field; `available_var_fields` de-dups so it appears once.
- **`unphased_union`**: custom fields share genotype offsets, so the existing
  `row_offsets[::ploidy]` fold applies unchanged.
- **dtype** comes from `metadata.json` — never hardcoded.
- **`dosage` stays special-cased** (own block, `DOSAGE_TYPE`) — lower risk,
  preserves the byte-identical guarantee; not unified into the generic path.

## Testing — hand-crafted fixture

Unit test in `tests/unit/dataset/`:

1. Copy an existing `.svar` fixture into `tmp_path`.
2. Write a custom `int8` field `<name>.npy` whose length matches the genotype
   call count (so it shares the genotype `offsets`), and patch
   `metadata.json["fields"]` to register `{"<name>": "int8"}`.
3. Open the dataset pointing at the patched svar and assert:
   - `<name>` appears in `available_var_fields`;
   - `Dataset.open(..., var_fields=["<name>"])[0]` returns `RaggedVariants`
     where `<name>` is `int8` and per-cell length-equal to `start` / the
     genotypes (issue acceptance criteria);
   - the field rides along in **flat** mode and **variant-windows** mode.

This tests gvl's reader directly, with no dependency on genoray's writer, and
matches the byte layout gvl actually reads.

## Documentation

Per `CLAUDE.md`, this changes the accepted `var_fields` values, so:

- Update `skills/genvarloader/SKILL.md` (the `var_fields` discussion) to note
  that genoray custom per-call FORMAT fields are discoverable and requestable.
- Docstring / changelog note on `available_var_fields` and `var_fields`.

## Acceptance criteria

Given a `.svar` with a custom `int8` FORMAT field `f`:

- `available_var_fields` lists `f`;
- `Dataset.open(..., var_fields=["f"])[0]` returns `RaggedVariants` where `f`
  is `int8` and per-cell length-equal to `start` / the genotypes;
- `f` is present in flat-mode and variant-windows output.
