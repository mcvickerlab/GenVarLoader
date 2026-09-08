# Design: buffered & double-buffered dataloader support for variant-window output

**Date:** 2026-07-20
**Status:** design (pending spec review)
**Target branch:** `main` (map-style `Dataset` torch transports — **not** the StreamingDataset effort;
this touches `_torch.py`/`_chunked.py`/`_shm_layout.py`/`_producer.py`/`_double_buffered_loader.py`,
none of `_streaming.py`/`src/stream/`)
**Issue:** _TBD_ — file a `type: enhancement` issue and link it here before the first PR.
**Line refs:** against `main` @ `e38d5094`. Cite symbols first; line numbers are a drifting aid.

## Summary

`gvl.Dataset.to_dataloader(...)` offers two custom torch transports beyond the default per-item
`DataLoader` (`mode=None`): `mode="buffered"` (in-process chunked fetch) and
`mode="double_buffered"` (subprocess producer + 2-slot shared-memory ping-pong). Both **hard-reject**
two flat-only output configs today:

- **Config A — `with_seqs("variant-windows")`** → `_FlatVariantWindows` output.
- **Config B — `with_seqs("variants")` carrying ride-along flank tokens** (`with_settings(flank_length=...)`)
  → `_FlatVariants.flank_tokens`.

Both are rejected up front in `get_dataloader` (`_torch.py:164-192`) for the same root cause: the
transport machinery does not understand these two output shapes end-to-end (byte accounting, chunk
slicing, shared-memory serialization, and producer-schema replay). This design lifts both
restrictions for **both transports and both configs**, keeping byte-identical parity with the
`mode=None` per-item path.

Variant-windows supports `ref`/`alt` chosen **independently** (`VarWindowOpt.ref`/`.alt ∈
{"window","allele"}`, `_flat_variants.py:957-958`), so any subset of the four output slots
(`ref_window`, `alt_window`, `ref`, `alt`) can be present. The design accommodates arbitrary present
subsets rather than a fixed set.

## Background: what exists

`get_dataloader` (`_torch.py:94`) dispatches on `mode`:

- **`mode=None`** — the existing per-item `td.DataLoader` over `TorchDataset.__getitem__` →
  `dataset[r, s]`. Supports every output mode, including variant-windows. This is the parity oracle.
- **`mode="buffered"`** (`_buffered_loader.py`) — a `td.IterableDataset` that runs **entirely
  in-process**: a `ChunkPlanner` groups the epoch's `(r, s)` indices into memory-bounded chunks, calls
  `dataset[chunk_r, chunk_s]` directly, and slices each chunk into mini-batches via `slice_chunk`. No
  serialization, no subprocess.
- **`mode="double_buffered"`** (`_double_buffered_loader.py`) — a `spawn`ed producer subprocess
  reopens the dataset from `ds.path`, replays a `schema` dict of `with_*` settings
  (`_producer.py:_apply_schema`), and writes chunks into two fixed-size shared-memory slots
  (`_shm_layout.write_chunk`) that the consumer reads back (`read_chunk`) and slices. The producer
  ping-pongs the two slots; only 2 buffers ever exist.

Both custom transports share `ChunkPlanner`/`slice_chunk` (`_chunked.py`) and the byte table from
`Dataset._output_bytes_per_instance` (`_impl.py`, called at `_torch.py:83`). Existing guards
(`_double_buffered_loader.py:181-197`, `:231-235`) already reject `is_spliced`, non-default
`insertion_fill`, and the dummy in-memory dataset for double-buffered; those are unaffected.

### The output types in question

- **`_FlatVariantWindows`** (`_flat_variants.py:299`): `fields` dict (`start`/`ilen`/`dosage`/INFO →
  `_Flat`) plus up to four optional **`_FlatWindow`** slots (`ref_window`, `alt_window`, `ref`, `alt`;
  `_WINDOW_FIELD_NAMES`, `:295`). `_present()` returns only the non-`None` slots. It is **flat-only and
  has no ragged form** (`:299` docstring); reverse-complement is intentionally unsupported.
- **`_FlatWindow`** (`_flat_variants.py:191`): a token buffer with **two ragged axes** — `data`
  (`uint8` or `int32` tokens, dtype from the token LUT), `seq_offsets` (inner, per-variant token
  boundaries, `n_variants+1`), `var_offsets` (outer, per-`b*p` group, `b*p+1`); shape `(b, p, ~v, ~win)`.
  Window length is **variable per variant**.
- **`_FlatVariants.flank_tokens`** (`_flat_variants.py:382`): a ride-along `_Flat` of shape
  `(b, ploidy, None, 2L)`, built at `_flat_variants.py` (`_Flat.from_offsets(tok, (b, p, None, 2*L), off)`).
  It is a **separate dataclass attribute, not a member of `.fields`**, and its ragged axis sits in the
  middle — which is why its own `__getitem__` refuses to slice it (`:425`) and the shm writer never
  sees it.

## Gap analysis

| # | Gap | Location | buffered | double_buffered |
|---|-----|----------|:---:|:---:|
| 1 | **Byte accounting** raises `AssertionError("unknown sequence_type 'variant-windows'")`; the `"variants"` branch **silently omits `flank_tokens`** (undersizes) | `_impl.py:_output_bytes_per_instance` (~:1308; `else` raise ~:1451; variants branch ~:1406) | ✅ | ✅ |
| 2 | **`slice_chunk`** can't slice `_FlatVariantWindows` (`_FLAT_TYPES`, `_chunked.py:126`, excludes it → `TypeError`); `_FlatVariants.__getitem__` **raises** when `flank_tokens` is set (`_flat_variants.py:425`) | `_chunked.py` + `_flat_variants.py` | ✅ | ✅ |
| 3 | **shm serialize/read**: `write_chunk` → `TypeError` for `_FlatVariantWindows` (`_shm_layout.py:381`); `_write_flat_variants` iterates only `.fields`, **dropping `flank_tokens`** (`:627`) | `_shm_layout.py` | — | ✅ |
| 4 | **producer schema** carries `sequence_type` (a string) but **not** `VarWindowOpt`/flank config; `with_seqs` is replayed with `kind` only | `_double_buffered_loader.py` schema build; `_producer.py:_apply_schema` | — | ✅ |
| 5 | **Remove the two rejection guards** | `_torch.py:164-192` | ✅ | ✅ |

**Key asymmetry:** buffered runs in-process, so it needs only gaps **1, 2, 5**. Double-buffered
additionally needs **3, 4** — the shared-memory format and the schema replay. This is what makes
buffered the smaller, independently-shippable first step.

## Design

### PR 1 — buffered support (in-process; gaps 1, 2, 5)

1. **Byte accounting (gap 1).** Add a `"variant-windows"` branch to
   `_output_bytes_per_instance`'s `seq_kind` switch that sums, over `_present()` slots only,
   `n_variants × window_len × token_itemsize` plus offset overhead for **two** ragged levels
   (`8 × (n_variants+1)` inner + `8 × (b·p+1)` outer per present slot). Window length follows the
   builder's own formula (`fill_empty_groups`, `_flat_variants.py:332`): `2·flank_length + allele_len`
   for the `_window` slots, bare `allele_len` for the `ref`/`alt` slots. Add a `flank_tokens` term to
   the existing `"variants"` branch: `n_variants × 2·flank_length × token_itemsize` + offsets.
   Mirror the exact ragged pre-pass the variants branch already does over the `(n_regions, n_samples)`
   grid — undersizing is only a memory concern for buffered, but a **slot-overflow bug** for
   double-buffered, so accuracy is required.

2. **Chunk slicing (gap 2).** Add `_FlatVariantWindows` to `_chunked._FLAT_TYPES` and give it (and
   `_FlatWindow`) an instance-axis `__getitem__`/`__len__` that rebases the scalar `.fields` **and**
   each present `_FlatWindow`'s two offset levels. Make `_FlatVariants.__getitem__` stop raising when
   `flank_tokens` is set: slice the mid-ragged `(b, p, None, 2L)` ride-along alongside the fields.
   `slice_chunk`'s `_len`/`_slice_one` then dispatch to these.

3. **Guards (gap 5).** Drop `buffered` from the two `get_dataloader` rejections; keep them for
   `double_buffered` until PR 2 lands (narrow the guard by `mode`).

PR 1 delivers buffered support for **both A and B** end-to-end, entirely in-process.

### PR 2 — double-buffered support (serialization; gaps 3, 4)

1. **Producer schema (gap 4).** In `_double_buffered_loader._spawn_producer`, when the seqs config is
   a variant-windows `Haps`, serialize the `VarWindowOpt` — all pure primitives
   (`flank_length: int`, `token_alphabet: bytes`, `unknown_token: int`, `ref`, `alt`;
   `_flat_variants.py:267-293`) — into the schema. Also serialize the plain-variants flank config
   (`flank_length` + `token_alphabet` + `unknown_token`) for Config B. In `_apply_schema`, reconstruct
   the opt and replay `ds.with_seqs("variant-windows", VarWindowOpt(...))` (Config A) or
   `ds.with_settings(flank_length=..., token_alphabet=..., unknown_token=...)` (Config B). **The token
   LUT is rebuilt inside `with_seqs`/`with_settings` from the raw alphabet+unknown_token — no LUT array
   crosses the process boundary.**

2. **Shared-memory format (gap 3).** See the decision below. Add a new **`kind=4`** for
   `_FlatVariantWindows` and extend `kind=2` with an optional `flank_tokens` sub-descriptor.

3. **Consumer reshape.** Teach `_reshape_ragged_for_chunk` (`_double_buffered_loader.py:44-103`) to
   re-introduce the ploidy axis on a read-back `_FlatVariantWindows` (and its `_FlatWindow` slots),
   analogous to its `_FlatVariants`/`RaggedAnnotatedHaps` handling.

4. **Guards (gap 5).** Remove the `double_buffered` arm of the two rejections.

### The one real design choice — shm format for `_FlatVariantWindows`

`_FlatVariantWindows` is not dense: up to four `_FlatWindow` buffers, each with two ragged axes, plus
the scalar `.fields`. The shm layout has four kinds today (`_shm_layout.py:832-849`): `0` dense, `1`
ragged/`_Flat`, `2` variants (`RaggedVariants`/`_FlatVariants`), `3` annotated. Its per-field
`FieldDescriptor` carries a single `inner_offsets` pair — insufficient for the window's second ragged
axis.

- **(A — recommended) New `kind=4`.** Symmetric with kinds 2/3: a presence bitmask over the four
  window slots, the scalar `.fields` (reuse kind=2's field encoding), and each present `_FlatWindow`'s
  `(data, seq_offsets, var_offsets)`. Extend `kind=2` with **one optional `flank_tokens`
  sub-descriptor** for Config B (`flank_tokens` genuinely belongs to `_FlatVariants`). Most code, but
  keeps each kind focused and preserves the fixed-layout zero-copy discipline.
- **(B) Overload `kind=2`'s `FieldDescriptor`** with a second inner ragged axis + ride-along. Less new
  code, but bloats an already-complex descriptor and conflates "fields" with "windows."
- **(C) pickle these types across the boundary.** Trivial, but breaks the fixed-layout zero-copy
  contract that is the entire point of `double_buffered`. **Rejected.**

**Chosen: (A).** The presence bitmask makes arbitrary present-subsets round-trip
(`{ref_window, alt}`, `{ref_window, alt_window}`, …); the token dtype (`uint8`/`int32`) rides in each
sub-descriptor's `dtype_str` as existing descriptors already do.

## Testing / parity

- **Config A oracle — `mode=None` per-item.** Variant-windows has no ragged form, so parity is against
  the default per-item path: iterate `dataset[r, s]` and compare each transported
  `_FlatVariantWindows`/`_FlatWindow` field (`.to_ragged()` / offset-aware `to_padded`) against the
  per-item output. Cover asymmetric configs explicitly: `ref="window", alt="allele"` and the reverse,
  plus `{ref_window, alt_window}` and bare `{ref, alt}`.
- **Config B oracle.** Flat variants with flank tokens vs the `mode=None` per-item output; also
  flat-vs-ragged where a ragged form exists.
- **Transport parity.** `double_buffered` vs `buffered` (mirrors
  `test_double_buffered_iter_matches_buffered`), both vs `mode=None`.
- **Flip the rejection tests** (`tests/unit/test_buffered_loader.py`): `test_flat_buffered_rejects_variant_windows`
  (:135) and `test_flat_buffered_rejects_variants_flank_tokens` (:118) become positive parity tests,
  mirroring `test_flat_buffered_plain_variants_still_works` (:154). Same in
  `tests/unit/test_double_buffered_loader.py`.
- **Fixtures.** `get_dummy_dataset()` (buffered, in-process); the `file_backed_ds` fixture
  (double_buffered, since the producer reopens `ds.path`).
- **Rebuild note.** No `src/` change is expected (the Rust `_assemble_variant_buffers_rust` kernel is
  unchanged); if any lands, `maturin develop --release` before pytest per CLAUDE.md. Run the full tree
  before pushing (`slice_chunk`/`_output_bytes_per_instance` are shared code).

## Decisions

- **Both configs, both transports** (chosen). A and B share one serialization gap; fixing both closes
  the whole "flat + windows/flank-tokens over the buffered transports" hole in one effort.
- **Buffered first, as a 2-PR stack** (chosen). Buffered is in-process (gaps 1/2/5 only) and ships
  independently; double-buffered layers the shm/schema work (gaps 3/4) on top. Narrow the guards by
  `mode` between the two PRs.
- **shm `kind=4` + `kind=2` flank_tokens extension** (chosen, option A) over overloading kind=2 or
  pickling.
- **LUT rebuilt in the child from primitives** (chosen) — no LUT array serialized; the schema carries
  only `VarWindowOpt`/flank primitives.

## Decomposition (PR stack)

| PR | Scope | Depends on | Configs |
|----|-------|-----------|---------|
| **PR 1** | Buffered: byte-accounting branch (gap 1) + `slice_chunk`/`_FlatVariantWindows`/`_FlatWindow`/`_FlatVariants` slicing incl. flank tokens (gap 2) + narrow guards to `double_buffered` (gap 5). Positive + `mode=None` parity tests. | — | A + B |
| **PR 2** | Double-buffered: producer schema serializes `VarWindowOpt`/flank primitives + `_apply_schema` replay (gap 4); shm `kind=4` + `kind=2` `flank_tokens` (gap 3); `_reshape_ragged_for_chunk` (consumer); drop the double_buffered guards. Transport + `mode=None` parity tests. | PR 1 | A + B |

Docs/skill fold into the PR that lifts the restriction (CLAUDE.md public-API + docs-audit gates):
update the `mode=`/output-mode compatibility notes in `docs/source/*.md` (`dataset.md`, `faq.md`) and
`skills/genvarloader/SKILL.md`. `api.md`/`__all__` are unaffected: no new public symbol
(`VarWindowOpt` is already exported, `__init__.py:71`); this only widens which `mode=` values accept
an existing output config.

## Risks

- **Byte-accounting accuracy.** Ragged window lengths vary per variant; the exact pre-pass must match
  the builder's `2·flank + allele_len` formula. An undersize is a **slot overflow** for
  double-buffered (`ChunkPlanner` sizes the shm slot from `peak_chunk_bytes`). Parity tests plus an
  explicit slot-fit assertion guard this.
- **Two-level ragged rebasing.** `_FlatWindow` slicing must rebase both offset levels; the mid-ragged
  `flank_tokens` `(b, p, None, 2L)` is the subtle one. Covered by per-slot parity.
- **Present-subset combinatorics.** Four optional window slots → the shm bitmask and the reshape path
  must handle every subset; enumerate the `ref`/`alt` × `window`/`allele` matrix in tests.
- **No `src/` change expected**, so the risk is confined to Python serialization/slicing — but
  `slice_chunk`/`_output_bytes_per_instance` are shared by the haplotype/variants paths; the full-tree
  run guards regressions there.

## References

- Transports: `python/genvarloader/_torch.py` (`get_dataloader:94`, guards `:164-192`),
  `_buffered_loader.py`, `_double_buffered_loader.py` (schema `:199-228`, reshape `:44-103`),
  `_producer.py` (`producer_main`, `_apply_schema`), `_shm_layout.py` (`write_chunk:341`,
  `_write_flat_variants:613`, kinds dispatch `:832-849`), `_chunked.py` (`ChunkPlanner`,
  `slice_chunk`, `_FLAT_TYPES:126`).
- Output types: `python/genvarloader/_flat.py` (`_Flat`, `_FlatAnnotatedHaps`),
  `python/genvarloader/_dataset/_flat_variants.py` (`_FlatWindow:191`, `VarWindowOpt:267`,
  `_WINDOW_FIELD_NAMES:295`, `_FlatVariantWindows:299`, `_FlatVariants:374`, `flank_tokens:382`,
  builder `ref_mode`/`alt_mode:957-958`).
- Byte accounting: `python/genvarloader/_dataset/_impl.py` (`_output_bytes_per_instance:1308`;
  `with_seqs`/variant-windows LUT build).
- Tests to mirror/flip: `tests/unit/test_buffered_loader.py`,
  `tests/unit/test_double_buffered_loader.py`.
