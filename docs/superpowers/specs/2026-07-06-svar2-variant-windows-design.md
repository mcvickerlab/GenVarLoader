# Design: variant-windows mode for svar2-backed gvl

> **Status:** design approved · **Date:** 2026-07-06 · **Branch:** `svar2-m6b-kernel` (PR #266, draft)
>
> Related: `docs/superpowers/specs/2026-06-25-target7-variant-windows-rust-assembly-design.md`
> (the `assemble_variant_buffers` kernel this reuses), and
> `docs/superpowers/specs/2026-07-05-svar2-readbound-getitem-perf-design.md`
> (which lists variant-windows as a guarded, out-of-scope svar2 mode — this
> design removes that guard).

## 1. Problem

`Dataset.open(path, reference=REF, svar2=<store>).with_seqs("variant-windows",
VarWindowOpt(...))[regions, samples]` currently raises `NotImplementedError`.
`Svar2Haps.__call__` (`python/genvarloader/_dataset/_svar2_haps.py`, ~lines
282–285) guards `_FlatVariantWindows`:

```python
if issubclass(self.kind, _FlatVariantWindows):
    raise NotImplementedError(
        "svar2 datasets do not support with_seqs('variant-windows') yet."
    )
```

Every other piece needed already exists and is parity-validated:

- The svar2 **`variants` decode** (`decode_variants_from_svar2_readbound`) is
  byte-identical to the svar2 decode oracle and to SVAR1 for the variant SET
  (`tests/dataset/test_svar2_dataset.py::test_svar2_variants_positions_match_svar1`,
  `::test_svar2_variants_match_svar2_oracle`).
- The **window-assembly Rust kernel** `assemble_variant_buffers`
  (`src/variants/windows.rs`, shimmed by `_assemble_variant_buffers_rust` in
  `python/genvarloader/_dataset/_flat_variants.py`) is validated end-to-end for
  SVAR1 and under both rust/numba backends (target-7 work).

The **only** thing svar2 variant-windows adds over the working svar2 `variants`
decode is the window-assembly step (reference fetch + flank + tokenize). So the
implementation composes two already-validated kernels; **no new Rust**.

## 2. Scope

**In scope** — the live `variant-windows` read path for svar2 datasets:

- `ref="window"` with `alt ∈ {"window", "allele"}`.
- Single-contig (fast path) and multi-contig (contig-group stitch).
- Empty (region, sample, ploid) groups with `dummy_variant` fill.

**Out of scope** (existing guards stay in force; each must raise, not silently
diverge):

- `ref="allele"` — **already** rejected upstream in `with_seqs`
  (`_impl.py` ~line 720: raises `ValueError` when `window_opt.ref == "allele"`
  and `self._seqs.variants.ref is None`, which is always true for `Svar2Haps` —
  its dummy `_Variants` carries `ref=None`, and svar2 stores no REF allele
  bytes). No new guard needed; a test pins it.
- `max_jitter > 0` at write or `jitter > 0` at read — the read-bound decode has
  no right-clip, so a padded/slid window over-includes variants past the
  (unpadded) read window. Same issue and same guard as svar2 `variants`.
- `min_af`/`max_af`, `unphased_union`, spliced, annotated, in-kernel
  reverse-complement — all already guarded (`_guard_unsupported` +
  `__call__`). Note: SVAR1 *does* support `unphased_union` + variant-windows,
  but svar2 guards `unphased_union` wholesale; that remains deferred.
- No on-disk **format** change, no **public API** signature change
  (`with_seqs`/`VarWindowOpt` are unchanged; this only makes an existing kind
  reachable for svar2).
- A **fused** single-call svar2 windows kernel (decode+assemble in one FFI
  crossing) is a perf optimization, deferred — see §7.

## 3. Architecture — compose two validated kernels per contig group

Add `Svar2Haps._reconstruct_variant_windows(idx, regions) -> _FlatVariantWindows`,
structured exactly like the existing `Svar2Haps._reconstruct_variants`: group
queries by contig (store readers are per-contig), process each group, then
stitch back to global `(b, P)` row order with the single-group identity fast
path and the multi-group inverse-row-permutation.

The window config fields (`token_lut`, `token_dtype`, `window_opt`,
`unknown_token`, `flank_length`, `dummy_variant`) are set on the `Svar2Haps`
instance by `with_seqs("variant-windows", ...)` via `replace(self._seqs, ...)`
— `Svar2Haps` is a dataclass subclass of `Haps`, so this already works
(`_impl.py` ~lines 729–744). No change to `Svar2Haps.from_path` construction is
required for the config to arrive.

### Per contig group `(ci, qsel)`

1. **Decode** (existing): cache-slice via `_gather_inputs`, then
   `decode_variants_from_svar2_readbound(...)` →
   `(pos, ilen, alt_bytes, str_off, var_off)`. These per-variant arrays are the
   already-gathered analog of SVAR1's global arrays.

2. **Assemble windows** (existing kernel, identity gather): call
   `_assemble_variant_buffers_rust` with:

   | kernel arg | svar2 value |
   |---|---|
   | `mode` | `1` (windows) |
   | `v_idxs` | `np.arange(n_var, dtype=np.int32)` (identity — data is pre-gathered) |
   | `row_offsets` | `var_off` (per-`(len(qsel)*P)`-row variant boundaries) |
   | `alt_global`, `alt_off_global` | `alt_bytes`, `str_off` |
   | `ref_global`, `ref_off_global` | `None`, `None` (`ref="allele"` blocked upstream) |
   | `want_ref_bytes`, `want_flank` | `False`, `False` |
   | `ref_mode` | `1` (window) |
   | `alt_mode` | `1` if `window_opt.alt == "window"` else `2` |
   | `flank_len` | `window_opt.flank_length` |
   | `lut` | `self.token_lut` |
   | `v_contigs` | `np.zeros(n_var, np.int32)` (single-contig slice) |
   | `v_starts`, `ilens` | `pos`, `ilen` |
   | `reference`, `ref_offsets` | `_ref_for_contig(ci)` slice + `[0, len]` |
   | `pad_char` | `self.reference.pad_char` |

   Coordinate note: `pos` is 0-based within-contig (proven `==` SVAR1 `start`),
   and the per-group reference is the single-contig slice starting at contig
   position 0, so the kernel's `contig_offset(=0) + start` indexing and its
   `[start-L, end+L)` OOB-padded read are correct — identical math to the SVAR1
   path, just fed pre-gathered arrays.

   Returned `bufs` keys: `ref_window` (always, since `ref="window"`) and either
   `alt_window` (alt="window") or `alt` (alt="allele"), each `(data, seq_off)`.

3. **Wrap** each `(data, seq_off)` into a per-group `_FlatWindow`
   (`data`, `seq_offsets=seq_off`, `var_offsets=var_off`, `shape=(b,P,None,None)`
   at group scope).

### Stitch groups → global `(b, P)` order

- **Single group** (the common single-contig read): grouped order already equals
  global `(b, P)` order → return directly, no reorder, no concatenate. Mirrors
  the `_reconstruct_variants` / `_assemble_haps` fast path.
- **Multiple groups**: compute `perm = _inverse_row_perm(cat_query_order, b, P)`
  once, then
  - scalar fields (`start`, `ilen`): reorder via `_ragged_arange_src(grouped_row_offsets, perm)` → `src`, index `pos_c[src]` / `ilen_c[src]` (exactly as `_reconstruct_variants` does for pos/ilen);
  - each window token buffer: reorder via the existing
    `_ragged_arange_gather_2level(token_data, grouped_row_offsets, grouped_seq_offsets, perm)`
    → `(new_data, new_var_off, new_seq_off)`. The returned `new_var_off` equals
    the reordered scalar `var_off_g`, so scalar and window offsets stay
    consistent by construction.

### Build the result

- Scalar `fields`: `start` always; `ilen` when `"ilen" in self.var_fields`
  (mirrors `get_variants_flat`; svar2 has no dosage/info/custom fields —
  `available_var_fields = ["alt", "ilen", "start"]`). Each wrapped as `_Flat`
  with the global `row_offsets` and `shape=(b, P, None)`.
- Assemble `_FlatVariantWindows(fields, ref_window=..., alt_window=... | alt=...)`
  (set the present window fields per `window_opt`).
- If `self.dummy_variant is not None`, apply
  `win.fill_empty_groups(self.dummy_variant, unk=self.unknown_token, flank_length=window_opt.flank_length)`
  (the existing `_FlatVariantWindows.fill_empty_groups`).

### `Svar2Haps.__call__` wiring

Replace the NotImplementedError branch with:

1. the **same jitter/max_jitter guard** already used for svar2 `variants`
   (raise `NotImplementedError(... "right-clip" ...)` when
   `self.max_jitter > 0 or jitter > 0`);
2. `return cast(_H, self._reconstruct_variant_windows(idx, regions))`.

`_guard_unsupported(splice_plan)` is called first (as today), covering
splice/exonic/min_af/max_af/unphased_union. `to_rc` is not applicable to
variant-windows (reference-oriented; RC intentionally unsupported for the kind).

## 4. Parity strategy

Both halves gated; a matched SVAR1 dataset cannot be the oracle for the alt side
because svar2 and SVAR1 encode a deletion's ALT differently (svar2 `""` vs SVAR1
anchor base, e.g. `G` for `GTA>G`) — so `alt`/`alt_window` bytes legitimately
differ. Split the gate:

1. **`ref_window` → byte-identical to a matched SVAR1 variant-windows dataset.**
   `ref_window` is a pure reference read over the same decoded variant SET
   (positions identical to SVAR1), so it must match exactly. Free, strong check
   on the reference-read + tokenize half. Compare `data` and both offset levels
   (`var_offsets`, `seq_offsets`) after `to_ragged()`.

2. **`alt`/`alt_window` (and the full bundle) → independent numpy oracle.**
   A small (~40-line) reference that consumes the *already-validated* svar2
   `variants` output (`ds2.with_seqs("variants")[:, :]`, i.e. the decoded
   `start`/`ilen`/`alt`) and, per variant, reproduces the kernel exactly:
   - `ends = start - min(ilen, 0) + 1`;
   - fetch `reference[start-L : end+L)` with absolute-coordinate `pad_char` OOB
     padding;
   - `alt="window"`: `flank5 (first L) · alt_bytes · flank3 (last L)`;
     `alt="allele"`: bare `alt_bytes`;
   - tokenize via the same LUT (built from `window_opt.token_alphabet` /
     `unknown_token`), out-of-alphabet bytes → `unknown_token`.
   Assert the oracle's `(data, seq_offsets, var_offsets)` equals the svar2
   `_FlatVariantWindows` window buffers. This is the true correctness gate for
   the alt side and independently re-derives `ref_window` too.

3. **Multi-contig stitch** — reuse the interleaved chr2/chr1 fixture
   (`_src2`/`svar2_fixture2`) so the single-contig fast path is bypassed and the
   inverse-row-perm reorder of both scalar and 2-level window buffers is
   exercised; assert against both oracles above.

4. **Empty-group / dummy fill** — a variant-free tail region (the existing bed's
   `chr1 [20, 40)`) plus a `dummy_variant`; assert each empty `(b*p)` row gets
   one all-`unknown_token` entry of the right width (`2L + len(dummy allele)` for
   `alt_window`, `2L + len(dummy ref)` for `ref_window`, `len(dummy alt)` for
   bare `alt`). Also assert the no-fill path is unchanged.

5. **Guard contracts** — tests that `ref="allele"` raises `ValueError` at
   `with_seqs`, and that `max_jitter>0`/`jitter>0` raises `NotImplementedError`
   with the "right-clip" message for variant-windows (mirrors the existing
   `test_svar2_variants_jitter_guard_raises`).

Backends: run the parity tests under both `GVL_BACKEND` values where the shim
dispatches (as the existing svar2 suite does), since `_assemble_variant_buffers`
is registered with rust + numba.

## 5. Testing & gates

- `maturin develop --release` is **not** needed (no Rust change) — but the
  branch may carry unbuilt Rust from prior work; run it once if the `.so` is
  stale, per CLAUDE.md.
- `pixi run -e dev pytest tests/dataset/test_svar2_dataset.py -q` (new tests),
  then the full svar2 suite `pixi run -e dev pytest tests -k svar2 -q`.
- Full tree before push: `pixi run -e dev pytest tests -q` (scoped runs skip
  `tests/unit/`), and `pixi run -e dev cargo-test` (no Rust change, but cheap
  insurance).
- Lint/format/typecheck: `ruff check python/ tests/ && ruff format python/ tests/
  && typecheck`.
- HPC gotcha: `--basetemp=$(pwd)/.pytest_tmp` so the write path's `os.link`
  hardlink does not fail cross-device (Errno 18).

## 6. Docs / skill / roadmap

Public API surface is unchanged, but svar2's **supported-mode matrix** changes
(variant-windows becomes supported), so:

- `skills/genvarloader/SKILL.md` — svar2 mode support / gotchas.
- `docs/source/*.md` where svar2 output-mode support is enumerated
  (`dataset.md`, `faq.md` as applicable).
- The svar2 read-bound perf design's "out of scope" list — note variant-windows
  is now implemented (feature complete; perf/fusion still deferred).
- No `api.md`/`__all__` change (no new public symbol).

## 7. Deferred / future

- **Fused svar2 windows kernel.** The two-call compose (decode then assemble,
  per group) mirrors the existing svar2 variants/tracks split-kernel structure
  and reuses validated code. A single fused
  `assemble_variant_windows_from_svar2` kernel could cut one FFI crossing +
  intermediate buffers, but that is a perf optimization to pursue only after
  profiling the live path (consistent with the read-bound perf design's
  measure-first discipline). Not in this work.
- `unphased_union` + variant-windows for svar2 (SVAR1 supports it; svar2 guards
  `unphased_union` wholesale).
- `ref="allele"` for svar2 (would require the decode to also return REF bytes,
  or reconstructing REF from `reference[start:start+ref_len]` where
  `ref_len = len(alt) - ilen`; a genoray-touching or extra-work change, deferred).

## 8. Files

- **Edit** `python/genvarloader/_dataset/_svar2_haps.py`:
  - new `_reconstruct_variant_windows(self, idx, regions)`;
  - `__call__`: replace the NotImplementedError branch with the jitter guard +
    dispatch;
  - imports: `_assemble_variant_buffers_rust`, `_FlatWindow` (from
    `_flat_variants`), and reuse existing `_ragged_arange_src` /
    `_ragged_arange_gather_2level` / `_inverse_row_perm`.
- **Edit** `tests/dataset/test_svar2_dataset.py`: ref_window-vs-SVAR1,
  alt-vs-numpy-oracle, multi-contig, empty-group/dummy, `ref="allele"` guard,
  jitter guard. Add a variant-windows bigwig/track combo only if trivial;
  otherwise seqs-only.
- **Edit** docs/skill per §6.
