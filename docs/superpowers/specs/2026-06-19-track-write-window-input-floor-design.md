# Fix #233 follow-up — track-write window must cover the input region

**Issue:** [#233](https://github.com/mcvickerlab/GenVarLoader/issues/233) (variant-dependent residual; supersedes the off-by-one fix in #234)
**Date:** 2026-06-19
**Type:** Bug fix (behavior-breaking for existing variant+track datasets; on-disk format unchanged)

## Problem

When a dataset has **both variants and tracks**, annotation (and sample) tracks can read
back with their tail zeroed — e.g. the canonical region `chr22:16902120-16902391` (271 bp)
returns `0/271` covered under `with_settings(realign_tracks=False)`, even though the source
bigWig has 98 nonzero positions in `[16902293, 16902391)`.

This is **variant-dependent, not scale- or offset-related** (the earlier chunk-boundary
hypothesis was wrong). Root cause proven by inspection of the stored windows:

1. The variant writers set each region's stored `chromEnd` to the **end of the furthest
   retained variant** (`v_ends[max idx]`), falling back to the input end **only when a
   region has zero variants**:
   - `_region_end` — `python/genvarloader/_dataset/_write.py:649-658`
   - `_region_ends_from_list` — `:661-671`
   - `_write_from_svar` inline `max_ends` — `:1010-1011`, returned at `:1033`
   The `fallback_end` passed in is `q_end` = the input window end, but it is used *only*
   for variant-free regions; regions **with** variants ignore it.
2. `_write_regions` saves this variant-derived bed as `regions.npy` (`:314`).
3. The track writers read `regions.npy` for their windows — annot at `:332-333` (and the
   `update()` append path at `:428`), sample tracks at `:319` / `:429`.
4. At open, `_full_regions` (the read-side window) is rebuilt from `input_regions.arrow`
   — the true user windows — **never** from `regions.npy` (`_open.py:111-125`).

So tracks are **written against variant-truncated windows** but **read against the true
input windows**. Any region whose rightmost retained variant ends before the input end
loses all stored signal past that point. With sparse variants this is the common case.

### Why the read side is already correct

For `realign_tracks=False`, readback output length = input-window length
(`regions[:,1:3]` from `_full_regions`) and `intervals_to_tracks`
(`_dataset/_intervals.py:54-78`) clips/`break`s any interval outside that window. The
proof region confirms it: read spanned the full 271 bp input window; only the stored
intervals were missing past `…272`. **Nothing changes on the read path.** `realign=True`
masks the bug because realignment only maps where variants are.

### Scope of corruption

Datasets with **both** `genotypes/` and (`intervals/` or `annot_intervals/`), written
before this fix, where at least one region's rightmost retained variant precedes the input
end. Variant-only and track-only datasets are unaffected (track-only never passes through
the variant writers; variant-only has no `regions.npy` track consumer).

## Fix — floor the stored `chromEnd` at the input window

The stored window must be the **union** of the input window (already `input ± max_jitter`
in `gvl_bed`) and the variant extension — never a truncation below the input. `chromStart`
is never modified by the variant writers, so only `chromEnd` needs a floor. Three sites,
all flooring at the existing `fallback_end` / input `chromEnd`:

```python
# _region_end (:658)
return max(fallback_end, v_ends[int(rag.data.max())])

# _region_ends_from_list (:671)
return max(fallback_end, v_ends[max_idx])

# _write_from_svar (:1033) — clamp the assembled array before writing
return bed.with_columns(
    chromEnd=pl.max_horizontal(pl.Series(max_ends), pl.col("chromEnd"))
), svar_link
```

### Why this is the whole fix

- **`realign=False` annot + sample tracks:** fixed — readback already clips to the input
  window, which now always has stored data.
- **`realign=True`:** preserved — the extension still applies wherever a variant reaches
  past the input end; the floor only ever *raises* `chromEnd` to the input window.
- **`Dataset.regions`:** unchanged (= input, from `input_regions.arrow`). Added as a
  tested invariant below.
- **Genotype reconstruction:** unchanged — it reads `_full_regions`, not `regions.npy`;
  widening the stored window adds no variants (none exist past the furthest retained one).
- **On-disk format:** unchanged — no `format_version` bump. Existing variant+track
  datasets have truncated track tails and must be **rewritten** (`gvl.write` / re-`update`).

## Signal silently-corrupt existing datasets

Add a soft, precise open-time warning (does **not** raise). `validate_dataset`
(`_dataset/_validate.py`) is called from open and `_check_integrity` already memmaps
`regions.npy`, so the check slots in there.

- **Gate:** only when both `genotypes/` and (`intervals/` or `annot_intervals/`) exist.
- **Detect:** load `input_regions.arrow`, sort it the same way the writer does
  (`sp.bed.sort`), and compare stored `regions.npy[:, 2]` against
  `sorted_input_chromEnd + metadata.max_jitter` (same sorted order by construction). If any
  stored `chromEnd` is **less** than the input floor → emit a `logger.warning`.
- **Message:** state that track tails may be truncated for regions with variants, name the
  writing `metadata.version`, and instruct to rewrite the dataset with `gvl.write` /
  `gvl.update`.

This fires on exactly the corrupt set (tracks+haps where truncation actually occurred),
regardless of the writer version, and never on post-fix datasets (whose stored `chromEnd`
is floored ≥ the input window). The package `version` field is auto-set from the installed
gvl version (`_write.py:178`), so post-fix datasets remain identifiable; no fragile
version-number constant is introduced.

## Tests

`tests/integration/tracks/test_annot_tracks.py` (and the sample-track e2e where it fits):

1. **Regression (the corruption):** build a variant source where a region's rightmost
   retained variant ends well before the input region end, plus an annot bigWig with signal
   in the tail. Assert `realign=False` readback covers the **full input window** and the
   per-region mean matches the source over the span. Cover **VCF, PGEN, and SVAR** (all
   three writers).
2. **Sample-track parity:** the same truncated-tail region read as a per-sample track with
   `realign=False` is also full-width (the latent sample-track case).
3. **Invariant:** `Dataset.regions == input regions` for a variant-containing dataset.
4. **Warning:** opening a dataset with a hand-truncated `regions.npy` (or a fixture built
   by the pre-fix writer) emits the truncation warning; a clean post-fix dataset does not.

## Out of scope / non-changes

- Read path (`_reconstruct.py` / `_tracks.py` / `_intervals.py`): untouched.
- `realign=True` length-guarantee semantics: unchanged.
- `format_version` / on-disk layout: unchanged (no migration tooling; rewrite is the
  remedy).
- `skills/genvarloader/SKILL.md`: no public-API signature/default change — re-check the
  "Common gotchas" note on tracks + variants, but no surface change expected.

## Verification

- `pixi run -e dev maturin develop` (no Rust change here, but keep the env consistent),
  then the new regression tests pass.
- Full tree before push: `pixi run -e dev pytest tests -q`, `ruff check python/ tests/`,
  `ruff format python/ tests/`, `typecheck`.
