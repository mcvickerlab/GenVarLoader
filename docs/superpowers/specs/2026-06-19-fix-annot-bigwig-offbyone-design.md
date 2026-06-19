# Fix #233 — annot bigWig readback collapses wide intervals (off-by-one in Rust reader)

**Issue:** [#233](https://github.com/mcvickerlab/GenVarLoader/issues/233)
**Date:** 2026-06-19
**Type:** Bug fix

## Problem

Reading an `annot_tracks` bigWig back with `with_settings(realign_tracks=False)`
collapses any interval **wider than (or starting before) the query region** to its
value at a single base, zero elsewhere. The per-region mean degrades to
`interval_value / region_length`.

- Coarse/binned tracks (e.g. 1 kb-binned replication timing) are destroyed
  (~100–350× too small, correlation ~0.03 vs source).
- Fine/~1 bp tracks are mildly diluted (correlation ~0.94).

The bigWig reader returns the right **values**; the corruption is in the
storage→readback coordinate handling.

## Root cause (confirmed by reproduction)

`src/bigwig.rs:124` stores each interval's start as `start - 1` instead of `start`:

```rust
coords_ptr.add((offset + i) * 2).write(MaybeUninit::new(start - 1)); // <-- off-by-one
```

bigtools' `get_interval` already **clips** intervals to the query range and returns
correct 0-based half-open coordinates. The `- 1` is therefore a pure off-by-one that
shifts every interval one base toward 5'. It explains both symptoms:

- **Coarse tracks (destroyed):** an interval clipped at the region start is stored with
  start = `query_start - 1`. In `intervals_to_tracks` the relative start is
  `itv_start - query_start = -1`, and the numba slice `_out[-1:end]` wraps `-1` to index
  `length - 1`, producing an **empty slice** — the wide interval writes nothing. Only a
  small trailing interval survives → `value / span_length`.
- **Fine tracks (mild):** every interval shifted one base → the ~0.94 edge-dilution
  correlation.
- It is also a latent **`u32` underflow** when `start == 0` (a contig-start interval),
  since `start` is unsigned.

### Reproduction evidence

A synthetic 1 kb-binned chr22 bigWig, annot-only dataset over sub-bin regions, read back
with `realign_tracks=False`:

| region | source mean | readback (before) | readback (after fix) |
|---|---|---|---|
| chr22:16774654-16775004 | 0.4011 | 0.0071 (5/350 bp filled) | **0.4011** (350/350) |
| chr22:16829376-16829599 | 0.3000 | 0.0013 (1/223 bp filled) | **0.3000** (223/223) |

The stored intervals for region 0 were starts `[16774653, 16774999]` — the first start is
exactly `query_start - 1`.

## Fix

Single-line change in `src/bigwig.rs`:

```rust
coords_ptr.add((offset + i) * 2).write(MaybeUninit::new(start)); // was: start - 1
```

This corrects coordinates for **all** bigWig reads — both per-sample `tracks` and
`annot_tracks` flow through this one function and the shared `intervals_to_tracks` kernel.

**Scope decision:** minimal fix only. After removing the off-by-one, bigtools clipping
guarantees relative start ≥ 0, so the kernel's negative-index branch can no longer
trigger; no defensive `max(0, start)` clamp is added (keeps the diff parity-preserving).

## Blast radius & snapshot regeneration

The coordinate correction shifts track output by one base, breaking 4 byte-identical
characterization snapshots in `tests/dataset/test_flat_getitem_snapshot.py`
(`tracks_ragged`, `tracks_fixed`, `haps_tracks_fixed`, `haps_tracks_ragged`). The diffs
show the **new** output filling the correct extra base at interval edges; seq-only
snapshots (`haps_*`, `ref`, `annot`, `variants`) are unchanged.

These snapshots are golden captures of gvl's own output and baked in the bug. They will be
**regenerated**, after hand-verifying each diff is exactly a one-base edge correction and
nothing else.

## Tests

Add a regression test to `tests/integration/tracks/test_annot_tracks.py` (the
non-experimental bigWig annot path):

1. Build a synthetic wide-interval (1 kb-binned) bigWig with pyBigWig.
2. Write an annot-only dataset over regions that each fall inside a bin.
3. Read back with `with_seqs(None).with_output_format("flat").with_settings(realign_tracks=False).with_tracks([...])`.
4. Assert per-region readback mean ≈ source bigWig mean over the span (catches the coarse
   collapse).
5. Include a region whose start sits on a bin edge to lock the off-by-one direction.

## Out of scope / non-changes

- No public API signature, output mode, or default changes.
- `skills/genvarloader/SKILL.md` needs **no update** — it documents dataset-level track
  reading, not the internal per-base interval coordinate this corrects.
- No kernel (`_dataset/_intervals.py`) change.

## Verification

- `pixi run -e dev maturin develop --release` then the new regression test passes.
- Regenerated snapshots pass `tests/dataset/test_flat_getitem_snapshot.py`.
- Full tree before push: `pixi run -e dev pytest tests -q` plus `cargo test`, `ruff
  check`/`format`, `typecheck`.
