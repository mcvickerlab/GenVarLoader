# Flat-buffer `__getitem__` pipeline — design

**Date:** 2026-05-31
**Branch:** `feat/bench-codspeed-profiling`
**Status:** approved design; pending writing-plans

## Problem

On the regenerated (correct) chr22 GEUVADIS benchmark slice, the per-batch
`Dataset.__getitem__` hot path is dominated by awkward-array work — `_carry` +
`_kernels.__call__` + `concat` account for **39–53%** of hot-path self-time
across the three profiled modes (tracks 43% / haps 53% / variants 39%), the #1
bucket in every mode. gvl's own numba kernels (`intervals_to_tracks` ~10%,
`_reconstruct_haplotypes` ~6%) are a clear second tier.

This awkward churn is *not* the reconstruction itself — every reconstructor
already produces flat `(buffer, offsets)` from buffer-writer numba kernels. Nor
is it the two transforms already moved to flat buffers this branch
(`reverse_complement_masked` via seqpro 0.12.1, `to_padded` via seqpro 0.13.0).
The remaining cost is the **glue**: wrapping flat buffers into awkward-backed
seqpro `Ragged` at the reconstructor boundary, then dispatching the
post-reconstruction transforms and materialization through awkward:

- **`to_numpy()` on fixed-length output** (every `with_len` int mode) —
  awkward `ak.to_numpy` densify. For fixed length, every row is exactly
  `output_length`, so this is just `data.reshape(outer + (length,))`.
- **RC-float path** (`_query.py:340`, `ak.where(to_rc, rag[..., ::-1], rag)`) —
  the `var_idxs` / `ref_coords` int32 reverse in the haps (`RaggedAnnotatedHaps`)
  mode. The S1 `haps` array already goes flat via `reverse_complement_masked`.
- **`RaggedVariants` assembly + `rc_`** — the variants mode.
- **`Ragged` wrapping / `.lengths` / `from_offsets`** — builds an awkward
  `ListOffsetArray`; any internal op on it then dispatches awkward kernels.

**Caveat carried in from prior profiling:** the slice is small and noisy (hot
path is only 28–61% of wall; the rest is one-time JIT/import). The earlier
REGRESSIONS before/after deltas were confounded by a dataset fix landing
between baseline and re-profile, so they do not isolate the transform effect.
This design front-loads a clean baseline (step 0) to fix that.

## Goal / success criterion

Awkward-backed seqpro `Ragged` is constructed **only at the `getitem` return
boundary**, and only when the output is ragged. Everything internal — recon →
reverse-complement → densify — operates on flat numpy `(data, offsets, shape)`
plus numba kernels. Outputs are **byte-identical** to today. The awkward
`_carry` / `_kernels` / `concat` frames disappear from the per-batch hot path.

## Architecture

### 1. Internal flat container (`_ragged.py`)

A lightweight gvl-only frozen dataclass (working name `_Flat`). It deliberately
does **not** wrap awkward — it is pure numpy + offsets, and converts to seqpro
`Ragged` only on request:

- fields: `data: NDArray`, `offsets: NDArray[np.int64]`,
  `shape: tuple[int | None, ...]` (outer fixed dims; `None` marks the ragged axis)
- `.to_ragged() -> Ragged` — `Ragged.from_offsets(data, shape, offsets)`; the
  **only** place awkward is touched
- `.to_fixed(length) -> NDArray` — `data.reshape(outer + (length,))`; replaces
  the awkward `to_numpy` densify for fixed-length (int) output
- `.to_padded(pad_value) -> NDArray` — variable-length densify; delegates to the
  already-flat seqpro `to_padded` (constructing one transient `Ragged` is
  acceptable here, or a gvl flat path if measurement shows the wrap matters)
- `.reverse_masked(mask, comp=None) -> _Flat` — flat masked reverse / RC: DNA
  (S1, `comp` given) via seqpro's already-flat masked-RC; int32
  (`var_idxs`/`ref_coords`, reverse only) via a small gvl numba kernel
- `.reshape`, `.squeeze` — pure offset/shape ops, no data movement

Composite flat analogs hold `_Flat` instances:
- `_FlatAnnotatedHaps` (`haps`, `var_idxs`, `ref_coords`)
- `_FlatVariants` (variant fields + the `rc_` logic, reworked on flat buffers)
- intervals are already flat-friendly (`intervals_to_tracks` reads
  `.starts.data` / `.offsets` directly)

`_Flat` is *not* a reimplementation of `Ragged` for public use — it is an
internal transport that never runs awkward kernels. `Ragged` remains the public
return container and the on-disk storage representation.

### 2. Reconstructor boundary

`Haps`, `Ref`, `Tracks`, `HapsTracks`, `RefTracks` return the flat container(s)
instead of `Ragged` / `RaggedAnnotatedHaps` / `RaggedTracks`. They already
compute `(out, out_offsets, out_shape)` and call `Ragged.from_offsets` as the
last step — that wrap is removed; they return `_Flat` (or a composite) instead.

The `kind` dispatch — the `RaggedSeqs` / `RaggedAnnotatedHaps` / `RaggedVariants`
Phantom selectors that pick reconstruction behavior — is preserved as the
selector; only the produced container changes.

### 3. Transforms (`_query.py`)

`reverse_complement_ragged` and `pad()` / `to_numpy()` operate on `_Flat`:
- RC: DNA via flat masked-RC; int (`var_idxs`/`ref_coords`) via the flat int
  reverse kernel; no `ak.where`, no `rag[..., ::-1]`.
- densify: int output → `to_fixed`; "variable" → `to_padded`.

### 4. Return boundary (`getitem`)

- `output_length` is `int` → dense numpy via `to_fixed`; no `Ragged`.
- `output_length == "variable"` → dense padded numpy via `to_padded`; no `Ragged`.
- `output_length == "ragged"` → `.to_ragged()` into the documented public return
  types (`Ragged`, `RaggedAnnotatedHaps`, `RaggedVariants`, `RaggedIntervals`).

**Public return types are unchanged.** Callers requesting ragged output still
get seqpro `Ragged`-backed objects; callers requesting fixed/variable still get
dense numpy.

## Kernel placement decision

**Hybrid: gvl-local flat kernels now, upstream to seqpro later if broadly
useful.** The remaining flat ops are small: fixed-length densify is a pure
reshape (no kernel); the int32 masked-reverse is a tiny numba kernel; DNA RC and
`to_padded` reuse the already-released seqpro 0.13 entry points. This avoids
blocking the rewrite on another seqpro release + genoray cap-lift (the recurring
seqpro↔genoray coupling tax). Any kernel that proves generally reusable can be
upstreamed to seqpro in a later pass.

## Phasing

Each stage ends with: full gvl dataset test suite green, byte-identical output
vs. the awkward reference, and a re-profile delta recorded in
`docs/superpowers/REGRESSIONS.md`.

0. **Clean baseline.** Bump `profile.py` `N_BATCHES` 200 → 2000 (the run is
   near-instant at 200, so more iterations are needed to accumulate stack
   samples; BATCH stays 32, which already satisfies ≥16 instances/batch).
   Re-profile all three modes to establish a confound-free A/B reference.
1. **`_Flat` container** + round-trip and byte-identical unit tests.
2. **Tracks** path flat (simplest — `_call_float32` is already flat; densify via
   reshape). Re-profile.
3. **Haps** path flat (`RaggedSeqs` + `RaggedAnnotatedHaps`; flat int
   masked-reverse for `var_idxs`/`ref_coords`). Re-profile.
4. **HapsTracks / RefTracks** compound reconstructors. Re-profile.
5. **Variants** path flat (`RaggedVariants` assembly + `rc_` — the gnarliest;
   sequenced last). Re-profile.
6. **Spliced** path (`_getitem_spliced` + `_regroup` offset rewrap) parity on
   flat buffers.
7. **Guard.** Assert no awkward ops remain in the hot path (a profile check
   and/or a static guard); confirm `Ragged` is constructed only at the boundary.

## Testing & tooling

- Existing 248+ dataset unit tests must stay green (byte-identical outputs are
  the correctness contract).
- New unit tests: `_Flat` round-trip (`from_offsets`/`to_ragged`/`to_fixed`/
  `to_padded` equivalence), and each flat transform vs. its awkward reference.
- Per-stage profiling on the 2000-batch bench: py-spy speedscope + memray, with
  hot-path self-time and allocation deltas logged in `REGRESSIONS.md`. This
  same-dataset A/B finally isolates the transform effect (resolving the
  confound).
- **Profiling handoff:** py-spy requires root on macOS and cannot be invoked by
  the agent. For each re-profile step, emit a runnable bash script (under
  `tests/benchmarks/profiling/`) containing the exact `py-spy record …` commands
  with absolute pixi-env python paths; David runs it with sudo. `memray run`
  does not need root and may be run directly.

## Non-goals / risks

- **Non-goal:** removing `Ragged` from the public API or the on-disk storage
  format. It stays as the return container and storage representation.
- **Risk:** the spliced `_regroup` step rewraps offsets via awkward — it needs a
  flat equivalent to keep the spliced path off awkward.
- **Risk:** `RaggedVariants` (~526 lines, variant-specific fields + `rc_`) is the
  hardest surface; sequenced last so earlier wins land first.
- **Risk:** recon return types are heavily generic (`_H`, `_T`, Phantom `kind`
  dispatch); changing the produced container touches type plumbing. Accepted as
  the cost of the full rewrite.
- `to_nested_tensor_batch` (torch) is outside the getitem hot path — left as-is.
- Throughput vs. allocation: wins demonstrated so far are allocation/structure;
  the ~18–20× tracks throughput regression vs 0.6.1 is not yet attributed to a
  frame and remains a separate open item (cluster-scale bisect).
