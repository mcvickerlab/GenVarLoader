# Track Insertion Fill Options — Design

**Branch:** `feat/track-insertion-options`
**Date:** 2026-05-11
**Scope:** Read-time only — the kernel `shift_and_realign_track_sparse` in `python/genvarloader/_dataset/_tracks.py`, invoked exclusively via `HapsTracks.__call__` in `python/genvarloader/_dataset/_reconstruct.py`. `write_transformed_track` is currently `NotImplementedError` and out of scope.

## Problem

When a haplotype contains an insertion (`v_diff > 0`), the read-time kernel currently writes `v_len = v_diff + 1` consecutive output positions, all equal to `track[v_rel_pos]` (the value at the variant POS). This single hardcoded behavior — "5' repeat" — is not always desirable. Users want to:

- mask inserted positions (e.g. `NaN`),
- preserve the sum of the original signal across the insertion,
- inject neighborhood-aware noise (sample from flanking values),
- interpolate smoothly between the variant position and the next reference base.

## Public API

New module: `python/genvarloader/_dataset/_insertion_fill.py`. Strategy classes use `attrs` (consistent with the rest of the codebase) and are re-exported from `genvarloader/__init__.py`.

```python
@define
class InsertionFill: ...  # base

@define
class Repeat5p(InsertionFill): pass

@define
class Repeat5pNormalized(InsertionFill): pass

@define
class Constant(InsertionFill):
    value: float = float("nan")

@define
class FlankSample(InsertionFill):
    flank_width: int = 5

@define
class Interpolate(InsertionFill):
    order: int = 1  # 1=linear, 2=quadratic, 3=cubic. Capped at 3.
```

**Dataset method** (mirrors `with_tracks`):

```python
dataset.with_insertion_fill(Constant(float("nan")))                   # all active tracks
dataset.with_insertion_fill({"atac": Repeat5p(), "mask": Constant(0)}) # per-track
```

Returns a new lazy view via `evolve`. Default (never called) = `Repeat5p()`, which is the existing behavior — fully backwards compatible. In the dict form, tracks not in the mapping fall back to `Repeat5p()`.

Raises a clear error if invoked on a dataset without haplotypes (no insertions are possible) or without tracks.

## Strategy Semantics

For an insertion with `v_diff > 0`, the kernel writes `v_len = v_diff + 1` consecutive output positions. All strategies fill **all** v_diff+1 positions, including the shared anchor base. This matches the current kernel structure and is the simplest contract.

| Strategy | Value written across the v_diff+1 positions |
|---|---|
| `Repeat5p` | `track[v_rel_pos]` (current behavior) |
| `Repeat5pNormalized` | `track[v_rel_pos] / (v_diff + 1)` — total written sum equals `track[v_rel_pos]`. |
| `Constant(value=c)` | `c` |
| `FlankSample(flank_width=W)` | each output position independently samples (with replacement) from `track[v_rel_pos - W : v_rel_pos + W + 1]`, clamped to in-bounds. |
| `Interpolate(order=k)` | k=1: linear between `track[v_rel_pos]` and `track[v_rel_pos + 1]` across the v_diff+1 positions. k=2,3: Lagrange polynomial through `ceil((k+1)/2)` reference values on each side. Out-of-bounds neighbors clamp to the nearest in-bounds value. |

**Edge handling:** all neighbor reads clamp to the bounds of the per-query track slice — no extra I/O, no extended fetch windows.

**Determinism:** `FlankSample` uses a parallel-safe inline hash (xorshift64 over `(base_seed, query, hap, position)`) rather than `np.random` globals — this avoids thread-affinity issues inside `@njit(parallel=True)`. The base seed is drawn from the Dataset `rng` on each call. When `deterministic=True` is passed to `__call__`, the base seed is derived from `idx` so the same input always produces the same fill.

## Implementation Plan

### 1. `python/genvarloader/_dataset/_insertion_fill.py` (new)

Defines the strategy classes plus a lowering helper that packs a list of strategies into numba-friendly arrays:

```python
def lower(strategies: list[InsertionFill]) -> tuple[NDArray[np.int8], NDArray[np.float64]]:
    # returns (strategy_ids, params)
    # strategy_ids: (n_active_tracks,) int8 enum
    # params: (n_active_tracks, MAX_PARAMS) float64
```

Enum: `REPEAT_5P=0, REPEAT_5P_NORM=1, CONSTANT=2, FLANK_SAMPLE=3, INTERPOLATE=4`. Param slot conventions: `Constant→[value]`, `FlankSample→[flank_width]`, `Interpolate→[order]`. Unused slots zero.

### 2. Extend the numba kernel in `_tracks.py`

`shift_and_realign_tracks_sparse` and `shift_and_realign_track_sparse` gain three parameters:

```python
strategy_id: int,
params: NDArray[np.float64],   # shape (MAX_PARAMS,) for this track
base_seed: int,                # uint64, for FlankSample determinism
```

The single line `out[out_idx : out_idx + writable_length] = track[v_rel_pos]` at `_tracks.py:215` is replaced by an inlined `if/elif` ladder on `strategy_id`:

- `REPEAT_5P`: unchanged.
- `REPEAT_5P_NORM`: `track[v_rel_pos] / (v_diff + 1)`.
- `CONSTANT`: `params[0]`.
- `FLANK_SAMPLE`: for each of the `writable_length` positions, derive a uniform integer in `[0, 2W+1)` via inline xorshift64 hash of `(base_seed, query, hap, out_idx + i)`, index into the clamped pool, write.
- `INTERPOLATE`: linear is two-anchor lerp across v_diff+1 positions. Quadratic/cubic use Lagrange polynomial coefficients computed from `ceil((order+1)/2)` clamped flanking values per side, evaluated at each inserted position. Local fixed-size arrays only — numba-friendly.

The `track` slice and indices `v_rel_pos`, `v_diff`, `v_len`, `out_idx`, `writable_length` are all already in scope at the insertion site.

### 3. `_reconstruct.py`

`Tracks` gains a field:

```python
insertion_fill: dict[str, InsertionFill] = field(factory=dict)
```

Initialized in `from_path` to `{name: Repeat5p() for name in active_tracks}`. Add:

```python
def with_insertion_fill(
    self, fill: InsertionFill | Mapping[str, InsertionFill]
) -> Tracks:
    if isinstance(fill, InsertionFill):
        fills = {name: fill for name in self.active_tracks}
    else:
        fills = {name: fill.get(name, Repeat5p()) for name in self.active_tracks}
    return evolve(self, insertion_fill=fills)
```

`with_tracks` is updated to prune `insertion_fill` to the new active set (preserve existing assignments where applicable, default to `Repeat5p()` for newly-activated tracks).

`HapsTracks.__call__` (the existing per-track loop at `_reconstruct.py:1151`) looks up `self.tracks.insertion_fill[name]`, lowers it to `(strategy_id, params)`, generates `base_seed` once per call from the `rng` (or deterministically from `idx` when `deterministic=True`), and passes them to `shift_and_realign_tracks_sparse`.

### 4. `_impl.py` — Dataset surface

Add `with_insertion_fill(...)` on `Dataset` / `RaggedDataset`. It:
- validates that the reconstructor is `HapsTracks` (tracks + haplotypes present),
- delegates to the inner `Tracks.with_insertion_fill`,
- returns a new dataset via `evolve`.

Re-export `InsertionFill`, `Repeat5p`, `Repeat5pNormalized`, `Constant`, `FlankSample`, `Interpolate` from `python/genvarloader/__init__.py`.

### 5. Tests — `tests/dataset/test_insertion_fill.py`

For each strategy, construct a small dataset with one known insertion and assert:

- **`Repeat5p`**: byte-identical to a pre-change baseline (regression guard).
- **`Repeat5pNormalized`**: sum across the v_diff+1 inserted positions == `track[v_rel_pos]`.
- **`Constant(NaN)`**: those positions are NaN; surrounding positions untouched.
- **`FlankSample(W=5)`**: every output value is in the flank pool; `deterministic=True` gives reproducible output across repeated calls.
- **`Interpolate(1)`**: linear sequence between the two anchors.
- **`Interpolate(3)`**: passes through the four anchors when sampled at anchor positions.
- **Per-track dict**: two tracks with different strategies in one call; outputs differ correctly per-track.
- **Edge**: insertion within `flank_width` of the region boundary — values are clamped, no crash.

### 6. Docs

- Short section in the Dataset API docs page covering `with_insertion_fill` and the strategy classes.
- Docstring example on `with_insertion_fill`.

## Files Touched

- new: `python/genvarloader/_dataset/_insertion_fill.py`
- modified: `python/genvarloader/_dataset/_tracks.py` — kernel signature + branching
- modified: `python/genvarloader/_dataset/_reconstruct.py` — `Tracks` field, `HapsTracks` call site, `with_insertion_fill` plumbing, `with_tracks` pruning
- modified: `python/genvarloader/_dataset/_impl.py` — Dataset method
- modified: `python/genvarloader/__init__.py` — re-exports
- new: `tests/dataset/test_insertion_fill.py`

## Open Risk Notes

- **Numba RNG**: avoided by using an inline xorshift64 hash. No reliance on `np.random` global or thread-local state inside `prange`.
- **Interpolation order**: capped at 3 (cubic). Spline is out of scope; if needed later, add a separate `Spline()` class.
- **Performance**: per-insertion branch on `strategy_id` is cheap (one branch per indel, not per base). No measurable regression expected for the default `Repeat5p` path.
