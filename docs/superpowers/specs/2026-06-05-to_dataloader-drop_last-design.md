# `to_dataloader()` — honor `drop_last` across all modes

**Date:** 2026-06-05
**Status:** Approved, ready for implementation plan

## Problem

A bug report says `to_dataloader()` does not respect `drop_last=False`. Empirical
investigation (`phased_vcf_gvl` + `reference`, with `batch_size` chosen so it does
not divide `N = len(ds)`) confirmed the report **and** surfaced a second, related
defect. Counting batches across both code paths and both `drop_last` values:

| mode               | `drop_last=False`              | `drop_last=True`                                   |
|--------------------|--------------------------------|----------------------------------------------------|
| `None` (default)   | ✅ keeps partial batch          | ❌ **crashes** (`ValueError: batch_size=None … mutually exclusive with drop_last`) |
| `buffered`         | ❌ **drops partial batch** (bug) | ✅ drops                                            |
| `double_buffered`  | ❌ **drops partial batch** (bug) | ✅ drops                                            |

Both defects live in `python/genvarloader/_torch.py` and its buffered-loader
helpers.

### Bug 1 — buffered modes ignore `drop_last=False` (the reported bug)

`_resolve_buffered_inputs` (`_torch.py`) unconditionally truncates the epoch to a
whole multiple of `batch_size`:

```python
n_keep = (len(flat) // batch_size) * batch_size
flat = flat[:n_keep]
```

This drops the final partial batch regardless of `drop_last`. The truncation
cannot simply be removed, because `ChunkPlanner` (`_chunked.py`) hard-requires
divisibility:

```python
if n % batch_size != 0:
    raise ValueError(...)              # line 38-42
...
batch_totals = per_inst.reshape(-1, batch_size).sum(-1)   # line 50
```

### Bug 2 — default mode crashes on `drop_last=True`

In `get_dataloader`, the `mode=None` branch builds a `BatchSampler` (via
`get_sampler`) that already applies `drop_last`, then **also** forwards
`drop_last=drop_last` to `td.DataLoader` alongside `batch_size=None`. PyTorch
rejects that combination:

> `batch_size=None option disables auto-batching and is mutually exclusive with drop_last`

So `drop_last=True` is broken in the default path; `drop_last=False` works only
because it is a no-op. This path is shared by `_dataset/_impl.py` **and**
`_dataset/_reference.py` `to_dataloader`, so both are affected.

## Approach

**Bug 2 (trivial):** stop forwarding `drop_last` to `td.DataLoader` in the
`mode=None` branch. The `BatchSampler` is the sole authority on dropping; the
DataLoader's own `drop_last` defaults to `False`, a no-op under `batch_size=None`.
Fixes both `_impl` and `_reference` default paths at once.

**Bug 1 (teach `ChunkPlanner` about a trailing partial batch):** chosen over
"reject `drop_last=False` in buffered modes" (cheapest but leaves the feature
unimplemented for buffered users) and "pad-and-trim" (adds bookkeeping and risks
leaking duplicated data). `ChunkPlanner` is the single chokepoint — both buffered
loaders delegate to it plus `slice_chunk`, and `slice_chunk` already yields a
partial tail via `range(0, n, batch_size)`.

Changes:
1. `_torch.py::_resolve_buffered_inputs` — gate the `n_keep` truncation on the
   `drop_last` argument (only truncate when `drop_last=True`).
2. `_chunked.py::ChunkPlanner` —
   - drop the `n % batch_size != 0` guard;
   - build `batch_totals` as the full-batch sums (`reshape` over the divisible
     prefix) **plus** one trailing entry for the remainder instances' summed
     bytes, when a remainder exists;
   - clamp each chunk's slice end to `n` (`end = min(j * batch_size, n)`) so the
     final partial batch slices correctly.
   The per-batch `too_big` check and `peak_chunk_bytes` both read `batch_totals`,
   so they stay correct with the partial entry included.
3. `_buffered_loader.py::__len__` — floor (`len(flat_r) // batch_size`) → ceil
   (`ceil(len(flat_r) / batch_size)`). The double-buffered loader's `__len__`
   already sums per-chunk batch counts from the planner, so it inherits the
   correct count once `ChunkPlanner` counts the partial batch.

## DDP interaction

Option 1 is safe under `DistributedDataParallel` and improves on the status quo:

- DDP correctness requires every rank to produce the **same number of batches**
  (otherwise the gradient all-reduce collectives mismatch and training hangs).
  That evenness is the **sampler's** job, not the loader's. `DistributedSampler`
  equalizes per-rank sample counts (pads with duplicates when `drop_last=False`,
  drops to the largest common multiple when `drop_last=True`), so every rank ends
  up with an identically-sized final partial batch — symmetric, no hang.
- `ChunkPlanner` operates per-process on whatever `(r,s)` sequence it is handed
  and has no cross-rank coupling. The change makes it faithfully reflect the
  sampler's output instead of silently flooring. Today's buffered path silently
  drops the tail on every rank (symmetric, so no hang, but data is lost); the fix
  removes that silent loss without adding any new asymmetry.
- The only footgun — DDP with `drop_last=False` but *without* a count-equalizing
  sampler — is pre-existing PyTorch behavior, not introduced here. Standard
  guidance applies: use `DistributedSampler`, choose `drop_last` consistently
  across ranks.

A note documenting this will accompany the change, and the test matrix includes a
custom-sampler case that locks in the DDP-relevant property (partial batch
preserved regardless of where the indices came from).

## Test strategy (TDD — written failing first)

Parametrized tests in `tests/unit/test_torch.py`, run in an env that has torch
(the `default` pixi env — `dev` does not bundle torch). `batch_size` is chosen so
it does not divide `N = len(ds)`.

- **Matrix:** `mode ∈ {None, "buffered", "double_buffered"}` ×
  `drop_last ∈ {False, True}`.
  - `drop_last=False` → `n_batches == ceil(N / batch_size)`, and the final batch's
    instance count equals `N % batch_size` (not `batch_size`).
  - `drop_last=True` → `n_batches == floor(N / batch_size)`, all batches full.
- **DDP-shaped case:** a custom sampler yielding an index count that is not a
  multiple of `batch_size`; assert the partial batch survives under buffered modes
  with `drop_last=False`.
- **Bug 2 regression:** `mode=None, drop_last=True` no longer raises.

Before the fix these produce exactly: 2 failures (buffered / double_buffered
`drop_last=False`) + 1 error (`mode=None, drop_last=True`).

## Touch-points summary

| File | Change |
|------|--------|
| `python/genvarloader/_torch.py` | gate `n_keep` on `drop_last` in `_resolve_buffered_inputs`; drop `drop_last=` forward to `td.DataLoader` in `get_dataloader` mode=None |
| `python/genvarloader/_chunked.py` | `ChunkPlanner`: allow non-divisible length, `batch_totals` += remainder entry, clamp chunk slice end to `n` |
| `python/genvarloader/_buffered_loader.py` | `__len__` floor → ceil |
| `tests/unit/test_torch.py` | parametrized drop_last × mode matrix + DDP-shaped sampler case + Bug 2 regression |

## Out of scope

- The buffered path's assumption that a user-provided `sampler` yields *batches*
  (lists), inconsistent with the default path's "pass a plain sampler, gvl wraps
  it" contract. Pre-existing; not part of this bug.
- Any public API or signature change — `drop_last` already exists on every
  `to_dataloader`; this only corrects its behavior.
