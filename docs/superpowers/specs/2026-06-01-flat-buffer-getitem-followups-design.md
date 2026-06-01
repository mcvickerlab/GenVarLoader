# Flat-buffer `__getitem__` follow-ups — design

**Date:** 2026-06-01
**Branch:** `feat/bench-codspeed-profiling` (follow-ups to PR #205)
**Status:** design; pending writing-plans

Two follow-ups identified during review of the flat-buffer `__getitem__` refactor
(PR #205). Both concern track paths that the main refactor deliberately left out of
scope. Independent; FU-1 is trivial, FU-2 is the substantive work.

---

## FU-1: Awkward-guard cases for `tracks_ragged` / `haps_tracks_ragged`

### Problem

`tests/dataset/test_no_awkward_in_hotpath.py` asserts the fixed/ragged hot paths
dispatch zero awkward kernels. It covers `tracks_fixed`, `haps_fixed`, `ref_fixed`,
`haps_tracks_fixed`, and `haps_ragged` — but **not** the ragged-output track paths
`tracks_ragged` and `haps_tracks_ragged`. The byte-identity snapshot gate covers
those for correctness, but there is no executable assertion that they stay
awkward-free.

### Key facts (from code investigation)

- `Tracks.__call__` dispatches on `self.kind`: the **default** `kind` (`RaggedTracks`)
  routes to `_call_float32`, which returns `_Flat`. Only an explicit
  `with_tracks(..., kind="intervals")` routes to `_call_intervals` (awkward — see FU-2).
- For default-kind tracks in ragged-output mode, the `_Flat` is converted via
  `_Flat.to_ragged()` → `Ragged.from_offsets` at the boundary (`_query.py:107–110`).
  That is a seqpro call, **not** an awkward kernel. So these guards should pass as-is.
- The existing `guard_dataset` fixture (session-scoped, track `"5ss"`, VCF genotypes)
  already supports both new cases.

### Approach

Add two cases to the guard test, both using the **default track kind** (no
`kind="intervals"`), ragged output (no `with_len`):

```python
def test_tracks_ragged_no_awkward(monkeypatch, guard_dataset):
    calls = _install_ak_counters(monkeypatch)
    ds = guard_dataset.with_seqs(None).with_tracks("5ss")   # ragged output
    _ = ds[regions, samples]
    assert calls["n"] == 0

def test_haps_tracks_ragged_no_awkward(monkeypatch, guard_dataset):
    calls = _install_ak_counters(monkeypatch)
    ds = guard_dataset.with_seqs("haplotypes").with_tracks("5ss")  # ragged output
    _ = ds[regions, samples]
    assert calls["n"] == 0
```

Add a module-level comment documenting the carve-out (mirroring the existing
`RaggedVariants` note): **the `kind="intervals"` track path is intentionally
excluded from this guard — it uses awkward natively until FU-2 lands.** Optionally
add an `xfail` test pinning the current `kind="intervals"` awkward behavior so FU-2
flips it to `xpass`.

### Risks / acceptance

- Risk: near-zero. If a guard unexpectedly fails, it reveals a missed awkward
  dispatch in the float32 ragged path — investigate rather than weaken the assert.
- Done when: both new tests pass with `calls["n"] == 0`; the intervals carve-out is
  documented.

---

## FU-2: Flat `_FlatIntervals` for `Tracks._call_intervals`

### Problem

`Tracks._call_intervals` (`_tracks.py:678–704`) is the only remaining awkward kernel
dispatch in any getitem track path. It assembles `RaggedIntervals` output for the
`with_tracks(kind="intervals")` mode entirely via awkward: per active track it indexes
the on-disk interval store, calls `.to_packed()` (→ `ak.to_packed` ×3 components), then
`ak.concatenate(..., axis=1)` ×3 to stack tracks. No numba kernel is involved.

This path was out of scope for PR #205 (the hot path targeted there is the float32
tracks / haps / ref paths). But it carries the same awkward churn the refactor removed
elsewhere, for users who request raw intervals.

### Key facts (from code investigation)

- `RaggedIntervals` (`_ragged.py:32`) is a **plain dataclass** (not an `ak.Array`
  subclass) of three parallel `seqpro.Ragged`: `starts: Ragged[int32]`,
  `ends: Ragged[int32]`, `values: Ragged[float32]`. Logical shape
  `(batch, n_tracks, ~n_intervals)` — **single-level ragged** (one `None` axis, two
  fixed outer dims). This is exactly the structure `_Flat`/`_FlatAnnotatedHaps`
  already handle — *not* the variable-length-of-variable-length case that kept
  `RaggedVariants` awkward-native.
- On disk (`_open_intervals`, `_tracks.py:525–543`) intervals are stored as a flat
  structured array + a **shared** `offsets` array. The three logical fields share one
  offsets array.
- `_call_intervals` has **no numba kernel** — the gather is `intervals[r_idx, s_idx]`
  (awkward fancy-index) + `to_packed` + `concatenate`. The flat rewrite can index the
  underlying flat `.data`/`.offsets` directly in numpy (or a small gather kernel),
  with no awkward at all.
- Boundary handling today: `RaggedIntervals` is **bypassed** in
  `reverse_complement_ragged` (returned unchanged), in both densify branches
  (returned as-is even for fixed/variable `output_length` — i.e. intervals output is
  *always* the ragged container regardless of `with_len`), and in the final
  `to_ragged` wrap. So a `_FlatIntervals` only needs `to_ragged()` (→ `RaggedIntervals`)
  + `reshape`/`squeeze`; it needs **no** `reverse_masked`, `to_fixed`, or `to_padded`.

### Approach

1. **Add `_FlatIntervals` to `python/genvarloader/_flat.py`** — a composite mirroring
   `_FlatAnnotatedHaps`, but over three `_Flat`s sharing one offsets array and with the
   reduced method set actually used:
   ```python
   @dataclass(slots=True)
   class _FlatIntervals:
       starts: _Flat   # int32
       ends: _Flat     # int32
       values: _Flat   # float32

       @property
       def shape(self): return self.starts.shape
       def reshape(self, shape): ...   # delegate to each _Flat
       def squeeze(self, axis=None): ...
       def to_ragged(self):            # boundary import RaggedIntervals
           return RaggedIntervals(self.starts.to_ragged(),
                                  self.ends.to_ragged(),
                                  self.values.to_ragged())
   ```
   (Three component dtypes differ, so it cannot be a single `_Flat`. No
   `reverse_masked`/`to_fixed`/`to_padded` — intervals bypass RC and densify.)

2. **Rewrite `_call_intervals` to build flat buffers** — index directly into each
   track's stored `.starts.data` / `.ends.data` / `.values.data` using the shared
   on-disk offsets, concatenating per-track slices into flat numpy buffers and building
   the combined `(batch, n_tracks, None)` offsets in C order (batch-major, track-minor —
   matching the current `ak.concatenate(axis=1)` ordering). A small numba gather kernel
   is natural but optional; pure-numpy slice-concatenation also works. Return
   `_FlatIntervals.from_offsets(...)` instead of `RaggedIntervals(...)`.

3. **Boundary wiring in `_query.py`** — add `_FlatIntervals` to the final `to_ragged`
   wrap (so a still-flat intervals element becomes `RaggedIntervals` at return) and to
   the densify-bypass guards (a `_FlatIntervals` must be passed through, not densified —
   same treatment `RaggedIntervals` gets today). Confirm `reverse_complement_ragged`
   either bypasses `_FlatIntervals` or is never called with it.

4. **Tests:**
   - Extend the byte-identity snapshot gate with an `intervals` case
     (`with_tracks("5ss", kind="intervals")`, ragged) — regenerate that one `.npz`.
     The `_flatten_output` helper must learn the `RaggedIntervals` shape (3 ragged
     fields × data+offsets) — this is the branch a prior review already noted was
     missing.
   - Flip the FU-1 `kind="intervals"` carve-out: the awkward-guard `tracks_ragged`
     with `kind="intervals"` should now assert **0** awkward dispatches.
   - Re-profile `--mode variants`/an intervals-mode driver with memray to record the
     allocation delta (the `ak.to_packed`/`ak.concatenate` frames should drop out).

### Risks

- **Ordering**: the per-track concatenation order (`axis=1`, batch-major) must be
  reproduced exactly or the byte-identity gate fails — this is the main correctness
  pin. Build offsets carefully and lean on the snapshot.
- **Multi-track offset assembly** is fiddlier than the single-buffer reconstructors
  (FU-2 stacks N tracks per batch row); a numba gather kernel keeps it readable.
- **Scope**: keep `RaggedIntervals` as the public return type (unchanged API). Only the
  internal transport changes. No skill update.

### Acceptance

- `with_tracks(kind="intervals")` output byte-identical to pre-change (new snapshot
  case + existing dataset suite green).
- Awkward guard for `kind="intervals"` asserts 0 dispatches.
- memray shows the `ak.to_packed`/`ak.concatenate` interval-assembly allocations gone.

### Worth-it check

Lower priority than the main refactor: `kind="intervals"` is an opt-in mode (default is
float32 tracks), so fewer users hit it. But it is single-level ragged (unlike variants),
so the flat conversion is genuinely viable and removes the last awkward dispatch from the
track hot path. Do it if/when intervals-mode throughput matters; otherwise the FU-1 guard
+ documented carve-out is a clean holding state.
