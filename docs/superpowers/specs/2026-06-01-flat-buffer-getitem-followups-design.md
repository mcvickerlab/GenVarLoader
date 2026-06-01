# Flat-buffer `__getitem__` follow-ups — design

**Date:** 2026-06-01
**Branch:** `feat/bench-codspeed-profiling` (follow-ups to PR #205)
**Status:** design; pending writing-plans

Three follow-ups to the flat-buffer `__getitem__` refactor (PR #205). FU-1/FU-2 were
identified during review (track paths the main refactor left out of scope); FU-3 takes
on the **variants** path, which PR #205 deliberately left awkward-native (Task 10
decision, based on *allocation* profiling) but which the final *CPU* py-spy A/B revealed
still spends ~4.21 s / 55% of getitem self-time in awkward. Independent of each other.
FU-1 is trivial, FU-2 and FU-3 are substantive.

FU-3's approach was unlocked by **seqpro 0.14.0**, which exposes a numba-parallel
`seqpro.rag.to_packed()` / `Ragged.to_packed()` — a flat replacement for
`Ragged(ak.to_packed(rag))`. Empirically (verified during design), `Ragged[idx]` fancy
indexing (1-D and 2-D) and `.to_packed()` dispatch **zero** awkward kernels, so the
per-batch allele/genotype/dosage *gathers* become one-line swaps with no hand-written
kernel.

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

---

## FU-3: Flat-ify the variants path (`to_packed` gathers + flat `rc_`)

### Problem

PR #205 left `RaggedVariants` awkward-native — the Task 10 spike decided so on the basis
of *memray* (variant assembly was not a top allocator). But the final *py-spy* A/B showed
variants still spends **~4.21 s / 55%** of getitem self-time in awkward — the largest
residual of any mode. `RaggedVariants` being an `ak.Array` subclass (public, in `__all__`)
forces the **container** to be awkward, but not the **work**. The per-batch awkward work is:

- **Gathers** — `ak.to_packed(<Ragged>[idx])` to pull each batch's rows from the on-disk
  stores: alleles `self.variants.{alt,ref}[v_idxs]` (`_haps.py:_get_alleles`), sparse
  genotypes `self.genotypes[r,s]` and dosages `self.dosages[r,s]` (`_haps.py:_get_variants`),
  and the AF-filter pack `ak.to_packed(ak.to_regular(genos[_keep], 1))`.
- **`rc_`** (`_rag_variants.py:200`) — `ak.where(to_rc, reverse_complement(alt), alt)`
  + `ak.to_packed`, ×2 for alt/ref. RCs the **whole batch eagerly** then `ak.where`-selects
  (the same anti-pattern removed for haps in PR #205), on the slow `ak.str` path.
- **`RaggedVariants.to_packed()`** (`_rag_variants.py:195`) — `ak.to_packed(self)` over the
  whole record array.

### Constraints / decisions (from brainstorming)

- **`RaggedVariants` stays an `ak.Array` subclass** — public API frozen, non-breaking. Move
  the work to flat buffers; build/keep the `ak.Array` as a thin layout wrapper at the boundary.
- **seqpro 0.14.0 `to_packed`** is the enabling primitive. Verified during design: `Ragged[idx]`
  fancy index (1-D `[v_idxs]` and 2-D `[r,s]`) returns a `Ragged` view with **0 awkward calls**,
  and `.to_packed()` materializes the gathered/reordered rows into a contiguous zero-based buffer
  with **0 awkward calls**, byte-correct. So `Ragged[idx].to_packed()` *is* an awkward-free,
  numba-parallel gather — no hand-written kernel needed.
- **Architecture B** (surgical, no new container): swap the awkward kernels in place; reuse the
  already-shipped seqpro `reverse_complement_masked` for `rc_`.

### Components

**C1 — gather swaps (`ak.to_packed(<Ragged>[idx])` → `<Ragged>[idx].to_packed()`):**
- `_get_alleles`: `self.variants.{alt,ref}[v_idxs].to_packed()` (×2). The downstream layout
  surgery (`ListOffsetArray`→`ListOffsetArray`→`RegularArray` → `(b,p,~v,~l)` `ak.Array`) is
  unchanged — it is cheap offset-wrapping, not a kernel.
- `_get_variants`: `genos = self.genotypes[r,s].to_packed()`; `dosages = self.dosages[r,s].to_packed()`.
- AF-filter path (only when `min_af`/`max_af` set): swap the `to_packed`. **Verify** the
  boolean-mask index `genos[_keep]` and `ak.to_regular(..., 1)` stay flat; if `to_regular` still
  dispatches awkward, either keep it (documented) or replace with offset arithmetic. Settle in the plan.

**C2 — flat `rc_`** (the one piece with real new logic; keep `rc_`'s signature + in-place contract):
- `to_rc` is per batch (length `shape[0] = b`). Build a **per-allele mask**:
  `np.repeat(np.repeat(to_rc, ploidy), variants_per_group)`, where `variants_per_group =
  np.diff(group_offsets)` (RC depends only on strand/batch, broadcast across ploidy then variants).
- Extract alt/ref's leaf byte buffer + innermost allele-offsets from the `ak.Array` layout, view as
  a 1-level `Ragged(n_alleles, ~l)`, and call seqpro `reverse_complement_masked(…, per_allele_mask)`
  (`copy=False`, in place, reuses `_COMP`). No `ak.where` / `ak.str.reverse` / `ak.to_packed`. ×2.
- The `ak.Array` shares the mutated buffer → in-place semantics preserved. Keep the early-return for
  `to_rc.any() == False`; `to_rc=None` → all-True mask.

**C3 — field-wise `RaggedVariants.to_packed()`** (replace `ak.to_packed(self)` with composition):
- Pack each **numeric** field (`start`, `ilen`, `dosage`, info fields) via `Ragged.to_packed()`
  (seqpro 0.14, single ragged dim).
- `alt`/`ref` are the **doubly-nested** case (ragged-of-ragged-of-uint8: `(b,p)→variants` and
  `variant→bytes`) — seqpro `to_packed` targets one ragged dim, so they need a dedicated 2-level
  pack. Approach: compose seqpro inner allele-byte pack (`Ragged(n_alleles, ~bytes).to_packed()`,
  which contiguates the bytes in the existing `(b,p,variant)` row order) + rebuild the outer
  group-offsets (cumsum of variant counts — structural, no second data copy); or a small numba
  2-level pack kernel. Settle the exact mechanism in the plan, gated by byte-identity.
- Reassemble the record from the packed fields.

### Left awkward (documented, irreducible-ish)

- `ak.zip` in `RaggedVariants.__init__` — the record-array construction. Cheap layout wrap; keeping
  it is the cost of the frozen `ak.Array` public type. This is the documented limit of "remove
  awkward as much as possible" for variants.

### Testing

- **Byte-identity gate:** add a `variants` case to `tests/dataset/test_flat_getitem_snapshot.py`
  (`with_seqs("variants")`, ragged). Extend `_flatten_output` to serialize a `RaggedVariants`
  (each field's data + both offset levels for alt/ref). Regenerate that one `.npz`.
- **Unit:** flat `rc_` vs old awkward `rc_` byte-identical (cases: `to_rc=None`, all-False early
  return, mixed-strand mask, multi-ploidy); field-wise `to_packed()` vs `ak.to_packed(self)`
  byte-identical, including a **sliced/scattered** input (the case `to_packed` exists to handle);
  gather swaps vs the old `ak.to_packed(<idx>)` byte-identical.
- **Awkward guard:** add a `variants` case to `tests/dataset/test_no_awkward_in_hotpath.py` asserting
  the gathers + `rc_` + `to_packed` no longer dispatch awkward — i.e. `ak.where`/`ak.str`/`ak.to_packed`
  call counts are 0; document that `ak.zip` (record construction) is the only remaining awkward and is
  intentionally excluded (mirroring the existing `RaggedVariants` carve-out note).
- **Profiling:** re-run `--mode variants` (memray + `sudo py-spy`) and refresh the variants row of the
  REGRESSIONS.md A/B; the gather `_carry` / `ak.str` RC / `to_packed` frames should drop out, leaving
  `ak.zip` + the gvl numba kernels as the residual.

### Risks

- **C2 leaf-buffer extraction + mask broadcast**: reaching the correct innermost byte buffer/offsets
  across the `RegularArray`/`ListOffsetArray` nesting, and the batch→ploidy→variants mask broadcast,
  are the correctness pins. Byte-identity unit test + snapshot guard them.
- **C2/C3 in-place buffer sharing**: confirm the extracted buffer is the `ak.Array`'s actual buffer
  (not a copy) so mutation is visible; otherwise rebuild the layout from the packed buffer.
- **AF-filter flatness**: `genos[_keep]` (bool-mask) + `to_regular` may still touch awkward; verify and
  document if so (this path is opt-in via `min_af`/`max_af`).

### Acceptance

- `with_seqs("variants")` output byte-identical to pre-change (new snapshot case + dataset suite green).
- Awkward guard: variants gathers + `rc_` + `to_packed` dispatch 0 awkward; only `ak.zip` remains
  (documented).
- memray + py-spy A/B show the variants awkward self-time (~4.21 s) substantially reduced; record the
  residual (`ak.zip` + numba kernels).

### Worth-it check

Higher value than FU-2: variants is a primary output mode and carried the **largest** residual awkward
CPU after PR #205. seqpro 0.14 collapses the gathers to one-liners, so the cost is concentrated in C2
(flat `rc_`) and C3's alt/ref 2-level pack — bounded, well-understood work. The frozen `ak.Array`
container means `ak.zip` stays; that is the acknowledged floor.
