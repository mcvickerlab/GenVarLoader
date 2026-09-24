# Analytic run planning for `concat` — design

**Issues:** #403 (primary), #406 (bundled, test-only)
**Status:** approved for planning, 2026-09-15
**Target:** `main`, one PR

## Problem

`gvl.concat` decides, for each *merged* flat slot, which input dataset and which
source slot it comes from. Today it does that by materializing the whole mapping
and then compressing it:

```python
prov = provenance(axis, shapes, ploidy, order=order)  # (n_slots, 2) int64
runs = coalesce(prov)  # list[Run]
```

Both intermediates are sized by the `(R, S[, P])` grid, which is exactly the
quantity #357 just removed from the svar2 range cache. At the All of Us chr22
grid (`R` = 3,734, `S` = 535,662, `P` = 2):

| intermediate | shape | bytes |
|---|---|---|
| `provenance`, genotypes (`ploidy=2`) | `(4,000,323,816, 2)` int64 | **64.0 GB** |
| `provenance`, tracks (`ploidy=1`) | `(2,000,161,908, 2)` int64 | **32.0 GB** |
| `coalesce` list, genotypes | 4.00e9 × 184 B + pointers | **768 GB** |
| `coalesce` list, tracks | 2.00e9 × 184 B + pointers | **384 GB** |
| `copy_runs` `lengths` temp, tracks | 2.00e9 int64 | **16.0 GB** |

`Run` is a 4-field `NamedTuple` measured at 184 B on this codebase (72 B tuple +
112 B for its four `int` objects). A run is a maximal span over which the source
dataset is constant *and* the source slot increases by exactly 1, so on an
interleaved sample merge — two cohorts whose sample IDs interleave in sorted
order, the common case — every run is one slot long and the list degenerates to
one `Run` per slot.

### Scope correction to #403

#403 names only the per-sample-tracks call site and states that an svar2 concat
without per-sample tracks holds nothing `R x S`-sized. That is true for the svar2
range cache specifically, but **three** call sites remain, and the two #403 does
not name are the larger ones:

| site | call | at chr22 |
|---|---|---|
| `_concat.py:529` (`pgen_vcf`) | `provenance(axis, shapes, ploidy, ...)` | 64.0 GB |
| `_concat.py:557` (`svar`) | `provenance(axis, shapes, ploidy, ...)` | 64.0 GB |
| `_concat.py:585` (per-sample tracks) | `provenance(axis, shapes, 1, ...)` | 32.0 GB |

Two further sites are already bounded and are **out of scope**: `_concat.py:341`
(`region_runs`, `(R, 2)`) and `_concat.py:631` (annot tracks, region axis,
`(R, 2)`). Both pass `[(r, 1) for r, _ in shapes]` with `ploidy=1`, so they are R
rows, not `R*S`.

So `concat` is unbounded on all three backends that store per-slot ragged
payloads, not on the tracks path alone.

### Correction to #403's proposed fix

#403 proposes emitting runs "as an iterator instead of a list," noting that
`copy_runs` consumes runs in destination order and never indexes backwards. That
is true but insufficient: `copy_runs` iterates `runs` **twice** — once at
`_concat_io.py:59-66` to build `lengths`, once at `:75-83` to stream bytes — and
`_gather_svar_offsets` (`_concat.py:232-233`) does the same. A bare generator
would yield an empty second pass and silently produce a truncated or zero-length
output rather than an error. The replacement must be **re-iterable**, not a
one-shot iterator.

## Design

### The key property

For both axes, the run-break pattern is a function of `order` alone and is
**independent of the region index `r`**. The break condition between adjacent
merged slots is `(source dataset changes) OR (source slot does not increment)`.

*Samples axis, within a region.* Between merged samples `j` and `j+1`, the source
slots are `(r*S_{d_j} + w_j)*P + P-1` and `(r*S_{d_{j+1}} + w_{j+1})*P`. If the
dataset changes it is a break unconditionally. If it does not, the `r*S_d` terms
cancel and the condition reduces to `w_{j+1} == w_j + 1` — no `r`.

*Samples axis, across a region boundary.* Continuity from `(r, S-1, P-1)` to
`(r+1, 0, 0)` requires `d_0 == d_{S-1}` and `S_d + w_0 == w_{S-1} + 1`. The `r`
terms cancel here too.

*Regions axis.* Merged region `i` occupies one contiguous `S*P` block in both
source and destination, so breaks are found by scanning `order` (length R) alone.

Therefore the entire run structure is derivable from `order` in `O(R + S)` work
and memory, and the runs themselves stream in `O(1)`.

### `RunPlan`

Replace the `provenance` → `coalesce` pipeline on the hot path with a small
re-iterable object in `_concat_plan.py`:

```python
class RunPlan:
    def __init__(self, axis, shape_per_ds, ploidy, *, order=None): ...
    def __iter__(self) -> Iterator[Run]: ...
    def slot_batches(
        self,
    ) -> Iterator[tuple[int, NDArray[np.int64], NDArray[np.int64]]]: ...
    @property
    def n_slots(self) -> int: ...
```

**`__iter__`** yields `Run`s in destination order, recomputing from `order` on
each call, so the two-pass consumers work unchanged. Implementation carries a
single pending `Run` and extends it when the next segment is contiguous in both
source and destination. That carry subsumes the region-boundary case entirely —
**no special-casing of `cross` is needed**, which the validation below confirms.

**`n_slots`** is computed arithmetically (`R * S_merged * ploidy`, or
`len(order) * S * ploidy` on the regions axis), removing `copy_runs`'
`sum(r.src_stop - r.src_start for r in runs)` pre-pass.

**`slot_batches`** exists because of a second bottleneck found during validation.
`copy_runs` builds its `lengths` array with one numpy slice-assignment *per run*.
On an interleaved merge that is one tiny numpy call per slot — 2.0e9 of them at
chr22, far slower than the streaming it feeds. `slot_batches` instead yields
destination-contiguous batches as `(dst_start, ds_vec, slot_vec)`, so `copy_runs`
fills lengths with `R` vectorized gathers (3,734 at chr22) instead of `R*S`
scalar ones. On the samples axis a batch is one region — `S*P` int64 = 8.6 MB at
chr22. On the regions axis a batch is a **fixed-size chunk of a run**, capped at
`_SLOT_BATCH_SLOTS = 1 << 20` (8 MiB of int64): one batch per run would not do,
because a regions-axis run can span the entire merged grid, and materializing its
slot indices would rebuild the 32 GB array this whole change exists to remove.

### Consumer changes

1. **`copy_runs`** (`_concat_io.py:39`) — take `RunPlan`; build merged offsets
   from `slot_batches()`; write lengths directly into `merged[1:]` and cumsum in
   place, dropping the separate `lengths` array (16.0 GB at chr22); stream bytes
   from `__iter__`.
2. **`gather_fixed`** (`_concat_io.py:92`) — take `RunPlan`; single pass, only the
   type annotation and docstring change.
3. **`_gather_svar_offsets`** (`_concat.py:203`) — take `RunPlan`; replace its
   `sum(...)` pre-pass with `plan.n_slots`. Its two materialized planes are
   `n_slots` int64 each and stay as they are: they are the same order of
   magnitude as the offsets array `copy_runs` must build anyway, and the
   docstring already justifies them.
4. **`_concat.py`** — the three unbounded sites construct `RunPlan` instead of
   calling `provenance` + `coalesce`. The two bounded sites (`:341`, `:631`) also
   move over, for one obvious way to do it, but gain nothing.

### What remains after the change

| quantity | before | after |
|---|---|---|
| provenance map | 32–64 GB | 0 |
| run list | 384–768 GB | one live `Run` |
| `copy_runs` `lengths` | 16.0 GB | 0 |
| merged `offsets.npy` | 16.0 GB | 16.0 GB — **irreducible** |

The merged offsets array is the output file itself, required by the ragged
format. Removing it is a format change and explicitly not in scope.

**Planning is not free.** On a fully interleaved sample merge the plan still
yields one `Run` per slot — 2.0e9 of them at chr22, at roughly a microsecond
each. That is real time, but it is dominated by the per-run file IO those runs
describe (one seek per run), and it replaces an allocation that cannot complete
at all. If planning ever becomes the measured bottleneck, a batched array-valued
run form is the follow-up; it is not proposed here.

### `provenance` and `coalesce` are retained

They stay in `_concat_plan.py`, no longer called on the hot path, documented as
the reference implementation that `RunPlan` is verified against. They are already
tested, they are small, and they are what makes the new path provably correct.
Deleting them would delete the oracle. `_concat_plan` is private
(`python/genvarloader/_dataset/_concat_plan.py`), so nothing here is a public API
change.

## Testing

### Differential property test (the primary gate)

```python
list(RunPlan(axis, shapes, ploidy, order=order)) == coalesce(
    provenance(axis, shapes, ploidy, order=order)
)
```

Exact equality, swept over: both axes; `ploidy` 1–3; 1–3 input datasets; region
and sample counts 0–4 inclusive (so empty grids are covered); and four `order`
modes — `None`, explicit block-default, uniformly shuffled, and a sorted
key-interleave that models the real two-cohort merge.

This prototype has already been run at 2,880 configurations with **zero
mismatches**, including every degenerate case in the sweep. The shipped test is
that sweep, seeded, as a normal pytest — not a hypothesis strategy, so failures
are reproducible by config rather than by shrink.

### Memory bound

`tracemalloc` peak over `RunPlan` construction plus one full iteration on a
synthetic grid whose `provenance` map provably exceeds the bound (e.g. `R = 200`,
`S = 2000`, `ploidy = 2` → 800k slots, 12.8 MB under the old path). Assert peak
stays below a small multiple of `O(R + S)`. The point is to fail loudly if
someone reintroduces a materialized slot-space array.

### Consumer equivalence

For each of `copy_runs`, `gather_fixed`, `_gather_svar_offsets`: byte-identical
output when driven by `RunPlan` versus by `coalesce(provenance(...))` on small
fixtures, across both axes.

### Existing suite

The full tree is the regression gate: `pixi run -e dev test`. Concat has existing
coverage that must stay green without modification — if a concat test needs
editing to pass, that is a signal the change is not behavior-preserving.

## #406 — `svar2_store` fixture (bundled, test-only)

Independent of the above; bundled into the same PR because it is test-only and
the project prefers one PR per initiative.

### Scope correction to #406

#406 refers to "the `svar2_store` fixture." There are **seven** definitions of
it, each carrying its own copy-pasted `_REF` (byte-identical 40 bp string) and
`_VCF` (the same three variants: SNP at 3, insertion at 7, deletion at 12),
differing only in whether sample `S2` is present:

| file | samples |
|---|---|
| `tests/dataset/conftest.py:52` | `S0, S1, S2` — the 18-cell grid #406 measured |
| `tests/test_svar2_reconstruct.py:34` | `S0, S1` |
| `tests/unit/dataset/test_svar2_store.py:28` | `S0, S1` |
| `tests/unit/dataset/test_svar2_link.py:59` | `S0, S1` |
| `tests/dataset/test_svar2_readbound_variants.py:36` | `S0, S1` |
| `tests/dataset/test_svar2_readbound_haps.py:34` | `S0, S1` |
| `tests/dataset/test_svar2_readbound_tracks.py:39` | `S0, S1` |

Enriching one of seven leaves the other six degenerate, so consolidation comes
first.

**Two of #406's three follow-ons are already done.** Verified against `main` at
`d0e2f498`:

- The per-channel oracle masks it reports at `test_write_svar2.py:117-119` and
  `:497-499` were fixed during #357 by `c3da6bda` ("test(svar2): fix oracle
  masking bug, strengthen weak assertions, add divergence pin"). The site now
  reads `present = (widths_snp > 0) | (widths_indel > 0)` — cell-level — with a
  comment explaining why per-channel masking is wrong. Nothing to do.
- `test_dense_layout_dataset_still_opens_and_reads` (`:737`) no longer overclaims:
  its docstring already says "This is a round-trip smoke test, not a parity pin"
  and points at `test_dense_ranges_matches_fancy_indexing` as what actually pins
  `_DenseRanges.lookup`. It does, however, name the one-non-empty-cell grid as its
  reason, so that docstring becomes stale when the fixture is enriched and must be
  updated — and the test can then be promoted to a real parity pin.

So #406's live scope is the degenerate grid, the seven duplicates, and the
withdrawn sample-axis ordering guard.

**A further weakness the issue does not mention:** the current fixture's
`dense_snp_range` is `[[0,0],[0,0],[0,0]]` — the per-region dense *SNP* channel
is never exercised at all. Only `dense_indel_range` is populated.

### Change

1. Hoist one shared `_REF`, `_VCF`, `vcf_and_ref`, and `svar2_store` into the
   top-level `tests/conftest.py`; delete the six local shadows. This also removes
   six redundant `samtools faidx` + `bcftools view` + `bcftools index` subprocess
   triples from the suite.
2. Replace `_VCF` with the enriched genotypes below. Every sparse variant carries
   exactly **one** call, which is what keeps genoray's
   `cost_model.rs::choose_representation` from routing it to the per-region dense
   channel; the two 3-carrier variants exist to populate the dense channel,
   including the SNP half that is currently unexercised. `_REF` is unchanged
   (`ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT`, 40 bp), as are the three regions
   `[0,20)`, `[5,15)`, `[25,40)`.

   ```
   #CHROM POS ID REF ALT QUAL FILTER INFO FORMAT S0    S1    S2
   chr1   3   .  A   G   .    .      .    GT     1|0   0|0   0|0
   chr1   7   .  C   CAT .    .      .    GT     0|1   0|0   0|0
   chr1   9   .  T   C   .    .      .    GT     0|0   0|1   0|0
   chr1   12  .  GTA G   .    .      .    GT     0|0   1|0   0|0
   chr1   17  .  A   C   .    .      .    GT     0|1   0|0   0|0
   chr1   19  .  C   CGG .    .      .    GT     1|1   1|1   1|1
   chr1   30  .  G   A   .    .      .    GT     1|1   1|1   1|1
   ```

   This was not reasoned about — it was built and measured. Realized vk widths
   (`snp`/`indel` per `(region, sample, ploid)`), against the current fixture's
   1-of-18:

   | | ploid 0 | ploid 1 |
   |---|---|---|
   | r0 S0 | snp1/ind0 | snp1/ind1 |
   | r0 S1 | snp0/ind1 | snp1/ind0 |
   | r0 S2 | empty | empty |
   | r1 S0 | empty | snp0/ind1 |
   | r1 S1 | snp0/ind1 | snp1/ind0 |
   | r1 S2 | empty | empty |
   | r2 (all) | empty | empty |

   **7 of 18 cells occupied.** `dense_snp_range` becomes `[[0,0],[0,0],[0,1]]`
   and `dense_indel_range` `[[0,1],[0,0],[1,1]]`, so both dense channels are
   exercised where previously `dense_snp_range` was all zeros.

   Every property #406 asks for is satisfied and was checked mechanically:
   empty column (S2) kept; empty row (region 2) kept; ≥2 non-empty sample
   columns; a present cell with one empty channel (r0 S0 p0, r0 S1 p0); a present
   cell with both channels (r0 S0 p1); an absent cell *inside* a non-empty region
   (r1 S0 p0); S0's and S1's occupancy patterns differ, so a transposed sample
   axis is detectable; dense channel non-empty.

3. Rewrite `test_fixture_has_empty_cells` (`test_write_svar2.py:~645-690`) against
   the new grid. Its structure is right and must be preserved: assert the empty
   column and empty row **separately** (a single `not grid.all()` is a disjunction
   that stays green when either regresses alone), and keep the non-vacuity pin so
   a future genoray cost-model change that pushed every variant dense fails loudly
   instead of making every sparse/dense parity test pass against an empty table.
   Replace the "exactly one cell is occupied" pin with the 7-cell pattern above.
4. Restore the sample-axis ordering guard that had to be withdrawn during #357 as
   unsatisfiable — it is satisfiable now that S0 and S1 differ at (r1, p0).
5. Update `test_dense_layout_dataset_still_opens_and_reads`'s docstring, which
   currently justifies itself by the one-non-empty-cell grid, and promote it to a
   real parity pin now that the grid can distinguish a correct dense reader from
   one returning a constant. Mutation check: injecting
   `r_q = np.zeros_like(r_q)` into `_DenseRanges.lookup` must turn it red.

### Gate

`pixi run -e dev test` over the full tree. Every test that changes meaning must
change *because* the fixture got stronger, and each such change is called out in
its commit message.

## #404 — explicitly not in scope

Analyzed and left deferred; the reasoning is recorded as a comment on the issue.
Summary: the concern was that the probe's share of batch wall rises with thread
count (0.86% single-threaded, 1.68% at 8 threads) and would cross the 2% gate on
a wide machine. Fitting Amdahl's law to the two recorded batch figures
(`T(1) = 171 ms`, `T(8) = 88 ms`) gives serial 76.1 ms and parallel 94.9 ms, so
the batch wall asymptotes at 76.1 ms — above the 73.9 ms break-even the gate
implies. Worst-row share at infinite threads is 1.94%, under gate. Thread count
alone never trips it. Caveat: a two-point fit to a two-parameter model has no
residual and cannot be validated against its own data; it is a strong prior, not
a measurement. What would make #404 live is the *serial* 76.1 ms shrinking from
other work, and the existing benchmark gate already catches that.

## Out of scope

- The merged `offsets.npy` itself (16.0 GB at chr22) — a format change.
- A batched array-valued run form — only if planning is measured as the
  bottleneck.
- `_concat.py:341` and `:631` beyond the mechanical move to `RunPlan` — already
  bounded at `(R, 2)`.
- Any change to `Run`'s field set or to `_concat_io`'s IO strategy
  (`CONCAT_CHUNK_BYTES`, the 16 MiB NFSv3 knee) — unrelated.
- #404, #405.

## Risks

**The one real correctness risk is a silent wrong merge.** A plan that is wrong
in a way the exactness test misses produces a dataset whose slots point at the
wrong samples — readable, not obviously corrupt. The differential test against
the retained oracle is the mitigation, which is why the oracle is kept rather
than deleted, and why the sweep includes the interleaved order mode that models
the real failure case rather than only shuffled and default orders.

**Secondary:** `copy_runs`' in-place cumsum into `merged[1:]` aliases the array it
reads. `np.cumsum(x, out=x)` is documented as safe for exact aliasing, but the
change must be pinned by a test comparing against the out-of-place form rather
than assumed.
