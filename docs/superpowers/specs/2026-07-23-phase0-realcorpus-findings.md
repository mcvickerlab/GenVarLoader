# Phase 0 findings — #315 SVAR2 variant-windows slot under-count, real corpus

**Date:** 2026-07-23
**Issue:** [#315](https://github.com/mcvickerlab/GenVarLoader/issues/315)
**Gate for:** Task 3 (correct `_output_bytes_per_instance` for the `Svar2Haps` path)
**Predecessors:**
- [`2026-07-21-phase0-findings.md`](./2026-07-21-phase0-findings.md) — Phase 0 on the `Haps`
  (SVAR1/VCF/PGEN) path; concluded the estimate is a correct per-instance upper bound there,
  and (incorrectly) that `Svar2Haps` was unreachable from the released `to_dataloader` path.
- [`2026-07-23-svar2-variant-windows-slot-fit-design.md`](./2026-07-23-svar2-variant-windows-slot-fit-design.md) —
  corrects that: the real Hartwig corpus is SVAR2-format and does reach `Svar2Haps`. This
  doc is that design's "Phase 0-real" gate, executed against the real corpus.

## Corpus confirmation

Symlinked in from `/carter/users/dlaub/projects/aster/{data/corpus/hartwig,refs}` per the
task brief. Opened with `gvl.Dataset.open("data/corpus/hartwig/hartwig.gvl",
reference="refs/GRCh38.ensembl.fa.bgz")`: full shape `(1044 regions, 7089 samples)`.
Subset to the reported 40 regions (`ds.subset_to(regions=slice(40))`), configured exactly as
reported (flat `variant-windows`, `ref="window"`, `alt="allele"`, `flank_length=128`,
`unphased_union=True`, `jitter=0`, tracks off):

```
reconstructor: Svar2Haps  is Svar2Haps: True
shape: 40 regions x 7089 samples
unphased_union: True
var_fields: ['alt', 'ilen', 'start']
window_opt: VarWindowOpt(flank_length=128, token_alphabet=b'ACGT', unknown_token=4, ref='window', alt='allele')
```

Confirmed: **this dataset reconstructs through `Svar2Haps`**, not `Haps`.

## est / overhead / real table (growing N)

Instrumentation script: `scratch/diag_315_realcorpus.py` (adapted from the brief — `view.shape`
is already `(R, S)`; `unphased_union` lives on `view._seqs`, not `view`). For N ∈ {1, 4, 16}
regions' worth of instances (all 7089 samples each), comparing
`_output_bytes_per_instance(..., include_offsets=True)` against a direct `write_chunk` of the
real reconstructed `_FlatVariantWindows`:

| N (instances) | est | overhead | real | est+overhead−real | per-instance gap `(real−est)/N` |
|---:|---:|---:|---:|---:|---:|
| 7,089 (1 region) | 226,848 | 4,096 | 121,039,709 | **−120,808,765** | 17,042.3 |
| 28,356 (4 regions) | 907,392 | 4,096 | 447,023,623 | **−446,112,135** | 15,732.7 |
| 113,424 (16 regions) | 3,629,568 | 4,096 | 1,939,754,801 | **−1,936,121,137** | 17,069.8 |

`est + overhead − real` is **massively negative** and grows in magnitude with N (not a fixed
per-chunk constant): `est` itself is *exactly* `32 bytes × N` at every N (`226848/7089 =
907392/28356 = 3629568/113424 = 32.0`), i.e. the estimate is a **flat, content-independent
per-instance constant**, while `real` scales with N and the actual variant density (up to
~17 KB/instance here). `per_inst_gap` stays in the same 15.7–17.1 K band across a 16×
increase in N — this is the per-instance under-count signature the brief asks to
distinguish from a per-chunk constant (Phase 0's `Haps`-path finding was the latter, at
~50 B/chunk; this is the former, at **~16 KB/instance**, ~500× larger and scaling with N).
Reported over the full 40×7089 = 283,560-instance subset this is why the real chunk
(`n_instances=283560`) blew a single ~17 MB slot with room to spare only for the
estimate's own delusion — the trace's `SlotOverflowError` matches the design doc's repro
exactly.

## The pinned diverging term: `n_vars_total` (M) is unconditionally 0

Decomposed one region (`r=0`, all 7089 samples):

```
sum n_vars_total (estimate M): 0
real_ploidy: 2   unphased_union: True
sum emitted window count (W): 427592
M == W (per-instance) all equal: False
n mismatched instances: 7044 / 7089        # the 45 matches are (0==0): samples with no real variants in this region
estimate's alt_alleles term (_allele_bytes_sum): all zero (sum = 0), first 5 = [0, 0, 0, 0, 0]
sum real alt (bare) token-bytes:  388,485
Across the FULL 40-region subset (283,560 instances): estimate's alt_alleles term all zero: True
```

**`M` (the estimate's `n_vars_total`, from `Dataset.n_variants()`) is identically `0` for
every one of the 283,560 instances in the subset**, regardless of how many real variants
that `(region, sample)` group actually has. `W` (the real emitted window count) is not:
427,592 total over the single sampled region alone (mean ≈ 60/instance, but heavily skewed —
see the worst instance below).

### Root cause (traced to source, not just measured)

`Dataset.n_variants()` (`_impl.py:1293-1337`) does `n_vars =
self._seqs.n_variants[r_idx, s_idx]` for any `isinstance(self._seqs, Haps)` — `Svar2Haps`
*is* a `Haps` subclass, so it takes this branch. But `Svar2Haps.__post_init__`
(`_svar2_haps.py:208`) sets `self.n_variants = self.genotypes.lengths`, and
`self.genotypes` is the **permanently-empty SVAR1-shaped placeholder** `Svar2Haps.from_path`
constructs once at open time (`_svar2_haps.py:281-285`):

```python
empty_geno = Ragged.from_offsets(
    np.empty(0, V_IDX_TYPE), (R, S, P, None), np.zeros(R * S * P + 1, np.int64)
)
```

— all offsets zero, by construction, for every `(region, sample, ploid)` group, forever. It
exists **only** so the many `isinstance(_, Haps)` / `case Haps()` checks scattered through
`_impl.py`/`_haps.py` keep type-checking; `Svar2Haps` never populates it with real per-region
membership, because SVAR2 reconstructs read-bound directly from the on-disk `.svar2` store via
Rust FFI (`decode_variants_from_svar2_readbound`), not through an in-memory sparse genotypes
table the way SVAR1 `Haps` does. So `.lengths` (`seqpro.rag.Ragged.lengths`, which derives
purely from the offsets array) is an all-zero array of shape `(R, S, P)` — **not** a lazily-
computed real count. `M = 0` always.

The same empty placeholder poisons two more terms in `_output_bytes_per_instance`'s
`"variant-windows"` branch (`_impl.py:1564-1743`), independently of `n_vars_total`:

- **`alt_alleles`** (`_impl.py:1590-1594`) calls `Haps._allele_bytes_sum(ds_idx, "alt")`
  (`_haps.py:770-809`), which reads `self.genotypes[r, s].to_packed()` (empty → `v_idxs`
  always empty) *and* `self.variants.alt.offsets` — `self.variants` is likewise a
  `dummy_variants` placeholder with an empty `RaggedAlleles` (`_svar2_haps.py:286-296`).
  **Directly confirmed by measurement**: `haps._allele_bytes_sum(ds_idx, "alt")` is
  identically `0` for all 283,560 instances in the subset (printed above).
- **`ref_span`** for `ref="window"` (`_impl.py:1605-1642`) computes
  `haps_obj.genotypes[r_idx_grp, s_idx_grp].to_packed()` — the *same* empty placeholder —
  so `v_idxs` is empty and `ref_span = 0` for every instance too (same code path, not
  separately re-measured but structurally identical to the `alt_alleles` case above; both
  read `self.genotypes` with the same all-zero offsets).

With `n_vars_total = ref_span = alt_alleles = 0` for every instance, the entire payload term
of the branch collapses to 0 bytes; the only surviving (nonzero) contribution is the
schema-derived, content-independent offset-array overhead:
`8 bytes × folded_ploidy(1) × (n_scalar_fields(2: start+ilen) + n_window_slots(2)) = 32
bytes/instance` — which is exactly the flat `32 × N` observed above. **This is not a
`p_eff`/ploidy-grouping bug** (the folded-ploidy bookkeeping is internally consistent, and
the design doc's original `p_eff` hypothesis is refuted by this trace) **and not a
formula bug in the `ref_span`/`alt_alleles` expressions themselves** — it is a **wrong
data source**: three independent terms all read `Svar2Haps.genotypes`/`.variants`, objects
that are structurally incapable of ever holding real content for this reconstructor.

### The captured worst instance

`(r_idx=0, s_idx=5856)`, within the sampled region:

```
est_bytes=32   n_vars_total (M)=0   W (real emitted windows)=4438
real_ref_tok=1,140,590 bytes   real_alt_tok=4,544 bytes
```

`real_ref_tok / W = 1,140,590 / 4,438 = 257.0` exactly = `1 + 2×flank_length(128)` — i.e.
**every one of the 4,438 real variants in this instance has ref-window span exactly 1**
(no net deletion: SNVs and/or non-deleting insertions, `ilen ≥ 0` for all of them).
`real_alt_tok / W = 4,544 / 4,438 ≈ 1.02` bytes/variant — almost all bare ALT alleles are
1 byte (biallelic SNVs), with a handful of longer (multi-byte) insertion ALTs pulling the
mean above 1. `W = 4,438` is the `unphased_union` fold of `P=2` haplotypes via
`var_off[::P]` in `Svar2Haps._reconstruct_variant_windows` (`_svar2_haps.py:1010-1013`) —
the **naive sum of both haplotypes' per-sample variant counts, no dedup** — which is
semantically the same "no-dedup union" contract `Dataset.n_variants()`'s own docstring
describes for the `Haps` path (`_impl.py:1323-1327`); the two are not in semantic conflict,
`Svar2Haps` just never reports its side of that count into the shared `self.n_variants`.
This sample sits in a SNP-dense region (likely covering a segmental-duplication /
high-variant-density locus of the tumor cohort) with `P=2, unphased_union=True` — a common,
unremarkable record class, not an edge case (multiallelic, exotic ALT, or missing-genotype
pattern). **The overflow is triggered by ordinary dense SNP content, not a rare record.**

## Fixture decision: synthetic — confirmed, not just recommended

The defect is **structural and content-independent** (it is a wrong data *source*, not a
formula error keyed to any particular record shape), so it does not require reproducing the
real corpus's specific variant density or record types. Verified directly with
`scratch/diag_315_synthetic.py`, built from the exact fixture pattern
`tests/dataset/test_svar2_dataset.py` already uses (`_src`/`svar2_fixture`/`bed`: a 40 bp
`chr1` reference, a 4-record VCF — SNP, insertion, dense SNP, deletion — 2 samples,
converted via `genoray._core.run_conversion_pipeline` into a `.svar2` store), opened via
the release path (`gvl.write` + `gvl.Dataset.open(...).with_seqs("variant-windows", opt)`,
`ref="window", alt="allele", unphased_union=True`) over a single 15 bp region:

```
reconstructor: Svar2Haps  is Svar2Haps: True
estimate n_vars_total (M) per sample: [0, 0]
estimate bytes per sample: [32, 32]
real total payload bytes (both samples): 445
estimate total (both samples): 64
PIN REPRODUCES on synthetic fixture: True
```

**A tiny synthetic SVAR2 store — 1 region, 2 samples, 4 ordinary variants — reproduces the
identical `M=0`-while-`real>0` defect** at the same 32-bytes/instance constant seen on the
real corpus. **Fixture ladder option 1 (preferred, per the design doc) applies: no real-data
slice is required.** `tests/dataset/test_svar2_dataset.py` already has ready-made
`svar2_fixture`/`bed`/`_src` pytest fixtures building exactly this kind of store and already
exercises `with_seqs("variant-windows", ...)` on it (`_open_windows_pair`,
`test_svar2_variant_windows_ref_window_matches_svar1`) — Task 2 can reuse that
fixture/pattern directly rather than inventing a new one.

## What Task 3 must change

In `Dataset._output_bytes_per_instance`'s `"variant-windows"` branch (`_impl.py`, currently
`elif seq_kind == "variant-windows":` at line 1564), when `isinstance(self._seqs,
Svar2Haps)`, **do not** derive `n_vars_total` from `self.n_variants(...)` (→
`Svar2Haps.n_variants`, permanently empty) nor `ref_span`/`alt_alleles` from
`haps_obj.genotypes[...]`/`Haps._allele_bytes_sum` (same empty placeholders). Instead source
all three from the real per-instance decode `Svar2Haps` itself uses at read time
(`decode_variants_from_svar2_readbound`, already invoked by
`_reconstruct_variant_windows`/`_reconstruct_variants`): factor out a lightweight
count/measure entry point (real per-instance `var_off` boundaries — folded by the same
`p_eff = 1 if unphased_union else P` rule `_reconstruct_variant_windows` already applies —
plus `ilen`-derived ref span and ALT byte lengths) that both the reconstructor and the
estimator call, so the two cannot drift apart again (this mirrors the design doc's stated
preference: "prefer reusing `Svar2Haps`' own counting entry point over duplicating its
logic"). The `Haps`-path branch logic proved correct in the predecessor Phase 0 should be
left untouched; only the `Svar2Haps` data source needs replacing. Also apply the *same* fix
to the `"variants"` (non-window) branch (`_impl.py:1437-1563`), which reads
`_allele_bytes_sum` unconditionally too, and is presumably similarly under-counting for
`Svar2Haps` even though it is out of this issue's reported repro.

## What Task 2's fixture must contain

A `Svar2Haps`-backed dataset (any `.svar2` store — the existing
`tests/dataset/test_svar2_dataset.py::svar2_fixture` pattern is sufficient and preferred: a
handful of regions/samples, ≥1 ordinary variant of any type) opened through the released
`Dataset.open(...).with_seqs("variant-windows", VarWindowOpt(...))` path with
`ref="window"` (or `alt="window"`) and/or `var_fields` including `"alt"`/`"ilen"`, at least
one `(region, sample)` group with `n_variants > 0`, `unphased_union` both `True` and
`False`. The regression test asserts `sum(_output_bytes_per_instance(include_offsets=True))
+ slot_overhead_bytes(view) >= real write_chunk payload bytes` — **red** before Task 3 (per
this doc, the estimate side is `32 × n_instances` regardless of real content, so any
instance with a real variant already fails the inequality) and **green** after. No
real-corpus data needs to be committed.
