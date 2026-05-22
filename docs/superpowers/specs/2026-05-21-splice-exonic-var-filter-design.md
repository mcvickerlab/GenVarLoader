# Fix #176 — `Dataset.open(splice_info, var_filter)` vs `with_settings`

## Problem

The reporter observes that `Dataset.open(splice_info=..., var_filter="exonic")`
crashes inside `choose_exonic_variants` on 1KG cohort data (`MemoryError`
or `ValueError: negative dimensions not allowed`), while the equivalent
sequence
`Dataset.open(...).with_seqs("haplotypes").with_settings(splice_info=..., var_filter=...)`
appears to succeed on the same data. The reporter inferred that
`with_settings` should be the canonical entry point and `open` is broken.

Investigation shows three concrete bugs:

### Bug 1 — `with_settings(var_filter=...)` silently fails to propagate to `_recon`

`python/genvarloader/_dataset/_impl.py:453-463` evolves `self._seqs` for
`var_filter` but never updates `self._recon`. After `with_seqs("haplotypes")`,
`_recon` is a *separate* `Haps` instance (created by
`haps.to_kind(RaggedSeqs)` at `_impl.py:692`, which uses
`evolve(self, kind=kind)`). The min_af / max_af / var_fields block at
`_impl.py:401-434` already does the right thing — it evolves `_seqs` *and*
re-derives `_recon` — but the `var_filter` block tacked on at the bottom
does not.

`__getitem__` and `_getitem_spliced` invoke `inner_ds._recon(...)`, never
`_seqs`. So Path B does not exercise the exonic filter at all. It "works"
only in the sense that no filter runs; the returned haplotypes are
un-filtered.

### Bug 2 — `choose_exonic_variants` mis-indexes 2-D SVAR offsets

`python/genvarloader/_dataset/_genotypes.py:455-458` and `:479-482`:

```python
if geno_offsets.ndim == 1:
    o_s, o_e = geno_offsets[o_idx], geno_offsets[o_idx + 1]
else:
    o_s, o_e = geno_offsets[o_idx]
```

Real SVAR offsets are constructed in `Haps.from_path` at `:284` via
`offsets.reshape(2, -1)` — shape `(2, n_slices)`. So `geno_offsets[o_idx]`
returns a length-`n_slices` row, not a 2-tuple. Numba unpacking against
two scalars produces garbage values, leading to `o_e - o_s` that is
either negative (→ `ValueError: negative dimensions not allowed` at
`keep = np.empty(n_variants, ...)`) or astronomically large (→
`MemoryError`).

The sibling `filter_af` kernel does it correctly at `:562`:
`o_s, o_e = geno_offsets[:, o_idx]`. Both branches of
`choose_exonic_variants` need the same form.

### Bug 3 — regression test from #170 used the wrong 2-D layout

`tests/dataset/genotypes/test_choose_exonic_variants.py:50` constructs
2-D offsets as `[[0, 1], [1, 2]]` — shape `(total_variants, 2)`. This
is not the SVAR layout. The wrong indexing happened to return a
2-element row by coincidence, so the test passed and #170 closed without
catching Bug 2. The test must be rebuilt on the real `(2, n_slices)`
layout, with `n_slices > 2`.

## Goal

1. `Dataset.open(splice_info=..., var_filter=...)` and
   `Dataset.open(...).with_seqs("haplotypes").with_settings(splice_info=..., var_filter=...)`
   must produce equivalent dataset state and equivalent `__getitem__`
   output on SVAR-backed data. No silent disagreement.
2. `choose_exonic_variants` must correctly handle the real
   `(2, n_slices)` SVAR offsets layout.
3. The asymmetry cannot recur — collapse the two configuration paths to
   one source of truth.

## Design

### Change 1: `choose_exonic_variants` — fix 2-D indexing

In `python/genvarloader/_dataset/_genotypes.py`, change both occurrences
of the `else` branch:

```python
else:
    o_s, o_e = geno_offsets[o_idx]
```

to:

```python
else:
    o_s, o_e = geno_offsets[:, o_idx]
```

Matches `filter_af:562`. No other logic in the function changes.

### Change 2: Unify `open` and `with_settings` for splice/filter config

Today `Dataset.open` reaches into `Haps(filter=...)` directly at
`_impl.py:229` and constructs `_sp_idxer`/`_spliced_bed` inline at
`:264-278`. `with_settings` has its own copy of the splice logic at
`:436-451` and a separate (broken) var_filter block at `:453-463`. The
reporter explicitly asked: "fix the `open()` path internally to route
through the same code as `with_settings()`".

Plan:

- `Dataset.open` constructs the dataset with `splice_info=None` and
  `var_filter=None` (i.e. with `Haps(filter=None)` and
  `_sp_idxer=None`/`_spliced_bed=None`).
- Immediately after, if either argument was supplied, it delegates:
  `dataset = dataset.with_settings(splice_info=..., var_filter=...)`.
- All splice / var_filter configuration logic now lives in one place:
  `with_settings`.

This requires `with_settings` to be correct for these arguments, which
is Change 3.

### Change 3: `with_settings(var_filter=...)` propagates to `_recon`

`_recon` is a `Haps[<kind>]` whose `kind` field reflects the user's
`with_seqs(...)` choice (e.g. `RaggedSeqs` for `"haplotypes"`,
`RaggedAnnotatedHaps` for `"annotated"`). `_seqs`, by contrast, is
always `Haps[RaggedVariants]`. We must update `_recon`'s filter while
preserving its `kind`.

Update the var_filter block at `_impl.py:453-463` to also propagate to
`_recon`:

```python
if var_filter is not None:
    if not isinstance(self._seqs, Haps):
        raise ValueError(
            "Filtering variants can only be done when the dataset has variants."
        )
    if var_filter is False:
        var_filter = None

    if var_filter != self._seqs.filter:
        haps = to_evolve.get("_seqs", self._seqs)
        haps = evolve(haps, filter=var_filter)
        to_evolve["_seqs"] = haps

        # Propagate filter to _recon, preserving its kind (the user's
        # with_seqs choice). Do NOT replace _recon with _seqs wholesale.
        if isinstance(self._recon, Haps):
            recon_haps = to_evolve.get("_recon", self._recon)
            to_evolve["_recon"] = evolve(recon_haps, filter=var_filter)
        elif isinstance(self._recon, HapsTracks):
            recon = to_evolve.get("_recon", self._recon)
            new_haps = evolve(recon.haps, filter=var_filter)
            to_evolve["_recon"] = evolve(recon, haps=new_haps)
```

Note: there is an analogous (pre-existing) issue in the
`min_af`/`max_af`/`var_fields` recon-rebuild block at `:427-434` —
`to_evolve["_recon"] = recon` (where `recon = haps`) replaces `_recon`
with `_seqs` wholesale, clobbering `_recon.kind`. That's a separate bug
not in scope for #176; we deliberately avoid relying on that block here
so this fix is independently correct.

### Change 4: Tests

#### 4a — Rebuild `test_choose_exonic_variants_2d_geno_offsets`

`tests/dataset/genotypes/test_choose_exonic_variants.py`: replace the
`(total_variants, 2)` layout with the real `(2, n_slices)` layout, and
size it so `n_slices > 2`. Concretely: at least 3 slices so the wrong
indexing cannot accidentally yield 2 elements.

The 1-D test stays. Both must produce identical `keep` / `keep_offsets`.

#### 4b — Path-parity end-to-end test

New test in `tests/dataset/` (e.g. `test_open_vs_settings_parity.py`)
that:

1. Builds a tiny SVAR-backed GVL dataset using the existing
   `tests/data/filtered.svar` fixture and the `test_rc_packing.py`
   pattern of writing into a tmp dir and injecting
   `transcript_id` / `exon_number` columns into `input_regions.arrow`.
2. Opens it two ways:
   - `Dataset.open(path, reference=ref, splice_info=("transcript_id", "exon_number"), var_filter="exonic").with_seqs("haplotypes")`
   - `Dataset.open(path, reference=ref).with_seqs("haplotypes").with_settings(splice_info=("transcript_id", "exon_number"), var_filter="exonic")`
3. Asserts both paths produce identical output for at least
   `ds[0, :]`. Use `numpy.array_equal` on the materialized awkward
   arrays.
4. Asserts `ds_a._recon.filter == "exonic"` and
   `ds_b._recon.filter == "exonic"` — the direct probe for Bug 1.

Optional, if simple: also assert that a configuration with at least one
non-exonic variant differs from the un-filtered configuration. That
catches "both paths produce un-filtered output" as a single-failure mode.

## Non-goals

- We do **not** add a deprecation path for `Dataset.open(splice_info=..., var_filter=...)`. The user is right that it's a documented entry point; we fix it.
- We do **not** change the public API or any signatures.
- We do **not** touch `gvl.write` or related issue #162.

## Risk / blast radius

- Change 1 is a one-character-class fix on a hot numba kernel, with a
  direct sibling pattern (`filter_af`) to mirror.
- Change 2 is the largest behavioral shift: `Dataset.open` now flows
  through `with_settings`. The user-visible state should be identical
  *after* Change 3 lands, but ordering of validation errors may shift
  (e.g. an invalid `splice_info` now raises from inside `with_settings`
  rather than from `open`). Acceptable.
- Change 3 affects all callers of `with_settings(var_filter=...)`. The
  prior behavior was a silent no-op on `_recon`, so any code relying on
  the buggy behavior is itself buggy. Worth a CHANGELOG note.

## Verification

- `pixi run -e dev pytest tests/dataset/genotypes/test_choose_exonic_variants.py` — rebuilt test passes on fixed kernel, fails on unfixed.
- New path-parity test passes.
- Full `pixi run -e dev test` continues to pass.
- Spot-check existing splice tests in `tests/dataset/test_rc_packing.py` still pass.
