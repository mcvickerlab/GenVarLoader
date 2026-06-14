# Ploidy-1 Unphased Union View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `unphased_union` view that folds a diploid dataset's two haplotypes onto one haploid sequence (union of called ALTs per `(region, sample)`), so `n_variants(...).shape[-1] == 1`, `variant-windows`/`variants` decode at ploidy 1, and `ds.ploidy == 1` — for haploid somatic modeling (issue #222).

**Architecture:** The stored genotypes stay diploid on disk; this is a read-time view. A `unphased_union: bool` flag lives on the `Haps` reconstructor, set via `Dataset.with_settings(unphased_union=True)`. The variant decode path (`get_variants_flat`) already gathers variant rows in C-order `(b, ploidy)`, so the union is a naive offset re-grouping (`row_offsets[::ploidy]`) — no sort, no dedup — which is safe because the downstream consumer is permutation-invariant. `Dataset.ploidy` and `Dataset.n_variants` collapse the ploidy axis when the flag is set. Phased haplotype/annotated output is disallowed under the flag. The retired, order-sensitive germline-CCF inference path is removed (it was the only consumer that required start-ordering).

**Tech Stack:** Python, NumPy, awkward, seqpro `Ragged`, frozen `Dataset` dataclass + slotted `Haps` dataclass, pytest + `snap_dataset` fixture (phased VCF, ploidy 2, opened with reference).

**Spec:** `docs/superpowers/specs/2026-06-14-unphased-union-ploidy1-design.md`

**Scoping note (read before starting):** The spec lists `Dataset.open(..., unphased_union=...)` as an *optional* parity entry point. This plan deliberately implements **only `with_settings`** as the entry point. Reason: `Dataset.open` promotes a genotype-backed dataset to the default `"haplotypes"` sequence type, which the flag's haplotype-guard (Task 3) rejects — so `open(unphased_union=True)` would raise unless the caller also switched to `variant-windows`/`variants` first, making it a footgun. `with_settings` is the clean, documented entry point and matches the user's chosen API. If open-parity is wanted later, it's a separate follow-up.

**Conventions:**
- Run a single test with: `pixi run -e dev pytest <path>::<name> -v`
- Generate test data once before the first run if not already present: `pixi run -e dev gen`
- Commit messages use conventional-commit prefixes (`feat:`, `test:`, `refactor:`, `docs:`).
- Prefix git commands with `rtk` (see CLAUDE.md).
- Line numbers below were accurate at plan-writing time; match on the quoted code, not the number, since earlier tasks shift later lines.

---

### Task 1: Add `unphased_union` flag to `Haps` and `with_settings`

**Files:**
- Modify: `python/genvarloader/_dataset/_haps.py` (add field after `window_opt`, ~line 272)
- Modify: `python/genvarloader/_dataset/_impl.py` (`with_settings`, signature ~line 222 and body ~line 419)
- Test: `tests/dataset/test_unphased_union.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/dataset/test_unphased_union.py`:

```python
"""Tests for the ploidy-1 unphased union view (issue #222).

Uses the session-scoped ``snap_dataset`` fixture (tests/dataset/conftest.py):
a phased VCF dataset, ploidy 2, opened with a reference.
"""

from __future__ import annotations

import numpy as np
import pytest

from genvarloader._dataset._flat_variants import VarWindowOpt


def test_with_settings_stores_unphased_union(snap_dataset):
    ds = snap_dataset.with_seqs("variants").with_settings(unphased_union=True)
    assert ds._seqs.unphased_union is True
    # Original dataset is unchanged (immutability).
    assert snap_dataset._seqs.unphased_union is False


def test_with_settings_unphased_union_requires_genotypes(
    reference, source_bed, tmp_path
):
    import pyBigWig

    import genvarloader as gvl

    # Reference-only dataset (no variants) -> _seqs is Ref, not Haps.
    bw_path = tmp_path / "dummy.bw"
    contig_sizes = [("chr1", 2_000_000), ("chr2", 2_000_000)]
    with pyBigWig.open(str(bw_path), "w") as bw:
        bw.addHeader(contig_sizes, maxZooms=0)
        bw.addEntries(["chr1"], [499_990], ends=[500_030], values=[1.0])

    out = tmp_path / "ref_only.gvl"
    gvl.write(
        path=out, bed=source_bed, tracks=gvl.BigWigs("sig", {"dummy": str(bw_path)})
    )
    ds = gvl.Dataset.open(out, reference=reference)
    with pytest.raises(ValueError, match="genotypes"):
        ds.with_settings(unphased_union=True)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_unphased_union.py::test_with_settings_stores_unphased_union -v`
Expected: FAIL — `with_settings() got an unexpected keyword argument 'unphased_union'` (or `AttributeError: ... 'unphased_union'`).

- [ ] **Step 3: Add the field to `Haps`**

In `python/genvarloader/_dataset/_haps.py`, find the `window_opt` field at the end of the `Haps` dataclass:

```python
    window_opt: VarWindowOpt | None = None
    """Options for variant-windows output mode. Set via ``with_seqs('variant-windows', opt)``."""
```

Add the new field immediately after it:

```python
    window_opt: VarWindowOpt | None = None
    """Options for variant-windows output mode. Set via ``with_seqs('variant-windows', opt)``."""
    unphased_union: bool = False
    """When True, fold the stored ``ploidy`` haplotypes onto a single haploid sequence
    (union of called ALTs per region/sample) for variant/variant-windows output. Phase is
    discarded; suited to unphased somatic calls. Set via ``with_settings(unphased_union=True)``.
    See issue #222."""
```

- [ ] **Step 4: Add the `with_settings` parameter and body**

In `python/genvarloader/_dataset/_impl.py`, add the parameter to the `with_settings` signature. Find:

```python
        dummy_variant: "DummyVariant | Literal[False] | None" = None,
    ) -> Self:
```

Replace with:

```python
        dummy_variant: "DummyVariant | Literal[False] | None" = None,
        unphased_union: bool | None = None,
    ) -> Self:
```

Add a docstring entry for it just after the `dummy_variant` docstring block (before the closing `"""` of the method docstring). Find the end of the `dummy_variant` description:

```python
            and the variant-window token buffers) the dummy entry is filled entirely with
            ``unknown_token``. Pass :code:`False` to disable.
        """
```

Replace with:

```python
            and the variant-window token buffers) the dummy entry is filled entirely with
            ``unknown_token``. Pass :code:`False` to disable.
        unphased_union
            When :code:`True`, fold the stored ``ploidy`` haplotypes onto a single haploid
            sequence: the union of called ALTs per ``(region, sample)``. ``ds.ploidy`` and
            ``n_variants(...)`` then report ploidy ``1``, and ``"variants"`` /
            ``"variant-windows"`` output decode at ploidy ``1``. Phase is discarded (suited
            to unphased somatic calls); ALT occurrences are concatenated across haplotypes
            with no sort or dedup (a hom call appears once per haplotype). Requires a dataset
            with genotypes and is incompatible with ``"haplotypes"`` / ``"annotated"``
            output (raises). See issue #222.
        """
```

Then add the handling block. Find the `dummy_variant` handling block that ends with:

```python
                haps = to_evolve.get("_seqs", self._seqs)
                to_evolve["_seqs"] = replace(haps, dummy_variant=dummy_variant)

        # If any source state changed, rebuild _recon via the factory.
```

Insert the new block between them:

```python
                haps = to_evolve.get("_seqs", self._seqs)
                to_evolve["_seqs"] = replace(haps, dummy_variant=dummy_variant)

        if unphased_union is not None:
            if not isinstance(self._seqs, Haps):
                raise ValueError(
                    "unphased_union requires a dataset with genotypes (variants)."
                )
            haps = to_evolve.get("_seqs", self._seqs)
            to_evolve["_seqs"] = replace(haps, unphased_union=unphased_union)

        # If any source state changed, rebuild _recon via the factory.
```

- [ ] **Step 5: Run both Task-1 tests to verify they pass**

Run: `pixi run -e dev pytest tests/dataset/test_unphased_union.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
rtk git add python/genvarloader/_dataset/_haps.py python/genvarloader/_dataset/_impl.py tests/dataset/test_unphased_union.py
rtk git commit -m "feat(dataset): add unphased_union flag on Haps + with_settings (#222)"
```

---

### Task 2: Collapse `ploidy` and `n_variants` under the flag

**Files:**
- Modify: `python/genvarloader/_dataset/_impl.py` (`ploidy` property ~line 908, `n_variants` method ~line 1207)
- Test: `tests/dataset/test_unphased_union.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/dataset/test_unphased_union.py`:

```python
def test_ploidy_reports_one_under_union(snap_dataset):
    baseline = snap_dataset.with_seqs("variants")
    assert baseline.ploidy == 2  # stored diploid
    u = baseline.with_settings(unphased_union=True)
    assert u.ploidy == 1


def test_n_variants_collapses_to_union_count(snap_dataset):
    baseline = snap_dataset.with_seqs("variants")
    # (R, S, ploidy)
    n2 = baseline.n_variants()
    assert n2.shape[-1] == 2

    u = baseline.with_settings(unphased_union=True)
    nu = u.n_variants()
    # Folded to a single haploid slot.
    assert nu.shape[-1] == 1
    # Naive union count == sum of per-haplotype counts (no dedup).
    np.testing.assert_array_equal(nu[..., 0], n2.sum(-1))


def test_n_variants_collapse_preserves_leading_shape(snap_dataset):
    u = snap_dataset.with_seqs("variants").with_settings(unphased_union=True)
    n2 = snap_dataset.with_seqs("variants").n_variants()
    nu = u.n_variants()
    # Region/sample axes unchanged, only ploidy axis folded 2 -> 1.
    assert nu.shape == (*n2.shape[:-1], 1)
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest tests/dataset/test_unphased_union.py::test_ploidy_reports_one_under_union tests/dataset/test_unphased_union.py::test_n_variants_collapses_to_union_count -v`
Expected: FAIL — `assert 2 == 1` for ploidy; `assert 2 == 1` for `nu.shape[-1]`.

- [ ] **Step 3: Update the `ploidy` property**

In `python/genvarloader/_dataset/_impl.py`, find:

```python
    @property
    def ploidy(self) -> int | None:
        """The ploidy of the dataset."""
        if isinstance(self._seqs, Haps):
            return self._seqs.genotypes.shape[-2]
```

Replace with:

```python
    @property
    def ploidy(self) -> int | None:
        """The ploidy of the dataset.

        Reports ``1`` when ``unphased_union`` is set (the two stored haplotypes are
        folded onto a single haploid sequence); otherwise the stored ploidy.
        """
        if isinstance(self._seqs, Haps):
            if self._seqs.unphased_union:
                return 1
            return self._seqs.genotypes.shape[-2]
```

- [ ] **Step 4: Update `n_variants`**

In `python/genvarloader/_dataset/_impl.py`, find the Haps branch inside `n_variants`:

```python
        if not isinstance(self._seqs, Haps):
            n_vars = np.zeros((len(r_idx), len(s_idx), 1), dtype=np.int32)
        else:
            # ((...), P)
            n_vars = self._seqs.n_variants[r_idx, s_idx]
```

Replace with:

```python
        if not isinstance(self._seqs, Haps):
            n_vars = np.zeros((len(r_idx), len(s_idx), 1), dtype=np.int32)
        else:
            # ((...), P)
            n_vars = self._seqs.n_variants[r_idx, s_idx]
            if self._seqs.unphased_union:
                # Fold the ploidy axis: union count per (region, sample) is the
                # naive sum of per-haplotype counts (no dedup). ((...), 1)
                n_vars = n_vars.sum(-1, keepdims=True)
```

- [ ] **Step 5: Run the Task-2 tests to verify they pass**

Run: `pixi run -e dev pytest tests/dataset/test_unphased_union.py -k "ploidy or n_variants" -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
rtk git add python/genvarloader/_dataset/_impl.py tests/dataset/test_unphased_union.py
rtk git commit -m "feat(dataset): report ploidy=1 and fold n_variants under unphased_union (#222)"
```

---

### Task 3: Disallow phased haplotype/annotated output under the flag

**Files:**
- Modify: `python/genvarloader/_dataset/_impl.py` (`with_seqs` ~line 666-684, `_check_valid_state` ~line 482-488)
- Test: `tests/dataset/test_unphased_union.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/dataset/test_unphased_union.py`:

```python
@pytest.mark.parametrize("kind", ["haplotypes", "annotated"])
def test_union_then_phased_seqs_raises(snap_dataset, kind):
    # Flag set first, then request a phased sequence type via with_seqs.
    u = snap_dataset.with_seqs("variants").with_settings(unphased_union=True)
    with pytest.raises(ValueError, match="unphased_union"):
        u.with_seqs(kind)


@pytest.mark.parametrize("kind", ["haplotypes", "annotated"])
def test_phased_seqs_then_union_raises(snap_dataset, kind):
    # Phased sequence type first, then the flag via with_settings.
    ds = snap_dataset.with_seqs(kind)
    with pytest.raises(ValueError, match="unphased_union"):
        ds.with_settings(unphased_union=True)


def test_union_allows_variant_windows(snap_dataset):
    # variant-windows is the supported output and must NOT raise.
    ds = (
        snap_dataset.with_seqs("variants")
        .with_settings(unphased_union=True)
        .with_output_format("flat")
        .with_seqs(
            "variant-windows",
            VarWindowOpt(flank_length=4, token_alphabet=b"ACGT", unknown_token=4),
        )
    )
    assert ds._seqs.unphased_union is True
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest tests/dataset/test_unphased_union.py -k "phased or union_allows" -v`
Expected: FAIL — `test_union_then_phased_seqs_raises` and `test_phased_seqs_then_union_raises` do not raise (DID NOT RAISE ValueError). `test_union_allows_variant_windows` should already pass.

- [ ] **Step 3: Add the guard to `_check_valid_state`**

This covers the "phased seqs first, then flag" ordering (`with_settings` calls `_check_valid_state` at its end). In `python/genvarloader/_dataset/_impl.py`, find the start of the variant-windows check in `_check_valid_state`:

```python
        if self.sequence_type == "variant-windows":
            haps = self._seqs
            if not isinstance(haps, Haps) or haps.window_opt is None:
                raise ValueError(
                    "with_seqs('variant-windows') requires a VarWindowOpt"
                    " (pass it to with_seqs)."
                )
```

Insert this block immediately *before* it:

```python
        if (
            isinstance(self._seqs, Haps)
            and self._seqs.unphased_union
            and self.sequence_type in ("haplotypes", "annotated")
        ):
            raise ValueError(
                "unphased_union is incompatible with 'haplotypes'/'annotated' output"
                " (a union of phased sequences is ill-defined). Use 'variant-windows'"
                " or 'variants', or clear the flag with"
                " with_settings(unphased_union=False)."
            )

        if self.sequence_type == "variant-windows":
```

- [ ] **Step 4: Add the guard to `with_seqs`**

This covers the "flag first, then with_seqs" ordering (`with_seqs` does not call `_check_valid_state`). In `python/genvarloader/_dataset/_impl.py`, find the end of `with_seqs`:

```python
        new_recon = _build_reconstructor(new_seqs, self._tracks, kind)
        return replace(self, _seqs=new_seqs, _seqs_kind=kind, _recon=new_recon)
```

Replace with:

```python
        if (
            kind in ("haplotypes", "annotated")
            and isinstance(new_seqs, Haps)
            and new_seqs.unphased_union
        ):
            raise ValueError(
                "unphased_union is incompatible with 'haplotypes'/'annotated' output"
                " (a union of phased sequences is ill-defined). Use 'variant-windows'"
                " or 'variants', or clear the flag with"
                " with_settings(unphased_union=False)."
            )
        new_recon = _build_reconstructor(new_seqs, self._tracks, kind)
        return replace(self, _seqs=new_seqs, _seqs_kind=kind, _recon=new_recon)
```

- [ ] **Step 5: Run the Task-3 tests to verify they pass**

Run: `pixi run -e dev pytest tests/dataset/test_unphased_union.py -k "phased or union_allows" -v`
Expected: PASS (5 passed: 2 + 2 parametrized + 1).

- [ ] **Step 6: Commit**

```bash
rtk git add python/genvarloader/_dataset/_impl.py tests/dataset/test_unphased_union.py
rtk git commit -m "feat(dataset): reject haplotypes/annotated output under unphased_union (#222)"
```

---

### Task 4: Implement the union fold in `get_variants_flat`

This is the core change. The variant decode gathers rows in C-order `(b, ploidy)` with `row_offsets` of length `b*ploidy + 1`. Folding = keep every `ploidy`-th offset (`row_offsets[::ploidy]`), giving `b + 1` offsets that span both haplotypes' variants per region/sample. `v_idxs` is untouched (hap-0's calls then hap-1's, concatenated). Output ploidy axis becomes `1`. This path serves **both** flat `variants` and `variant-windows`, and the ragged `variants` output (which decodes flat then converts) — so all variant outputs honor the flag.

**Files:**
- Modify: `python/genvarloader/_dataset/_flat_variants.py` (`get_variants_flat` ~lines 709-828)
- Test: `tests/dataset/test_unphased_union.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/dataset/test_unphased_union.py`:

```python
def _windows_ds(snap_dataset, union: bool):
    ds = snap_dataset.with_tracks(False).with_seqs("variants")
    if union:
        ds = ds.with_settings(unphased_union=True)
    return ds.with_output_format("flat").with_seqs(
        "variant-windows",
        VarWindowOpt(flank_length=4, token_alphabet=b"ACGT", unknown_token=4),
    )


def test_variant_windows_union_collapses_ploidy_axis(snap_dataset):
    ploidy = snap_dataset._seqs.genotypes.shape[-2]
    assert ploidy == 2

    u = _windows_ds(snap_dataset, union=True)
    out = u[[[0, 1]], [[0, 1]]]  # out_reshape == (1, 2)
    # Ploidy axis folded 2 -> 1; scalar field shape (1, 2, 1, None).
    assert out.shape == (1, 2, 1, None)
    # Window buffers carry the extra ragged window axis: (1, 2, 1, None, None).
    assert out.ref_window.shape == (1, 2, 1, None, None)
    assert out.alt_window.shape == (1, 2, 1, None, None)
    # to_ragged must still work (offsets/data consistent after the fold).
    out.ref_window.to_ragged()
    out.alt_window.to_ragged()


def test_variant_windows_union_count_matches_sum_over_haplotypes(snap_dataset):
    baseline = _windows_ds(snap_dataset, union=False)
    union = _windows_ds(snap_dataset, union=True)

    r_idx = np.arange(min(4, snap_dataset.shape[0]))
    s_idx = np.arange(snap_dataset.shape[1])

    b = baseline[r_idx, s_idx]  # shape (R, S, ploidy, None)
    u = union[r_idx, s_idx]  # shape (R, S, 1, None)

    # Per (region, sample): total variants across both haplotypes == union count.
    rb = b.fields["start"].to_ragged()  # Ragged (R, S, ploidy, ~v)
    ru = u.fields["start"].to_ragged()  # Ragged (R, S, 1, ~v)
    import awkward as ak

    base_counts = ak.to_numpy(ak.sum(ak.num(rb, axis=-1), axis=-1))  # (R, S)
    union_counts = ak.to_numpy(ak.sum(ak.num(ru, axis=-1), axis=-1))  # (R, S)
    np.testing.assert_array_equal(union_counts, base_counts)


def test_ragged_variants_union_folds(snap_dataset):
    # The ragged "variants" output also honors the flag (decodes flat, then converts).
    u = snap_dataset.with_seqs("variants").with_settings(unphased_union=True)
    out = u[0, 0]  # RaggedVariants
    # alt layout shape is (ploidy, ~v, ~l) for a single (region, sample); ploidy == 1.
    import awkward as ak

    assert len(ak.to_list(out["alt"])) == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest tests/dataset/test_unphased_union.py -k "variant_windows_union or ragged_variants_union" -v`
Expected: FAIL — `assert (1, 2, 2, None) == (1, 2, 1, None)` (ploidy axis not folded).

- [ ] **Step 3: Insert the fold after AF compaction**

In `python/genvarloader/_dataset/_flat_variants.py`, find the AF-compaction block followed by the `shape` assignment:

```python
    # Apply AF compaction to v_idxs / row_offsets / dosage.
    if keep is not None:
        v_idxs, row_offsets = _compact_keep(v_idxs, unfiltered_row_offsets, keep)
        if dosage_data is not None:
            dosage_data, _ = _compact_keep(dosage_data, unfiltered_row_offsets, keep)

    shape: tuple[int | None, ...] = (b, ploidy, None)
```

Replace with:

```python
    # Apply AF compaction to v_idxs / row_offsets / dosage.
    if keep is not None:
        v_idxs, row_offsets = _compact_keep(v_idxs, unfiltered_row_offsets, keep)
        if dosage_data is not None:
            dosage_data, _ = _compact_keep(dosage_data, unfiltered_row_offsets, keep)

    # Unphased ploidy-1 union: fold the C-order (b, ploidy) rows onto b rows by
    # keeping every ploidy-th offset. row_offsets has length b*ploidy + 1, so the
    # slice yields b + 1 offsets that span each region/sample's variants across all
    # stored haplotypes. v_idxs is untouched: hap-0's calls then hap-1's, concatenated
    # (no sort, no dedup; a hom call appears once per haplotype). Safe because the
    # downstream consumer is permutation-invariant (issue #222). eff_ploidy drives the
    # output shape and per-variant contig broadcasting below.
    eff_ploidy = ploidy
    if haps.unphased_union:
        row_offsets = np.ascontiguousarray(row_offsets[::ploidy])
        eff_ploidy = 1

    shape: tuple[int | None, ...] = (b, eff_ploidy, None)
```

- [ ] **Step 4: Use `eff_ploidy` in the variant-windows branch**

In the same function, find the variant-windows branch's contig broadcast and `wshape`:

```python
        regions = np.asarray(regions)
        group_contigs = np.repeat(regions[:, 0], ploidy)
        v_contigs = np.repeat(group_contigs, np.diff(row_offsets))
        wshape = (b, ploidy, None, None)
```

Replace with:

```python
        regions = np.asarray(regions)
        group_contigs = np.repeat(regions[:, 0], eff_ploidy)
        v_contigs = np.repeat(group_contigs, np.diff(row_offsets))
        wshape = (b, eff_ploidy, None, None)
```

- [ ] **Step 5: Use `eff_ploidy` in the ride-along flank-tokens branch**

Still in `get_variants_flat`, find the plain-variants flank-token branch:

```python
        regions = np.asarray(regions)
        group_contigs = np.repeat(regions[:, 0], ploidy)  # (b*p,)
        v_contigs = np.repeat(group_contigs, np.diff(row_offsets))  # (n_var,)
```

Replace with:

```python
        regions = np.asarray(regions)
        group_contigs = np.repeat(regions[:, 0], eff_ploidy)  # (b*p,)
        v_contigs = np.repeat(group_contigs, np.diff(row_offsets))  # (n_var,)
```

- [ ] **Step 6: Run the Task-4 tests to verify they pass**

Run: `pixi run -e dev pytest tests/dataset/test_unphased_union.py -k "variant_windows_union or ragged_variants_union" -v`
Expected: PASS (3 passed).

- [ ] **Step 7: Run the whole union test file**

Run: `pixi run -e dev pytest tests/dataset/test_unphased_union.py -v`
Expected: PASS (all tests).

- [ ] **Step 8: Run the existing flat/variant-windows suites to confirm no regression**

Run: `pixi run -e dev pytest tests/dataset/test_flat_flanks.py tests/unit/dataset/test_flat_variants_type.py tests/dataset/test_flat_mode_equivalence.py -q`
Expected: PASS (no failures). The default `eff_ploidy == ploidy` path is unchanged when the flag is off.

- [ ] **Step 9: Commit**

```bash
rtk git add python/genvarloader/_dataset/_flat_variants.py tests/dataset/test_unphased_union.py
rtk git commit -m "feat(dataset): fold haplotypes into ploidy-1 union in variant decode (#222)"
```

---

### Task 5: Remove the retired germline-CCF inference path

The order-sensitive `infer_germline_ccfs_` / `_infer_germline_ccfs` are retired (simultaneous germline+somatic, unused ~1 year) and are the only consumers that assumed start-ordering. Removing them prevents misuse against the unsorted union and is endorsed by the spec.

**Files:**
- Modify: `python/genvarloader/_dataset/_rag_variants.py` (delete method ~lines 271-298 and function ~lines 578-678)
- Modify: `tests/unit/ragged/test_rag_variants.py` (delete CCF cases + test + import, keep `rc_*` / `test_rc`)

- [ ] **Step 1: Confirm there are no remaining call sites**

Run: `rtk grep "infer_germline_ccfs" python/`
Expected: no matches in `python/` after this task (currently only the definitions themselves; nothing in the read path calls them).

- [ ] **Step 2: Delete the `infer_germline_ccfs_` method**

In `python/genvarloader/_dataset/_rag_variants.py`, delete the entire method (from its decorator-free `def` line through its `return self`). Find and remove:

```python
    def infer_germline_ccfs_(
        self, ccf_field: str = "dosages", max_ccf: float = 1.0
    ) -> Self:
        """Infer germline CCFs in-place.

        Germline variants are identified by having missing CCFs i.e. they have a variant
        index but missing CCFs. Missing CCFs are inferred to be :code:`max_ccf` - sum(overlapping CCFs).

        Parameters
        ----------
        max_ccf
            Maximum CCF value.
        """
        if not hasattr(self, ccf_field):
            raise ValueError(f"Cannot infer germline CCFs without {ccf_field}.")

        ccfs = self[ccf_field]
        if not isinstance(ccfs, Ragged) or not is_rag_dtype(ccfs, DOSAGE_TYPE):
            raise ValueError(f"{ccf_field} must be a Ragged array of {DOSAGE_TYPE}.")

        _infer_germline_ccfs(
            ccfs.data,
            self.start.offsets,
            self.start.data,
            self.ilen.data,
            max_ccf=max_ccf,
        )
        return self
```

(Leave the surrounding methods `squeeze` above and `to_packed` below intact.)

- [ ] **Step 3: Delete the `_infer_germline_ccfs` numba kernel**

In the same file, delete the entire `@nb.njit(...)`-decorated `_infer_germline_ccfs` function — from the decorator line through the end of the function (it is the last function in the file). Find and remove the block beginning:

```python
@nb.njit(parallel=True, nogil=True, cache=True)
def _infer_germline_ccfs(
    ccfs: NDArray[DOSAGE_TYPE],
    v_offsets: NDArray[OFFSET_TYPE],
    v_starts: NDArray[POS_TYPE],
    ilens: NDArray[np.int32],
    max_ccf: float = 1.0,
):
```

…through its final lines:

```python
            else:
                # sign of pos, with 0 being positive
                running_ccf += (2 * (pos >= 0) - 1) * pos_ccf
```

- [ ] **Step 4: Remove the test cases, test, and import**

In `tests/unit/ragged/test_rag_variants.py`:

(a) Delete the import line:

```python
from genvarloader._dataset._rag_variants import _infer_germline_ccfs
```

(b) Delete all six `ccfs_*` case functions (`ccfs_no_overlaps`, `ccfs_no_germs`, `ccfs_all_nonoverlap_germs`, `ccfs_overlap_som`, `ccfs_overlap_germ`, `ccfs_spanning_del`) and the `test_infer_germ_ccfs` function (everything from `def ccfs_no_overlaps():` through the end of `test_infer_germ_ccfs`, i.e. up to but not including `def _bpv(`).

(c) The `rc_*` cases use `POS_TYPE` and `lengths_to_offsets`; `OFFSET_TYPE` is now unused. Change the import:

```python
from seqpro.rag import OFFSET_TYPE, lengths_to_offsets
```

to:

```python
from seqpro.rag import lengths_to_offsets
```

- [ ] **Step 5: Fix any now-unused imports in `_rag_variants.py`**

Run ruff to detect imports left unused by the deletions (e.g. `is_rag_dtype`, `DOSAGE_TYPE`, `OFFSET_TYPE`, `POS_TYPE`, `nb`, `V_IDX_TYPE` — only those NOT used elsewhere in the file):

Run: `pixi run -e dev ruff check python/genvarloader/_dataset/_rag_variants.py`
Expected: either clean, or `F401` unused-import warnings. Remove exactly the imports ruff flags as unused, then re-run until clean.

- [ ] **Step 6: Verify the kept tests still pass and nothing imports the removed symbols**

Run: `pixi run -e dev pytest tests/unit/ragged/test_rag_variants.py -v`
Expected: PASS (`test_rc` cases only).

Run: `pixi run -e dev python -c "import genvarloader; from genvarloader._dataset import _rag_variants; assert not hasattr(_rag_variants, '_infer_germline_ccfs'); assert not hasattr(_rag_variants.RaggedVariants, 'infer_germline_ccfs_'); print('ok')"`
Expected: prints `ok`.

- [ ] **Step 7: Commit**

```bash
rtk git add python/genvarloader/_dataset/_rag_variants.py tests/unit/ragged/test_rag_variants.py
rtk git commit -m "refactor(variants): drop retired order-dependent germline-CCF inference (#222)"
```

---

### Task 6: Update the `genvarloader` skill

`unphased_union` is a new public `with_settings` option, so per `CLAUDE.md` the skill MUST be updated.

**Files:**
- Modify: `skills/genvarloader/SKILL.md`

- [ ] **Step 1: Locate the `with_settings` / output-mode documentation**

Run: `rtk grep "with_settings\|variant-windows\|ploidy" skills/genvarloader/SKILL.md`
Expected: line numbers for the `with_settings` options table/section and the variant-windows description.

- [ ] **Step 2: Document `unphased_union`**

Add `unphased_union` to the `with_settings` parameter list/table in `skills/genvarloader/SKILL.md`, with text equivalent to:

> `unphased_union` (bool, default `False`): fold the stored diploid haplotypes onto a single haploid sequence — the union of called ALTs per `(region, sample)`. With it set, `ds.ploidy` and `n_variants(...)` report ploidy `1`, and `"variants"` / `"variant-windows"` output decode at ploidy `1`. Phase is discarded (for unphased somatic calls); ALT occurrences are concatenated across haplotypes with no sort or dedup. Requires genotypes; incompatible with `"haplotypes"`/`"annotated"` output (raises). Intended for haploid somatic modeling.

If the skill has a "Common gotchas" section, add a one-liner: "`unphased_union` + `with_seqs('haplotypes'|'annotated')` raises — it only applies to `variants`/`variant-windows`."

- [ ] **Step 3: Commit**

```bash
rtk git add skills/genvarloader/SKILL.md
rtk git commit -m "docs(skill): document unphased_union option (#222)"
```

---

### Task 7: Full verification

- [ ] **Step 1: Run the union test file plus variant-decode and dataset suites**

Run: `pixi run -e dev pytest tests/dataset/test_unphased_union.py tests/dataset/test_flat_flanks.py tests/unit/dataset/test_flat_variants_type.py tests/unit/ragged/test_rag_variants.py tests/dataset/test_flat_mode_equivalence.py -q`
Expected: all PASS.

- [ ] **Step 2: Lint and type-check the changed files**

Run: `pixi run -e dev ruff check python/`
Expected: clean.

Run: `pixi run -e dev typecheck`
Expected: no new errors in `_impl.py`, `_haps.py`, `_flat_variants.py`, `_rag_variants.py`.

- [ ] **Step 3: Run the broader dataset test suite for regressions**

Run: `pixi run -e dev pytest tests/dataset tests/unit -q`
Expected: all PASS.

- [ ] **Step 4: Final review of acceptance criteria against the spec**

Confirm, by re-reading the spec's acceptance criteria:
1. `n_variants(...).shape[-1] == 1` under the flag — covered by `test_n_variants_collapses_to_union_count`.
2. variant-windows decode at ploidy 1 (`P=1` layout) — covered by `test_variant_windows_union_collapses_ploidy_axis`.
3. union count == sum over haplotypes (naive combine) — covered by `test_variant_windows_union_count_matches_sum_over_haplotypes` and `test_n_variants_collapses_to_union_count`.
4. `haplotypes`/`annotated` raise under the flag — covered by Task 3 tests.
5. `ds.ploidy == 1` — covered by `test_ploidy_reports_one_under_union`.

---

## Self-Review Notes

- **Spec coverage:** API (`with_settings(unphased_union=True)`) → Task 1; scope = all variant outputs (`variant-windows`, flat + ragged `variants`, `n_variants`) → Tasks 2 & 4; ploidy reporting = 1 → Task 2; naive-combine union semantics → Task 4; phased-output guard (error) → Task 3; drop CCF path → Task 5; skill update → Task 6. The spec's optional `Dataset.open` entry point is deliberately deferred (see scoping note) — flagged for the user.
- **Type consistency:** flag named `unphased_union` everywhere; `eff_ploidy` introduced and used consistently in `get_variants_flat` for `shape`, `wshape`, and both `np.repeat(regions[:, 0], …)` calls.
- **No placeholders:** every code step shows exact before/after text.
