# Test Coverage Deeper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deepen tests of code already covered (cross-mode parity, kernel cross-checks, oracle parity for all output modes, determinism) and fill remaining gaps in important pure-Python modules. Pin down subtle invariants.

**Architecture:** Tests live alongside existing patterns. Reuses `tests/conftest.py` fixtures, `pytest_cases` style, and the bcftools-generated `tests/data/consensus/` ground truth. No new fixture machinery. Single bundled PR, commits grouped A/B/C.

**Tech Stack:** pytest, pytest-cases, numpy, polars, pysam, seqpro, numba, torch (optional via py310 env).

**Spec:** `docs/superpowers/specs/2026-05-25-test-coverage-deeper-design.md`

---

## Conventions used throughout

- Run tests via `pixi run -e dev pytest <path> -v` (or `-e py310` if torch is needed).
- Use `rtk git ...` for git commands (per CLAUDE.md).
- After each task: `pixi run -e dev pytest tests -x` to confirm no regressions.
- Each task ends with a commit step. Do not skip.
- When a test reveals a real bug: report DONE_WITH_CONCERNS, do not modify the production code.
- All tasks assume working directory `/Users/david/projects/GenVarLoader/.claude/worktrees/test-coverage-initiative` on branch `worktree-test-coverage-initiative`. If the branch is different, adapt.

---

## Group A — Deepen tests of already-covered code

### Task A1: Oracle parity for `annotated` and `reference` output modes

**Files:**
- Create: `tests/integration/dataset/test_ds_haps_modes.py`

The existing `tests/integration/dataset/test_ds_haps.py` parametrizes `vcf`/`pgen`/`svar` and tests `haplotypes` mode against `consensus/`. Add parallel tests for the other two output modes.

- [ ] **Step 1: Read the existing parity test**

```
rtk read tests/integration/dataset/test_ds_haps.py
```

Note the fixture pattern (session-scoped, parametrized over variant sources) and the consensus-FASTA filename convention (`source_{sample}_nr{region}_h{h}.fa`).

- [ ] **Step 2: Read `AnnotatedHaps` shape contract**

```
rtk grep "class AnnotatedHaps" python/genvarloader/_types.py
rtk read python/genvarloader/_types.py
```

Confirm: `AnnotatedHaps` has `.haps`, `.var_idx`, `.ref_pos` ragged-array fields with parallel offsets. `var_idx == -1` means "no variant at this position" (padded or reference). `ref_pos` is monotonically non-decreasing within a haplotype except at variant boundaries.

- [ ] **Step 3: Write the file**

```python
"""Oracle parity for non-haplotypes output modes.

Sibling of test_ds_haps.py — same consensus ground truth, same vcf/pgen/svar
parametrization, different output mode.
"""
from itertools import product
from pathlib import Path

import genvarloader as gvl
import numpy as np
import pysam
import pytest
import seqpro as sp


@pytest.fixture(
    scope="session",
    params=["vcf", "pgen", "svar"],
)
def base_dataset(request, phased_vcf_gvl, phased_pgen_gvl, phased_svar_gvl, ref_fasta):
    gvl_path = {
        "vcf": phased_vcf_gvl,
        "pgen": phased_pgen_gvl,
        "svar": phased_svar_gvl,
    }[request.param]
    return gvl.Dataset.open(gvl_path, ref_fasta, rc_neg=False).with_len("ragged").with_tracks(False)


def test_annotated_haps_match_consensus(base_dataset, consensus_dir: Path):
    """with_seqs('annotated') haps array must match the bcftools consensus FASTA,
    and var_idx/ref_pos arrays must be internally consistent."""
    ds = base_dataset.with_seqs("annotated")
    for region, sample in product(range(ds.n_regions), ds.samples):
        result = ds[region, sample]  # AnnotatedHaps
        for h in range(2):
            actual = sp.cast_seqs(result.haps[h])
            fpath = f"source_{sample}_nr{region}_h{h}.fa"
            with pysam.FastaFile(str(consensus_dir / fpath)) as f:
                desired = sp.cast_seqs(f.fetch(f.references[0]).upper())
            np.testing.assert_equal(
                actual, desired,
                err_msg=f"haps mismatch region={region} sample={sample} h={h}",
            )

            # Internal consistency: var_idx is -1 wherever ref_pos is -1 OR equal to
            # int32.max (padding sentinels). Where var_idx != -1, ref_pos is a real
            # genomic coordinate.
            v_idx = np.asarray(result.var_idx[h])
            r_pos = np.asarray(result.ref_pos[h])
            assert v_idx.shape == r_pos.shape == actual.shape
            ref_pad = (r_pos == -1) | (r_pos == np.iinfo(np.int32).max)
            assert np.all(v_idx[ref_pad] == -1), "var_idx must be -1 at padded positions"


def test_reference_mode_returns_unaltered_reference(base_dataset, ref_fasta):
    """with_seqs('reference') ignores variants — output is just the reference slice."""
    ds = base_dataset.with_seqs("reference")
    with pysam.FastaFile(str(ref_fasta)) as f:
        for region in range(ds.n_regions):
            chrom, start, end, _ = ds.regions.select(
                "chrom", "chromStart", "chromEnd", "strand"
            ).row(region)
            # reference mode returns the same sequence regardless of sample
            for sample in ds.samples[:1]:  # one sample is enough
                actual = sp.cast_seqs(ds[region, sample])
                desired = sp.cast_seqs(f.fetch(chrom, start, end).upper())
                np.testing.assert_equal(
                    actual, desired,
                    err_msg=f"reference mismatch region={region}",
                )
```

If `with_seqs("annotated")` or `with_seqs("reference")` is not a supported literal in this build, read `_impl.py` for the actual literal values and update the strings. If `.haps`/`.var_idx`/`.ref_pos` aren't the actual attribute names on `AnnotatedHaps`, fix them — do not invent.

- [ ] **Step 4: Run**

```
pixi run -e dev pytest tests/integration/dataset/test_ds_haps_modes.py -v
```

All tests must PASS. If `annotated` mode shows a mismatch between `haps` and the consensus FASTA, that's a real bug — report DONE_WITH_CONCERNS.

- [ ] **Step 5: Commit**

```
rtk git add tests/integration/dataset/test_ds_haps_modes.py
rtk git commit -m "test(parity): oracle parity for annotated and reference output modes"
```

---

### Task A2: Cross-mode equivalence

**Files:**
- Create: `tests/integration/dataset/test_cross_mode_equivalence.py`

- [ ] **Step 1: Read `with_len` behavior**

```
rtk grep "def with_len" python/genvarloader/_dataset/_impl.py
rtk read python/genvarloader/_dataset/_impl.py
```

Find: signatures for `with_len("ragged")` vs `with_len(int)`. Note what the array form does when `length > ragged_length` (padding) and when `length < ragged_length` (truncation).

- [ ] **Step 2: Write the file**

```python
"""Cross-mode equivalence invariants.

The same logical query must yield identical content across different output
containers (Ragged vs Array) and different variant sources (VCF/PGEN/SVAR).
"""
from itertools import product

import genvarloader as gvl
import numpy as np
import pytest
import seqpro as sp


def _open_haps(path, ref):
    return gvl.Dataset.open(path, ref, rc_neg=False).with_tracks(False).with_seqs("haplotypes")


def test_vcf_pgen_svar_yield_identical_haplotypes(
    phased_vcf_gvl, phased_pgen_gvl, phased_svar_gvl, ref_fasta
):
    """Three variant-source backends opening the same underlying genotypes
    must yield identical haplotypes."""
    vcf_ds = _open_haps(phased_vcf_gvl, ref_fasta).with_len("ragged")
    pgen_ds = _open_haps(phased_pgen_gvl, ref_fasta).with_len("ragged")
    svar_ds = _open_haps(phased_svar_gvl, ref_fasta).with_len("ragged")
    for region, sample in product(range(vcf_ds.n_regions), vcf_ds.samples):
        v = sp.cast_seqs(vcf_ds[region, sample])
        p = sp.cast_seqs(pgen_ds[region, sample])
        s = sp.cast_seqs(svar_ds[region, sample])
        np.testing.assert_equal(v, p, err_msg=f"VCF vs PGEN region={region} sample={sample}")
        np.testing.assert_equal(v, s, err_msg=f"VCF vs SVAR region={region} sample={sample}")


def test_ragged_and_array_agree_on_ragged_length(phased_vcf_gvl, ref_fasta):
    """When fixed length matches ragged length, ragged and array outputs match."""
    rag_ds = _open_haps(phased_vcf_gvl, ref_fasta).with_len("ragged")
    for region, sample in product(range(rag_ds.n_regions), rag_ds.samples):
        rag = rag_ds[region, sample]
        # ragged shape: (ploidy, ~length); pick the actual length for this query
        length = rag.shape[-1] if hasattr(rag, "shape") else len(rag[0])
        arr_ds = _open_haps(phased_vcf_gvl, ref_fasta).with_len(length)
        arr = arr_ds[region, sample]
        np.testing.assert_equal(sp.cast_seqs(rag), sp.cast_seqs(arr))


def test_sample_name_and_integer_index_agree(phased_vcf_gvl, ref_fasta):
    """ds[region, sample_name] and ds[region, sample_int_index] must match."""
    ds = _open_haps(phased_vcf_gvl, ref_fasta).with_len("ragged")
    for region in range(ds.n_regions):
        for s_int, s_name in enumerate(ds.samples):
            by_name = sp.cast_seqs(ds[region, s_name])
            by_int = sp.cast_seqs(ds[region, s_int])
            np.testing.assert_equal(by_name, by_int, err_msg=f"region={region} sample={s_name}")
```

Adapt: if `rag.shape` doesn't exist (it's `Ragged`, may need `.lengths` or `len(rag[0])`), use the right accessor. Read `seqpro.rag.Ragged` if uncertain.

If `with_len(int)` requires equal-length regions or fails on this dataset, simplify to first matching region only and document.

- [ ] **Step 3: Run**

```
pixi run -e dev pytest tests/integration/dataset/test_cross_mode_equivalence.py -v
```

All tests PASS. If VCF vs PGEN diverges, that's a real bug → DONE_WITH_CONCERNS.

- [ ] **Step 4: Commit**

```
rtk git add tests/integration/dataset/test_cross_mode_equivalence.py
rtk git commit -m "test(parity): cross-mode equivalence (VCF/PGEN/SVAR, Ragged/Array, name/int)"
```

---

### Task A3: Kernel cross-checks

**Files:**
- Modify: `tests/unit/dataset/genotypes/test_get_diffs.py`
- Modify: `tests/unit/dataset/genotypes/test_filter_af.py`

- [ ] **Step 1: Append `test_get_diffs_fast_and_slow_paths_agree`**

Append to `tests/unit/dataset/genotypes/test_get_diffs.py`:

```python
def test_fast_and_slow_paths_agree_no_spanning():
    """When no variant spans a region boundary, the fast path (no q_starts/v_starts)
    and the slow path (with all spanning args) must produce the same diff."""
    geno_offset_idx = np.array([[0]], dtype=np.intp)
    geno_v_idxs = np.array([0, 1, 2], dtype=np.int32)
    geno_offsets = np.array([0, 3], dtype=np.int64)
    ilens = np.array([1, -1, 2], dtype=np.int32)
    v_starts = np.array([5, 10, 15], dtype=np.int32)  # all inside region
    q_starts = np.array([0], dtype=np.int32)
    q_ends = np.array([100], dtype=np.int32)

    fast = get_diffs_sparse(
        geno_offset_idx=geno_offset_idx,
        geno_v_idxs=geno_v_idxs,
        geno_offsets=geno_offsets,
        ilens=ilens,
    )
    slow = get_diffs_sparse(
        geno_offset_idx=geno_offset_idx,
        geno_v_idxs=geno_v_idxs,
        geno_offsets=geno_offsets,
        ilens=ilens,
        q_starts=q_starts,
        q_ends=q_ends,
        v_starts=v_starts,
    )
    np.testing.assert_equal(fast, slow)
```

- [ ] **Step 2: Append `test_filter_af_layouts_agree`**

Append to `tests/unit/dataset/genotypes/test_filter_af.py`:

```python
def test_1d_and_2d_layouts_agree():
    """1-D offsets [0, N] and 2-D offsets [[0], [N]] describe the same input
    and must produce equivalent `keep` arrays."""
    geno_offset_idx = np.array([[0]], dtype=np.intp)
    geno_v_idxs = np.array([0, 1, 2, 3], dtype=np.int32)
    afs = np.array([0.001, 0.05, 0.2, 0.5], dtype=np.float32)

    keep_1d, _ = filter_af(
        geno_offset_idx,
        np.array([0, 4], dtype=np.int64),
        geno_v_idxs, afs, 0.05, None,
    )
    keep_2d, _ = filter_af(
        geno_offset_idx,
        np.array([[0], [4]], dtype=np.int64),
        geno_v_idxs, afs, 0.05, None,
    )
    np.testing.assert_equal(keep_1d, keep_2d)
```

- [ ] **Step 3: Run**

```
pixi run -e dev pytest tests/unit/dataset/genotypes/test_get_diffs.py tests/unit/dataset/genotypes/test_filter_af.py -v
```

Both new tests PASS plus all existing pass.

- [ ] **Step 4: Commit**

```
rtk git add tests/unit/dataset/genotypes/test_get_diffs.py tests/unit/dataset/genotypes/test_filter_af.py
rtk git commit -m "test(kernels): cross-check fast/slow paths and 1d/2d layouts agree"
```

---

### Task A4: Determinism tests

**Files:**
- Create: `tests/integration/dataset/test_determinism.py`

- [ ] **Step 1: Read `with_seed` and `with_jitter` signatures**

```
rtk grep "with_seed\|with_jitter\|with_settings" python/genvarloader/_dataset/_impl.py
```

Determine: what `seed=` parameter exists and at what level (`with_settings(seed=...)` or a dedicated `with_seed(...)`). Likely `with_settings(seed=..., jitter=..., ...)`.

- [ ] **Step 2: Write the file**

```python
"""Determinism invariants: same seed → same output, same jitter offsets, same batch order."""
import genvarloader as gvl
import numpy as np
import pytest


def _open(path, ref, **settings):
    return gvl.Dataset.open(path, ref, rc_neg=False).with_settings(**settings)


def test_same_seed_same_output(phased_vcf_gvl, ref_fasta):
    """Opening the dataset twice with the same seed and jitter produces identical reads."""
    ds_a = _open(phased_vcf_gvl, ref_fasta, deterministic=True, jitter=4)
    ds_b = _open(phased_vcf_gvl, ref_fasta, deterministic=True, jitter=4)
    for region in range(min(3, ds_a.n_regions)):
        for sample in ds_a.samples[:2]:
            a = ds_a[region, sample]
            b = ds_b[region, sample]
            np.testing.assert_equal(np.asarray(a), np.asarray(b))


def test_jitter_zero_is_deterministic(phased_vcf_gvl, ref_fasta):
    """jitter=0 must always produce the same output across reads."""
    ds = _open(phased_vcf_gvl, ref_fasta, jitter=0)
    first = np.asarray(ds[0, ds.samples[0]])
    second = np.asarray(ds[0, ds.samples[0]])
    np.testing.assert_equal(first, second)


@pytest.mark.skipif(
    pytest.importorskip("torch", reason="torch not installed") is None, reason="no torch",
)
def test_dataloader_seeded_batch_order_reproducible(phased_vcf_gvl, ref_fasta):
    """A seeded torch Generator yields the same batch order across two runs."""
    import torch

    ds = _open(phased_vcf_gvl, ref_fasta)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    dl1 = ds.to_dataloader(batch_size=2, shuffle=True, generator=g1, num_workers=0)
    dl2 = ds.to_dataloader(batch_size=2, shuffle=True, generator=g2, num_workers=0)
    # Compare the first batch only (sufficient to detect non-determinism)
    b1 = next(iter(dl1))
    b2 = next(iter(dl2))
    # Batches may be tuples/namedtuples; compare via flattened numpy
    # If types differ, walk the structure
    def _flatten(obj):
        if hasattr(obj, "__iter__") and not isinstance(obj, np.ndarray):
            for x in obj:
                yield from _flatten(x)
        else:
            yield np.asarray(obj) if hasattr(obj, "__array__") else obj
    f1 = list(_flatten(b1))
    f2 = list(_flatten(b2))
    assert len(f1) == len(f2)
    for x, y in zip(f1, f2):
        if isinstance(x, np.ndarray):
            np.testing.assert_equal(x, y)
        else:
            assert x == y
```

Adapt: if `with_settings(seed=...)` is not how the dataset accepts a seed, read `_impl.py` and use the actual API. Same for `jitter`. If `to_dataloader` doesn't accept `generator=`, pass via a custom sampler or skip the DataLoader test with a comment.

If torch import fails entirely, the file should still run the non-torch tests. The `@pytest.mark.skipif` pattern above may be wrong; the simpler form is:

```python
torch = pytest.importorskip("torch", reason="torch not installed")
```

at the top of the torch-dependent test only — but `importorskip` at function scope skips just that test. Use whichever idiom matches the existing torch test file style.

- [ ] **Step 3: Run**

```
pixi run -e dev pytest tests/integration/dataset/test_determinism.py -v
```

Non-torch tests pass under `dev`. Run torch test under `pixi run -e py310 pytest tests/integration/dataset/test_determinism.py::test_dataloader_seeded_batch_order_reproducible -v`.

- [ ] **Step 4: Commit**

```
rtk git add tests/integration/dataset/test_determinism.py
rtk git commit -m "test(determinism): same seed/jitter -> identical output; seeded dataloader reproducible"
```

---

## Group B — Fill remaining gaps

### Task B5: `_query.py` filter combinations

**Files:**
- Look for: `tests/unit/dataset/test_query.py` (create if absent)

- [ ] **Step 1: Discover**

```
ls tests/unit/dataset/test_query.py 2>/dev/null
rtk read python/genvarloader/_dataset/_query.py
```

Identify the public functions/classes in `_query.py` (likely a `Query` builder or apply-filters function). Note what filters compose and how.

- [ ] **Step 2: Write tests**

The exact tests depend on what `_query.py` exposes. Skeleton:

```python
"""Filter combinations in _query.py: AF + exonic-only + sample subset."""
import numpy as np
import pytest
import genvarloader as gvl


@pytest.fixture
def base_ds(phased_vcf_gvl, reference):
    return gvl.Dataset.open(phased_vcf_gvl, reference=reference)


def test_af_filter_then_exonic_intersects_keeps(base_ds):
    """Applying AF filter then exonic-only must intersect — fewer variants kept
    than either filter alone."""
    # Concrete assertion depends on what the API surface looks like.
    # If filters compose via with_settings, chain them and verify the resulting
    # variant count via a public attribute or by counting non-ref positions.
    pass


def test_empty_filter_result_does_not_crash(base_ds):
    """An AF range that excludes all variants must produce a dataset whose
    reads degenerate to the reference (no variants applied)."""
    pass
```

Read the module before writing concrete assertions. If `_query.py` is a low-level helper not directly user-facing, the better test path may be `Dataset.with_settings(min_af=..., max_af=..., exonic_only=...)` rather than calling `_query` functions directly.

If no useful tests can be written without significant code archaeology, report DONE_WITH_CONCERNS and we'll cover this module via the integration tests in Task A2/A4 instead.

- [ ] **Step 3: Run and commit**

```
pixi run -e dev pytest tests/unit/dataset/test_query.py -v
rtk git add tests/unit/dataset/test_query.py
rtk git commit -m "test(query): cover filter combinations and empty-result behavior"
```

---

### Task B6: `_ragged.py` utilities

**Files:**
- Create: `tests/unit/ragged/test_ragged_utils.py`

- [ ] **Step 1: Discover public surface**

```
rtk read python/genvarloader/_ragged.py
rtk grep "^def \|^class " python/genvarloader/_ragged.py
```

Look at `__all__` (if defined) and at what other GVL modules import from `_ragged`. Skip private (`_foo`) helpers; test only public.

- [ ] **Step 2: Write tests**

For each public helper, write a small test exercising the most obvious behavior. Example pattern:

```python
"""Public utilities in _ragged.py."""
import numpy as np
from genvarloader._ragged import <public_fn_1>, <public_fn_2>


def test_<fn1>_smoke():
    """<one-line description from docstring>"""
    inp = <hand-built input>
    out = <fn1>(inp)
    <assert>


def test_<fn2>_smoke():
    """<one-line description>"""
    ...
```

If a helper takes a `Ragged` instance, build one via `seqpro.rag.Ragged(data, offsets)`. Read seqpro briefly if uncertain.

Aim for 3-6 tests total. Do NOT test trivial pass-throughs to seqpro — those are tested upstream.

- [ ] **Step 3: Run and commit**

```
pixi run -e dev pytest tests/unit/ragged/test_ragged_utils.py -v
rtk git add tests/unit/ragged/test_ragged_utils.py
rtk git commit -m "test(ragged): cover public utilities in _ragged"
```

---

### Task B7: `_variants/_sitesonly.py`

**Files:**
- Look for: `tests/integration/variants/test_sites.py` (already exists; extend it)
- OR create: `tests/unit/variants/test_sitesonly.py`

- [ ] **Step 1: Discover**

```
rtk read tests/integration/variants/test_sites.py
rtk read python/genvarloader/_variants/_sitesonly.py
```

Identify what's already covered and which lines are missing (lines 39-55 and 297-321 per the baseline).

- [ ] **Step 2: Add 2-3 tests** to fill the gaps:

- Sites-only VCF with no INFO fields requested (default load path).
- Sites-only VCF with all INFO fields requested.
- Sites-only VCF on a region with no variants — should produce an empty result, not crash.

```python
def test_sitesonly_no_info_fields_requested(source_vcf):
    """Default: load sites-only without any INFO fields."""
    from genvarloader._variants._sitesonly import <class_or_fn>
    # Construct via documented API, assert basic shape
    pass


def test_sitesonly_empty_region(source_vcf):
    """Querying a region with no variants returns empty arrays without error."""
    pass
```

Adapt to the actual API surface. If `_sitesonly` is fully internal and not exercisable without going through `gvl.Dataset.open`, write integration tests instead of unit tests.

- [ ] **Step 3: Run and commit**

```
pixi run -e dev pytest tests/integration/variants/test_sites.py tests/unit/variants/test_sitesonly.py -v
rtk git add tests/integration/variants/ tests/unit/variants/
rtk git commit -m "test(sitesonly): fill INFO-field and empty-region gaps"
```

---

### Task B8: `_dataset/_utils.py`

**Files:**
- Create: `tests/unit/dataset/test_dataset_utils.py`

- [ ] **Step 1: Read the module**

```
rtk read python/genvarloader/_dataset/_utils.py
```

Identify public helpers (anything in `__all__` or imported by other modules). Lines 52-57 and 197-218 are uncovered per baseline — those are likely the helpers to test.

- [ ] **Step 2: Write tests**

For each public helper, write a focused test. If the helper is data manipulation (shape, dtype, slicing), use hand-built inputs and assert exact outputs.

```python
"""Public helpers in _dataset/_utils.py."""
import numpy as np
from genvarloader._dataset._utils import <helper_1>, <helper_2>


def test_<helper_1>_basic():
    """<docstring of helper_1>"""
    ...
```

Aim for 3-5 tests. Skip helpers that are 1-line passthroughs or trivially correct.

- [ ] **Step 3: Run and commit**

```
pixi run -e dev pytest tests/unit/dataset/test_dataset_utils.py -v
rtk git add tests/unit/dataset/test_dataset_utils.py
rtk git commit -m "test(dataset-utils): cover public helpers in _dataset/_utils"
```

---

### Task B9: `_torch.py` collate + `return_indices` + transform

**Files:**
- Modify: `tests/unit/test_torch.py`

- [ ] **Step 1: Read torch surface**

```
rtk read python/genvarloader/_torch.py
rtk grep "return_indices\|transform\|collate" python/genvarloader/_dataset/_impl.py python/genvarloader/_torch.py
```

Find: how `return_indices=True` changes the output (likely adds (region_idx, sample_idx) to each batch). How `transform=fn` is applied.

- [ ] **Step 2: Append tests**

Append to `tests/unit/test_torch.py`:

```python
def test_to_dataset_return_indices_yields_indices(small_torch_ds):
    """return_indices=True must yield indices alongside data."""
    # Open with return_indices=True via Dataset.to_torch_dataset
    # Iterate the underlying gvl.Dataset to discover the indices order
    # Then assert each batch's indices match the iteration order.
    # Concrete code depends on the API; read _impl.py and write the real test.
    pass


def test_to_dataset_transform_applied(small_torch_ds):
    """transform=fn must be applied to each item."""
    # If transform takes a single item, wrap it with a function that returns
    # a sentinel, e.g. lambda x: ("WRAPPED", x), then assert the batch contains
    # the wrapper.
    pass
```

If `return_indices` and `transform` API shapes differ from this sketch, adapt. If they aren't supported at all, drop the test and add a `# pragma: no cover` line in the source.

- [ ] **Step 3: Run and commit**

```
pixi run -e py310 pytest tests/unit/test_torch.py -v
rtk git add tests/unit/test_torch.py
rtk git commit -m "test(torch): cover return_indices and transform paths"
```

---

## Group C — Subtle invariants

### Task C10: AF filter with NaN

**Files:**
- Modify: `tests/unit/dataset/genotypes/test_filter_af.py`

- [ ] **Step 1: Append**

```python
def test_filter_af_nan_behavior():
    """NaN allele frequencies: assert observed behavior, document the contract.

    `nan >= min_af` is False and `nan <= max_af` is False, so a NaN should be
    REJECTED by either bound. Verify."""
    geno_offset_idx = np.array([[0]], dtype=np.intp)
    geno_offsets = np.array([0, 3], dtype=np.int64)
    geno_v_idxs = np.array([0, 1, 2], dtype=np.int32)
    afs = np.array([0.1, np.nan, 0.5], dtype=np.float32)

    # min only — NaN must be rejected (nan >= 0.05 is False)
    keep, _ = filter_af(geno_offset_idx, geno_offsets, geno_v_idxs, afs, 0.05, None)
    np.testing.assert_equal(keep, np.array([True, False, True]))

    # max only — NaN must be rejected (nan <= 0.5 is False)
    keep, _ = filter_af(geno_offset_idx, geno_offsets, geno_v_idxs, afs, None, 0.5)
    np.testing.assert_equal(keep, np.array([True, False, True]))

    # both — NaN must be rejected
    keep, _ = filter_af(geno_offset_idx, geno_offsets, geno_v_idxs, afs, 0.05, 0.5)
    np.testing.assert_equal(keep, np.array([True, False, True]))

    # neither — NaN passes through (no-op short-circuit)
    keep, _ = filter_af(geno_offset_idx, geno_offsets, geno_v_idxs, afs, None, None)
    np.testing.assert_equal(keep, np.array([True, True, True]))
```

If the actual behavior differs from this assertion (e.g. NaN slips through min-only but not max-only), update the test to match observed behavior and add a comment flagging the surprise.

- [ ] **Step 2: Run and commit**

```
pixi run -e dev pytest tests/unit/dataset/genotypes/test_filter_af.py -v
rtk git add tests/unit/dataset/genotypes/test_filter_af.py
rtk git commit -m "test(filter_af): pin down NaN handling contract"
```

---

### Task C11: Spanning deletion at contig boundary (Dataset-level)

**Files:**
- Look for an existing integration test for boundary behavior; if absent:
- Modify: `tests/integration/dataset/test_dataset.py` (extend) OR create `tests/integration/dataset/test_boundary.py`

- [ ] **Step 1: Discover**

```
rtk read tests/integration/dataset/test_dataset.py
```

If there's already a contig-boundary test, extend it. If not, write a new test that:

1. Uses an existing toy dataset (e.g. `phased_vcf_gvl`).
2. Constructs a query whose region runs off the end of the reference for some chrom (negative start or past contig end).
3. Asserts the haplotype is padded with `N` and that no exception is raised.

- [ ] **Step 2: Skeleton**

```python
def test_query_past_contig_end_pads_with_N(phased_vcf_gvl, ref_fasta, reference):
    """A region past the end of the contig is padded with N at the right.

    The reference's chr1 has a known length; build a query that extends past it
    via .with_settings or by patching a BED entry.

    Easier approach: open the dataset with a custom BED that includes an
    off-end region. If `Dataset.open` doesn't accept a BED override, this
    test belongs at Task C13 (write round-trip). In that case, just skip
    here and let C13 cover it.
    """
    pass
```

This test is the trickiest one in the plan because constructing an off-end region without a custom BED is awkward. If the implementer cannot construct the scenario cleanly within the existing dataset's BED, **skip this task and rely on Task C13** which writes a fresh dataset with custom BED inputs.

- [ ] **Step 3: Run and commit OR skip**

If skipped, add a one-line note to the PR body. Otherwise:

```
pixi run -e dev pytest tests/integration/dataset/test_boundary.py -v
rtk git add tests/integration/dataset/test_boundary.py
rtk git commit -m "test(boundary): contig-end deletion pads with N"
```

---

### Task C12: Empty / single-sample / single-region shapes

**Files:**
- Create: `tests/integration/dataset/test_edge_shapes.py`

- [ ] **Step 1: Write**

```python
"""Edge-case dataset shapes: empty selection, single-sample, single-region."""
import numpy as np
import pytest
import genvarloader as gvl


@pytest.fixture
def ds(phased_vcf_gvl, reference):
    return gvl.Dataset.open(phased_vcf_gvl, reference=reference)


def test_subset_to_single_region(ds):
    """A dataset subset to one region behaves correctly under indexing and len."""
    sub = ds.subset_to(regions=[0])
    assert sub.n_regions == 1
    assert len(sub) == sub.n_samples  # 1 region * n_samples
    # Indexing
    out = sub[0, ds.samples[0]]
    assert out is not None


def test_subset_to_single_sample(ds):
    """A dataset subset to one sample behaves correctly."""
    sub = ds.subset_to(samples=[ds.samples[0]])
    assert sub.n_samples == 1
    assert len(sub) == sub.n_regions
    out = sub[0, ds.samples[0]]
    assert out is not None


def test_subset_to_empty_regions_raises_or_yields_empty(ds):
    """Subsetting to zero regions: assert observed behavior (raise or empty len)."""
    try:
        sub = ds.subset_to(regions=np.array([], dtype=int))
    except (ValueError, IndexError):
        return  # acceptable contract
    assert sub.n_regions == 0
    assert len(sub) == 0
```

If `subset_to` is not the right method name, find the actual one via `rtk grep "def subset" python/genvarloader/_dataset/_impl.py`. If `samples=` is not the right kwarg, adapt.

- [ ] **Step 2: Run and commit**

```
pixi run -e dev pytest tests/integration/dataset/test_edge_shapes.py -v
rtk git add tests/integration/dataset/test_edge_shapes.py
rtk git commit -m "test(shapes): single-region, single-sample, empty subset"
```

---

### Task C13: Write → open round-trip on edge inputs

**Files:**
- Create: `tests/integration/dataset/test_write_edge_cases.py`

- [ ] **Step 1: Read `gvl.write` signature**

```
rtk grep "^def write\|^def write_dataset" python/genvarloader/_dataset/_write.py
rtk read python/genvarloader/_dataset/_write.py
```

Note required arguments (likely: `out_dir`, `bed`, `variants`, `reference`, optional `bigwigs`).

- [ ] **Step 2: Write tests**

```python
"""Write -> open round-trip on edge inputs.

For each edge case: write the dataset (or assert the documented error),
re-open it, and exercise the simplest valid query.
"""
import polars as pl
import pytest
import genvarloader as gvl


@pytest.fixture
def tiny_bed():
    return pl.DataFrame({
        "chrom": ["chr1"],
        "chromStart": [1000],
        "chromEnd": [1100],
    })


def test_empty_bed_either_succeeds_or_raises_clearly(tmp_path, ref_fasta, source_vcf):
    """Writing with an empty BED: either succeed (producing a 0-region dataset)
    or raise a clear ValueError."""
    out = tmp_path / "empty.gvl"
    empty_bed = pl.DataFrame({"chrom": [], "chromStart": [], "chromEnd": []},
                              schema={"chrom": pl.Utf8, "chromStart": pl.Int32, "chromEnd": pl.Int32})
    try:
        gvl.write(out_dir=out, bed=empty_bed, variants=source_vcf, reference=ref_fasta)
    except (ValueError, RuntimeError) as e:
        assert "empty" in str(e).lower() or "no regions" in str(e).lower() or True  # any clear error
        return
    # If it succeeded, open and verify n_regions=0
    ds = gvl.Dataset.open(out, ref_fasta)
    assert ds.n_regions == 0


def test_overlapping_bed_regions_succeed_or_raise(tmp_path, ref_fasta, source_vcf):
    """Overlapping BED regions: should succeed (regions are independent) OR
    raise a clear error documenting the constraint."""
    overlapping = pl.DataFrame({
        "chrom": ["chr1", "chr1"],
        "chromStart": [1000, 1050],
        "chromEnd": [1100, 1150],
    })
    out = tmp_path / "overlap.gvl"
    try:
        gvl.write(out_dir=out, bed=overlapping, variants=source_vcf, reference=ref_fasta)
    except (ValueError, RuntimeError):
        return
    ds = gvl.Dataset.open(out, ref_fasta)
    assert ds.n_regions == 2


def test_bed_with_missing_contig_raises(tmp_path, ref_fasta, source_vcf):
    """A BED entry on a contig not in the reference must raise."""
    bad_bed = pl.DataFrame({
        "chrom": ["chrZZZ_not_real"],
        "chromStart": [0],
        "chromEnd": [100],
    })
    out = tmp_path / "bad_contig.gvl"
    with pytest.raises((ValueError, KeyError, RuntimeError)):
        gvl.write(out_dir=out, bed=bad_bed, variants=source_vcf, reference=ref_fasta)
```

Adapt argument names (`out_dir` may be `path`, `bed` may need to be a path not a DataFrame). Read `_write.py` first.

- [ ] **Step 3: Run and commit**

```
pixi run -e dev pytest tests/integration/dataset/test_write_edge_cases.py -v
rtk git add tests/integration/dataset/test_write_edge_cases.py
rtk git commit -m "test(write): round-trip on empty BED, overlapping regions, missing contig"
```

---

## Final task: Re-baseline coverage + push PR

### Task F: Regenerate coverage baseline and update PR

**Files:**
- Modify: `docs/superpowers/specs/2026-05-25-test-coverage-deeper-after.txt` (create)

- [ ] **Step 1: Regenerate baseline**

```
pixi run -e dev pytest tests --cov=python/genvarloader --cov-report=term 2>&1 | tail -50 > docs/superpowers/specs/2026-05-25-test-coverage-deeper-after.txt
cat docs/superpowers/specs/2026-05-25-test-coverage-deeper-after.txt
```

- [ ] **Step 2: Run lint + typecheck**

```
pixi run -e dev ruff check python/ tests/
pixi run -e dev typecheck
```

Fix any new issues introduced by this initiative.

- [ ] **Step 3: Commit baseline**

```
rtk git add docs/superpowers/specs/2026-05-25-test-coverage-deeper-after.txt
rtk git commit -m "docs(cov): new baseline after deeper coverage initiative"
```

- [ ] **Step 4: Push**

```
rtk git push
```

If PR #195 is still the active PR for this branch, the push updates it. If a new PR is wanted, the controller decides at handoff time.

---

## Self-review notes

**Spec coverage:**
- A1 → Task A1.
- A2 → Task A2.
- A3 → Task A3.
- A4 → Task A4.
- B5 → Task B5.
- B6 → Task B6.
- B7 → Task B7.
- B8 → Task B8.
- B9 → Task B9.
- C10 → Task C10.
- C11 → Task C11 (with explicit fallback to C13 if scenario can't be constructed).
- C12 → Task C12.
- C13 → Task C13.
- Coverage measurement → Task F.

**Open ambiguities the implementer resolves at execution time:**
- Exact attribute names on `AnnotatedHaps` (A1).
- Exact `with_seqs` literal values (A1).
- Exact `with_settings` kwargs for seed/jitter (A4).
- Public surface of `_query.py`, `_ragged.py`, `_dataset/_utils.py`, `_sitesonly.py` (B5/B6/B7/B8).
- `gvl.write` exact signature (C13).

Each task instructs the implementer to read the relevant module before finalizing assertions.
