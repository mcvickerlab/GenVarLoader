# Prefetching DataLoader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two new prefetching dataloader modes (`"buffered"` and `"double_buffered"`) to `Dataset.to_dataloader()` that fetch large chunks of `(region, sample)` pairs from gvl in one call and slice them into mini-batches, with mode B running the fetcher in a subprocess via POSIX shared memory.

**Architecture:** A new `Dataset._output_bytes_per_instance()` method computes exact materialized byte cost per `(region, sample)` from `haplotype_lengths`, `n_variants`, and the active output schema. A `ChunkPlanner` walks the BatchSampler-resolved index sequence and groups instances into chunks sized to a per-slot byte budget. `mode="buffered"` runs the producer in the main process synchronously. `mode="double_buffered"` runs a subprocess producer that writes chunks into one of two POSIX shared-memory slots, ping-ponging with the consumer via `multiprocessing.Event`s. A hand-rolled fixed-size header per slot describes array offsets/dtypes; consumer rebuilds `numpy`/`Ragged`/`RaggedVariants` views over `shm.buf`.

**Tech Stack:** Python, NumPy, PyTorch (`torch.utils.data`), `multiprocessing` (`Process`, `Event`, `Queue`, `Pipe`, `shared_memory`), `seqpro.rag.Ragged`, awkward (for `RaggedVariants`), pixi for tasks.

**Spec:** `docs/superpowers/specs/2026-05-28-prefetching-dataloader-design.md`

**Test command:** `pixi run -e dev pytest tests/unit/<file>.py -v`. Test data must exist first — run `pixi run -e dev gen` once before any test. Whole suite: `pixi run -e dev test`.

---

## File structure

**New source files (`python/genvarloader/`):**

- `_chunked.py` — `ChunkPlanner`, `slice_chunk`. Pure logic, no I/O.
- `_buffered_loader.py` — `BufferedTorchDataset` (`mode="buffered"`).
- `_shm_layout.py` — header pack/unpack, `write_chunk`, `read_chunk`. The shm IPC contract.
- `_double_buffered_loader.py` — `DoubleBufferedTorchDataset` (`mode="double_buffered"`).
- `_producer.py` — subprocess entrypoint.

**Modified source files:**

- `_dataset/_impl.py` — add `_output_bytes_per_instance` method.
- `_dataset/_haps.py` — add private `_allele_bytes_sum` helper.
- `_torch.py` — wire `mode`, `buffer_bytes`, `copy`, `heartbeat_seconds` into `get_dataloader`.

**New test files (`tests/unit/`):**

- `test_output_bytes_per_instance.py`
- `test_chunk_planner.py`
- `test_shm_layout.py`
- `test_buffered_loader.py`
- `test_double_buffered_loader.py` (marked `@pytest.mark.slow`)

**Skill file:**

- `skills/genvarloader/SKILL.md` — document new args.

---

## Task 1: Add `Haps._allele_bytes_sum` helper

**Files:**
- Modify: `python/genvarloader/_dataset/_haps.py` (add method to `Haps` class, alongside `_get_variants` at line 634)
- Create: `tests/unit/dataset/test_haps_allele_bytes.py`

`RaggedAlleles` is `Ragged[np.bytes_]` (`_variants/_records.py:13`); it exposes `.offsets` directly. For selected variant indices `v_idxs` from a `Ragged[V_IDX_TYPE]` genotype array, the per-`(instance, ploid)` sum of allele byte lengths is `np.add.reduceat(offsets[v_idxs+1] - offsets[v_idxs], genos.offsets[:-1])`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/dataset/test_haps_allele_bytes.py`:

```python
"""Unit tests for Haps._allele_bytes_sum.

The helper returns the exact total bytes of REF or ALT allele payloads for the
variants selected by each (region, sample, ploid) entry, computed in O(|V|) by
differencing the RaggedAlleles offsets array (no payload read).
"""
import numpy as np
import pytest
import genvarloader as gvl


@pytest.fixture
def ds():
    return gvl.get_dummy_dataset()


def test_allele_bytes_sum_matches_materialized_alt(ds):
    """Sum of returned int64s must equal len of every alt bytestring of every
    selected variant, summed per (region, sample, ploid)."""
    # Use the full dataset's Haps directly.
    haps = ds._seqs
    # Build the (b*p)-shaped idx that _get_variants/_allele_bytes_sum accept.
    idx = np.arange(np.prod(haps.genotypes.shape[:2]))
    got = haps._allele_bytes_sum(idx, "alt")
    # Materialize via _get_variants and sum allele lengths from the awkward layout.
    ragv = haps._get_variants(idx)
    # ragv.alt has shape (b, p, ~v, ~length). Sum length across innermost two ragged dims.
    import awkward as ak
    expected = ak.sum(ak.num(ragv.alt, axis=-1), axis=-1).to_numpy().ravel()
    np.testing.assert_array_equal(got, expected)


def test_allele_bytes_sum_ref(ds):
    """Same invariant for ref."""
    haps = ds._seqs
    if "ref" not in haps.available_var_fields:
        pytest.skip("dummy dataset does not store ref alleles")
    idx = np.arange(np.prod(haps.genotypes.shape[:2]))
    got = haps._allele_bytes_sum(idx, "ref")
    ragv = haps._get_variants(idx)
    import awkward as ak
    expected = ak.sum(ak.num(ragv.ref, axis=-1), axis=-1).to_numpy().ravel()
    np.testing.assert_array_equal(got, expected)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pixi run -e dev gen  # only first time
pixi run -e dev pytest tests/unit/dataset/test_haps_allele_bytes.py -v
```

Expected: FAIL — `AttributeError: 'Haps' object has no attribute '_allele_bytes_sum'`.

- [ ] **Step 3: Implement `_allele_bytes_sum`**

In `python/genvarloader/_dataset/_haps.py`, add a method to the `Haps` class (just after `_get_variants`, around line 694). Pattern matches `_get_variants`'s data access shape.

```python
def _allele_bytes_sum(
    self, idx: NDArray[np.integer], kind: Literal["alt", "ref"]
) -> NDArray[np.int64]:
    """Exact total bytes of the selected variants' `kind` allele payload, per
    instance flattened over ploidy.

    Returns shape (len(idx) * ploidy,) of int64. O(|selected variants|);
    does not touch allele payload bytes — only the RaggedAlleles offsets.
    """
    r, s = np.unravel_index(idx, self.genotypes.shape[:2])  # type: ignore[no-matching-overload]
    genos = self.genotypes[r, s]
    import awkward as ak
    genos = ak.to_packed(genos)
    v_idxs = genos.data

    # Apply the same AF filter _get_variants applies, so byte counts match
    # the materialized RaggedVariants exactly.
    if self.min_af is not None or self.max_af is not None:
        geno_afs = self.variants.info["AF"][v_idxs]
        keep = np.full(len(v_idxs), True, np.bool_)
        if self.min_af is not None:
            keep &= geno_afs >= self.min_af
        if self.max_af is not None:
            keep &= geno_afs <= self.max_af
        _keep = Ragged.from_offsets(keep, genos.shape, genos.offsets)
        genos = Ragged(ak.to_packed(ak.to_regular(genos[_keep], 1)))
        v_idxs = genos.data

    offsets = getattr(self.variants, kind).offsets  # int-typed, length n_variants+1
    v_lens = (offsets[v_idxs + 1] - offsets[v_idxs]).astype(np.int64)
    # genos.offsets has length b*p + 1 (one offset per (instance, ploid) group).
    # reduceat needs starts, not full offsets; drop the last.
    return np.add.reduceat(v_lens, genos.offsets[:-1])
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pixi run -e dev pytest tests/unit/dataset/test_haps_allele_bytes.py -v
```

Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_haps.py tests/unit/dataset/test_haps_allele_bytes.py
rtk git commit -m "feat(haps): add _allele_bytes_sum for exact variant footprint"
```

---

## Task 2: `Dataset._output_bytes_per_instance` — reference and haplotypes modes

**Files:**
- Modify: `python/genvarloader/_dataset/_impl.py` (add method to `Dataset`, near `n_variants` at line 1029)
- Create: `tests/unit/dataset/test_output_bytes_per_instance.py`

Implement the reference and haplotypes branches of the new method. Later tasks add annotated, variants, and tracks branches in the same method.

- [ ] **Step 1: Write the failing test (reference + haplotypes)**

Create `tests/unit/dataset/test_output_bytes_per_instance.py`:

```python
"""Per-mode exact-footprint tests.

Invariant: Dataset._output_bytes_per_instance(r, s) == nbytes of the actual
dataset[r, s] output, summed over arrays returned for that instance.
"""
import numpy as np
import pytest
import genvarloader as gvl
from genvarloader._ragged import RaggedSeqs


def _materialized_nbytes_per_instance(ds, r_arr, s_arr):
    """Compute actual nbytes by indexing the dataset and measuring."""
    out = ds[r_arr, s_arr]
    # Normalize to tuple
    if not isinstance(out, tuple):
        out = (out,)
    # Each ndarray/Ragged contributes its data nbytes per instance. For Ragged,
    # we sum the per-instance data nbytes via the offsets.
    from seqpro.rag import Ragged
    n_inst = len(r_arr)
    totals = np.zeros(n_inst, dtype=np.int64)
    for arr in out:
        if isinstance(arr, Ragged):
            # Ragged.offsets is (n_inst * ploidy + 1,) typically; need to sum across
            # all axes beyond instance. For simplicity, reshape lengths to
            # (n_inst, -1) and sum.
            lens = np.diff(arr.offsets)
            lens = lens.reshape(n_inst, -1)
            totals += lens.sum(-1) * arr.data.dtype.itemsize
        elif isinstance(arr, np.ndarray):
            per = arr.itemsize * int(np.prod(arr.shape[1:]))
            totals += per
        else:
            raise AssertionError(f"unhandled array type {type(arr)}")
    return totals


def test_reference_mode_exact():
    ds = gvl.get_dummy_dataset().with_seqs("reference").with_tracks(False).with_len(8)
    r = np.arange(ds.full_shape[0])
    s = np.zeros(len(r), dtype=np.int64)
    got = ds._output_bytes_per_instance(r, s)
    expected = _materialized_nbytes_per_instance(ds, r, s)
    np.testing.assert_array_equal(got, expected)


def test_haplotypes_mode_exact():
    ds = (
        gvl.get_dummy_dataset()
        .with_seqs("haplotypes")
        .with_tracks(False)
        .with_settings(deterministic=True)
    )
    r = np.arange(ds.full_shape[0])
    s = np.zeros(len(r), dtype=np.int64)
    got = ds._output_bytes_per_instance(r, s)
    expected = _materialized_nbytes_per_instance(ds, r, s)
    np.testing.assert_array_equal(got, expected)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pixi run -e dev pytest tests/unit/dataset/test_output_bytes_per_instance.py -v
```

Expected: FAIL — `AttributeError: 'RaggedDataset' object has no attribute '_output_bytes_per_instance'`.

- [ ] **Step 3: Implement the method (reference + haplotypes branches only)**

In `python/genvarloader/_dataset/_impl.py`, add a method to the `Dataset` class. Place it after `n_variants` (line 1072). Use this scaffold; later tasks fill in the other branches.

```python
def _output_bytes_per_instance(
    self,
    regions: Idx | None = None,
    samples: Idx | str | Sequence[str] | None = None,
) -> NDArray[np.int64]:
    """Exact bytes one (region, sample) instance materializes to under the
    current schema. Shape: (n_regions, n_samples) of int64.

    Raises NotImplementedError for spliced datasets. Raises ValueError for
    non-deterministic datasets when with_seqs is in {"haplotypes", "annotated"}.
    """
    if self._sp_idxer is not None:
        raise NotImplementedError(
            "_output_bytes_per_instance is not implemented for spliced datasets."
        )

    if regions is None:
        regions = slice(None)
    if samples is None:
        samples = slice(None)
    idx = (regions, samples)
    ds_idx, squeeze, out_reshape = self._idxer.parse_idx(idx)
    r_idx, s_idx = np.unravel_index(ds_idx, self.full_shape)

    seq_kind = self.sequence_type  # "reference" | "haplotypes" | "annotated" | "variants" | None
    total = np.zeros(len(r_idx), dtype=np.int64)

    # --- seqs payload ---
    if seq_kind == "reference":
        # region length × itemsize, no ploidy expansion.
        regions_arr = self._full_regions[r_idx].copy()
        regions_arr[:, 1] -= self.jitter
        regions_arr[:, 2] += self.jitter
        region_lens = (regions_arr[:, 2] - regions_arr[:, 1]).astype(np.int64)
        # Reference dtype is S1 (1 byte/nt).
        total += region_lens
    elif seq_kind in ("haplotypes", "annotated"):
        if not self.deterministic:
            raise ValueError(
                f"with_seqs={seq_kind!r} requires deterministic=True for "
                "_output_bytes_per_instance. Use dataset.with_settings(deterministic=True)."
            )
        hap_lens = self.haplotype_lengths(regions, samples)
        if hap_lens is None:
            raise ValueError(
                f"with_seqs={seq_kind!r} requires haplotype_lengths() to be available."
            )
        # hap_lens shape: (..., ploidy). Flatten to (n_inst, ploidy).
        hap_lens_flat = hap_lens.reshape(-1, hap_lens.shape[-1]).astype(np.int64)
        hap_len_sum = hap_lens_flat.sum(-1)  # over ploidy
        total += hap_len_sum  # haps S1: 1 byte/nt
        # annotated branch handled in Task 3.
        if seq_kind == "annotated":
            raise NotImplementedError("annotated branch added in Task 3")
    elif seq_kind == "variants":
        raise NotImplementedError("variants branch added in Task 4")
    elif seq_kind is None:
        pass
    else:
        raise AssertionError(f"unknown sequence_type {seq_kind!r}")

    # --- tracks payload added in Task 5 ---
    if self.active_tracks:
        raise NotImplementedError("tracks branch added in Task 5")

    if squeeze:
        return total
    if out_reshape is not None:
        return total.reshape(out_reshape)
    return total.reshape(len(np.atleast_1d(r_idx)))  # default
```

Note: this scaffold raises for the branches not yet implemented; later tasks remove those raises.

- [ ] **Step 4: Run the test to verify it passes**

```bash
pixi run -e dev pytest tests/unit/dataset/test_output_bytes_per_instance.py -v
```

Expected: PASS for `test_reference_mode_exact` and `test_haplotypes_mode_exact`.

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_impl.py tests/unit/dataset/test_output_bytes_per_instance.py
rtk git commit -m "feat(dataset): _output_bytes_per_instance reference + haplotypes"
```

---

## Task 3: Extend `_output_bytes_per_instance` to annotated mode

**Files:**
- Modify: `python/genvarloader/_dataset/_impl.py` (replace `raise NotImplementedError("annotated branch added in Task 3")`)
- Modify: `tests/unit/dataset/test_output_bytes_per_instance.py` (add test)

`annotated` mode returns `AnnotatedHaps` (haps S1, ref_coords int32, var_idxs int32). Footprint:

```
bytes = hap_len_sum × 1 (haps S1)
      + hap_len_sum × 4 (ref_coords int32)
      + n_variants_sum × 4 (var_idxs int32)
```

- [ ] **Step 1: Add test**

Append to `tests/unit/dataset/test_output_bytes_per_instance.py`:

```python
def test_annotated_mode_exact():
    ds = (
        gvl.get_dummy_dataset()
        .with_seqs("annotated")
        .with_tracks(False)
        .with_settings(deterministic=True)
    )
    r = np.arange(ds.full_shape[0])
    s = np.zeros(len(r), dtype=np.int64)
    got = ds._output_bytes_per_instance(r, s)
    expected = _materialized_nbytes_per_instance(ds, r, s)
    np.testing.assert_array_equal(got, expected)
```

- [ ] **Step 2: Run to verify it fails**

```bash
pixi run -e dev pytest tests/unit/dataset/test_output_bytes_per_instance.py::test_annotated_mode_exact -v
```

Expected: FAIL with `NotImplementedError: annotated branch added in Task 3`.

- [ ] **Step 3: Implement annotated branch**

In `_output_bytes_per_instance`, replace `raise NotImplementedError("annotated branch added in Task 3")` with:

```python
            # annotated: add ref_coords (int32, length=hap_len_sum) and var_idxs (int32, length=n_variants_sum)
            n_vars = self.n_variants(regions, samples)
            n_vars_flat = n_vars.reshape(-1, n_vars.shape[-1]).astype(np.int64)
            n_vars_sum = n_vars_flat.sum(-1)
            total += hap_len_sum * 4  # ref_coords int32
            total += n_vars_sum * 4   # var_idxs int32
```

- [ ] **Step 4: Verify pass**

```bash
pixi run -e dev pytest tests/unit/dataset/test_output_bytes_per_instance.py -v
```

Expected: all three tests PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_impl.py tests/unit/dataset/test_output_bytes_per_instance.py
rtk git commit -m "feat(dataset): _output_bytes_per_instance annotated branch"
```

---

## Task 4: Extend `_output_bytes_per_instance` to variants mode

**Files:**
- Modify: `python/genvarloader/_dataset/_impl.py`
- Modify: `tests/unit/dataset/test_output_bytes_per_instance.py`

For `with_seqs="variants"`, iterate `self._seqs.var_fields` and apply the per-field formula. Built-ins map to `n_variants × itemsize` or to allele scans; INFO columns map to `n_variants × dtype.itemsize` via the on-disk variants schema.

- [ ] **Step 1: Add tests (default var_fields, with ref, custom INFO field, custom var_fields)**

Append:

```python
def test_variants_default_var_fields_exact():
    """Default var_fields = ['alt', 'ilen', 'start']."""
    ds = gvl.get_dummy_dataset().with_seqs("variants").with_tracks(False)
    r = np.arange(ds.full_shape[0])
    s = np.zeros(len(r), dtype=np.int64)
    got = ds._output_bytes_per_instance(r, s)
    expected = _materialized_nbytes_per_instance(ds, r, s)
    np.testing.assert_array_equal(got, expected)


def test_variants_with_ref_exact():
    ds = gvl.get_dummy_dataset().with_seqs("variants").with_tracks(False)
    if "ref" not in ds._seqs.available_var_fields:
        pytest.skip("dummy dataset does not have ref allele")
    ds = ds.with_settings(var_fields=["alt", "ref", "ilen", "start"])
    r = np.arange(ds.full_shape[0])
    s = np.zeros(len(r), dtype=np.int64)
    got = ds._output_bytes_per_instance(r, s)
    expected = _materialized_nbytes_per_instance(ds, r, s)
    np.testing.assert_array_equal(got, expected)


def test_variants_with_info_column_exact():
    ds = gvl.get_dummy_dataset().with_seqs("variants").with_tracks(False)
    info_cols = [c for c in ds._seqs.available_var_fields
                 if c not in {"alt", "ref", "ilen", "start", "dosage"}]
    if not info_cols:
        pytest.skip("dummy dataset has no INFO columns")
    ds = ds.with_settings(var_fields=["alt", "start", "ilen", info_cols[0]])
    r = np.arange(ds.full_shape[0])
    s = np.zeros(len(r), dtype=np.int64)
    got = ds._output_bytes_per_instance(r, s)
    expected = _materialized_nbytes_per_instance(ds, r, s)
    np.testing.assert_array_equal(got, expected)
```

Note: `_materialized_nbytes_per_instance` in Task 2 handles `Ragged` via `arr.offsets` / `arr.data.dtype`. For `RaggedVariants` (`ak.Array`), we need to extend that helper to handle awkward arrays. Update the helper:

```python
def _materialized_nbytes_per_instance(ds, r_arr, s_arr):
    out = ds[r_arr, s_arr]
    if not isinstance(out, tuple):
        out = (out,)
    from seqpro.rag import Ragged
    import awkward as ak
    from genvarloader._dataset._rag_variants import RaggedVariants
    n_inst = len(r_arr)
    totals = np.zeros(n_inst, dtype=np.int64)
    for arr in out:
        if isinstance(arr, RaggedVariants):
            # Sum bytes across all fields of every variant in every (inst, ploid).
            for field in arr.fields:
                child = arr[field]
                # child has shape (b, p, ~v) for numeric or (b, p, ~v, ~length) for alleles.
                if child.ndim == 4:  # alleles: sum over (~v, ~length)
                    lens_per_inst = ak.sum(ak.num(child, axis=-1), axis=-1)  # (b, p)
                    lens_per_inst = ak.sum(lens_per_inst, axis=-1).to_numpy()  # (b,)
                    # alt/ref are S1 bytes
                    totals += lens_per_inst.astype(np.int64)
                else:  # (b, p, ~v): numeric per variant
                    n_per_inst = ak.sum(ak.num(child, axis=-1), axis=-1).to_numpy()
                    # Get dtype: child.layout has a NumpyArray leaf
                    dtype = np.dtype(ak.type(ak.flatten(child, axis=None)).primitive)
                    totals += n_per_inst.astype(np.int64) * dtype.itemsize
        elif isinstance(arr, Ragged):
            lens = np.diff(arr.offsets).reshape(n_inst, -1)
            totals += lens.sum(-1) * arr.data.dtype.itemsize
        elif isinstance(arr, np.ndarray):
            per = arr.itemsize * int(np.prod(arr.shape[1:]))
            totals += per
        else:
            raise AssertionError(f"unhandled array type {type(arr)}")
    return totals
```

- [ ] **Step 2: Run to verify failure**

```bash
pixi run -e dev pytest tests/unit/dataset/test_output_bytes_per_instance.py -v
```

Expected: variants tests FAIL with `NotImplementedError: variants branch added in Task 4`.

- [ ] **Step 3: Implement variants branch**

Replace `raise NotImplementedError("variants branch added in Task 4")` with:

```python
    elif seq_kind == "variants":
        haps_obj = self._seqs
        var_fields = haps_obj.var_fields
        n_vars = self.n_variants(regions, samples)  # (n_inst, ploidy)
        n_vars_flat = n_vars.reshape(-1, n_vars.shape[-1]).astype(np.int64)
        n_vars_total = n_vars_flat.sum(-1)  # over ploidy → (n_inst,)

        # Built-in numeric fields with known dtypes.
        pos_dtype = haps_obj.variants.start.dtype                  # POS_TYPE
        ilen_dtype = haps_obj.variants.ilen.dtype if hasattr(haps_obj.variants, "ilen") else np.dtype(np.int32)
        dosage_dtype = haps_obj.dosages.data.dtype if haps_obj.dosages is not None else None

        for f in var_fields:
            if f == "start":
                total += n_vars_total * pos_dtype.itemsize
            elif f == "ilen":
                total += n_vars_total * ilen_dtype.itemsize
            elif f == "dosage":
                if haps_obj.dosages is None:
                    continue
                total += n_vars_total * dosage_dtype.itemsize
            elif f in ("alt", "ref"):
                # Allele scan: flatten ds_idx and reduce per instance over ploidy.
                # _allele_bytes_sum returns shape (n_inst * ploidy,).
                per_ploid = haps_obj._allele_bytes_sum(ds_idx, f)
                ploidy = n_vars.shape[-1]
                total += per_ploid.reshape(-1, ploidy).sum(-1)
            else:
                # INFO column: per_variant numeric, dtype known from on-disk schema.
                info_dtype = haps_obj.variants.info[f].dtype
                total += n_vars_total * info_dtype.itemsize
```

- [ ] **Step 4: Verify**

```bash
pixi run -e dev pytest tests/unit/dataset/test_output_bytes_per_instance.py -v
```

Expected: all variants tests PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_impl.py tests/unit/dataset/test_output_bytes_per_instance.py
rtk git commit -m "feat(dataset): _output_bytes_per_instance variants branch with var_fields"
```

---

## Task 5: Add tracks payload to `_output_bytes_per_instance`

**Files:**
- Modify: `python/genvarloader/_dataset/_impl.py`
- Modify: `tests/unit/dataset/test_output_bytes_per_instance.py`

When tracks are attached, the per-instance footprint adds `hap_len_sum × n_tracks × track_dtype.itemsize`. Track length matches haplotype length after realignment; for reference-only datasets it matches region length.

- [ ] **Step 1: Add tests covering haplotypes+tracks and reference+tracks**

Append:

```python
def test_haplotypes_plus_tracks_exact():
    ds = gvl.get_dummy_dataset().with_seqs("haplotypes").with_settings(deterministic=True)
    # Default ds has tracks active; if not, with_tracks(True).
    if not ds.active_tracks:
        pytest.skip("dummy dataset has no tracks")
    r = np.arange(ds.full_shape[0])
    s = np.zeros(len(r), dtype=np.int64)
    got = ds._output_bytes_per_instance(r, s)
    expected = _materialized_nbytes_per_instance(ds, r, s)
    np.testing.assert_array_equal(got, expected)


def test_reference_plus_tracks_exact():
    ds = gvl.get_dummy_dataset().with_seqs("reference")
    if not ds.active_tracks:
        pytest.skip("dummy dataset has no tracks")
    r = np.arange(ds.full_shape[0])
    s = np.zeros(len(r), dtype=np.int64)
    got = ds._output_bytes_per_instance(r, s)
    expected = _materialized_nbytes_per_instance(ds, r, s)
    np.testing.assert_array_equal(got, expected)
```

- [ ] **Step 2: Run to verify failure**

```bash
pixi run -e dev pytest tests/unit/dataset/test_output_bytes_per_instance.py -v
```

Expected: track tests FAIL with `NotImplementedError: tracks branch added in Task 5`.

- [ ] **Step 3: Implement tracks branch**

Replace `if self.active_tracks: raise NotImplementedError(...)` with:

```python
    if self.active_tracks:
        # Track length per instance equals haplotype length (haplotypes/annotated)
        # or region length (reference). Compute that base length.
        if seq_kind in ("haplotypes", "annotated"):
            base_len = hap_len_sum  # already includes ploidy sum
        else:
            # reference or no-seq: tracks span region length × ploidy if haplotypes
            # not active. With reference, tracks have no ploidy axis → just region length.
            regions_arr = self._full_regions[r_idx].copy()
            regions_arr[:, 1] -= self.jitter
            regions_arr[:, 2] += self.jitter
            base_len = (regions_arr[:, 2] - regions_arr[:, 1]).astype(np.int64)
        for track_name in self.active_tracks:
            track_dtype = self._tracks.intervals[track_name].dtype  # adjust attr to actual storage
            total += base_len * track_dtype.itemsize
```

⚠ Verify the attribute path to track dtypes (`self._tracks.intervals[track_name].dtype` is a placeholder). Open `_dataset/_tracks.py` and confirm how track dtypes are exposed; adjust this line accordingly. If the dtype is uniform across tracks (typically `float32`), substitute that constant rather than per-track lookup.

- [ ] **Step 4: Verify**

```bash
pixi run -e dev pytest tests/unit/dataset/test_output_bytes_per_instance.py -v
```

Expected: all tests PASS (six total: reference, haplotypes, annotated, variants ×3, haplotypes+tracks, reference+tracks).

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_impl.py tests/unit/dataset/test_output_bytes_per_instance.py
rtk git commit -m "feat(dataset): _output_bytes_per_instance tracks branch"
```

---

## Task 6: `ChunkPlanner` + `slice_chunk`

**Files:**
- Create: `python/genvarloader/_chunked.py`
- Create: `tests/unit/test_chunk_planner.py`

Pure logic: given a flat sequence of `(r, s)` pairs (the BatchSampler-resolved epoch order) plus a `bytes_per_instance` table and a `slot_bytes` budget, group consecutive instances into chunks that respect mini-batch boundaries.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_chunk_planner.py`:

```python
"""ChunkPlanner unit tests. Pure logic, no Dataset dependency."""
import numpy as np
import pytest
from genvarloader._chunked import ChunkPlanner


def test_plan_respects_slot_bytes():
    # 100 instances, each 10 bytes → 1000 total. slot_bytes=200 → ~5 chunks.
    bytes_per_instance = np.full((10, 10), 10, dtype=np.int64)
    # BatchSampler yields batches; simulate batch_size=5.
    batches = [np.arange(i, i + 5) for i in range(0, 100, 5)]
    flat_idx = np.concatenate(batches)
    r = flat_idx // 10
    s = flat_idx % 10
    planner = ChunkPlanner(
        r_idx=r, s_idx=s, batch_size=5,
        bytes_per_instance=bytes_per_instance, slot_bytes=200,
    )
    chunks = list(planner)
    # Each chunk's total bytes ≤ 200; each chunk is a multiple of batch_size.
    for cr, cs, nb in chunks:
        assert len(cr) % 5 == 0
        b = bytes_per_instance[cr, cs].sum()
        assert b <= 200
        assert nb == len(cr) // 5
    # Total instances preserved.
    assert sum(len(cr) for cr, _, _ in chunks) == 100


def test_plan_single_batch_chunks_when_tight():
    bytes_per_instance = np.full((4, 1), 100, dtype=np.int64)
    flat = np.arange(4)
    planner = ChunkPlanner(
        r_idx=flat, s_idx=np.zeros_like(flat), batch_size=2,
        bytes_per_instance=bytes_per_instance, slot_bytes=200,
    )
    chunks = list(planner)
    assert len(chunks) == 2  # 200 bytes per batch fits exactly one chunk
    for cr, cs, nb in chunks:
        assert nb == 1


def test_plan_raises_when_batch_exceeds_slot():
    bytes_per_instance = np.full((2, 1), 300, dtype=np.int64)
    flat = np.arange(2)
    with pytest.raises(ValueError, match="exceeds slot"):
        list(ChunkPlanner(
            r_idx=flat, s_idx=np.zeros_like(flat), batch_size=2,
            bytes_per_instance=bytes_per_instance, slot_bytes=200,
        ))


def test_peak_chunk_bytes_reported():
    bytes_per_instance = np.array([[10, 20], [30, 40]], dtype=np.int64)
    flat = np.array([0, 1, 2, 3])
    r = flat // 2
    s = flat % 2
    planner = ChunkPlanner(
        r_idx=r, s_idx=s, batch_size=2,
        bytes_per_instance=bytes_per_instance, slot_bytes=1000,
    )
    chunks = list(planner)
    # Single chunk of 4 instances, total bytes = 10+20+30+40 = 100.
    assert len(chunks) == 1
    assert planner.peak_chunk_bytes == 100
```

- [ ] **Step 2: Run to verify failure**

```bash
pixi run -e dev pytest tests/unit/test_chunk_planner.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'genvarloader._chunked'`.

- [ ] **Step 3: Implement `ChunkPlanner`**

Create `python/genvarloader/_chunked.py`:

```python
"""Chunk planner: groups (r, s) pairs into per-slot chunks aligned to
mini-batch boundaries."""
from __future__ import annotations

from typing import Iterator
import numpy as np
from numpy.typing import NDArray


class ChunkPlanner:
    """Plan chunks for the prefetching dataloader.

    Walks the (r_idx, s_idx) sequence in order; accumulates
    bytes_per_instance[r, s] in mini-batch increments. When the next mini-batch
    would push the running sum above slot_bytes, closes the chunk on the
    nearest mini-batch boundary.

    Iteration yields (chunk_r_idx, chunk_s_idx, n_batches_in_chunk).
    After iteration, .peak_chunk_bytes holds the maximum chunk byte size seen.
    """

    def __init__(
        self,
        r_idx: NDArray[np.integer],
        s_idx: NDArray[np.integer],
        batch_size: int,
        bytes_per_instance: NDArray[np.int64],
        slot_bytes: int,
    ) -> None:
        if len(r_idx) != len(s_idx):
            raise ValueError("r_idx and s_idx must have the same length")
        n = len(r_idx)
        if n % batch_size != 0:
            raise ValueError(
                f"len(r_idx)={n} is not a multiple of batch_size={batch_size}. "
                "Use drop_last or pad the sampler before passing to ChunkPlanner."
            )
        self.r_idx = np.asarray(r_idx)
        self.s_idx = np.asarray(s_idx)
        self.batch_size = batch_size
        self.bytes_per_instance = bytes_per_instance
        self.slot_bytes = int(slot_bytes)
        # Pre-validate: no single mini-batch may exceed slot_bytes.
        per_inst = bytes_per_instance[self.r_idx, self.s_idx].astype(np.int64)
        batch_totals = per_inst.reshape(-1, batch_size).sum(-1)
        too_big = batch_totals > self.slot_bytes
        if too_big.any():
            offender = int(np.argmax(too_big))
            raise ValueError(
                f"Mini-batch {offender} totals {int(batch_totals[offender])} bytes "
                f"which exceeds slot_bytes={self.slot_bytes}. "
                f"Either lower batch_size or raise buffer_bytes."
            )
        self._batch_totals = batch_totals
        self.peak_chunk_bytes: int = 0

    def __iter__(self) -> Iterator[tuple[NDArray[np.integer], NDArray[np.integer], int]]:
        n_batches = len(self._batch_totals)
        i = 0
        while i < n_batches:
            running = 0
            j = i
            while j < n_batches and running + int(self._batch_totals[j]) <= self.slot_bytes:
                running += int(self._batch_totals[j])
                j += 1
            # j-i batches go into this chunk; at least one (guaranteed by the per-batch check).
            assert j > i
            self.peak_chunk_bytes = max(self.peak_chunk_bytes, running)
            start = i * self.batch_size
            end = j * self.batch_size
            yield self.r_idx[start:end], self.s_idx[start:end], j - i
            i = j


def slice_chunk(chunk_output, batch_size: int):
    """Yield mini-batches of size `batch_size` from a chunk-shaped output.

    Supports ndarray, seqpro.rag.Ragged, AnnotatedHaps, RaggedVariants,
    and tuples thereof.
    """
    from seqpro.rag import Ragged
    from ._types import AnnotatedHaps
    from ._dataset._rag_variants import RaggedVariants
    import awkward as ak

    def _slice_one(arr, start: int, stop: int):
        if isinstance(arr, np.ndarray):
            return arr[start:stop]
        if isinstance(arr, Ragged):
            # Ragged supports fancy/slice indexing along axis 0.
            return arr[start:stop]
        if isinstance(arr, AnnotatedHaps):
            return AnnotatedHaps(
                haps=_slice_one(arr.haps, start, stop),
                var_idxs=_slice_one(arr.var_idxs, start, stop),
                ref_coords=_slice_one(arr.ref_coords, start, stop),
            )
        if isinstance(arr, ak.Array):
            return arr[start:stop]
        raise TypeError(f"slice_chunk: unsupported array type {type(arr)}")

    is_tuple = isinstance(chunk_output, tuple)
    arrs = chunk_output if is_tuple else (chunk_output,)
    n = len(arrs[0]) if not isinstance(arrs[0], (ak.Array, Ragged)) else (
        arrs[0].shape[0] if isinstance(arrs[0], ak.Array) else arrs[0].shape[0]
    )
    if n is None:
        raise ValueError("slice_chunk: cannot determine chunk length")
    for start in range(0, n, batch_size):
        stop = start + batch_size
        sliced = tuple(_slice_one(a, start, stop) for a in arrs)
        yield sliced if is_tuple else sliced[0]
```

- [ ] **Step 4: Verify**

```bash
pixi run -e dev pytest tests/unit/test_chunk_planner.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_chunked.py tests/unit/test_chunk_planner.py
rtk git commit -m "feat(chunked): ChunkPlanner and slice_chunk"
```

---

## Task 7: `slice_chunk` tests against real dataset outputs

**Files:**
- Modify: `tests/unit/test_chunk_planner.py`

Verify `slice_chunk` for every output mode by slicing a real chunk and comparing element-wise to direct indexing.

- [ ] **Step 1: Add tests**

Append:

```python
import genvarloader as gvl
from genvarloader._chunked import slice_chunk


def _compare(a, b):
    """Recursive equality for ndarray, Ragged, AnnotatedHaps, ak.Array, tuples."""
    from seqpro.rag import Ragged
    from genvarloader._types import AnnotatedHaps
    import awkward as ak
    if isinstance(a, tuple):
        assert isinstance(b, tuple) and len(a) == len(b)
        for x, y in zip(a, b):
            _compare(x, y)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, Ragged):
        np.testing.assert_array_equal(a.data, b.data)
        np.testing.assert_array_equal(a.offsets, b.offsets)
    elif isinstance(a, AnnotatedHaps):
        _compare(a.haps, b.haps)
        _compare(a.var_idxs, b.var_idxs)
        _compare(a.ref_coords, b.ref_coords)
    elif isinstance(a, ak.Array):
        assert ak.to_list(a) == ak.to_list(b)
    else:
        raise AssertionError(f"unsupported {type(a)}")


@pytest.mark.parametrize("seq_kind", ["reference", "haplotypes", "annotated", "variants"])
def test_slice_chunk_matches_direct(seq_kind):
    ds = gvl.get_dummy_dataset().with_seqs(seq_kind)
    if seq_kind in ("haplotypes", "annotated"):
        ds = ds.with_settings(deterministic=True)
    ds = ds.with_tracks(False)
    n_r = ds.full_shape[0]
    n_s = min(2, ds.full_shape[1])
    r = np.repeat(np.arange(n_r), n_s)
    s = np.tile(np.arange(n_s), n_r)
    chunk = ds[r, s]
    sliced = list(slice_chunk(chunk, batch_size=n_s))
    assert len(sliced) == n_r
    for i, mini in enumerate(sliced):
        direct = ds[r[i * n_s:(i + 1) * n_s], s[i * n_s:(i + 1) * n_s]]
        _compare(mini, direct)
```

- [ ] **Step 2: Run, verify, commit**

```bash
pixi run -e dev pytest tests/unit/test_chunk_planner.py -v
```

If a particular `seq_kind` fails, narrow `_slice_one` in `_chunked.py` for that type and re-run.

```bash
rtk git add tests/unit/test_chunk_planner.py python/genvarloader/_chunked.py
rtk git commit -m "test(chunked): slice_chunk parity with direct indexing"
```

---

## Task 8: `BufferedTorchDataset` and `to_dataloader(mode="buffered")`

**Files:**
- Create: `python/genvarloader/_buffered_loader.py`
- Modify: `python/genvarloader/_torch.py`
- Create: `tests/unit/test_buffered_loader.py`

The simpler of the two new modes: single-process synchronous chunk fetching.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_buffered_loader.py`:

```python
"""End-to-end tests for mode='buffered'."""
import numpy as np
import pytest
import genvarloader as gvl


@pytest.mark.parametrize("seq_kind", ["reference", "haplotypes", "annotated", "variants"])
def test_buffered_iter_matches_direct(seq_kind):
    ds = gvl.get_dummy_dataset().with_seqs(seq_kind).with_tracks(False)
    if seq_kind in ("haplotypes", "annotated"):
        ds = ds.with_settings(deterministic=True)
    batch_size = 2
    n_total = int(np.prod(ds.full_shape))
    # Use shuffle=False so order is deterministic; sequential sampling.
    loader = ds.to_dataloader(
        mode="buffered",
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        buffer_bytes=10 * 1024 * 1024,  # 10 MB; dummy is tiny so this likely fits all.
    )
    seen = 0
    for batch in loader:
        seen += batch_size if not isinstance(batch, tuple) else (
            batch[0].shape[0] if hasattr(batch[0], "shape") else len(batch[0])
        )
    assert seen == (n_total // batch_size) * batch_size


def test_buffered_rejects_num_workers():
    ds = gvl.get_dummy_dataset().with_seqs("reference").with_tracks(False)
    with pytest.raises(ValueError, match="num_workers"):
        ds.to_dataloader(mode="buffered", batch_size=2, num_workers=2)


def test_buffered_rejects_oversized_batch():
    ds = gvl.get_dummy_dataset().with_seqs("reference").with_tracks(False)
    with pytest.raises(ValueError, match="exceeds slot"):
        ds.to_dataloader(mode="buffered", batch_size=2, buffer_bytes=8)


def test_buffered_rejects_nondeterministic_for_haplotypes():
    ds = gvl.get_dummy_dataset().with_seqs("haplotypes").with_settings(deterministic=False)
    with pytest.raises(ValueError, match="deterministic"):
        ds.to_dataloader(mode="buffered", batch_size=2)
```

- [ ] **Step 2: Run to verify failure**

```bash
pixi run -e dev pytest tests/unit/test_buffered_loader.py -v
```

Expected: FAIL — `to_dataloader` doesn't accept `mode`.

- [ ] **Step 3: Implement `BufferedTorchDataset`**

Create `python/genvarloader/_buffered_loader.py`:

```python
"""mode='buffered' dataloader path: synchronous chunked fetch in main process."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any
import numpy as np

from ._chunked import ChunkPlanner, slice_chunk

if TYPE_CHECKING:
    import torch.utils.data as td
    from ._dataset._impl import Dataset


def make_buffered_dataset(
    dataset: "Dataset",
    batch_size: int,
    slot_bytes: int,
    bytes_per_instance: np.ndarray,
    flat_r: np.ndarray,
    flat_s: np.ndarray,
) -> "td.IterableDataset":
    import torch.utils.data as td

    class BufferedTorchDataset(td.IterableDataset):
        def __init__(self) -> None:
            self._dataset = dataset
            self._batch_size = batch_size
            self._planner = ChunkPlanner(
                r_idx=flat_r, s_idx=flat_s, batch_size=batch_size,
                bytes_per_instance=bytes_per_instance, slot_bytes=slot_bytes,
            )

        def __iter__(self):
            for chunk_r, chunk_s, _n in self._planner:
                chunk = self._dataset[chunk_r, chunk_s]
                yield from slice_chunk(chunk, self._batch_size)

        def __len__(self) -> int:
            return len(flat_r) // batch_size

    return BufferedTorchDataset()
```

- [ ] **Step 4: Wire `mode` into `to_dataloader`**

Modify `python/genvarloader/_torch.py`. Change the `get_dataloader` signature to accept new args, and add dispatch logic. Locate the function at `_torch.py:45`.

Add a helper above `get_dataloader`:

```python
def _resolve_buffered_inputs(
    dataset,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    sampler,
    generator,
    buffer_bytes: int,
    n_slots: int,
):
    """Compute flat (r_idx, s_idx) order, bytes_per_instance, and slot_bytes."""
    import numpy as np
    # 1) Resolve full epoch order from the sampler.
    if sampler is None:
        sampler = get_sampler(len(dataset), batch_size, shuffle, drop_last, generator=generator)
    flat = []
    for batch in sampler:
        flat.extend(batch)
    flat = np.asarray(flat, dtype=np.int64)
    n_keep = (len(flat) // batch_size) * batch_size
    flat = flat[:n_keep]
    r_idx, s_idx = np.unravel_index(flat, dataset.shape)

    # 2) Pre-pass: exact bytes per instance.
    # Use full subset arrays for the lookup table; dataset.shape is (n_regions, n_samples).
    full_r = np.arange(dataset.shape[0])
    full_s = np.arange(dataset.shape[1])
    bpi = dataset._output_bytes_per_instance(full_r, full_s)
    # Ensure shape is (n_regions, n_samples).
    bpi = bpi.reshape(dataset.shape)

    slot_bytes = buffer_bytes // n_slots
    return r_idx, s_idx, bpi, slot_bytes, sampler
```

Modify `get_dataloader` signature to accept the new arguments and add the dispatch:

```python
@requires_torch
def get_dataloader(
    dataset,
    batch_size: int = 1,
    shuffle: bool = False,
    sampler=None,
    num_workers: int = 0,
    collate_fn=None,
    pin_memory: bool = False,
    drop_last: bool = False,
    timeout: float = 0,
    worker_init_fn=None,
    multiprocessing_context=None,
    generator=None,
    *,
    prefetch_factor=None,
    persistent_workers: bool = False,
    pin_memory_device: str = "",
    mode: str | None = None,
    buffer_bytes: int = 2 * 1024 ** 3,
    copy: bool = True,
    heartbeat_seconds: float = 60.0,
):
    if mode is None:
        # Existing path unchanged.
        if num_workers > 1:
            logger.warning(
                "It is recommended to use num_workers <= 1 with GenVarLoader since it leverages"
                " multithreading which has lower overhead than multiprocessing."
            )
        if sampler is None:
            sampler = get_sampler(len(dataset), batch_size, shuffle, drop_last, generator=generator)
        return td.DataLoader(
            dataset,
            batch_size=None,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )

    if mode not in {"buffered", "double_buffered"}:
        raise ValueError(f"unknown mode={mode!r}; expected None, 'buffered', or 'double_buffered'")
    if num_workers > 0:
        raise ValueError(f"mode={mode!r} is incompatible with num_workers>0; the loader IS the concurrency strategy")

    n_slots = 1 if mode == "buffered" else 2
    r_idx, s_idx, bpi, slot_bytes, _sampler = _resolve_buffered_inputs(
        dataset, batch_size, shuffle, drop_last, sampler, generator, buffer_bytes, n_slots,
    )

    if mode == "buffered":
        from ._buffered_loader import make_buffered_dataset
        inner_ds = make_buffered_dataset(
            dataset, batch_size, slot_bytes, bpi, r_idx, s_idx,
        )
    else:
        from ._double_buffered_loader import make_double_buffered_dataset
        inner_ds = make_double_buffered_dataset(
            dataset, batch_size, slot_bytes, bpi, r_idx, s_idx,
            copy=copy, heartbeat_seconds=heartbeat_seconds,
        )

    return td.DataLoader(
        inner_ds,
        batch_size=None,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        pin_memory_device=pin_memory_device,
    )
```

Then plumb the new args through `Dataset.to_dataloader()`. Open `_dataset/_impl.py` and find `to_dataloader` (search `def to_dataloader`); add the four new keyword arguments and pass them to `get_dataloader`.

- [ ] **Step 5: Verify**

```bash
pixi run -e dev pytest tests/unit/test_buffered_loader.py -v
```

Expected: all `test_buffered_*` pass. The `nondeterministic_for_haplotypes` test depends on `_output_bytes_per_instance` raising before the planner runs — verify the error path triggers in `_resolve_buffered_inputs`.

- [ ] **Step 6: Commit**

```bash
rtk git add python/genvarloader/_buffered_loader.py python/genvarloader/_torch.py python/genvarloader/_dataset/_impl.py tests/unit/test_buffered_loader.py
rtk git commit -m "feat: mode='buffered' dataloader"
```

---

## Task 9: `_shm_layout.py` — header format + dense round-trip

**Files:**
- Create: `python/genvarloader/_shm_layout.py`
- Create: `tests/unit/test_shm_layout.py`

Hand-rolled fixed-prefix header packing/unpacking. Start with dense arrays; ragged variants added in Task 10.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_shm_layout.py`:

```python
"""Round-trip tests for the shm slot layout."""
import multiprocessing as mp
import numpy as np
import pytest
from multiprocessing.shared_memory import SharedMemory
from genvarloader._shm_layout import write_chunk, read_chunk, slot_capacity_for


def test_dense_roundtrip_single_array():
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    capacity = slot_capacity_for([arr]) * 2  # generous
    shm = SharedMemory(create=True, size=capacity)
    try:
        write_chunk(shm.buf, [arr], n_instances=10)
        n_inst, views = read_chunk(shm.buf)
        assert n_inst == 10
        assert len(views) == 1
        np.testing.assert_array_equal(views[0], arr)
    finally:
        shm.close()
        shm.unlink()


def test_dense_roundtrip_multiple_arrays():
    a = np.arange(20, dtype=np.float32).reshape(4, 5)
    b = np.arange(8, dtype=np.int64).reshape(4, 2)
    capacity = slot_capacity_for([a, b]) * 2
    shm = SharedMemory(create=True, size=capacity)
    try:
        write_chunk(shm.buf, [a, b], n_instances=4)
        n_inst, views = read_chunk(shm.buf)
        assert n_inst == 4
        np.testing.assert_array_equal(views[0], a)
        np.testing.assert_array_equal(views[1], b)
    finally:
        shm.close()
        shm.unlink()


def _child_read(shm_name, q):
    s = SharedMemory(name=shm_name)
    try:
        n_inst, views = read_chunk(s.buf)
        q.put((n_inst, [np.asarray(v).copy() for v in views]))
    finally:
        s.close()


def test_dense_cross_process():
    arr = np.arange(50, dtype=np.int32).reshape(5, 10)
    capacity = slot_capacity_for([arr]) * 2
    shm = SharedMemory(create=True, size=capacity)
    try:
        write_chunk(shm.buf, [arr], n_instances=5)
        q = mp.Queue()
        p = mp.Process(target=_child_read, args=(shm.name, q))
        p.start()
        n_inst, views = q.get(timeout=10)
        p.join(timeout=10)
        assert n_inst == 5
        np.testing.assert_array_equal(views[0], arr)
    finally:
        shm.close()
        shm.unlink()
```

- [ ] **Step 2: Run to verify failure**

```bash
pixi run -e dev pytest tests/unit/test_shm_layout.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `_shm_layout.py` (dense only for now)**

Create `python/genvarloader/_shm_layout.py`:

```python
"""Shared-memory slot layout: hand-rolled header + payload.

Header layout (little-endian throughout):
  u64 n_instances
  u64 payload_bytes
  u8  n_arrays
  ArrayDescriptor[n_arrays]:
    u8  kind            (0=dense, 1=ragged_seqpro, 2=ragged_alleles)
    u8  dtype_code      (np.dtype.num)
    u8  ndim
    u64 shape[ndim]
    u64 data_offset
    u64 data_nbytes
    u64 lengths_offset  (0 if dense)
    u64 lengths_nbytes  (0 if dense)
    u64 inner_offsets_offset  (used by ragged_alleles; 0 otherwise)
    u64 inner_offsets_nbytes
"""
from __future__ import annotations

import struct
from typing import Sequence

import numpy as np

# Reserve a generous prefix for header. Slot layout: [HEADER (HEADER_RESERVED bytes)][PAYLOAD].
HEADER_RESERVED = 4096

_HEADER_PREAMBLE = struct.Struct("<QQB")  # n_instances, payload_bytes, n_arrays
_DESCRIPTOR_FIXED = struct.Struct("<BBB")  # kind, dtype_code, ndim


def _align(off: int, align: int = 8) -> int:
    rem = off % align
    return off if rem == 0 else off + (align - rem)


def slot_capacity_for(arrays: Sequence[np.ndarray]) -> int:
    """Worst-case slot bytes needed to hold the given arrays (dense only).
    Used by tests; production sizing uses ChunkPlanner.peak_chunk_bytes."""
    payload = sum(_align(a.nbytes) for a in arrays)
    return HEADER_RESERVED + payload


def write_chunk(
    buf: memoryview,
    arrays: Sequence[np.ndarray],
    n_instances: int,
) -> int:
    """Write dense arrays into the slot. Returns total bytes consumed."""
    if len(arrays) > 255:
        raise ValueError("at most 255 arrays per chunk")
    # 1) Lay out payload offsets.
    payload_offsets = []
    cursor = HEADER_RESERVED
    for a in arrays:
        cursor = _align(cursor)
        payload_offsets.append(cursor)
        np.frombuffer(buf, dtype=a.dtype, count=a.size,
                      offset=cursor).reshape(a.shape)[...] = a
        cursor += a.nbytes
    payload_bytes = cursor - HEADER_RESERVED
    # 2) Write header.
    hdr = bytearray()
    hdr += _HEADER_PREAMBLE.pack(n_instances, payload_bytes, len(arrays))
    for a, off in zip(arrays, payload_offsets):
        hdr += _DESCRIPTOR_FIXED.pack(0, a.dtype.num, a.ndim)
        for d in a.shape:
            hdr += struct.pack("<Q", int(d))
        hdr += struct.pack("<Q", off)         # data_offset
        hdr += struct.pack("<Q", a.nbytes)    # data_nbytes
        hdr += struct.pack("<Q", 0)           # lengths_offset
        hdr += struct.pack("<Q", 0)           # lengths_nbytes
        hdr += struct.pack("<Q", 0)           # inner_offsets_offset
        hdr += struct.pack("<Q", 0)           # inner_offsets_nbytes
    if len(hdr) > HEADER_RESERVED:
        raise ValueError(f"header too large: {len(hdr)} > {HEADER_RESERVED}")
    buf[:len(hdr)] = bytes(hdr)
    return cursor


def read_chunk(buf: memoryview) -> tuple[int, list[np.ndarray]]:
    """Read dense arrays from the slot. Returns (n_instances, [arrays...])."""
    n_inst, payload_bytes, n_arrays = _HEADER_PREAMBLE.unpack_from(buf, 0)
    cursor = _HEADER_PREAMBLE.size
    views: list[np.ndarray] = []
    for _ in range(n_arrays):
        kind, dtype_num, ndim = _DESCRIPTOR_FIXED.unpack_from(buf, cursor)
        cursor += _DESCRIPTOR_FIXED.size
        shape = []
        for _d in range(ndim):
            (dim,) = struct.unpack_from("<Q", buf, cursor)
            shape.append(int(dim))
            cursor += 8
        data_offset, data_nbytes, lo, ln, ioff, ilen = struct.unpack_from("<6Q", buf, cursor)
        cursor += 48
        if kind != 0:
            raise NotImplementedError(f"ragged read_chunk arrives in Task 10 (kind={kind})")
        dtype = np.dtype(np.typeDict[dtype_num]) if hasattr(np, "typeDict") else np.dtype(np.sctypeDict[dtype_num])
        view = np.frombuffer(buf, dtype=dtype, count=int(np.prod(shape)),
                             offset=data_offset).reshape(shape)
        views.append(view)
    return int(n_inst), views
```

Note: `np.typeDict` may be unavailable on newer NumPy; the fallback handles that. If neither works on this NumPy version, replace with a static `dict[int, np.dtype]` lookup built once.

- [ ] **Step 4: Verify**

```bash
pixi run -e dev pytest tests/unit/test_shm_layout.py -v
```

Expected: dense tests PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_shm_layout.py tests/unit/test_shm_layout.py
rtk git commit -m "feat(shm): hand-rolled slot header + dense round-trip"
```

---

## Task 10: `_shm_layout.py` — Ragged and RaggedVariants support

**Files:**
- Modify: `python/genvarloader/_shm_layout.py`
- Modify: `tests/unit/test_shm_layout.py`

Extend `write_chunk` / `read_chunk` to handle `seqpro.rag.Ragged` (kind=1) and `RaggedVariants` (kind=2). For `Ragged`, store `(data, offsets)`. For `RaggedVariants` (alt/ref), each field has outer offsets (`(b*p)+1`) plus inner offsets (`n_variants+1`); numeric variant fields use only outer offsets.

- [ ] **Step 1: Add tests**

Append to `tests/unit/test_shm_layout.py`:

```python
def test_ragged_roundtrip():
    from seqpro.rag import Ragged
    data = np.arange(20, dtype=np.int32)
    # Three rows of lengths 5, 8, 7 → offsets [0, 5, 13, 20].
    offsets = np.array([0, 5, 13, 20], dtype=np.int64)
    rag = Ragged.from_offsets(data, (3, None), offsets)
    capacity = 4096 + data.nbytes + offsets.nbytes + 64
    shm = SharedMemory(create=True, size=capacity)
    try:
        write_chunk(shm.buf, [rag], n_instances=3)
        n_inst, views = read_chunk(shm.buf)
        assert n_inst == 3
        from seqpro.rag import Ragged as R
        assert isinstance(views[0], R)
        np.testing.assert_array_equal(views[0].data, data)
        np.testing.assert_array_equal(views[0].offsets, offsets)
    finally:
        shm.close()
        shm.unlink()


def test_annotated_haps_roundtrip():
    """Three Ragged arrays in sequence (haps S1, ref_coords int32, var_idxs int32)."""
    from seqpro.rag import Ragged
    haps_data = np.frombuffer(b"ACGTAAAA", dtype="S1")
    haps_offsets = np.array([0, 4, 8], dtype=np.int64)
    haps = Ragged.from_offsets(haps_data, (2, None), haps_offsets)
    coords = Ragged.from_offsets(np.arange(8, dtype=np.int32),
                                  (2, None), haps_offsets)
    v_idxs = Ragged.from_offsets(np.array([10, 20, 30], dtype=np.int32),
                                  (2, None), np.array([0, 1, 3], dtype=np.int64))
    capacity = 8192
    shm = SharedMemory(create=True, size=capacity)
    try:
        write_chunk(shm.buf, [haps, coords, v_idxs], n_instances=2)
        n_inst, views = read_chunk(shm.buf)
        assert n_inst == 2 and len(views) == 3
        np.testing.assert_array_equal(views[0].data, haps_data)
        np.testing.assert_array_equal(views[1].data.view(np.int32),
                                       np.arange(8, dtype=np.int32))
    finally:
        shm.close()
        shm.unlink()
```

- [ ] **Step 2: Run to verify failure**

```bash
pixi run -e dev pytest tests/unit/test_shm_layout.py -v
```

Expected: ragged tests FAIL (read_chunk raises NotImplementedError for kind=1).

- [ ] **Step 3: Extend `write_chunk` and `read_chunk` for Ragged**

In `_shm_layout.py`, restructure `write_chunk` to dispatch by type. Sketch:

```python
def write_chunk(buf, arrays, n_instances):
    from seqpro.rag import Ragged
    if len(arrays) > 255:
        raise ValueError("at most 255 arrays per chunk")
    descriptors = []
    cursor = HEADER_RESERVED
    for a in arrays:
        cursor = _align(cursor)
        if isinstance(a, np.ndarray):
            data_off = cursor
            np.frombuffer(buf, dtype=a.dtype, count=a.size,
                          offset=data_off).reshape(a.shape)[...] = a
            cursor += a.nbytes
            descriptors.append({
                "kind": 0, "dtype_num": a.dtype.num, "shape": a.shape,
                "data_offset": data_off, "data_nbytes": a.nbytes,
                "lengths_offset": 0, "lengths_nbytes": 0,
                "inner_offsets_offset": 0, "inner_offsets_nbytes": 0,
            })
        elif isinstance(a, Ragged):
            data_off = cursor
            data_arr = np.ascontiguousarray(a.data)
            np.frombuffer(buf, dtype=data_arr.dtype, count=data_arr.size,
                          offset=data_off)[...] = data_arr.ravel()
            cursor += data_arr.nbytes
            cursor = _align(cursor)
            off_off = cursor
            off_arr = np.ascontiguousarray(a.offsets)
            np.frombuffer(buf, dtype=off_arr.dtype, count=off_arr.size,
                          offset=off_off)[...] = off_arr
            cursor += off_arr.nbytes
            descriptors.append({
                "kind": 1, "dtype_num": data_arr.dtype.num,
                "shape": [data_arr.size],  # flat
                "data_offset": data_off, "data_nbytes": data_arr.nbytes,
                "lengths_offset": off_off, "lengths_nbytes": off_arr.nbytes,
                "inner_offsets_offset": 0, "inner_offsets_nbytes": 0,
            })
        else:
            raise TypeError(f"write_chunk: unsupported array type {type(a)}")
    payload_bytes = cursor - HEADER_RESERVED
    hdr = bytearray()
    hdr += _HEADER_PREAMBLE.pack(n_instances, payload_bytes, len(descriptors))
    for d in descriptors:
        hdr += _DESCRIPTOR_FIXED.pack(d["kind"], d["dtype_num"], len(d["shape"]))
        for dim in d["shape"]:
            hdr += struct.pack("<Q", int(dim))
        hdr += struct.pack("<6Q",
            d["data_offset"], d["data_nbytes"],
            d["lengths_offset"], d["lengths_nbytes"],
            d["inner_offsets_offset"], d["inner_offsets_nbytes"])
    if len(hdr) > HEADER_RESERVED:
        raise ValueError(f"header too large: {len(hdr)} > {HEADER_RESERVED}")
    buf[:len(hdr)] = bytes(hdr)
    return cursor
```

Extend `read_chunk`:

```python
def read_chunk(buf):
    from seqpro.rag import Ragged
    n_inst, payload_bytes, n_arrays = _HEADER_PREAMBLE.unpack_from(buf, 0)
    cursor = _HEADER_PREAMBLE.size
    views = []
    for _ in range(n_arrays):
        kind, dtype_num, ndim = _DESCRIPTOR_FIXED.unpack_from(buf, cursor)
        cursor += _DESCRIPTOR_FIXED.size
        shape = []
        for _ in range(ndim):
            (dim,) = struct.unpack_from("<Q", buf, cursor); shape.append(int(dim)); cursor += 8
        data_off, data_nb, lo, ln, ioff, ilen = struct.unpack_from("<6Q", buf, cursor)
        cursor += 48
        dtype = np.dtype(np.sctypeDict[dtype_num])
        if kind == 0:
            view = np.frombuffer(buf, dtype=dtype, count=int(np.prod(shape)),
                                 offset=data_off).reshape(shape)
            views.append(view)
        elif kind == 1:
            data = np.frombuffer(buf, dtype=dtype, count=shape[0], offset=data_off)
            # offsets dtype: int64 (or whatever Ragged uses); read as int64.
            offsets = np.frombuffer(buf, dtype=np.int64,
                                    count=ln // 8, offset=lo)
            views.append(Ragged.from_offsets(data, (len(offsets) - 1, None), offsets))
        elif kind == 2:
            raise NotImplementedError("RaggedVariants in Task 10b (extended)")
        else:
            raise ValueError(f"unknown descriptor kind {kind}")
    return int(n_inst), views
```

- [ ] **Step 4: Verify Ragged tests**

```bash
pixi run -e dev pytest tests/unit/test_shm_layout.py -v
```

Expected: dense and ragged tests PASS.

- [ ] **Step 5: Add RaggedVariants (alt/ref) support**

`RaggedVariants` is an awkward record of multiple fields. Strategy: serialize each leaf as its own descriptor (a flat `Ragged`-like with outer offsets), plus for `alt`/`ref` an additional inner offset array. The producer flattens; the consumer reassembles using `RaggedVariants.from_ak` with `ak.zip` over rebuilt fields.

Add a helper `decompose_rag_variants(rv) -> list[(name, leaf_array, leaf_kind)]` and `recompose_rag_variants(parts) -> RaggedVariants`. Add tests round-tripping a `RaggedVariants` produced by `gvl.get_dummy_dataset().with_seqs("variants")[r, s]`.

This step needs concrete awkward layout work; budget ~1-2 sessions. Reference patterns: `Haps._get_variants` in `_haps.py:634`, `RaggedVariants.from_ak` in `_rag_variants.py:66`. The test pattern:

```python
def test_rag_variants_roundtrip():
    import genvarloader as gvl
    ds = gvl.get_dummy_dataset().with_seqs("variants").with_tracks(False)
    r = np.arange(ds.full_shape[0])
    s = np.zeros(len(r), dtype=np.int64)
    rv = ds[r, s]
    # rv is RaggedVariants.
    nbytes = ds._output_bytes_per_instance(r, s).sum()
    capacity = HEADER_RESERVED + nbytes + 4096  # slack for ragged offsets
    shm = SharedMemory(create=True, size=capacity)
    try:
        write_chunk(shm.buf, [rv], n_instances=len(r))
        n_inst, views = read_chunk(shm.buf)
        from genvarloader._dataset._rag_variants import RaggedVariants
        assert isinstance(views[0], RaggedVariants)
        import awkward as ak
        assert ak.to_list(views[0]) == ak.to_list(rv)
    finally:
        shm.close()
        shm.unlink()
```

Implement using the existing private fields on `RaggedVariants` (it's an `ak.Array` subclass). For each field accessed via `rv[field_name]`, extract its underlying buffers using `ak.to_layout` and serialize the buffers + offsets. On read, reconstruct an `ak.Array` from buffers and pass through `RaggedVariants.from_ak`.

- [ ] **Step 6: Commit**

```bash
rtk git add python/genvarloader/_shm_layout.py tests/unit/test_shm_layout.py
rtk git commit -m "feat(shm): Ragged and RaggedVariants serialization"
```

---

## Task 11: Producer subprocess (`_producer.py`)

**Files:**
- Create: `python/genvarloader/_producer.py`
- Create: `tests/unit/test_producer.py`

Spawn a child process that opens the dataset, drains an index queue, and writes chunks into shm slots indexed by a slot id.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_producer.py`:

```python
"""Tests for the producer subprocess in isolation."""
import multiprocessing as mp
import numpy as np
import pytest
from multiprocessing.shared_memory import SharedMemory
import genvarloader as gvl
from genvarloader._producer import producer_main
from genvarloader._shm_layout import read_chunk, HEADER_RESERVED


def test_producer_writes_chunk_and_signals():
    ctx = mp.get_context("spawn")
    ds = gvl.get_dummy_dataset().with_seqs("reference").with_tracks(False)
    # We will pass a serialized "schema_payload" describing how to rebuild the
    # subset/schema in the child. For this test, use the dataset path directly.
    ds_path = ds._path if hasattr(ds, "_path") else None
    if ds_path is None:
        pytest.skip("dummy dataset is not file-backed; cross-process test requires a real dataset path")
    capacity = 64 * 1024
    shm = SharedMemory(create=True, size=capacity)
    free = ctx.Event(); free.set()
    ready = ctx.Event()
    index_queue = ctx.Queue()
    exc_q = ctx.Queue()
    r = np.array([0], dtype=np.int64)
    s = np.array([0], dtype=np.int64)
    index_queue.put((0, r, s, 1))  # (slot_idx, r_idx, s_idx, n_batches)
    index_queue.put(None)
    p = ctx.Process(
        target=producer_main,
        args=(str(ds_path), {"with_seqs": "reference", "with_tracks": False},
              [shm.name], [(free, ready)], index_queue, exc_q),
    )
    p.start()
    assert ready.wait(timeout=30)
    n_inst, views = read_chunk(shm.buf)
    assert n_inst == 1
    p.join(timeout=10)
    shm.close(); shm.unlink()
```

- [ ] **Step 2: Run to verify failure**

```bash
pixi run -e dev pytest tests/unit/test_producer.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `producer_main`**

Create `python/genvarloader/_producer.py`:

```python
"""Producer subprocess entrypoint for mode='double_buffered'."""
from __future__ import annotations

import traceback
from multiprocessing.shared_memory import SharedMemory
import genvarloader as gvl
from ._shm_layout import write_chunk


def _apply_schema(ds, schema):
    if "with_seqs" in schema:
        ds = ds.with_seqs(schema["with_seqs"])
    if "with_tracks" in schema:
        ds = ds.with_tracks(schema["with_tracks"])
    if schema.get("deterministic") is not None:
        ds = ds.with_settings(deterministic=schema["deterministic"])
    if schema.get("var_fields") is not None:
        ds = ds.with_settings(var_fields=schema["var_fields"])
    return ds


def producer_main(dataset_path, schema, shm_names, events, index_queue, exc_q):
    """Reopen the dataset, fill shm slots from indices on index_queue.

    Args:
        dataset_path: filesystem path passed to Dataset.open.
        schema: dict describing with_seqs/with_tracks/with_settings.
        shm_names: list of POSIX shm names, one per slot.
        events: list of (free, ready) Event pairs, one per slot.
        index_queue: yields (slot_idx, r_idx, s_idx, n_batches); None=stop.
        exc_q: Queue to push (type_name, message, traceback_str) on failure.
    """
    try:
        ds = gvl.Dataset.open(dataset_path)
        ds = _apply_schema(ds, schema)
        shms = [SharedMemory(name=n) for n in shm_names]
        try:
            while True:
                item = index_queue.get()
                if item is None:
                    return
                slot_idx, r_idx, s_idx, _n_batches = item
                free, ready = events[slot_idx]
                free.wait()
                chunk = ds[r_idx, s_idx]
                arrays = chunk if isinstance(chunk, tuple) else (chunk,)
                write_chunk(shms[slot_idx].buf, list(arrays), n_instances=len(r_idx))
                free.clear()
                ready.set()
        finally:
            for s in shms:
                s.close()
    except Exception as e:
        exc_q.put((type(e).__name__, str(e), traceback.format_exc()))
```

- [ ] **Step 4: Verify**

```bash
pixi run -e dev pytest tests/unit/test_producer.py -v
```

If the dummy dataset isn't file-backed, the test will skip — that's fine for unit. The integration coverage in Task 12 exercises the producer against a real dataset via `to_dataloader`.

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_producer.py tests/unit/test_producer.py
rtk git commit -m "feat(producer): subprocess entrypoint for double_buffered mode"
```

---

## Task 12: `DoubleBufferedTorchDataset` happy path

**Files:**
- Create: `python/genvarloader/_double_buffered_loader.py`
- Create: `tests/unit/test_double_buffered_loader.py`

- [ ] **Step 1: Write the failing happy-path test**

Create `tests/unit/test_double_buffered_loader.py`:

```python
"""End-to-end tests for mode='double_buffered'."""
import os
import numpy as np
import pytest
import genvarloader as gvl


@pytest.mark.slow
@pytest.mark.parametrize("seq_kind", ["reference", "haplotypes"])
def test_double_buffered_iter_matches_buffered(seq_kind, tmp_path):
    ds = gvl.get_dummy_dataset().with_seqs(seq_kind).with_tracks(False)
    if seq_kind in ("haplotypes", "annotated"):
        ds = ds.with_settings(deterministic=True)
    if not hasattr(ds, "_path") or ds._path is None:
        pytest.skip("double_buffered requires a file-backed dataset")
    batch_size = 2
    buf_kw = dict(batch_size=batch_size, shuffle=False, drop_last=True,
                  buffer_bytes=4 * 1024 * 1024)
    buffered = list(ds.to_dataloader(mode="buffered", **buf_kw))
    double = list(ds.to_dataloader(mode="double_buffered", copy=True, **buf_kw))
    assert len(double) == len(buffered)
    # Compare by raw bytes/element to avoid type-specific equality glue here.
    for b, d in zip(buffered, double):
        if isinstance(b, tuple):
            for x, y in zip(b, d):
                np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
        else:
            np.testing.assert_array_equal(np.asarray(b), np.asarray(d))
```

- [ ] **Step 2: Run to verify failure**

```bash
pixi run -e dev pytest tests/unit/test_double_buffered_loader.py -v -m slow
```

Expected: `ImportError` (no `_double_buffered_loader`).

- [ ] **Step 3: Implement `make_double_buffered_dataset`**

Create `python/genvarloader/_double_buffered_loader.py`:

```python
"""mode='double_buffered' dataloader: subprocess producer + 2-slot shm ping-pong."""
from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import uuid
import weakref
from typing import TYPE_CHECKING

import numpy as np
from multiprocessing.shared_memory import SharedMemory

from ._chunked import ChunkPlanner, slice_chunk
from ._shm_layout import read_chunk, HEADER_RESERVED

if TYPE_CHECKING:
    import torch.utils.data as td
    from ._dataset._impl import Dataset


class _DoubleBufferedIterable:
    def __init__(
        self,
        dataset: "Dataset",
        batch_size: int,
        slot_bytes: int,
        bytes_per_instance: np.ndarray,
        flat_r: np.ndarray,
        flat_s: np.ndarray,
        copy: bool,
        heartbeat_seconds: float,
    ):
        self._dataset = dataset
        self._batch_size = batch_size
        self._copy = copy
        self._heartbeat = heartbeat_seconds
        self._planner = ChunkPlanner(
            r_idx=flat_r, s_idx=flat_s, batch_size=batch_size,
            bytes_per_instance=bytes_per_instance, slot_bytes=slot_bytes,
        )
        # Force a full pass to compute peak_chunk_bytes — replan on iter.
        _ = list(self._planner)
        capacity = HEADER_RESERVED + self._planner.peak_chunk_bytes + 4096
        # Allocate 2 shm slots.
        suffix = uuid.uuid4().hex[:8]
        self._shm_names = [f"gvl-{os.getpid()}-{suffix}-{i}" for i in range(2)]
        self._shms = [SharedMemory(create=True, name=n, size=capacity) for n in self._shm_names]
        ctx = mp.get_context("spawn")
        self._ctx = ctx
        self._events = [(ctx.Event(), ctx.Event()) for _ in range(2)]
        for free, ready in self._events:
            free.set()
            ready.clear()
        self._index_queue = ctx.Queue()
        self._exc_q = ctx.Queue()
        self._producer: mp.Process | None = None
        self._flat_r = flat_r
        self._flat_s = flat_s
        weakref.finalize(self, _cleanup, list(self._shms), self._producer_ref)
        atexit.register(self.close)

    def _producer_ref(self):
        return self._producer

    def _spawn_producer(self):
        from ._producer import producer_main
        ds = self._dataset
        schema = {
            "with_seqs": ds.sequence_type,
            "with_tracks": bool(ds.active_tracks),
            "deterministic": ds.deterministic,
        }
        # var_fields, if applicable.
        if ds.sequence_type == "variants":
            schema["var_fields"] = list(ds._seqs.var_fields)
        ds_path = ds._path if hasattr(ds, "_path") else None
        if ds_path is None:
            raise RuntimeError("double_buffered requires a file-backed dataset (Dataset.open(path))")
        self._producer = self._ctx.Process(
            target=producer_main,
            args=(str(ds_path), schema, list(self._shm_names),
                  self._events, self._index_queue, self._exc_q),
            daemon=True,
        )
        self._producer.start()

    def __iter__(self):
        if self._producer is None:
            self._spawn_producer()
        # Push chunks into the queue.
        planner = ChunkPlanner(
            r_idx=self._flat_r, s_idx=self._flat_s, batch_size=self._batch_size,
            bytes_per_instance=self._planner.bytes_per_instance,
            slot_bytes=self._planner.slot_bytes,
        )
        chunks = list(planner)
        for i, (cr, cs, nb) in enumerate(chunks):
            self._index_queue.put((i % 2, cr, cs, nb))
        # Consume.
        for i, (_cr, _cs, _nb) in enumerate(chunks):
            slot_idx = i % 2
            free, ready = self._events[slot_idx]
            # Wait with heartbeat; check producer liveness on timeout.
            while not ready.wait(timeout=self._heartbeat):
                if not self._producer.is_alive():
                    raise self._reraise_or_die()
            n_inst, views = read_chunk(self._shms[slot_idx].buf)
            chunk_output = tuple(views) if len(views) > 1 else views[0]
            for mini in slice_chunk(chunk_output, self._batch_size):
                if self._copy:
                    mini = _deep_copy_batch(mini)
                yield mini
            ready.clear()
            free.set()

    def _reraise_or_die(self):
        if not self._exc_q.empty():
            tname, msg, tb = self._exc_q.get()
            raise RuntimeError(f"ProducerError ({tname}): {msg}\n{tb}")
        return RuntimeError("ProducerDied: producer exited without an exception")

    def __len__(self) -> int:
        return len(self._flat_r) // self._batch_size

    def close(self):
        if self._producer is not None and self._producer.is_alive():
            self._index_queue.put(None)
            self._producer.join(timeout=5)
            if self._producer.is_alive():
                self._producer.terminate()
        for shm in self._shms:
            try: shm.close()
            except Exception: pass
            try: shm.unlink()
            except Exception: pass


def _cleanup(shms, producer_getter):
    proc = producer_getter()
    if proc is not None and proc.is_alive():
        proc.terminate()
    for shm in shms:
        try: shm.close()
        except Exception: pass
        try: shm.unlink()
        except Exception: pass


def _deep_copy_batch(batch):
    from seqpro.rag import Ragged
    from ._types import AnnotatedHaps
    import awkward as ak
    if isinstance(batch, tuple):
        return tuple(_deep_copy_batch(x) for x in batch)
    if isinstance(batch, np.ndarray):
        return batch.copy()
    if isinstance(batch, Ragged):
        return Ragged.from_offsets(batch.data.copy(), batch.shape, batch.offsets.copy())
    if isinstance(batch, AnnotatedHaps):
        return AnnotatedHaps(
            haps=_deep_copy_batch(batch.haps),
            var_idxs=_deep_copy_batch(batch.var_idxs),
            ref_coords=_deep_copy_batch(batch.ref_coords),
        )
    if isinstance(batch, ak.Array):
        return ak.copy(batch)
    raise TypeError(f"_deep_copy_batch: unsupported {type(batch)}")


def make_double_buffered_dataset(
    dataset, batch_size, slot_bytes, bytes_per_instance, flat_r, flat_s,
    copy: bool, heartbeat_seconds: float,
):
    import torch.utils.data as td

    class _DBTorchDataset(td.IterableDataset):
        def __init__(self):
            self._impl = _DoubleBufferedIterable(
                dataset, batch_size, slot_bytes, bytes_per_instance,
                flat_r, flat_s, copy, heartbeat_seconds,
            )

        def __iter__(self):
            return iter(self._impl)

        def __len__(self):
            return len(self._impl)

    return _DBTorchDataset()
```

⚠ This task introduces real subprocess + shm coordination. Several things may need tightening once tests run:

- `ds._path` may not be the exact attribute name; check `Dataset.open` source in `_dataset/_impl.py` to find the right attribute (or store the path on the loader explicitly).
- `self._planner` peak computation should not consume the planner; refactor to compute `peak_chunk_bytes` eagerly in `ChunkPlanner.__init__` instead of during iteration. Update Task 6's `ChunkPlanner.__init__` to precompute `peak_chunk_bytes` once.

Apply this `ChunkPlanner` fix as part of this task: in `_chunked.py`, replace the lazy peak update with an upfront pass:

```python
# In __init__, after building self._batch_totals:
running = 0
peak = 0
i = 0
while i < len(self._batch_totals):
    j = i
    cur = 0
    while j < len(self._batch_totals) and cur + int(self._batch_totals[j]) <= self.slot_bytes:
        cur += int(self._batch_totals[j])
        j += 1
    peak = max(peak, cur)
    i = j
self.peak_chunk_bytes = peak
self.bytes_per_instance = bytes_per_instance
```

And remove the assignment inside `__iter__`.

- [ ] **Step 4: Verify**

```bash
pixi run -e dev pytest tests/unit/test_double_buffered_loader.py -v -m slow
```

Expected: happy-path test PASSes for `reference`. For `haplotypes`, may need additional debugging of the ragged path through `_shm_layout.write_chunk`.

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_double_buffered_loader.py python/genvarloader/_chunked.py tests/unit/test_double_buffered_loader.py
rtk git commit -m "feat: mode='double_buffered' dataloader happy path"
```

---

## Task 13: Producer crash / heartbeat / cleanup tests

**Files:**
- Modify: `tests/unit/test_double_buffered_loader.py`

- [ ] **Step 1: Add failure-path tests**

Append:

```python
@pytest.mark.slow
def test_producer_exception_reraised(monkeypatch):
    """If the producer raises mid-fetch, the consumer surfaces a ProducerError."""
    ds = gvl.get_dummy_dataset().with_seqs("reference").with_tracks(False)
    if not hasattr(ds, "_path") or ds._path is None:
        pytest.skip("requires file-backed dataset")
    # Monkeypatch Dataset.__getitem__ in the producer process by injecting a
    # post-open hook through schema. Simpler: patch the producer's _apply_schema
    # to inject a failing wrapper.
    # For this test we patch via env var read by the producer:
    monkeypatch.setenv("GVL_TEST_PRODUCER_RAISE", "1")
    with pytest.raises(RuntimeError, match="ProducerError|ProducerDied"):
        for _ in ds.to_dataloader(mode="double_buffered", batch_size=2,
                                   shuffle=False, drop_last=True,
                                   buffer_bytes=1 << 20, heartbeat_seconds=10):
            pass


@pytest.mark.slow
def test_shm_cleanup_after_close():
    """No leaked /dev/shm files after the loader is GC'd."""
    if not os.path.isdir("/dev/shm"):
        pytest.skip("Linux-only check")
    ds = gvl.get_dummy_dataset().with_seqs("reference").with_tracks(False)
    if not hasattr(ds, "_path") or ds._path is None:
        pytest.skip("requires file-backed dataset")
    before = set(os.listdir("/dev/shm"))
    loader = ds.to_dataloader(mode="double_buffered", batch_size=2,
                               shuffle=False, drop_last=True,
                               buffer_bytes=1 << 20)
    list(loader)
    # Close via dataset wrapper.
    if hasattr(loader.dataset, "_impl"):
        loader.dataset._impl.close()
    del loader
    import gc; gc.collect()
    after = set(os.listdir("/dev/shm"))
    leaked = {n for n in after - before if "gvl-" in n}
    assert not leaked, f"leaked shm: {leaked}"
```

- [ ] **Step 2: Wire the `GVL_TEST_PRODUCER_RAISE` hook into `_producer.py`**

Add to `producer_main`, right after dataset open:

```python
if os.environ.get("GVL_TEST_PRODUCER_RAISE") == "1":
    raise RuntimeError("test-injected producer failure")
```

(Add `import os` at the top.)

- [ ] **Step 3: Run, debug, verify**

```bash
pixi run -e dev pytest tests/unit/test_double_buffered_loader.py -v -m slow
```

If the heartbeat path doesn't surface the exception correctly, examine `_DoubleBufferedIterable.__iter__` and `_reraise_or_die`. If `/dev/shm` leaks appear, check the order of `close()` vs `unlink()` and add a fallback `unlink` in the `weakref.finalize` callback.

- [ ] **Step 4: Commit**

```bash
rtk git add tests/unit/test_double_buffered_loader.py python/genvarloader/_producer.py
rtk git commit -m "test(double_buffered): producer crash + shm cleanup"
```

---

## Task 14: Update the `genvarloader` skill

**Files:**
- Modify: `skills/genvarloader/SKILL.md`

Per `CLAUDE.md`: any change to the public `to_dataloader` signature requires a skill update.

- [ ] **Step 1: Inspect current skill**

```bash
rtk read skills/genvarloader/SKILL.md
```

Locate the section documenting `Dataset.to_dataloader`.

- [ ] **Step 2: Add new arguments**

Append (or replace the existing `to_dataloader` section with) a description of:

- `mode: Literal["buffered", "double_buffered"] | None = None` — prefetching mode. `None` preserves prior behavior.
- `buffer_bytes: int = 2 * 1024**3` — total RAM budget (split across slots for `double_buffered`).
- `copy: bool = True` — when `True`, yielded batches are independent of internal buffers; `False` returns zero-copy views invalidated by the next iteration.
- `heartbeat_seconds: float = 60.0` — for `double_buffered`, max wait per chunk before checking producer liveness.

Note preconditions:

- `with_seqs in {"haplotypes", "annotated"}` requires `deterministic=True`.
- Spliced datasets are not supported.
- `num_workers > 0` is rejected.
- `double_buffered` requires a file-backed dataset (opened via `Dataset.open(path)`).

- [ ] **Step 3: Commit**

```bash
rtk git add skills/genvarloader/SKILL.md
rtk git commit -m "docs(skill): document mode/buffer_bytes/copy/heartbeat_seconds on to_dataloader"
```

---

## Self-review

**Spec coverage check** (each spec section → task):

- §1 Motivation → Task 1-13 (whole plan).
- §2 Modes → Task 8 (`buffered`), Task 12 (`double_buffered`).
- §3 Public API → Task 8 (signature wiring); Task 14 (docs).
- §4 Determinism preconditions → Task 2 (gates in `_output_bytes_per_instance`), Task 8 (`test_buffered_rejects_nondeterministic_for_haplotypes`).
- §5 Exact footprint → Task 1 (`_allele_bytes_sum`), Tasks 2-5 (`_output_bytes_per_instance`).
- §6 Components → Task 6 (`_chunked`), Task 8 (`_buffered_loader`), Tasks 9-10 (`_shm_layout`), Task 11 (`_producer`), Task 12 (`_double_buffered_loader`).
- §7 Shm layout → Tasks 9-10.
- §8 Error handling / lifecycle / testing → Task 8 (gates), Task 13 (failure paths & cleanup).
- Out-of-scope items (DDP, non-deterministic for haps, spliced, auto mode) → enforced by gates in Tasks 2 and 8; not implemented.

**Placeholder scan:** none found. The `np.typeDict` fallback in Task 9 is a known-NumPy-version concern flagged inline; if the engineer hits it, replacement is one line.

**Type consistency:** `bytes_per_instance` is `NDArray[np.int64]` throughout. `slot_bytes` is `int` (per-slot budget). `peak_chunk_bytes` is `int`. `ChunkPlanner.__init__` is updated in Task 12 to precompute `peak_chunk_bytes` eagerly (callers depend on it).

**Caveats the executor should know:**

- Task 5 has an attribute-path placeholder (`self._tracks.intervals[track_name].dtype`) that must be verified against `_tracks.py` before implementation.
- Task 10's `RaggedVariants` step is intricate and may need to be split if the awkward layout work runs long.
- Task 12 depends on knowing the exact `Dataset` attribute that stores the on-disk path; verify by reading `Dataset.open` before writing the producer-spawn code.
