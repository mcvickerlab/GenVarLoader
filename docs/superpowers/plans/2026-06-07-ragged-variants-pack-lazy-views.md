# RaggedVariants pack/rc_ on lazy awkward views — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `RaggedVariants.to_packed()` and `rc_()` work on sliced/reversed/fancy-indexed views (currently crash), by fixing the upstream seqpro `unbox()` gap on `IndexedArray` and adding a numba allele-pack path in gvl.

**Architecture:** Two repos, hard ordering. **Part A (seqpro, land + release first):** teach the layout walkers `unbox()` and `_extract_list_offsets()` to project `IndexedArray`/`IndexedOptionArray`; this fixes every numeric field. **Part B (gvl, depends on new seqpro):** numeric fields then work with no change; the doubly-nested `alt`/`ref` fields get a numba `_pack_alleles` kernel + a layout-decomposition helper, gated by a canonical-layout check so the hot path is byte-identical. `rc_` on a non-canonical view becomes `self.to_packed().rc_(to_rc)` (returns a new object).

**Tech Stack:** Python, awkward-array, numba, seqpro (`/Users/david/projects/SeqPro`), pixi, pytest.

**Spec:** `docs/superpowers/specs/2026-06-07-ragged-variants-pack-lazy-views-design.md`

---

# Part A — seqpro upstream (`/Users/david/projects/SeqPro`)

> All Part A paths are under `/Users/david/projects/SeqPro`. Run commands from that repo root.
> Land and release this BEFORE touching gvl.

## Task A1: Failing tests for IndexedArray layouts

**Files:**
- Modify: `tests/test_rag_to_packed.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_rag_to_packed.py`:

```python
class TestIndexedLayouts:
    """An IndexedArray wrapper arises from indexing a *record* then pulling a
    field (e.g. ak.zip(...)[perm]["x"]). seqpro's layout walkers must traverse
    it. Plain Ragged[perm] yields a ListArray (already covered elsewhere)."""

    def test_indexed_field_unbox_and_to_packed(self):
        lengths = np.array([3, 0, 2, 4])
        perm = np.array([2, 0, 3, 1])
        data = np.arange(int(lengths.sum()), dtype=np.float64)
        r = Ragged.from_lengths(data, lengths)
        field = ak.zip({"x": r}, depth_limit=1)[perm]["x"]

        from awkward.contents import IndexedArray
        assert isinstance(field.layout, IndexedArray)

        rag = Ragged(field)              # used to raise: Expected 1 ragged dimension, got 0
        # accessors that route through unbox() must all work
        assert rag.offsets is not None
        assert rag.data is not None
        out = to_packed(rag)
        assert out.offsets.ndim == 1 and out.offsets[0] == 0
        assert out.is_contiguous
        assert ak.to_list(out) == ak.to_list(field)
        assert ak.to_list(out) == ak.to_list(ak.to_packed(field))

    def test_indexed_record_layout_offsets(self):
        # ak.zip(..., depth_limit=1) -> RecordArray; indexing it ->
        # IndexedArray(RecordArray(...)), which _extract_list_offsets must walk.
        r = Ragged.from_lengths(np.arange(9, dtype=np.float64), np.array([3, 2, 4]))
        rec = ak.zip({"a": r, "b": r}, depth_limit=1)[np.array([2, 0, 1])]

        from awkward.contents import IndexedArray
        assert isinstance(rec.layout, IndexedArray)

        rag = Ragged(rec)                # record-layout Ragged over an indexed layout
        # offsets extraction (via _extract_list_offsets) must not crash
        assert rag.offsets is not None
        assert ak.to_list(rag["a"]) == ak.to_list(rec["a"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_rag_to_packed.py::TestIndexedLayouts -v`
Expected: FAIL — `ValueError: Expected 1 ragged dimension, got 0` (first test) and
`ValueError: No list layer found ...` (second test).

> If `pixi run -e dev` is not the seqpro test task, use the repo's documented test command
> (check `pyproject.toml`/`pixi.toml`); the rest of the plan assumes `pixi run -e dev pytest`.

## Task A2: Project Indexed* in the layout walkers

**Files:**
- Modify: `python/seqpro/rag/_array.py` (imports; `_extract_list_offsets` `:85`; `unbox` `:759`)

- [ ] **Step 1: Add the IndexedArray imports**

In the `from awkward.contents import (...)` block (currently lines 9–17), add `IndexedArray` and
`IndexedOptionArray` so the block reads:

```python
from awkward.contents import (
    Content,
    EmptyArray,
    IndexedArray,
    IndexedOptionArray,
    ListArray,
    ListOffsetArray,
    NumpyArray,
    RecordArray,
    RegularArray,
)
```

- [ ] **Step 2: Handle Indexed* in `_extract_list_offsets`**

In `_extract_list_offsets` (the `while True:` loop), add an Indexed* branch as the first check:

```python
    node = layout
    while True:
        if isinstance(node, (IndexedArray, IndexedOptionArray)):
            node = node.project()
        elif isinstance(node, ListOffsetArray):
            return np.asarray(node.offsets.data)
        elif isinstance(node, ListArray):
            return np.stack([node.starts.data, node.stops.data], 0)  # type: ignore
        elif isinstance(node, RegularArray):
            node = node.content
        elif isinstance(node, RecordArray):
            node = node.content(0)
        else:
            raise ValueError(  # noqa: TRY004
                f"No list layer found while extracting offsets from layout:\n{layout.form}"
            )
```

- [ ] **Step 3: Handle Indexed* in `unbox`**

In `unbox`, extend the loop condition and add an Indexed* branch that projects and continues
without recording a dimension:

```python
    while isinstance(
        node,
        (ListArray, ListOffsetArray, RegularArray, RecordArray, IndexedArray, IndexedOptionArray),
    ):
        if isinstance(node, (IndexedArray, IndexedOptionArray)):
            node = node.project()
            continue
        if isinstance(node, RecordArray):
            raise ValueError(  # noqa: TRY004
                "Must extract a single field before unboxing a Ragged array of records."
            )
        elif isinstance(node, RegularArray):
            shape.append(node.size)
        else:
            shape.append(None)
            n_ragged += 1
            if isinstance(node, ListOffsetArray):
                offsets = node.offsets.data
            else:
                offsets = np.stack(  # pyrefly: ignore[no-matching-overload]  # awkward .data is ArrayLike, not _ArrayLike
                    [node.starts.data, node.stops.data],  # type: ignore
                    0,
                )

        node = node.content
```

- [ ] **Step 4: Update the `unbox` docstring (zero-copy caveat)**

Replace the line `Always zero-copy: the returned data is a view of the original array.` with:

```python
    Zero-copy when no IndexedArray/IndexedOptionArray is present. Indexed layouts
    (e.g. from fancy-indexing a record then extracting a field) are materialized via
    ``project()``, so the returned data is a fresh contiguous array, not a view.
```

- [ ] **Step 5: Run the new tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_rag_to_packed.py::TestIndexedLayouts -v`
Expected: PASS (both tests).

## Task A3: Full suite, version bump, release

- [ ] **Step 1: Run the full seqpro suite**

Run: `pixi run -e dev pytest -q`
Expected: PASS (existing `TestToPackedFlat`, ragged, rc, to_padded tests all green — proves no
regression on the `ListArray`/canonical paths).

- [ ] **Step 2: Bump version**

Bump `version` in `pyproject.toml` (e.g. `0.15.0 → 0.15.1`) per the repo's commitizen/semver
convention (a fix → patch bump).

- [ ] **Step 3: Commit and release**

```bash
git add python/seqpro/rag/_array.py tests/test_rag_to_packed.py pyproject.toml
git commit -m "fix(rag): traverse IndexedArray in unbox/_extract_list_offsets

unbox() and _extract_list_offsets() omitted IndexedArray/IndexedOptionArray
from their layout walk, so a Ragged over an indexed layout (ak.zip(...)[perm]
[field]) raised 'Expected 1 ragged dimension, got 0' / 'No list layer found'.
Project Indexed* nodes (materializes only when an index is present).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

Then publish per the repo's release process (the gvl pin in Part B must be able to resolve the new
version — local editable install or published artifact).

---

# Part B — gvl (`/Users/david/projects/GenVarLoader`)

> Depends on the released seqpro from Part A. All Part B paths are under
> `/Users/david/projects/GenVarLoader`.

## Task B1: Bump the seqpro pin and confirm numeric fields are fixed

**Files:**
- Modify: `pyproject.toml` (seqpro dependency constraint)
- Modify: `pixi.lock` (regenerated)
- Test: `tests/dataset/test_flat_variants.py`

- [ ] **Step 1: Write the numeric-field regression test**

Append to `tests/dataset/test_flat_variants.py` (helpers `_make_rv` already exist at the top of the
file):

```python
def test_to_packed_numeric_fields_reorder_after_fancy_index():
    # b=3, p=1; distinct starts so reorder is observable.
    group_off = [0, 2, 3, 5]
    rv = _make_rv(
        [b"A", b"C", b"G", b"T", b"N"],
        [b"a", b"c", b"g", b"t", b"n"],
        [10, 20, 30, 40, 50],
        group_off,
        ploidy=1,
    )
    fancy = RaggedVariants.from_ak(rv[np.array([2, 0])])
    got = fancy.to_packed()
    exp = ak.to_packed(ak.Array(fancy))
    np.testing.assert_array_equal(np.asarray(got["start"].data), np.asarray(exp["start"].data))
    assert ak.to_list(got["start"]) == ak.to_list(exp["start"])
```

- [ ] **Step 2: Bump the seqpro constraint**

In `pyproject.toml`, raise the seqpro lower bound to the Part A release (e.g. `seqpro>=0.15.1`).
Keep any existing genoray-compat upper bound (see the seqpro↔genoray version-coupling gotcha — if
genoray pins seqpro tightly, that pin must be compatible first).

- [ ] **Step 3: Regenerate the lock and install**

Run: `pixi install -e dev`
Expected: resolves seqpro to the Part A version. If resolution fails due to genoray coupling, use
the `pixi.lock` `merge=ours` rebase trick and reconcile genoray's seqpro pin before continuing.

- [ ] **Step 4: Run the numeric test — should now pass with no gvl code change**

Run: `pixi run -e dev pytest tests/dataset/test_flat_variants.py::test_to_packed_numeric_fields_reorder_after_fancy_index -v`
Expected: PASS (the existing field-wise `Ragged.to_packed()` path now handles the indexed numeric
field via the seqpro fix).

- [ ] **Step 5: Commit**

```bash
rtk git add pyproject.toml pixi.lock tests/dataset/test_flat_variants.py
rtk git commit -m "fix(deps): bump seqpro for IndexedArray unbox fix; numeric to_packed regression test"
```

## Task B2: `_pack_alleles` numba kernel + decomposition + canonical check

**Files:**
- Modify: `python/genvarloader/_dataset/_rag_variants.py` (imports; new module-level helpers)
- Test: `tests/dataset/test_flat_variants.py`

- [ ] **Step 1: Write failing unit tests for the helpers**

Append to `tests/dataset/test_flat_variants.py`:

```python
def test_pack_alleles_kernel_identity_and_reorder():
    from genvarloader._dataset._rag_variants import _pack_alleles

    # 3 variant rows, leaf "ACGTGG", alleles [ACG, T, GG]; rows: [v0,v1],[v2]
    leaf = np.frombuffer(b"ACGTGG", np.uint8)
    allele_starts = np.array([0, 3, 4], np.int64)
    allele_stops = np.array([3, 4, 6], np.int64)
    var_starts = np.array([0, 2], np.int64)   # row0 -> alleles[0:2], row1 -> alleles[2:3]
    var_stops = np.array([2, 3], np.int64)

    # identity order
    packed, allele_off, group_off = _pack_alleles(
        np.array([0, 1], np.int64), var_starts, var_stops, allele_starts, allele_stops, leaf
    )
    assert bytes(packed) == b"ACGTGG"
    assert allele_off.tolist() == [0, 3, 4, 6]
    assert group_off.tolist() == [0, 2, 3]

    # reversed row order
    packed, allele_off, group_off = _pack_alleles(
        np.array([1, 0], np.int64), var_starts, var_stops, allele_starts, allele_stops, leaf
    )
    assert bytes(packed) == b"GGACGT"
    assert allele_off.tolist() == [0, 2, 5, 6]
    assert group_off.tolist() == [0, 1, 3]


def test_is_canonical_alleles():
    from genvarloader._dataset._rag_variants import _is_canonical_alleles

    rv = _make_rv([b"A", b"C"], [b"a", b"c"], [1, 2], [0, 1, 2], ploidy=1)
    assert _is_canonical_alleles(rv["alt"].layout) is True
    fancy = RaggedVariants.from_ak(rv[np.array([1, 0])])
    assert _is_canonical_alleles(fancy["alt"].layout) is False


def test_decompose_alleles_reversed():
    from genvarloader._dataset._rag_variants import _decompose_alleles, _pack_alleles

    rv = _make_rv(
        [b"A", b"C", b"G", b"T", b"N"], [b"a", b"c", b"g", b"t", b"n"],
        [1, 2, 3, 4, 5], [0, 2, 3, 5], ploidy=1,
    )
    fancy = RaggedVariants.from_ak(rv[np.array([2, 0])])
    row_src, var_starts, var_stops, allele_starts, allele_stops, leaf, ploidy = _decompose_alleles(
        fancy["alt"]
    )
    assert ploidy == 1
    packed, allele_off, group_off = _pack_alleles(
        row_src, var_starts, var_stops, allele_starts, allele_stops, leaf
    )
    from genvarloader._dataset._haps import _build_allele_layout
    rebuilt = _build_allele_layout(packed, allele_off, group_off, ploidy)
    assert ak.to_list(rebuilt) == ak.to_list(fancy["alt"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run -e dev pytest tests/dataset/test_flat_variants.py -k "pack_alleles or canonical_alleles or decompose_alleles" -v`
Expected: FAIL — `ImportError: cannot import name '_pack_alleles'` (etc.).

- [ ] **Step 3: Add the IndexedArray imports**

In `python/genvarloader/_dataset/_rag_variants.py`, extend the awkward.contents import block
(currently lines 10–16) to include the indexed types:

```python
from awkward.contents import (
    Content,
    IndexedArray,
    IndexedOptionArray,
    ListArray,
    ListOffsetArray,
    NumpyArray,
    RegularArray,
)
```

- [ ] **Step 4: Implement the helpers**

Add at module level in `python/genvarloader/_dataset/_rag_variants.py` (after the imports, before
the `RaggedVariant`/`RaggedVariants` classes):

```python
def _is_canonical_alleles(layout: Content) -> bool:
    """True if an alt/ref layout is the canonical, directly-extractable chain
    ``RegularArray -> ListOffsetArray -> ListOffsetArray -> NumpyArray`` (possibly
    sliced, i.e. non-zero-based offsets — handled by the existing fast path). Any
    ``IndexedArray``/``ListArray`` wrapping (from fancy-index/reverse) returns False."""
    return (
        isinstance(layout, RegularArray)
        and isinstance(layout.content, ListOffsetArray)
        and isinstance(layout.content.content, ListOffsetArray)
        and isinstance(layout.content.content.content, NumpyArray)
    )


def _decompose_alleles(arr: ak.Array):
    """Decompose a (possibly non-canonical) (b, p, ~v, ~l) allele array into raw
    primitives for :func:`_pack_alleles`. Reads ``.starts``/``.stops`` (present on
    both ``ListArray`` and ``ListOffsetArray``) and the optional outer index.

    Returns ``(row_src, var_starts, var_stops, allele_starts, allele_stops, leaf, ploidy)``
    where ``row_src[b*p + h] = index[b]*p + h`` indexes the variant-list rows.
    """
    lay = arr.layout
    if isinstance(lay, (IndexedArray, IndexedOptionArray)):
        index = np.asarray(lay.index, np.int64)
        reg = lay.project() if isinstance(lay, IndexedOptionArray) else lay.content
        # For IndexedArray, content is the (un-indexed) RegularArray; for the option
        # case we project (gvl variants never contain None, but be safe).
        if isinstance(lay, IndexedOptionArray):
            index = None  # project() already applied the gather
    else:
        index = None
        reg = lay

    if not isinstance(reg, RegularArray):
        raise ValueError(
            f"Unsupported allele layout for packing: {arr.layout.form}"
        )
    ploidy = int(reg.size)

    var_node = reg.content
    var_starts = np.asarray(var_node.starts, np.int64)
    var_stops = np.asarray(var_node.stops, np.int64)

    allele_node = var_node.content
    allele_starts = np.asarray(allele_node.starts, np.int64)
    allele_stops = np.asarray(allele_node.stops, np.int64)
    leaf = np.asarray(allele_node.content.data).view(np.uint8)

    if index is None:
        n_out_rows = len(reg) * ploidy
        row_src = np.arange(n_out_rows, dtype=np.int64)
    else:
        row_src = (
            index[:, None] * ploidy + np.arange(ploidy, dtype=np.int64)
        ).reshape(-1)
    return row_src, var_starts, var_stops, allele_starts, allele_stops, leaf, ploidy


@nb.njit(nogil=True, cache=True)
def _pack_alleles(row_src, var_starts, var_stops, allele_starts, allele_stops, leaf):
    """Gather doubly-nested alleles into contiguous, zero-based byte buffers in
    canonical ``(b, p, ~v, ~l)`` row-major order. Sequential (offset accumulation);
    only invoked off the hot path for non-canonical layouts."""
    n_rows = row_src.shape[0]
    n_alleles = 0
    n_bytes = 0
    for i in range(n_rows):
        src = row_src[i]
        for a in range(var_starts[src], var_stops[src]):
            n_alleles += 1
            n_bytes += allele_stops[a] - allele_starts[a]

    packed = np.empty(n_bytes, np.uint8)
    allele_off = np.empty(n_alleles + 1, np.int64)
    group_off = np.empty(n_rows + 1, np.int64)
    allele_off[0] = 0
    group_off[0] = 0

    ai = 0
    bi = 0
    for i in range(n_rows):
        src = row_src[i]
        for a in range(var_starts[src], var_stops[src]):
            s = allele_starts[a]
            e = allele_stops[a]
            for k in range(s, e):
                packed[bi] = leaf[k]
                bi += 1
            ai += 1
            allele_off[ai] = bi
        group_off[i + 1] = ai
    return packed, allele_off, group_off
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pixi run -e dev pytest tests/dataset/test_flat_variants.py -k "pack_alleles or canonical_alleles or decompose_alleles" -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
rtk git add python/genvarloader/_dataset/_rag_variants.py tests/dataset/test_flat_variants.py
rtk git commit -m "feat(variants): numba allele-pack kernel + layout decomposition for non-canonical views"
```

## Task B3: Wire `to_packed` alt/ref to the kernel for non-canonical layouts

**Files:**
- Modify: `python/genvarloader/_dataset/_rag_variants.py` (`to_packed`, the `field in ("alt","ref")` branch ~213–237)
- Test: `tests/dataset/test_flat_variants.py`

- [ ] **Step 1: Write the failing to_packed tests (alt/ref on lazy views)**

Append to `tests/dataset/test_flat_variants.py`:

```python
@pytest.mark.parametrize("transform", ["reverse", "fancy"])
def test_to_packed_alt_ref_on_lazy_views(transform):
    group_off = [0, 2, 3, 5]
    rv = _make_rv(
        [b"ACG", b"T", b"GG", b"AA", b"C"], [b"A", b"CC", b"T", b"G", b"TT"],
        [1, 5, 9, 12, 20], group_off, ploidy=1,
    )
    view = rv[::-1] if transform == "reverse" else rv[np.array([2, 0, 3, 1])]
    view = RaggedVariants.from_ak(view)
    got = view.to_packed()
    exp = ak.to_packed(ak.Array(view))
    assert ak.to_list(got["alt"]) == ak.to_list(exp["alt"])
    assert ak.to_list(got["ref"]) == ak.to_list(exp["ref"])
    np.testing.assert_array_equal(np.asarray(got["start"].data), np.asarray(exp["start"].data))


def test_to_packed_explicit_listarray_variant_level():
    # Hand-build a variant-level ListArray (starts/stops), as the user bug report hit.
    from awkward.contents import ListArray, ListOffsetArray, RegularArray, NumpyArray
    from awkward.index import Index

    def listarray_alleles(joined_bytes, allele_off, starts, stops):
        leaf = NumpyArray(np.frombuffer(joined_bytes, np.uint8), parameters={"__array__": "byte"})
        allele = ListOffsetArray(
            Index(np.asarray(allele_off, np.int64)), leaf, parameters={"__array__": "bytestring"}
        )
        var = ListArray(Index(np.asarray(starts, np.int64)), Index(np.asarray(stops, np.int64)), allele)
        return ak.Array(RegularArray(var, 1))

    alt = listarray_alleles(b"ACGTGG", [0, 3, 4, 6], [0, 2], [2, 3])
    ref = listarray_alleles(b"ACCT", [0, 1, 3, 4], [0, 2], [2, 3])
    start = Ragged.from_offsets(np.array([1, 5, 9], np.int32), (2, None), np.array([0, 2, 3], np.int64))
    rv = RaggedVariants(alt=alt, start=start, ref=ref)

    got = rv.to_packed()
    exp = ak.to_packed(ak.Array(rv))
    assert ak.to_list(got["alt"]) == ak.to_list(exp["alt"])
    assert ak.to_list(got["ref"]) == ak.to_list(exp["ref"])
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest tests/dataset/test_flat_variants.py -k "lazy_views or explicit_listarray" -v`
Expected: FAIL — `AttributeError: 'ListArray'/'IndexedArray' object has no attribute ...`.

- [ ] **Step 3: Branch the alt/ref path on canonicality**

In `to_packed`, replace the body of the `if field in ("alt", "ref"):` branch (the block that calls
`_alt_layout_parts` and does the g0/a0 rebase, ~lines 213–237) with:

```python
            if field in ("alt", "ref"):
                if _is_canonical_alleles(arr.layout):
                    # fast path (unchanged): canonical (possibly sliced) layout
                    leaf, allele_off, group_off, ploidy = _alt_layout_parts(arr)
                    g0 = int(group_off[0])
                    rebased_group = np.asarray(group_off, np.int64) - g0
                    a0 = int(allele_off[g0])
                    sliced_allele_off = np.asarray(allele_off[g0:], np.int64) - a0
                    sliced_leaf = leaf[a0:]
                    allele_lvl = Ragged.from_offsets(
                        sliced_leaf.view("S1"),
                        (sliced_allele_off.size - 1, None),
                        sliced_allele_off,
                    ).to_packed()
                    packed[field] = _build_allele_layout(
                        np.asarray(allele_lvl.data).view(np.uint8),
                        np.asarray(allele_lvl.offsets),
                        rebased_group,
                        ploidy,
                    )
                else:
                    # non-canonical (IndexedArray/ListArray from slicing/reorder):
                    # numba gather, no ak.to_packed / awkward gather primitives.
                    (
                        row_src, var_starts, var_stops,
                        allele_starts, allele_stops, leaf, ploidy,
                    ) = _decompose_alleles(arr)
                    packed_bytes, allele_off, group_off = _pack_alleles(
                        row_src, var_starts, var_stops, allele_starts, allele_stops, leaf
                    )
                    packed[field] = _build_allele_layout(
                        packed_bytes, allele_off, group_off, ploidy
                    )
```

> The fast-path block is the existing code verbatim, now guarded by `_is_canonical_alleles`. The
> module-level `_decompose_alleles`/`_pack_alleles`/`_is_canonical_alleles` are in this same file —
> no import needed. `_alt_layout_parts` and `_build_allele_layout` keep their existing local import
> from `._haps` at the top of `to_packed`.

- [ ] **Step 4: Run the new + existing to_packed tests**

Run: `pixi run -e dev pytest tests/dataset/test_flat_variants.py -k "to_packed" -v`
Expected: PASS — new lazy-view/ListArray tests AND the existing
`test_to_packed_matches_awkward_contiguous` / `_sliced` (canonical fast path unchanged).

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_rag_variants.py tests/dataset/test_flat_variants.py
rtk git commit -m "fix(variants): to_packed() handles sliced/reordered alt/ref via numba kernel"
```

## Task B4: `rc_` on non-canonical views returns a new object

**Files:**
- Modify: `python/genvarloader/_dataset/_rag_variants.py` (`rc_`, ~246–281)
- Test: `tests/dataset/test_flat_variants.py`

- [ ] **Step 1: Write the failing rc_ test**

Append to `tests/dataset/test_flat_variants.py`:

```python
@pytest.mark.parametrize("transform", ["reverse", "fancy"])
def test_rc_on_lazy_views_matches_reference(transform):
    group_off = [0, 2, 3, 5]
    rv = _make_rv(
        [b"ACG", b"T", b"GG", b"AA", b"C"], [b"A", b"CC", b"T", b"G", b"TT"],
        [1, 5, 9, 12, 20], group_off, ploidy=1,
    )
    view = rv[::-1] if transform == "reverse" else rv[np.array([2, 0, 3, 1])]
    view = RaggedVariants.from_ak(view)

    n = view.shape[0]
    mask = np.ones(n, np.bool_)
    exp_alt, exp_ref = _ref_rc(view, mask)   # independent awkward reference (top of file)

    out = view.rc_(mask)
    assert ak.to_list(out["alt"]) == ak.to_list(exp_alt)
    assert ak.to_list(out["ref"]) == ak.to_list(exp_ref)
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest tests/dataset/test_flat_variants.py::test_rc_on_lazy_views_matches_reference -v`
Expected: FAIL — `ValueError: Expected 1 ragged dimension, got 0` (or `AttributeError`).

- [ ] **Step 3: Add the non-canonical short-circuit to `rc_`**

In `rc_`, immediately after the `to_rc is None` / `not to_rc.any()` handling and before the
`from ._haps import _alt_layout_parts` line, insert:

```python
        # Non-canonical (sliced/reordered) views can't be reverse-complemented in
        # place safely. Materialize a contiguous canonical copy, then recurse — the
        # recursion hits the in-place fast path below. Returns a new object; the sole
        # caller (reverse_complement_ragged) uses the return value.
        if any(
            not _is_canonical_alleles(self[f].layout)
            for f in ("alt", "ref")
            if f in self.fields
        ):
            return self.to_packed().rc_(to_rc)
```

- [ ] **Step 4: Run the rc_ tests (new + existing)**

Run: `pixi run -e dev pytest tests/dataset/test_flat_variants.py -k "rc" -v`
Expected: PASS — new lazy-view test AND existing `test_rc_matches_awkward` /
`test_rc_none_means_all` / `test_rc_ploidy2_broadcast` (canonical in-place path unchanged).

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_rag_variants.py tests/dataset/test_flat_variants.py
rtk git commit -m "fix(variants): rc_() on sliced/reordered views via materialized copy"
```

## Task B5: ploidy=2 reordered coverage + full regression

**Files:**
- Test: `tests/dataset/test_flat_variants.py`

- [ ] **Step 1: Add a ploidy=2 reordered test**

Append to `tests/dataset/test_flat_variants.py` (mirror the construction in the existing
`test_rc_ploidy2_broadcast`, which builds a (b, p, None) RaggedVariants directly):

```python
def test_to_packed_ploidy2_reordered():
    # b=2, p=2 -> 4 (b*p) rows; variant counts per row = [2, 1, 1, 1] -> 5 variants.
    # alt/ref carry exactly ONE allele per variant, so n_alleles == n_variants == 5.
    group_off = np.array([0, 2, 3, 4, 5], np.int64)
    # alt alleles: ["AC","G","T","GG","A"] -> b"ACGTGGA" (lengths 2,1,1,2,1)
    alt = _build_allele_layout(
        np.frombuffer(b"ACGTGGA", np.uint8),
        np.array([0, 2, 3, 4, 6, 7], np.int64),
        group_off, ploidy=2,
    )
    # ref alleles: ["a","c","g","t","n"] -> b"acgtn"
    ref = _build_allele_layout(
        np.frombuffer(b"acgtn", np.uint8),
        np.array([0, 1, 2, 3, 4, 5], np.int64),
        group_off, ploidy=2,
    )
    start = Ragged.from_offsets(
        np.array([1, 2, 3, 4, 5], np.int32), (2, 2, None), group_off
    )
    rv = RaggedVariants(alt=alt, start=start, ref=ref)
    fancy = RaggedVariants.from_ak(rv[np.array([1, 0])])  # swap the two batches
    got = fancy.to_packed()
    exp = ak.to_packed(ak.Array(fancy))
    assert ak.to_list(got["alt"]) == ak.to_list(exp["alt"])
    assert ak.to_list(got["ref"]) == ak.to_list(exp["ref"])
```

- [ ] **Step 2: Run the ploidy-2 test**

Run: `pixi run -e dev pytest tests/dataset/test_flat_variants.py::test_to_packed_ploidy2_reordered -v`
Expected: PASS.

- [ ] **Step 3: Full gvl test + lint + typecheck**

Run: `pixi run -e dev pytest tests/dataset/test_flat_variants.py -v`
Expected: PASS (all variants tests).

Run: `pixi run -e dev pytest -m "not slow"`
Expected: PASS (broader regression; CI also runs slow/torch/1kg tiers).

Run: `pixi run -e dev ruff check python/ && pixi run -e dev typecheck`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
rtk git add tests/dataset/test_flat_variants.py
rtk git commit -m "test(variants): ploidy=2 reordered to_packed coverage"
```

## Task B6: Skill / docs check

**Files:**
- Modify (if needed): `skills/genvarloader/SKILL.md`

- [ ] **Step 1: Check whether the skill mentions to_packed/rc_ behavior**

Run: `rtk grep -n "to_packed\|rc_\|RaggedVariants" skills/genvarloader/SKILL.md`

- [ ] **Step 2: Update if a documented gotcha applies**

These are bugfixes (no public-signature change), so a skill update is likely unnecessary. If a
"Common gotchas" entry would help (e.g. "you can slice/shuffle a `RaggedVariants` then `.to_packed()`
/ `.rc_()`"), add it; otherwise note "no change needed" and skip. Commit only if changed:

```bash
rtk git add skills/genvarloader/SKILL.md
rtk git commit -m "docs(skill): note to_packed/rc_ work on sliced RaggedVariants"
```

---

## Notes & non-goals

- **Ordering is hard:** Part A must be released and resolvable before Part B Step B1 can install it.
- **Canonical hot path is untouched:** `rc_` in `_getitem_unspliced`/`_getitem_spliced` always sees
  freshly-built canonical arrays → `_is_canonical_alleles` is True → existing in-place path, no
  kernel, no regression. Guarded by the existing `test_rc_matches_awkward*` tests.
- **Inner-axis slicing** (`rv[:, :, 1:]`) crashes inside awkward itself before reaching gvl — out of
  scope.
- **No `ak.to_packed`** is introduced in gvl; the only `project()` is inside seqpro's `unbox`, gated
  on an index actually being present.
