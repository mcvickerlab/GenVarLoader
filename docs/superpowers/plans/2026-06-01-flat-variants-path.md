# Flat variants path (FU-3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the per-batch awkward kernels from the `RaggedVariants` getitem path — the allele/genotype/dosage **gathers**, the eager-RC `rc_`, and the record-level `to_packed` — while `RaggedVariants` stays a public `ak.Array` subclass and outputs stay byte-identical.

**Architecture:** seqpro 0.14's numba-parallel `Ragged.to_packed()` (verified: `Ragged[idx]` fancy-index + `.to_packed()` dispatch zero awkward kernels) replaces the `ak.to_packed(<Ragged>[idx])` gathers as one-liners. `rc_` is reimplemented to extract alt/ref's shared leaf byte buffer from the `ak.Array` layout and reverse-complement only masked alleles in place via the already-shipped seqpro `reverse_complement_masked`. `RaggedVariants.to_packed()` becomes field-wise: `Ragged.to_packed()` per numeric field, and an allele-level seqpro pack + group-offset rebase + layout rebuild for the doubly-nested alt/ref. A shared `_build_allele_layout` helper (extracted from `_get_alleles`) is reused by the gather and the field-wise pack. `ak.zip` (record construction) is the documented remaining awkward.

**Tech Stack:** Python 3.10, numpy, numba, awkward (`ak.Array` container only), seqpro 0.14.0 (`seqpro.rag.Ragged`/`to_packed`/`reverse_complement_masked`), pytest, pixi (`pixi run -e dev …`), memray + py-spy for profiling.

---

## Spec

`docs/superpowers/specs/2026-06-01-flat-buffer-getitem-followups-design.md` (section **FU-3**).

## Orientation (read before starting)

- `python/genvarloader/_dataset/_rag_variants.py` — `RaggedVariants(ak.Array)` (`:35`), `__init__` (`ak.zip`, `:40`), `to_packed` (`:195`, `ak.to_packed(self)`), `rc_` (`:200`, eager `ak.where`+`reverse_complement`). **Modified: `to_packed`, `rc_`.**
- `python/genvarloader/_dataset/_haps.py` — `_get_variants` (`:635`, the gather call sites: `genos`/`dosage`/AF), `_get_alleles` (`:729`, allele gather + layout surgery). **Modified: both; `_build_allele_layout`/`_alt_layout_parts` helpers land here.**
- `python/genvarloader/_ragged.py` — `reverse_complement_masked(rag: Ragged[bytes], mask) -> Ragged[bytes]` (`:306`, in-place, seqpro flat kernel, reuses `_COMP`); `_COMP` LUT (`:280`).
- `python/genvarloader/_variants/_records.py` — `RaggedAlleles(Ragged[np.bytes_])` (`:13`); on-disk `alt`/`ref` are `Ragged` of `(n_variants, ~bytes)`.
- `tests/dataset/test_flat_getitem_snapshot.py` — byte-identity gate; `_flatten_output` (currently raises on `RaggedVariants`). **Modified: add variants case + RaggedVariants serialization.**
- `tests/dataset/test_no_awkward_in_hotpath.py` — awkward guard; `guard_dataset` fixture, `_install_ak_counters`. **Modified: add variants guard.**
- `tests/benchmarks/profiling/profile.py` — `--mode variants` driver.

### Key facts pinned from the code (verified empirically during design)

- `self.genotypes` and `self.dosages` are `Ragged`; `self.variants.alt`/`.ref` are `RaggedAlleles(Ragged)`. All support seqpro `.to_packed()`.
- `Ragged[idx]` (1-D `[v_idxs]` and 2-D `[r,s]`) returns a `Ragged` view with **0 awkward calls**; `.to_packed()` materializes a contiguous zero-based `Ragged` with **0 awkward calls**, byte-correct. So `<Ragged>[idx].to_packed()` is the awkward-free gather.
- `_get_alleles` builds the `(b,p,~v,~l)` `ak.Array` as: leaf `NumpyArray(uint8, parameters={"__array__":"byte"})` → `ListOffsetArray(allele_offsets, leaf, parameters={"__array__":"bytestring"})` → `ListOffsetArray(group_offsets, …)` → `RegularArray(…, ploidy)`.
- Inverse extraction from such an `ak.Array` `a`: `lay = a.layout`; `ploidy = lay.size`; `group_offsets = np.asarray(lay.content.offsets)`; `allele_offsets = np.asarray(lay.content.content.offsets)`; `leaf_uint8 = np.asarray(lay.content.content.content.data)`. The leaf **shares memory** with the underlying buffer (so in-place mutation is visible).
- `reverse_complement_masked(rag, mask)` mutates `rag` in place (`copy=False`), reverses+complements only `mask`-True rows, reuses `_COMP`; `mask` is one entry per flattened ragged row (replicated across inner fixed axes if smaller). For a 1-level `(n_alleles, None)` view pass a length-`n_alleles` mask (no replication).
- `to_rc` in `rc_` is length `shape[0] == b` (batch). Per-allele mask: `np.repeat(np.repeat(to_rc, ploidy), np.diff(group_offsets))`.
- AF path: `genos[_keep]` (bool Ragged-mask) is awkward-free; `ak.to_regular(genos[_keep], 1)` dispatches awkward (1 call). Keep `to_regular`, swap only the `to_packed`.

## Regression gate (run after each task)

1. **Byte-identity snapshot** (extended in Task 0): `pixi run -e dev pytest tests/dataset/test_flat_getitem_snapshot.py -q`
2. **Dataset suite:** `pixi run -e dev pytest tests/dataset -q -m "not slow"`

Commit with `--no-verify` (pre-commit pyrefly fails in worktrees without the built Rust ext — documented gotcha). Run `pixi run -e dev ruff format python/ tests/` before committing to satisfy the pre-push ruff-format hook.

---

## Task 0: Variants byte-identity characterization

Lock the current `with_seqs("variants")` output to a committed snapshot before any refactor, so every later task proves byte-identity.

**Files:**
- Modify: `tests/dataset/test_flat_getitem_snapshot.py`
- Create (generated, committed): `tests/dataset/_snapshots/variants_ragged.npz`

- [ ] **Step 1: Add `RaggedVariants` serialization to `_flatten_output`**

In `tests/dataset/test_flat_getitem_snapshot.py`, add an import and a branch. Import near the top:

```python
from genvarloader import RaggedVariants
```

Add this branch in `_flatten_output` **before** the final `isinstance(obj, np.ndarray)` / `raise` branch:

```python
    elif isinstance(obj, RaggedVariants):
        # Serialize each field to plain arrays. Numeric fields -> data+offsets.
        # alt/ref are doubly-nested bytes -> leaf bytes + both offset levels.
        for fld in sorted(obj.fields):
            v = obj[fld]
            if fld in ("alt", "ref"):
                lay = v.layout
                out[f"{fld}_group_off"] = np.asarray(lay.content.offsets)
                out[f"{fld}_allele_off"] = np.asarray(lay.content.content.offsets)
                out[f"{fld}_bytes"] = np.asarray(lay.content.content.content.data)
                out[f"{fld}_ploidy"] = np.asarray(lay.size)
            else:
                out[f"{fld}_data"] = np.asarray(v.data)
                out[f"{fld}_off"] = np.asarray(v.offsets)
```

- [ ] **Step 2: Add the variants case to `CASES`**

Add to the `CASES` list (variants is seqs-only; tracks must be off so the return is a bare `RaggedVariants`, not a tuple):

```python
    ("variants_ragged", dict(seqs="variants"), "ragged"),
```

Confirm `_build` turns tracks off for seqs-only cases (it calls `with_tracks(False)` when `tracks is None`). `with_seqs("variants")` is valid for the fixture (it has a VCF with variants).

- [ ] **Step 3: Generate the snapshot (first run writes + skips)**

Run: `pixi run -e dev pytest tests/dataset/test_flat_getitem_snapshot.py -k variants_ragged -q`
Expected: 1 SKIP — "wrote snapshot variants_ragged.npz".

If it instead ERRORS (e.g. `_flatten_output` can't reach `lay.content...`), the fixture's variants output layout differs from the documented `(b,p,~v,~l)` — inspect `obj.layout` and adjust the extraction indices, then re-run. If `with_seqs("variants")` raises (no variant fields), report NEEDS_CONTEXT.

- [ ] **Step 4: Re-run to confirm the gate passes**

Run: `pixi run -e dev pytest tests/dataset/test_flat_getitem_snapshot.py -q`
Expected: all cases PASS (now 10, incl. `variants_ragged`).

- [ ] **Step 5: Commit**

```bash
pixi run -e dev ruff format tests/
git add tests/dataset/test_flat_getitem_snapshot.py tests/dataset/_snapshots/variants_ragged.npz
git commit --no-verify -m "test(dataset): byte-identical snapshot for RaggedVariants output"
```

---

## Task 1: Extract `_build_allele_layout` + `_alt_layout_parts` helpers (pure refactor)

Factor the allele-layout build/extract used by both the gather (Task 2) and the field-wise pack (Task 4). Byte-identical — the snapshot must stay green.

**Files:**
- Modify: `python/genvarloader/_dataset/_haps.py` (`_get_alleles` + two new module/helper functions)

- [ ] **Step 1: Add `_build_allele_layout`**

In `_haps.py`, add a module-level helper (place near `_get_alleles`). It builds the `(b, p, ~v, ~l)` `ak.Array` from flat parts — extracted verbatim from the layout surgery currently inside `_get_alleles`:

```python
def _build_allele_layout(
    data: NDArray[np.uint8],
    allele_offsets: NDArray[np.integer],
    group_offsets: NDArray[np.integer],
    ploidy: int,
) -> ak.Array:
    """Wrap flat allele bytes + two offset levels into a (b, p, ~v, ~l) ak.Array.

    ``data`` is the contiguous allele byte buffer (uint8). ``allele_offsets`` are the
    per-variant byte boundaries (len n_alleles + 1). ``group_offsets`` are the
    per-(b*p)-row variant boundaries (len b*p + 1). Both offset arrays must be
    zero-based. ``ploidy`` groups the b*p rows into the outer regular axis.
    """
    leaf = NumpyArray(np.ascontiguousarray(data), parameters={"__array__": "byte"})
    l_content = ListOffsetArray(
        Index(np.asarray(allele_offsets, np.int64)),
        leaf,
        parameters={"__array__": "bytestring"},
    )
    vl_content = ListOffsetArray(Index(np.asarray(group_offsets, np.int64)), l_content)
    pvl_content = RegularArray(vl_content, ploidy)
    return ak.Array(pvl_content)
```

Ensure the imports exist at the top of `_haps.py` (they are used by the current `_get_alleles`): `from awkward.contents import ListOffsetArray, NumpyArray, RegularArray` and `from awkward.index import Index`. Add any that are missing.

- [ ] **Step 2: Add `_alt_layout_parts`**

```python
def _alt_layout_parts(
    arr: ak.Array,
) -> tuple[NDArray[np.uint8], NDArray[np.int64], NDArray[np.int64], int]:
    """Inverse of :func:`_build_allele_layout`: extract (leaf_uint8, allele_offsets,
    group_offsets, ploidy) from a (b, p, ~v, ~l) allele ak.Array.

    The returned ``leaf_uint8`` shares memory with ``arr``'s buffer, so mutating it
    mutates ``arr`` in place.
    """
    lay = arr.layout
    ploidy = int(lay.size)
    group_offsets = np.asarray(lay.content.offsets, np.int64)
    allele_offsets = np.asarray(lay.content.content.offsets, np.int64)
    leaf = np.asarray(lay.content.content.content.data).view(np.uint8)
    return leaf, allele_offsets, group_offsets, ploidy
```

- [ ] **Step 3: Rewrite `_get_alleles` to use `_build_allele_layout`**

Replace the layout-surgery block in `_get_alleles` (the `node = alleles.layout` … `RegularArray(...)` … `ak.Array(pvl_content)` lines) so it delegates to the helper. The gather still uses the current `ak.to_packed(... [v_idxs])` for now (Task 2 swaps it):

```python
    def _get_alleles(
        self, genos: Ragged[V_IDX_TYPE], kind: Literal["alt", "ref"]
    ) -> ak.Array:
        v_idxs = genos.data
        # (b*p*v ~l) packed allele bytes for the selected variants
        alleles = ak.to_packed(
            cast(RaggedAlleles, getattr(self.variants, kind)[v_idxs])
        )
        return _build_allele_layout(
            np.asarray(alleles.data).view(np.uint8),
            np.asarray(alleles.offsets),
            np.asarray(genos.offsets),
            genos.shape[-2],
        )
```

(`genos.shape[-2]` is ploidy `p`.)

- [ ] **Step 4: Run both gates (byte-identical — no behavior change)**

```bash
pixi run -e dev pytest tests/dataset/test_flat_getitem_snapshot.py -q
pixi run -e dev pytest tests/dataset -q -m "not slow"
```
Expected: all PASS (incl. `variants_ragged`). If `variants_ragged` regresses, the helper's offset dtypes or ploidy differ from the inline version — diff against the pre-refactor layout.

- [ ] **Step 5: Commit**

```bash
pixi run -e dev ruff format python/
git add python/genvarloader/_dataset/_haps.py
git commit --no-verify -m "refactor(haps): extract _build_allele_layout/_alt_layout_parts helpers"
```

---

## Task 2: C1 — gather swaps to seqpro `to_packed`

Replace every `ak.to_packed(<Ragged>[idx])` gather in the variants path with `<Ragged>[idx].to_packed()`. Byte-identical.

**Files:**
- Modify: `python/genvarloader/_dataset/_haps.py` (`_get_variants`, `_get_alleles`)

- [ ] **Step 1: Swap the genotype + dosage + AF gathers in `_get_variants`**

In `_get_variants`, change the genotype pack:

```python
        # (b p ~v)
        genos = cast(Ragged[V_IDX_TYPE], self.genotypes[r, s]).to_packed()
        v_idxs = genos.data
```

In the AF-filter branch, swap the pack but keep `ak.to_regular` (it is the only remaining awkward in this opt-in path):

```python
            _keep = Ragged.from_offsets(keep, genos.shape, genos.offsets)
            genos = ak.to_regular(genos[_keep], 1).to_packed()
            v_idxs = genos.data
```

> Note: `ak.to_regular(...)` returns an awkward Array; confirm `.to_packed()` resolves to seqpro's `Ragged.to_packed` — if `ak.to_regular` returns a non-`Ragged`, wrap as `Ragged(ak.to_regular(genos[_keep], 1)).to_packed()`. Verify by reading the type at runtime; the byte-identity gate confirms correctness either way.

In the dosage branch:

```python
        if self.dosages is not None and "dosage" in self.var_fields:
            dosages = self.dosages[r, s]
            if _keep is not None:
                dosages = ak.to_regular(dosages[_keep], 1)
            fields["dosage"] = Ragged(dosages).to_packed() if not isinstance(dosages, Ragged) else dosages.to_packed()
```

> Keep it simple: if `self.dosages[r, s]` already returns a `Ragged`, use `dosages.to_packed()`; the `_keep`/`to_regular` sub-branch may yield an awkward Array, so wrap with `Ragged(...)` before `.to_packed()`. Confirm the runtime types and pick the minimal correct form; the snapshot gate is the check.

- [ ] **Step 2: Swap the allele gather in `_get_alleles`**

```python
        v_idxs = genos.data
        alleles = cast(RaggedAlleles, getattr(self.variants, kind)[v_idxs]).to_packed()
        return _build_allele_layout(
            np.asarray(alleles.data).view(np.uint8),
            np.asarray(alleles.offsets),
            np.asarray(genos.offsets),
            genos.shape[-2],
        )
```

- [ ] **Step 3: Run both gates**

```bash
pixi run -e dev pytest tests/dataset/test_flat_getitem_snapshot.py -q
pixi run -e dev pytest tests/dataset -q -m "not slow"
```
Expected: all PASS. If the AF path is exercised by a slow/marked test, also run any `min_af`/`max_af` variants test you can find (`grep -rn "min_af\|max_af" tests/`).

- [ ] **Step 4: Commit**

```bash
pixi run -e dev ruff format python/
git add python/genvarloader/_dataset/_haps.py
git commit --no-verify -m "perf(variants): seqpro to_packed for allele/genotype/dosage gathers"
```

---

## Task 3: C2 — flat `rc_`

Reimplement `RaggedVariants.rc_` to reverse-complement only masked alleles in place on the shared leaf buffer via seqpro `reverse_complement_masked`, replacing the eager `ak.where(to_rc, reverse_complement(...), ...)` + `ak.to_packed`.

**Files:**
- Modify: `python/genvarloader/_dataset/_rag_variants.py` (`rc_`)
- Test: `tests/dataset/test_flat_variants.py` (new)

- [ ] **Step 1: Write the failing byte-identity unit test**

Create `tests/dataset/test_flat_variants.py`. It builds a `RaggedVariants` two ways isn't needed — instead it compares the NEW `rc_` against a reference computed with the OLD awkward idiom inline, so it is self-contained:

```python
import numpy as np
import awkward as ak
import pytest

from genvarloader import RaggedVariants
from genvarloader._ragged import reverse_complement  # the awkward reference
from seqpro.rag import Ragged
from genvarloader._dataset._haps import _build_allele_layout


def _make_rv(alt_rows, ref_rows, starts, group_off, ploidy):
    """alt_rows/ref_rows: list[bytes] per variant; group_off: variant boundaries per (b*p) row."""
    def alleles(rows):
        data = np.frombuffer(b"".join(rows), np.uint8)
        off = np.concatenate([[0], np.cumsum([len(r) for r in rows])]).astype(np.int64)
        return _build_allele_layout(data, off, np.asarray(group_off, np.int64), ploidy)
    alt = alleles(alt_rows)
    ref = alleles(ref_rows)
    n = len(starts)
    start = Ragged.from_offsets(np.asarray(starts, np.int32), (len(group_off) - 1, None), np.asarray(group_off, np.int64))
    return RaggedVariants(alt=alt, start=start, ref=ref)


def _ref_rc(rv, to_rc):
    """Old awkward idiom, computed independently."""
    alt = ak.to_packed(ak.where(to_rc, reverse_complement(rv["alt"]), rv["alt"]))
    ref = ak.to_packed(ak.where(to_rc, reverse_complement(rv["ref"]), rv["ref"]))
    return alt, ref


@pytest.mark.parametrize("mask", [
    np.array([True, True]),    # all
    np.array([False, False]),  # none (early return)
    np.array([True, False]),   # mixed
])
def test_rc_matches_awkward(mask):
    # b=2, p=1, group_off over 2 rows: row0 has 2 variants, row1 has 1
    group_off = [0, 2, 3]
    rv = _make_rv([b"ACG", b"T", b"GG"], [b"A", b"CC", b"T"], [1, 5, 9], group_off, ploidy=1)
    exp_alt, exp_ref = _ref_rc(rv, mask)
    rv.rc_(mask)
    np.testing.assert_array_equal(ak.to_list(rv["alt"]), ak.to_list(exp_alt))
    np.testing.assert_array_equal(ak.to_list(rv["ref"]), ak.to_list(exp_ref))


def test_rc_none_means_all():
    group_off = [0, 2, 3]
    rv = _make_rv([b"ACG", b"T", b"GG"], [b"A", b"CC", b"T"], [1, 5, 9], group_off, ploidy=1)
    exp_alt, exp_ref = _ref_rc(rv, np.ones(2, bool))
    rv.rc_(None)
    np.testing.assert_array_equal(ak.to_list(rv["alt"]), ak.to_list(exp_alt))
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest tests/dataset/test_flat_variants.py -q`
Expected: FAIL — the current `rc_` uses the eager awkward path; tests fail only if the new impl is wrong, so this primarily fails to compile against the new behavior once Step 3 lands. (If it PASSES against the old code, that is fine — it just means old==new; proceed to Step 3 and keep it green.)

> If the old `rc_` already makes these pass, the test still guards the refactor. The point is byte-identity; do not weaken it.

- [ ] **Step 3: Reimplement `rc_`**

In `python/genvarloader/_dataset/_rag_variants.py`, replace the body of `rc_` (keep the signature, docstring, and the `to_rc is None` / `not to_rc.any()` guards):

```python
    def rc_(self, to_rc: NDArray[np.bool_] | None = None) -> Self:
        if to_rc is None:
            to_rc = np.ones(self.shape[0], np.bool_)
        elif not to_rc.any():
            return self

        from .._ragged import _COMP, reverse_complement_masked
        from seqpro.rag import Ragged
        from ._haps import _alt_layout_parts

        for field in ("alt", "ref"):
            if field not in self.fields:
                continue
            arr = self[field]
            leaf, allele_off, group_off, ploidy = _alt_layout_parts(arr)
            # per-allele mask: to_rc is per-batch; broadcast across ploidy then variants
            per_bp = np.repeat(np.ascontiguousarray(to_rc, np.bool_), ploidy)
            per_allele = np.repeat(per_bp, np.diff(group_off))
            view = Ragged.from_offsets(
                leaf.view("S1"), (per_allele.size, None), allele_off
            )
            # in-place: mutates `leaf`, which shares memory with `arr`'s buffer
            reverse_complement_masked(view, per_allele)

        return self
```

> The leaf buffer shares memory with the `ak.Array` (verified), so mutating `view` (which wraps `leaf`) reverse-complements `self["alt"]`/`self["ref"]` in place — preserving `rc_`'s in-place contract without `ak.where`/`ak.to_packed`. Confirm `NDArray`/`Self` are imported in this module (they are used in the existing signature).

- [ ] **Step 4: Run unit + both gates**

```bash
pixi run -e dev pytest tests/dataset/test_flat_variants.py -q
pixi run -e dev pytest tests/dataset/test_flat_getitem_snapshot.py -q
pixi run -e dev pytest tests/dataset -q -m "not slow"
```
Expected: all PASS. (The snapshot `variants_ragged` exercises `rc_` only if the fixture has `-` strand regions + `rc_neg`; the dataset suite covers strand RC. If no snapshot delta, the unit test is the primary guard.)

- [ ] **Step 5: Commit**

```bash
pixi run -e dev ruff format python/ tests/
git add python/genvarloader/_dataset/_rag_variants.py tests/dataset/test_flat_variants.py
git commit --no-verify -m "perf(variants): flat in-place rc_ via seqpro reverse_complement_masked"
```

---

## Task 4: C3 — field-wise `RaggedVariants.to_packed()`

Replace `ak.to_packed(self)` with per-field packing: `Ragged.to_packed()` for numeric fields; allele-level seqpro pack + group-offset rebase + `_build_allele_layout` for alt/ref.

**Files:**
- Modify: `python/genvarloader/_dataset/_rag_variants.py` (`to_packed`)
- Test: `tests/dataset/test_flat_variants.py` (append)

- [ ] **Step 1: Write the failing byte-identity test (incl. a sliced/scattered input)**

Append to `tests/dataset/test_flat_variants.py`:

```python
def test_to_packed_matches_awkward_contiguous():
    group_off = [0, 2, 3]
    rv = _make_rv([b"ACG", b"T", b"GG"], [b"A", b"CC", b"T"], [1, 5, 9], group_off, ploidy=1)
    exp = ak.to_packed(ak.Array(rv))  # old behavior
    got = rv.to_packed()
    assert ak.to_list(got["alt"]) == ak.to_list(exp["alt"])
    assert ak.to_list(got["ref"]) == ak.to_list(exp["ref"])
    np.testing.assert_array_equal(np.asarray(got["start"].data), np.asarray(exp["start"].data))


def test_to_packed_matches_awkward_sliced():
    # a sliced RaggedVariants has non-zero-based / scattered offsets -> to_packed must contiguate
    group_off = [0, 2, 3, 5]
    rv = _make_rv(
        [b"ACG", b"T", b"GG", b"AA", b"C"],
        [b"A", b"CC", b"T", b"G", b"TT"],
        [1, 5, 9, 12, 20],
        group_off, ploidy=1,
    )
    sliced = rv[1:]            # drop the first (b,p) row
    exp = ak.to_packed(ak.Array(sliced))
    got = sliced.to_packed()
    assert ak.to_list(got["alt"]) == ak.to_list(exp["alt"])
    assert ak.to_list(got["ref"]) == ak.to_list(exp["ref"])
    np.testing.assert_array_equal(np.asarray(got["start"].data), np.asarray(exp["start"].data))
```

- [ ] **Step 2: Run to verify (guards the refactor)**

Run: `pixi run -e dev pytest tests/dataset/test_flat_variants.py -k to_packed -q`
Expected: PASS against the current `ak.to_packed(self)`; this pins behavior so Step 3 must stay byte-identical.

- [ ] **Step 3: Reimplement `to_packed` field-wise**

Replace `RaggedVariants.to_packed`:

```python
    def to_packed(self) -> Self:
        from seqpro.rag import Ragged
        from ._haps import _alt_layout_parts, _build_allele_layout

        packed = {}
        for field in self.fields:
            arr = self[field]
            if field in ("alt", "ref"):
                leaf, allele_off, group_off, ploidy = _alt_layout_parts(arr)
                # pack the allele (byte) level: contiguates bytes, zero-bases allele_off
                allele_lvl = Ragged.from_offsets(
                    leaf.view("S1"), (allele_off.size - 1, None), allele_off
                ).to_packed()
                # group_off may be non-zero-based (sliced view) -> rebase
                rebased_group = np.asarray(group_off, np.int64) - int(group_off[0])
                packed[field] = _build_allele_layout(
                    np.asarray(allele_lvl.data).view(np.uint8),
                    np.asarray(allele_lvl.offsets),
                    rebased_group,
                    ploidy,
                )
            else:
                packed[field] = Ragged(arr).to_packed() if not isinstance(arr, Ragged) else arr.to_packed()
        return type(self)(**packed)
```

> `type(self)(**packed)` reuses `RaggedVariants.__init__`, which `ak.zip`s the packed fields (the one remaining, documented awkward call — cheap layout wrap). For the allele level, `allele_off.size - 1` is the number of variants; `to_packed` reorders by the existing allele order, which is `(b,p,variant)` row-major — matching `ak.to_packed`'s canonical order. Confirm at runtime whether `self[field]` for numeric fields returns a seqpro `Ragged` (has `.to_packed()`) or a bare `ak.Array`; the `isinstance(arr, Ragged)` guard handles both, and the byte-identity test is the check.

- [ ] **Step 4: Run unit + both gates**

```bash
pixi run -e dev pytest tests/dataset/test_flat_variants.py -q
pixi run -e dev pytest tests/dataset/test_flat_getitem_snapshot.py -q
pixi run -e dev pytest tests/dataset -q -m "not slow"
```
Expected: all PASS. If `test_to_packed_matches_awkward_sliced` fails, the group-offset rebase or the allele-level row order diverges — compare `ak.to_list` of got vs exp to localize.

- [ ] **Step 5: Commit**

```bash
pixi run -e dev ruff format python/ tests/
git add python/genvarloader/_dataset/_rag_variants.py tests/dataset/test_flat_variants.py
git commit --no-verify -m "perf(variants): field-wise RaggedVariants.to_packed (seqpro + layout rebuild)"
```

---

## Task 5: Awkward guard + re-profile + REGRESSIONS write-up

**Files:**
- Modify: `tests/dataset/test_no_awkward_in_hotpath.py`
- Modify: `docs/superpowers/REGRESSIONS.md`

- [ ] **Step 1: Add a variants awkward-guard test**

Read the existing `guard_dataset` fixture + `_install_ak_counters` in `tests/dataset/test_no_awkward_in_hotpath.py`. `_install_ak_counters` patches `ak.to_numpy`/`ak.to_packed`/`ak.flatten`/`ak.where`. For variants, `ak.zip` (record construction) is expected and allowed, so assert on the kernels we removed — patch `ak.to_packed`, `ak.where`, and (if patchable) `ak.str.reverse` — and assert those are 0, while documenting `ak.zip` is the allowed remainder. Add:

```python
def test_variants_ragged_minimal_awkward(monkeypatch, guard_dataset):
    """Variants gather + rc_ + to_packed must dispatch no awkward kernels.
    ak.zip (record construction) is the documented remaining awkward and is NOT patched here."""
    calls = _install_ak_counters(monkeypatch)  # patches to_numpy/to_packed/flatten/where
    ds = guard_dataset.with_seqs("variants").with_tracks(False)
    regions = list(range(min(4, ds.shape[0])))
    samples = [i % ds.shape[1] for i in range(len(regions))]
    rv = ds[regions, samples]
    rv.rc_(np.ones(len(regions), np.bool_))   # exercise rc_ explicitly
    rv.to_packed()                             # exercise field-wise to_packed
    assert calls["n"] == 0, "variants gather/rc_/to_packed dispatched awkward (to_packed/where/flatten/to_numpy)"
```

> If `ds[regions, samples]` for variants returns a tuple (tracks somehow on), use `with_tracks(False)` (already applied) and unwrap. If the guard fails because a path still calls a patched function, investigate — it reveals a missed swap; report DONE_WITH_CONCERNS rather than weakening the assert. Note: `ak.zip` is intentionally not in the patched set.

- [ ] **Step 2: Run the guard + full gates**

```bash
pixi run -e dev pytest tests/dataset/test_no_awkward_in_hotpath.py -q
pixi run -e dev pytest tests/dataset -q -m "not slow"
pixi run -e dev ruff check python/
```
Expected: all PASS / clean.

- [ ] **Step 3: Re-profile variants (memray now; py-spy handed to David)**

```bash
pixi run -e dev memray run -fo tests/benchmarks/profiling/variants.fu3.memray.bin tests/benchmarks/profiling/profile.py --mode variants
pixi run -e dev memray stats tests/benchmarks/profiling/variants.fu3.memray.bin 2>&1 | head -40
```
Compare total allocated + top frames to `tests/benchmarks/profiling/variants.final.memray.bin` (the post-PR-#205 state). Tell David: "Run `sudo bash tests/benchmarks/profiling/run_pyspy.sh` (variants mode) for the FU-3 py-spy A/B." Do NOT run py-spy yourself. Do NOT commit `.memray.bin`.

- [ ] **Step 4: Update REGRESSIONS.md**

In `docs/superpowers/REGRESSIONS.md`, add an "FU-3: flat variants path" subsection: memray total-allocated + key-frame delta vs the PR-#205 variants baseline (the `ak.to_packed`/`_carry`/`ak.str` frames should drop; `ak.zip` remains). Mark the py-spy self-time line PENDING the maintainer's run (do not fabricate). Note `ak.zip` as the documented residual.

- [ ] **Step 5: Commit**

```bash
pixi run -e dev ruff format python/ tests/
git add tests/dataset/test_no_awkward_in_hotpath.py docs/superpowers/REGRESSIONS.md
git commit --no-verify -m "test(variants): awkward guard + FU-3 memray A/B write-up"
```

- [ ] **Step 6: Confirm public API unchanged**

`RaggedVariants` stays an `ak.Array` subclass with the same fields/return type; no `__init__.py __all__` change. Confirm `skills/genvarloader/SKILL.md` needs no update (FU-3 is internal-only). State "no skill change needed" or update if a signature drifted.

---

## Self-review notes (for the executor)

- **Task 0 is the linchpin** — the `variants_ragged` snapshot + the `_flatten_output` RaggedVariants serialization must land first and pass, or later tasks have no byte-identity gate. If the fixture's variants layout differs from the documented `(b,p,~v,~l)`, fix the extraction indices in Task 0 before proceeding.
- **Shared helpers (Task 1) before the swaps** — `_build_allele_layout`/`_alt_layout_parts` are used by Tasks 2/3/4; landing them as a pure byte-identical refactor first keeps each later diff small.
- **In-place leaf mutation** (Task 3) depends on `_alt_layout_parts` returning a buffer that shares memory with the `ak.Array` (verified in design). If a future awkward version copies on layout access, `rc_` would silently no-op — the unit test catches it.
- **AF-filter + dosage** (Task 2) keep one `ak.to_regular` in the opt-in `min_af`/`max_af` path; that is acceptable and documented. Verify runtime types when choosing `Ragged(...).to_packed()` vs `x.to_packed()`.
- **`ak.zip` stays** — the guard test (Task 5) intentionally does not patch it; it is the documented floor of the frozen `ak.Array` container.
- **Profiling is David-run on macOS** (sudo py-spy); the agent runs only memray.
