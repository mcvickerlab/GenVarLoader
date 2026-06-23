# RaggedVariants subclasses `Ragged` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make GenVarLoader's `RaggedVariants` a true subclass of `seqpro.rag.Ragged` (`__slots__ = ()`, no `_rag` composition), deleting every structural override now provided correctly by seqpro 0.18.0's numpy-consistent + subclass-preserving base, while preserving all current behavior.

**Architecture:** seqpro 0.18.0 shipped two changes that make composition unnecessary: (PR1) `Ragged.__getitem__` is numpy-consistent for multi-leading-axis records (`A[x] == A[(x,)]`), and (PR2) `__getitem__`/`reshape`/`squeeze`/`to_packed` preserve the concrete subclass via `_with_layout`. So `RaggedVariants` becomes `class RaggedVariants(Ragged)` holding only the record `_layout` it inherits; its `__init__` builds the record and calls `super().__init__`; the structural overrides (`__getitem__`, `__getattr__`, `reshape`, `squeeze`, `to_packed`, `shape`, `fields`, `__len__`) are deleted and inherited from base. Domain methods/properties (`alt`/`ref`/`start`/`dosage`/`ilen`/`end`, `rc_`, `pad`, `to_nested_tensor_batch`, `_alt_chars`) are kept, rewired from `self._rag[...]` to `self[...]`. One real downstream hazard — a generic `isinstance(x, Ragged)` dispatch branch that would newly capture `RaggedVariants` — is guarded explicitly.

**Tech Stack:** Python, `seqpro.rag` (`_core.Ragged`), pixi (`pixi run -e dev`, torch via `pixi run -e default`), pytest, ruff, pyrefly.

## Global Constraints

- **seqpro floor 0.18.0** — this refactor REQUIRES seqpro ≥ 0.18.0 (includes both `Ragged.__getitem__` numpy-consistency and subclass-preserving structural transforms). Bump the pin first (Task 1); nothing else works until then.
- **Behavior-preserving** — every existing `RaggedVariants` test (`tests/unit/ragged/test_rag_variants.py`, `test_ragged_rc_packing.py`, `tests/dataset/test_flat_variants.py`, `tests/unit/dataset/test_flat_variants_type.py`, `tests/integration/dataset/test_vcf_pgen_svar_parity.py`, `tests/unit/test_shm_layout.py`, loader tests) must stay green. The existing tests ARE the spec.
- **Subclassing contract (from seqpro PR2):** a `Ragged` subclass declares `__slots__ = ()` and holds **no instance state beyond `_layout`**. Structural transforms reconstruct via `object.__new__(type(self))`, bypassing `__init__`. `RaggedVariants` must honor this — no new instance attributes.
- **Run via pixi:** `pixi run -e dev <cmd>` for dev/lint/typecheck; torch + slow tiers via `KMP_DUPLICATE_LIB_OK=TRUE pixi run -e default pytest ...` (the `dev` env has no torch — see project memory).
- **Generate test data once** before the first run if not already present: `pixi run -e dev gen`.
- **Public-API change → update the skill** (per CLAUDE.md): `RaggedVariants` is in `__init__.py.__all__`; its method surface changes (`reshape` signature `*shape`; `squeeze` becomes a real axis-squeeze; it is now a `Ragged` subclass). Update `skills/genvarloader/SKILL.md`.
- **Lint + typecheck clean** before each push: `pixi run -e dev ruff check python/ tests/`, `pixi run -e dev ruff format python/ tests/`, `pixi run -e dev typecheck`.
- **Full-tree caveat** (per CLAUDE.md): scoped runs like `pytest tests/dataset` skip `tests/unit/`. Before the final commit run the whole tree.

**Reference (do not re-derive):** the base `Ragged` API this plan relies on, confirmed in seqpro 0.18.0 `python/seqpro/rag/_core.py`:
- `__slots__ = ("_layout",)`; `__init__(self, data, *, validate=False)` — if `data` is a `Ragged` it takes `data._layout`; a `RaggedLayout`/`RecordLayout` is stored directly.
- `_with_layout(self, layout)` → `object.__new__(type(self))` + `obj._layout = layout` (subclass-preserving).
- `__getattr__(name)` → for a record field returns `Ragged(field)`, else raises `AttributeError(name)` (identical to RaggedVariants' current override).
- `__getitem__` → numpy-consistent (PR1) and subclass-preserving for positional keys; a **string** key returns a **base** `Ragged` field (PR2 wrapper skips string keys).
- `reshape(*shape)`, `squeeze(axis=None)`, `to_packed(*, copy=True)` — all subclass-preserving; `shape`, `fields`, `__len__` provided.
- **No** `__eq__`/`__hash__` on base (identity semantics — same as RaggedVariants today). Base does define `__array__`; `np.asarray(rv)` behavior may change but is not relied on anywhere in GVL.

---

### Task 1: Bump the seqpro pin to 0.18.0 and establish a green baseline

**Files:**
- Modify: `pixi.toml:91` (`seqpro = "==0.17.0"` → `"==0.18.0"`)
- Modify: `pyproject.toml:13` (`"seqpro>=0.17",` → `"seqpro>=0.18",`)

**Interfaces:**
- Consumes: seqpro 0.18.0 from conda (pixi).
- Produces: a dev env on seqpro 0.18.0; a recorded green baseline of the RaggedVariants suite (the behavior this refactor must preserve).

- [ ] **Step 1: Edit the pins**

In `pixi.toml` line 91 change `seqpro = "==0.17.0"` to `seqpro = "==0.18.0"`.
In `pyproject.toml` line 13 change `"seqpro>=0.17",` to `"seqpro>=0.18",`.

- [ ] **Step 2: Re-lock + install the dev env**

Run: `pixi install -e dev`
Expected: solves with `seqpro 0.18.0`. If the solver cannot find 0.18.0, STOP and report — the release may still be propagating to the conda channel (the user said the release was "running"); do not downgrade or work around.

- [ ] **Step 3: Confirm the installed version**

Run: `pixi run -e dev python -c "import seqpro, seqpro.rag as r; print(seqpro.__version__); print(hasattr(r.Ragged, '_with_layout'))"`
Expected: `0.18.0` and `True` (the `_with_layout` seam from PR2 is present).

- [ ] **Step 4: Generate test data if needed, then record the green baseline**

Run (skip the first if `tests/data` fixtures already exist): `pixi run -e dev gen`
Run: `pixi run -e dev pytest tests/unit/ragged/test_rag_variants.py tests/unit/ragged/test_ragged_rc_packing.py tests/dataset/test_flat_variants.py tests/unit/dataset/test_flat_variants_type.py tests/unit/test_shm_layout.py tests/unit/test_chunk_planner.py tests/unit/test_double_buffered_loader.py tests/unit/test_buffered_loader.py -q`
Expected: all PASS. This is the behavior the refactor preserves. If anything fails on a clean pin bump (before any refactor), STOP and report — that's a seqpro 0.18.0 regression, not part of this plan.

- [ ] **Step 5: Commit**

```bash
git add pixi.toml pyproject.toml pixi.lock
git commit -m "build(deps): require seqpro>=0.18 for Ragged subclass-preserving transforms"
```

---

### Task 2: Convert `RaggedVariants` to a `Ragged` subclass

**Files:**
- Modify: `python/genvarloader/_dataset/_rag_variants.py` — the `class RaggedVariants` body (lines 200–548). Module-level helpers (`_empty_group_pad`, `_concat_string_ragged`, `_as_opaque`, `_share_offsets`) are unchanged.
- Test: `tests/unit/ragged/test_rag_variants_subclass.py` (create)

**Interfaces:**
- Consumes: base `Ragged` (already imported `from seqpro.rag import Ragged`), `Ragged._with_layout`, `Ragged.__init__`, base `__getattr__`/`__getitem__`/`reshape`/`squeeze`/`to_packed`/`shape`/`fields`/`__len__`.
- Produces: `class RaggedVariants(Ragged)` with `__slots__ = ()`; `RaggedVariants.from_record(rag)` classmethod (unchanged contract: wrap an existing record `Ragged` with no copy); kept domain API (`alt`/`ref`/`start`/`dosage`/`ilen`/`end`, `_alt_chars`, `rc_`, `pad`, `to_nested_tensor_batch`). `isinstance(rv, Ragged)` is now `True`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/ragged/test_rag_variants_subclass.py`:

```python
import numpy as np
import pytest
from seqpro.rag import Ragged

from genvarloader import RaggedVariants


def _rv():
    """A (2, 2, ~v) RaggedVariants: 2 batch × 2 ploidy, ragged variants."""
    alt = Ragged.from_offsets(
        np.frombuffer(b"ACGTACGT", dtype="S1").copy(),
        (2, 2, None),
        np.array([0, 1, 2, 3, 4], np.int64),  # 4 groups (b*p), 1-2 vars each
        str_offsets=np.array([0, 1, 3, 5, 8], np.int64),
    ).to_strings()
    start = Ragged.from_offsets(
        np.arange(4, dtype=np.int32), (2, 2, None), np.array([0, 1, 2, 3, 4], np.int64)
    )
    ilen = Ragged.from_offsets(
        np.zeros(4, np.int32), (2, 2, None), np.array([0, 1, 2, 3, 4], np.int64)
    )
    return RaggedVariants(alt=alt, start=start, ilen=ilen)


def test_raggedvariants_is_ragged_subclass():
    rv = _rv()
    assert isinstance(rv, Ragged)
    assert type(rv).__mro__[1] is Ragged
    assert rv.__slots__ == ()


def test_no_rag_composition_attribute():
    rv = _rv()
    assert not hasattr(rv, "_rag")  # composition field is gone
    assert rv._layout is not None  # holds the record layout directly


@pytest.mark.parametrize("key", [0, slice(0, 2), np.array([1, 0])], ids=["int", "slice", "fancy"])
def test_positional_indexing_preserves_subclass(key):
    rv = _rv()
    out = rv[key]
    assert type(out) is RaggedVariants


def test_int_index_collapses_leading_axis():
    rv = _rv()  # (2, 2, ~v)
    assert rv[0].shape == (2, None)        # int collapses batch -> (ploidy, ~v)
    assert rv[0:2].shape == (2, 2, None)   # slice keeps batch (ploidy preserved)


def test_string_key_returns_base_ragged():
    rv = _rv()
    field = rv["start"]
    assert isinstance(field, Ragged)
    assert type(field) is Ragged  # NOT RaggedVariants


def test_inherited_structural_transforms_preserve_subclass():
    rv = _rv()
    assert type(rv.reshape(1, 2, 2, None)) is RaggedVariants  # base *shape signature
    assert type(rv.to_packed()) is RaggedVariants
    sq = rv.reshape(1, 2, 2, None).squeeze(0)
    assert type(sq) is RaggedVariants
    assert sq.shape == (2, 2, None)


def test_squeeze_axis0_equals_index0_on_singleton():
    rv = _rv().reshape(1, 2, 2, None)  # (1, 2, 2, ~v)
    np.testing.assert_array_equal(
        np.asarray(rv.squeeze(0)["start"].data), np.asarray(rv[0]["start"].data)
    )


def test_extra_field_via_getattr():
    alt = _rv()["alt"]
    start = _rv()["start"]
    af = Ragged.from_offsets(
        np.arange(4, dtype=np.float32), (2, 2, None), np.array([0, 1, 2, 3, 4], np.int64)
    )
    rv = RaggedVariants(alt=alt.to_strings(), start=start, ilen=_rv()["ilen"], AF=af)
    assert "AF" in rv.fields
    np.testing.assert_array_equal(np.asarray(rv.AF.data), np.arange(4, dtype=np.float32))
    with pytest.raises(AttributeError):
        _ = rv.not_a_field
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/unit/ragged/test_rag_variants_subclass.py -v`
Expected: FAIL — `test_raggedvariants_is_ragged_subclass`/`test_no_rag_composition_attribute` fail because `RaggedVariants` is not yet a `Ragged` subclass (still has `_rag`, `__slots__ == ("_rag",)`).

- [ ] **Step 3: Rewrite the class declaration, `__slots__`, `__init__`, and `from_record`**

In `python/genvarloader/_dataset/_rag_variants.py`, change the class header (line 200) and the `__slots__`/`__init__`/`from_record` region (lines 207–238).

Replace `class RaggedVariants:` with `class RaggedVariants(Ragged):` (keep the existing docstring).

Replace `__slots__ = ("_rag",)` with `__slots__ = ()`.

Replace the `__init__` body's final line `self._rag = Ragged.from_fields(rec)` (line 231) with:

```python
        super().__init__(Ragged.from_fields(rec))
```

(The rest of `__init__` — the `ref`/`ilen` validation, `_as_opaque`, `_share_offsets`, the `rec` dict assembly — is unchanged.)

Replace the `from_record` classmethod (lines 233–238) with:

```python
    @classmethod
    def from_record(cls, rag: Ragged) -> "RaggedVariants":
        """Wrap an existing record Ragged directly (no copy), preserving subclass."""
        obj = object.__new__(cls)
        obj._layout = rag._layout
        return obj
```

- [ ] **Step 4: Delete the now-redundant overrides**

Delete these methods/properties entirely (base `Ragged` provides identical behavior — see the Global Constraints reference block):

- `fields` property (lines 240–242) — base `fields` is identical.
- `shape` property (lines 268–270) — base `shape` is identical.
- `__len__` (lines 304–305) — base `__len__` is identical.
- `__getitem__` (lines 307–321) — base PR1+PR2 `__getitem__` handles it: non-tuple keys are numpy-consistent, positional results preserve the subclass, and a **string** key returns a base `Ragged` field.
- `__getattr__` (lines 323–337) — base `__getattr__` returns `Ragged(field)` for record fields and raises `AttributeError` otherwise (identical).
- `reshape` (lines 339–342) — base `reshape(*shape)` preserves subclass.
- `squeeze` (lines 344–346) — base `squeeze(axis)` preserves subclass; the one caller passes `axis=0` on a singleton leading axis (equivalent to the old `self[0]`).
- `to_packed` (lines 348–349) — base `to_packed(*, copy=True)` preserves subclass.

- [ ] **Step 5: Rewire the kept methods from `self._rag[...]` to `self[...]`**

In the KEPT methods/properties only, replace each remaining `self._rag` with `self`. After deleting the methods in Step 4, the remaining occurrences are in `_alt_chars`, the `alt`/`ref`/`start`/`dosage`/`ilen`/`end` properties, `rc_`, `pad`, and `to_nested_tensor_batch`. Concretely:

- `_alt_chars` (246): `self._rag[field]` → `self[field]`
- `alt` (251): `self._rag["alt"]` → `self["alt"]`
- `ref` (256): `self._rag["ref"]` → `self["ref"]`
- `start` (261): `self._rag["start"]` → `self["start"]`
- `dosage` (266): `self._rag["dosage"]` → `self["dosage"]`
- `ilen` (276, 279, 282, 286): `self._rag["ilen"]`/`["alt"]`/`["ref"]`/`["start"]` → `self["ilen"]`/`["alt"]`/`["ref"]`/`["start"]`; `self.fields` stays.
- `end` (297): `self._rag["ref"]` → `self["ref"]`
- `rc_` (369): `field = self._rag[f]` → `field = self[f]`; the comment at 401 (`Non-allele fields from self._rag ...`) → reword to `from self ...`.
- `pad` (431, 437): `self._rag["start"]` → `self["start"]`, `base = self._rag[f]` → `base = self[f]`
- `to_nested_tensor_batch` (502, 509): `self._rag["start"]` → `self["start"]`, `field = self._rag[f]` → `field = self[f]`

After this step `grep -n "self\._rag" python/genvarloader/_dataset/_rag_variants.py` must return **zero** matches.

(`self.fields` in `rc_`/`pad`/`to_nested_tensor_batch` is the inherited base property — leave it. `self.start`/`self.shape` property accesses inside `ilen`/`end`/`rc_` resolve to the kept properties / inherited `shape`.)

- [ ] **Step 6: Run the new subclass test**

Run: `pixi run -e dev pytest tests/unit/ragged/test_rag_variants_subclass.py -v`
Expected: PASS (all parametrizations).

- [ ] **Step 7: Run the full RaggedVariants behavior suite (the preserved spec)**

Run: `pixi run -e dev pytest tests/unit/ragged/test_rag_variants.py tests/unit/ragged/test_ragged_rc_packing.py tests/dataset/test_flat_variants.py tests/unit/dataset/test_flat_variants_type.py tests/integration/dataset/test_vcf_pgen_svar_parity.py -q`
Expected: PASS. In particular `test_getitem_string` (string key → base `Ragged`), `test_getitem_int`/`_slice`/`_fancy` (positional → `RaggedVariants`), `test_getattr_extra_fields`, `test_rc`, `test_pad*` stay green. If `test_vcf_pgen_svar_parity` needs plink2/PGEN (linux-only) and skips on osx-arm64, that is expected (see project memory) — note it, don't treat the skip as failure.

- [ ] **Step 8: Lint + typecheck**

Run: `pixi run -e dev ruff format python/ tests/ && pixi run -e dev ruff check python/ tests/ && pixi run -e dev typecheck`
Expected: ruff clean. For pyrefly: `_with_layout`'s return type is annotated `Ragged[Any]`, so `rv[...]`, `rv.reshape(...)`, `rv.squeeze(...)`, `rv.to_packed()` are now typed as `Ragged` rather than `RaggedVariants`. If pyrefly reports new errors at sites that call a `RaggedVariants`-only method (`.rc_`/`.pad`) on such a result WITHOUT an `isinstance` narrow, add a minimal `# type: ignore[<code>]` at that exact site (do NOT re-add the deleted overrides). Record each such site in the report. If pyrefly is clean, change nothing.

- [ ] **Step 9: Commit**

```bash
git add python/genvarloader/_dataset/_rag_variants.py tests/unit/ragged/test_rag_variants_subclass.py
git commit -m "refactor(variants): RaggedVariants subclasses Ragged, drop _rag composition

seqpro 0.18 made base Ragged.__getitem__ numpy-consistent and subclass-
preserving across __getitem__/reshape/squeeze/to_packed, so the composition
wrapper and the structural overrides (__getitem__/__getattr__/reshape/squeeze/
to_packed/shape/fields/__len__) are redundant. RaggedVariants now holds only the
inherited record _layout; domain methods rewired from self._rag to self."
```

---

### Task 3: Guard `isinstance(x, Ragged)` dispatch sites against the new subclass

**Files:**
- Modify: `python/genvarloader/_double_buffered_loader.py:54–66` (add a `RaggedVariants` short-circuit before the generic `Ragged` branch)
- Modify: `python/genvarloader/_chunked.py:130–165` (remove the now-dead `RaggedVariants` branches; the generic `Ragged` branches handle them identically)
- Modify: `python/genvarloader/_dataset/_query.py:121–123` (drop the stale `# type: ignore` and comment now that `squeeze(axis)` is homogeneous)
- Test: `tests/unit/test_double_buffered_loader.py` (extend)

**Interfaces:**
- Consumes: `RaggedVariants` (now a `Ragged` subclass), base `Ragged` dispatch.
- Produces: dispatch that routes `RaggedVariants` correctly even though `isinstance(rv, Ragged)` is now `True`. No public-API change.

**Why:** Before this refactor `RaggedVariants` was NOT a `Ragged`, so `isinstance(x, Ragged)` branches never matched it. Now they do. Three production sites use `isinstance(x, Ragged)`:
- `_shm_layout.py:373` checks `RaggedVariants` BEFORE `(Ragged, _Flat)` in one elif chain → already correct, **no change**.
- `_chunked.py` checks generic `Ragged` (132, 148) BEFORE the `RaggedVariants` branches (136, 162); base now yields identical results for `RaggedVariants` (`shape[0] == len`; `arr[start:stop]` preserves subclass), so 136/162 are dead — remove them.
- `_double_buffered_loader.py:55` checks generic `Ragged` and would now pull `RaggedVariants` into the ploidy-reshape branch (which reads `.data`/`.offsets` — invalid on a record). It is masked today only because the guard `n_groups != n_instances` is false for ploidy-carrying `RaggedVariants` — fragile. Add an explicit short-circuit.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_double_buffered_loader.py`:

```python
def test_reshape_ragged_for_chunk_leaves_raggedvariants_untouched():
    """A RaggedVariants (record Ragged carrying ploidy) must pass through
    _reshape_ragged_for_chunk unchanged — it must NOT enter the generic Ragged
    ploidy-reshape branch (which assumes a single-field Ragged with .data/.offsets)."""
    import numpy as np
    from seqpro.rag import Ragged
    from genvarloader import RaggedVariants
    from genvarloader._double_buffered_loader import _reshape_ragged_for_chunk

    alt = Ragged.from_offsets(
        np.frombuffer(b"ACGT", dtype="S1").copy(), (2, 1, None),
        np.array([0, 1, 2], np.int64), str_offsets=np.array([0, 2, 4], np.int64),
    ).to_strings()
    start = Ragged.from_offsets(
        np.arange(2, dtype=np.int32), (2, 1, None), np.array([0, 1, 2], np.int64)
    )
    ilen = Ragged.from_offsets(
        np.zeros(2, np.int32), (2, 1, None), np.array([0, 1, 2], np.int64)
    )
    rv = RaggedVariants(alt=alt, start=start, ilen=ilen)  # shape (2, 1, ~v)
    out = _reshape_ragged_for_chunk([rv], n_instances=2)[0]
    assert out is rv  # untouched
    assert out.shape == (2, 1, None)
```

- [ ] **Step 2: Run to verify it fails (or errors)**

Run: `pixi run -e dev pytest tests/unit/test_double_buffered_loader.py::test_reshape_ragged_for_chunk_leaves_raggedvariants_untouched -v`
Expected: FAIL — without the guard, `rv` (shape `(2,1,~v)`, `n_groups == n_instances == 2`) currently slips past the reshape guard and is returned, so `out is rv` may pass by luck; to make the test meaningfully red, ALSO assert the branch is not entered. If it passes by luck at this point, proceed to Step 3 anyway (the guard makes the pass intentional rather than accidental) and re-run in Step 4.

- [ ] **Step 3: Add the `RaggedVariants` short-circuit in `_double_buffered_loader.py`**

In `python/genvarloader/_double_buffered_loader.py`, in `_reshape_ragged_for_chunk`, add `RaggedVariants` to the local imports (the function imports `from ._ragged import RaggedAnnotatedHaps` and `from ._flat import _Flat, _FlatAnnotatedHaps` near line 51–52) and short-circuit it at the top of `_reshape_one`:

```python
    from ._dataset._rag_variants import RaggedVariants
```

Then make `_reshape_one` (line 54) begin with:

```python
    def _reshape_one(arr):
        # RaggedVariants is a record Ragged that already carries the ploidy axis;
        # it is now a Ragged subclass, so it would otherwise enter the generic
        # `isinstance(arr, Ragged)` branch below, which assumes a single-field
        # Ragged with .data/.offsets (invalid on a record). Leave it unchanged.
        if isinstance(arr, RaggedVariants):
            return arr
        if isinstance(arr, Ragged):
            ...
```

(Leave the rest of `_reshape_one` unchanged.)

- [ ] **Step 4: Remove the dead `RaggedVariants` branches in `_chunked.py`**

In `python/genvarloader/_chunked.py`:
- In `_len`, delete the `RaggedVariants` branch (lines 136–138). The branch at line 132 (`isinstance(arr, (np.ndarray, Ragged, RaggedAnnotatedHaps))`) already returns `arr.shape[0]`, which equals `len(arr)` for a record `RaggedVariants`.
- In `_slice_one`, delete the `RaggedVariants` branch (lines 162–164). The branch at line 148 (`isinstance(arr, Ragged): return arr[start:stop]`) already handles `RaggedVariants` identically — base `__getitem__` preserves the subclass on a slice.

If `RaggedVariants` is now unused in `_chunked.py` after these deletions, remove its import too (check with `grep -n RaggedVariants python/genvarloader/_chunked.py`); if it is still referenced elsewhere in the file, keep the import.

- [ ] **Step 5: Drop the stale `type: ignore` in `_query.py`**

In `python/genvarloader/_dataset/_query.py` line 121–123, `squeeze` is now homogeneous (every kind, including `RaggedVariants`, accepts the `axis` argument). Change:

```python
    if squeeze:
        # (1 [p] l) -> ([p] l)
        recon = tuple(o.squeeze(0) for o in recon)  # type: ignore[bad-argument-count]  # RaggedVariants.squeeze() takes no args; other kinds do — heterogeneous dispatch
```

to:

```python
    if squeeze:
        # (1 [p] l) -> ([p] l); axis 0 is a singleton here, so squeeze(0) drops it.
        recon = tuple(o.squeeze(0) for o in recon)
```

- [ ] **Step 6: Run the new test + the loader/chunk/shm suites**

Run: `pixi run -e dev pytest tests/unit/test_double_buffered_loader.py tests/unit/test_buffered_loader.py tests/unit/test_chunk_planner.py tests/unit/test_shm_layout.py tests/unit/test_flat_slice.py tests/dataset/test_flat_getitem_snapshot.py -q`
Expected: PASS.

- [ ] **Step 7: Lint + typecheck**

Run: `pixi run -e dev ruff format python/ tests/ && pixi run -e dev ruff check python/ tests/ && pixi run -e dev typecheck`
Expected: clean (the removed `type: ignore` must NOT trigger an `unused-ignore`; if pyrefly now wants an ignore back somewhere, that's a real signal — record it).

- [ ] **Step 8: Commit**

```bash
git add python/genvarloader/_double_buffered_loader.py python/genvarloader/_chunked.py python/genvarloader/_dataset/_query.py tests/unit/test_double_buffered_loader.py
git commit -m "fix(dispatch): guard isinstance(Ragged) sites for RaggedVariants subclass

RaggedVariants is now a Ragged subclass, so generic isinstance(x, Ragged)
branches capture it. Short-circuit it in _reshape_ragged_for_chunk before the
ploidy-reshape branch (which assumes a single-field Ragged); drop the now-dead
RaggedVariants branches in _chunked and the stale squeeze type: ignore in _query."
```

---

### Task 4: Update the `genvarloader` skill for the changed `RaggedVariants` surface

**Files:**
- Modify: `skills/genvarloader/SKILL.md` (the `RaggedVariants` method-surface line ~342 and the surrounding section/gotchas)

**Interfaces:**
- Consumes: nothing (docs only).
- Produces: an accurate skill reflecting that `RaggedVariants` is a `Ragged` subclass with base structural semantics.

- [ ] **Step 1: Read the current RaggedVariants references**

Run: `pixi run -e dev rg -n "RaggedVariants" skills/genvarloader/SKILL.md`
Read each hit (notably the method-surface one-liner around line 342 listing `.rc_()`, `.pad()`, `.to_packed()`, `.reshape()`, `.squeeze()`, `.to_nested_tensor_batch()`).

- [ ] **Step 2: Update the method-surface description**

Edit the `RaggedVariants` method-surface line so it states:
- `RaggedVariants` is a subclass of `seqpro.rag.Ragged` (so `isinstance(rv, Ragged)` is `True`); indexing/`reshape`/`squeeze`/`to_packed` come from the base and preserve the `RaggedVariants` type.
- `reshape(*shape)` takes unpacked ints (base signature), e.g. `rv.reshape(1, 2, None)` (NOT a single tuple).
- `squeeze(axis=None)` is a real axis-squeeze (base semantics), not a fixed "drop axis 0".
- Domain methods retained: `.rc_()`, `.pad()`, `.to_nested_tensor_batch()`, derived `.ilen`/`.end`, fields `.alt`/`.ref`/`.start`/`.dosage`. A **string** key (`rv["start"]`) returns a bare `Ragged` field, not a `RaggedVariants`.

Do not invent behavior beyond the above; keep the rest of the section intact.

- [ ] **Step 3: Sanity-check the skill examples still parse**

Run: `pixi run -e dev rg -n "\.reshape\(|\.squeeze\(" skills/genvarloader/SKILL.md`
If any example calls `rv.reshape((...))` with a single tuple, fix it to unpacked ints. If none, no change.

- [ ] **Step 4: Commit**

```bash
git add skills/genvarloader/SKILL.md
git commit -m "docs(skill): RaggedVariants is now a Ragged subclass (base structural semantics)"
```

---

### Task 5: Whole-tree verification including torch + slow tiers

**Files:** none (verification + final tidy commit only).

**Interfaces:**
- Consumes: all prior tasks.
- Produces: a verified, fully green tree on seqpro 0.18.0.

- [ ] **Step 1: Full default-env tree (covers dataset + unit + integration)**

Run: `pixi run -e dev pytest tests -q`
Expected: PASS (modulo the documented linux-only skips: plink2/PGEN, basenji2-torch — see project memory). Any unexpected failure here is in scope — debug it (likely a missed `self._rag` site or an isinstance ordering case the plan didn't enumerate) before proceeding.

- [ ] **Step 2: Torch + slow tiers**

Run: `KMP_DUPLICATE_LIB_OK=TRUE pixi run -e default pytest tests -m "slow or not slow" -q`
Expected: PASS (the `default` env has torch; this exercises `to_nested_tensor_batch` and the dataloader paths that move `RaggedVariants` through shm + chunking). This is the deferred-plan's explicit gate.

- [ ] **Step 3: Final lint + typecheck across the tree**

Run: `pixi run -e dev ruff format python/ tests/ && pixi run -e dev ruff check python/ tests/ && pixi run -e dev typecheck`
Expected: clean.

- [ ] **Step 4: Confirm no composition residue**

Run: `pixi run -e dev rg -n "self\._rag\b|\.from_record\(|__slots__ = \(\"_rag\"" python/genvarloader/`
Expected: `self._rag` gone; `from_record` only at its definition and the legitimate call sites (`_shm_layout.py`, and inside `rc_`/`pad`); no `("_rag",)` slot.

- [ ] **Step 5: Commit any formatting-only changes**

```bash
git add -A
git commit -m "test(variants): full-tree + torch/slow verification for Ragged subclass refactor" --allow-empty
```

---

## Self-Review

- **Spec coverage (deferred section of `SeqPro/.../2026-06-23-ragged-subclass-getitem.md`):**
  - "Make `RaggedVariants` subclass `Ragged` (`__slots__ = ()`), dropping `_rag`" → Task 2.
  - "Keep domain methods/properties (`alt`/`ref`/`start`/`dosage`, derived `ilen`/`end`, `rc_`, `_alt_chars`, `__init__` invariants)" → Task 2 Steps 3–5 (kept + rewired; `pad`/`to_nested_tensor_batch` also kept).
  - "the allele-aware `concatenate` override" → **N/A / discrepancy:** there is no `concatenate` method on `RaggedVariants` in the current code; allele concatenation lives in `pad` via `_concat_string_ragged`, and the module function is `seqpro.rag._ops.concatenate`. Nothing to keep or change. Flagged here rather than inventing a method.
  - "Delete the structural overrides: `reshape`, `to_packed`, `squeeze`, and the entire `__getitem__` override (string-key field access returns a bare field `Ragged`)" → Task 2 Step 4. Also deletes `__getattr__`/`shape`/`fields`/`__len__` (base-identical) for DRY — a superset of the spec's deletions, justified per-item.
  - "Bump the seqpro pin … run the full GVL suite including torch + slow tiers" → Task 1 + Task 5.
  - Not in the deferred text but required by the refactor: the `isinstance(x, Ragged)` dispatch hazard → Task 3. This is the one place the spec's "just delete the overrides" omits a real consequence (the subclass newly satisfies `isinstance(_, Ragged)`).

- **Placeholder scan:** none — every step has concrete code, file:line targets, exact commands, and expected output.

- **Type consistency:** `RaggedVariants.from_record(rag) -> RaggedVariants` defined in Task 2 and consumed unchanged at `_shm_layout.py:605` and inside `rc_`/`pad`. The deleted-override list (Task 2 Step 4) and the rewire list (Step 5) partition the 24 `self._rag` sites: 6 live in deleted methods (`shape` 270, `__len__` 305, `__getitem__` 308, `reshape` 341/342, `to_packed` 349) and vanish; the other 18 are rewired to `self`. Post-refactor `grep self._rag` must be empty (Task 2 Step 5 / Task 5 Step 4 both assert this).

- **Risk notes for the implementer / reviewer:**
  - `squeeze` semantics narrow from "always `self[0]`" to "real axis-squeeze"; the sole caller (`_query.py:123`) passes `axis=0` on a guaranteed-singleton leading axis, so the result is identical. Pinned by `test_squeeze_axis0_equals_index0_on_singleton` (Task 2) and the dataset/integration suites (Task 5).
  - `reshape` signature changes from `reshape(tuple)` to base `reshape(*shape)`. No production or test caller passes a tuple to `RaggedVariants.reshape` (verified), so deletion is safe; the skill is updated (Task 4).
  - Type widening: `rv[...]`/`reshape`/`squeeze`/`to_packed` are now statically typed `Ragged`. Handled by the typecheck steps (Task 2 Step 8, Task 3 Step 7) with minimal `# type: ignore` only where pyrefly actually complains.
