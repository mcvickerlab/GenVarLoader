# Migrate GenVarLoader off awkward onto seqpro `_core.Ragged` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `awkward` from GenVarLoader's production data structures by making `RaggedVariants` a thin wrapper over a single record `seqpro.rag._core.Ragged`, so seqpro can delete its legacy awkward backend.

**Architecture:** `alt`/`ref` are stored as opaque-string (`'S'`) fields so their `~length` axis collapses, letting all variant fields share one offsets object inside one record `Ragged`. The record provides len/index/slice/field-access natively; a wrapper class preserves the documented `.alt`/`.rc_()` method API. seqpro gains a ragged `concatenate` Rust kernel, a `to_ak` record fix, and obligatory record `to_packed`, then drops `_array.py`.

**Tech Stack:** Python 3.10–3.13, numpy, numba, seqpro 0.16 (`_core` backend, Rust+PyO3+numba), genoray, awkward (test-only after this work), pixi, maturin.

**Spec:** `docs/superpowers/specs/2026-06-22-migrate-gvl-off-awkward-to-core-ragged-design.md`

## Global Constraints

- **GVL suite stays green throughout:** `pixi run -e dev pytest tests -q` → **800 passed, 0 failed** (data pre-generated via `pixi run -e dev gen`). Run the full tree before pushing renames/shared-code changes (scoped runs skip `tests/unit/`).
- **Lint gate (pre-push hook enforces both):** `pixi run -e dev ruff check python/ tests/` and `pixi run -e dev ruff format python/ tests/` over BOTH `python/` and `tests/`.
- **E501 (line length) is ignored** by ruff config; do not reflow for length alone.
- **No Python loops over residues/variants in hot paths** — numpy vectorized, numba kernels, or seqpro/Rust ops only (matches seqpro's hot-path rule).
- **awkward stays a test-only dependency** — production target is zero `ak.*` / `import awkward` in `python/genvarloader/**`; test oracles may keep awkward.
- **Byte-identical parity** is the contract: the existing green suite is the parity gate; new code is verified against awkward oracles where behavior is new.
- **seqpro work** is done in `~/projects/SeqPro` (sibling checkout; pixi already pins `seqpro = { path = "../SeqPro", editable = true }`). seqpro suite stays green: `cd ~/projects/SeqPro && pixi run -e dev pytest tests/` → **544 passed**. Update `~/projects/SeqPro/skills/seqpro/SKILL.md` for any new public op (seqpro CLAUDE.md requirement).
- **PR strategy:** one bundled GVL PR on branch `rust-ragged-audit`; one seqpro PR. Keep-branch after the plan. Repoint pixi pins from local editable paths to released versions before merge.
- **Commit style:** conventional commits; end commit messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Use `rtk git ...` per repo CLAUDE.md.

---

## Reference: the recurring awkward→`_core` swap recipes

Several consumer tasks apply these. Each is behavior-preserving (gated by the existing suite). `rag` denotes a `_core.Ragged`; `.data` is the flat contiguous buffer, `.offsets` the `(N+1,)` (or `(2,N)`) offsets, `.lengths` per-segment lengths.

| # | awkward idiom | `_core` replacement |
|---|---|---|
| R1 | `ak.max(rag.to_ak(), None)` / global max | `rag.to_packed().data.max()` |
| R2 | `ak.min(rag.to_ak(), None)` / global min | `rag.to_packed().data.min()` |
| R3 | `ak.flatten(rag, None).to_numpy()` (flatten all) | `rag.to_packed().data` |
| R4 | `ak.flatten(rag, -1).to_numpy()` (flatten innermost, single ragged axis) | `rag.to_packed().data` |
| R5 | `ak.count(rag, -1)` (per-group counts) | `rag.lengths` |
| R6 | `isinstance(x, ak.Array)` used to detect `RaggedVariants` | `isinstance(x, RaggedVariants)` |
| R7 | `ak.concatenate([a.to_ak() for a in ls], -1)` then flatten | `seqpro.rag.concatenate(ls, axis=-1).to_packed().data` (Task S3) |

When `rag` may carry a non-contiguous `(2,N)` offsets layout (after slicing), call `.to_packed()` first — these recipes already do.

---

# Phase S — seqpro upstream (do first; GVL depends on S1–S3)

All Phase-S work is in `~/projects/SeqPro`. Build/test via `cd ~/projects/SeqPro && pixi run -e dev <cmd>`.

### Task S1: Fix `to_ak()` on multi-leading-axis records

**Files:**
- Modify: `~/projects/SeqPro/python/seqpro/rag/_ingest.py:148-153` (`to_ak`)
- Test: `~/projects/SeqPro/tests/rag/test_ingest.py` (add; create if absent)

**Interfaces:**
- Produces: `Ragged.to_ak()` returns a correct `ak.Array` for record Rageds with ≥2 leading fixed axes (e.g. `(b, p, ~v)`).

- [ ] **Step 1: Write the failing test**

```python
# tests/rag/test_ingest.py
import numpy as np
import seqpro.rag as r
from seqpro.rag import Ragged

def test_to_ak_multi_leading_axis_record():
    # (b=2, p=2, ~v) record with an opaque-string and a numeric field
    var_off = np.array([0, 2, 3, 3, 4], dtype=np.int64)  # 4 groups
    char_off = np.array([0, 2, 3, 6, 7], dtype=np.int64)
    chars = np.frombuffer(b"ACGTTTX", dtype="S1").copy()
    alt = Ragged.from_offsets(chars, (2, 2, None, None), [var_off, char_off]).to_strings()
    start = Ragged.from_offsets(np.arange(4, dtype=np.int32), (2, 2, None), alt.offsets)
    rv = Ragged.from_fields({"alt": alt, "start": start})
    got = rv.to_ak()  # must not raise
    assert got["alt"].to_list() == [[b"AC", b"G"], [b"TTT"], [], [b"X"]]
    assert got["start"].to_list() == [[[0, 1], [2]], [[], [3]]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/projects/SeqPro && pixi run -e dev pytest tests/rag/test_ingest.py::test_to_ak_multi_leading_axis_record -v`
Expected: FAIL with `ValueError: cannot broadcast RegularArray of size 2 with ...`.

- [ ] **Step 3: Fix `to_ak` record branch**

In `_ingest.py`, the record branch is `return ak.zip({f: to_ak(rag[f]) for f in rag.fields}, depth_limit=1)`. The `ak.zip` broadcast fails because the per-field arrays carry multiple leading `RegularArray` axes. Reconstruct the record by zipping at the **innermost ragged level** and re-wrapping the shared leading regular axes, instead of relying on `ak.zip` broadcasting. Concretely: convert each field to its awkward layout, strip the shared leading `RegularArray`/`ListOffsetArray` levels (identical across fields by construction), `ak.zip` the leaf record at `depth_limit=1`, then re-apply the shared leading levels via `ak.Array(RegularArray(... ListOffsetArray(... record ...)))`. Use the existing `_array.py` `to_ak` (the awkward backend) as the parity oracle for the expected nesting.

- [ ] **Step 4: Run the new test + full seqpro suite**

Run: `cd ~/projects/SeqPro && pixi run -e dev pytest tests/rag/test_ingest.py -v && pixi run -e dev pytest tests/ -q`
Expected: new test PASS; suite **544 passed** (no regressions).

- [ ] **Step 5: Commit**

```bash
cd ~/projects/SeqPro && rtk git add python/seqpro/rag/_ingest.py tests/rag/test_ingest.py && rtk git commit -m "fix(rag): to_ak() on multi-leading-axis record Ragged

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task S2: Record / opaque-string `to_packed`

**Files:**
- Modify: `~/projects/SeqPro/python/seqpro/rag/_ops.py` (`to_packed`) and/or `_core.py` (`Ragged.to_packed`)
- Test: `~/projects/SeqPro/tests/rag/test_rag_to_packed.py` (add cases)

**Interfaces:**
- Produces: `rag.to_packed()` and `seqpro.rag.to_packed(rag)` succeed on (a) opaque-string-under-axis Rageds and (b) record Rageds whose fields include opaque-string fields, returning a packed (contiguous, zero-based) Ragged of the same logical content. Removes the `NotImplementedError("... Spec C ...")`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/rag/test_rag_to_packed.py (append)
import numpy as np
import seqpro.rag as r
from seqpro.rag import Ragged

def _record(var_off, char_off, chars):
    alt = Ragged.from_offsets(chars, (len(var_off) - 1, None, None),
                              [var_off, char_off]).to_strings()
    start = Ragged.from_offsets(np.arange(int(var_off[-1]), dtype=np.int32),
                                (len(var_off) - 1, None), alt.offsets)
    return Ragged.from_fields({"alt": alt, "start": start}), alt

def test_to_packed_opaque_string_under_axis():
    var_off = np.array([0, 2, 3], dtype=np.int64)
    char_off = np.array([0, 2, 3, 6], dtype=np.int64)
    rv, alt = _record(var_off, char_off, np.frombuffer(b"ACGTTT", "S1").copy())
    sl = alt[np.array([1, 0])]            # produces (2,N) gather offsets
    packed = sl.to_packed()              # must not raise
    assert packed.to_ak().to_list() == [[b"TTT"], [b"AC", b"G"]]

def test_to_packed_record_with_string_field():
    var_off = np.array([0, 2, 3], dtype=np.int64)
    char_off = np.array([0, 2, 3, 6], dtype=np.int64)
    rv, _ = _record(var_off, char_off, np.frombuffer(b"ACGTTT", "S1").copy())
    sl = rv[np.array([1, 0])]
    packed = sl.to_packed()             # must not raise
    assert packed["alt"].to_ak().to_list() == [[b"TTT"], [b"AC", b"G"]]
    assert packed["start"].to_ak().to_list() == [[2], [0, 1]]
```

- [ ] **Step 2: Run to verify failure**

Run: `cd ~/projects/SeqPro && pixi run -e dev pytest tests/rag/test_rag_to_packed.py -k "opaque or record_with_string" -v`
Expected: FAIL with `NotImplementedError(... Spec C ...)`.

- [ ] **Step 3: Implement record/opaque-string packing**

Implement `to_packed` for the string-under-axis layout: pack the inner string offsets (`str_offsets`) and char data into contiguous zero-based buffers, then pack the outer (variant) offsets — reuse the existing `_ragged_nested_pack` Rust kernel / numba path used for R=2 packing (the char view is structurally R=2). For records, pack each field (numeric fields via the existing path; string fields via the new string path) and rebuild the record sharing the repacked outer offsets object.

- [ ] **Step 4: Run new tests + full suite**

Run: `cd ~/projects/SeqPro && pixi run -e dev pytest tests/rag/test_rag_to_packed.py -q && pixi run -e dev pytest tests/ -q`
Expected: new tests PASS; suite **544 passed**.

- [ ] **Step 5: Commit**

```bash
cd ~/projects/SeqPro && rtk git add python/seqpro/rag/_ops.py python/seqpro/rag/_core.py tests/rag/test_rag_to_packed.py && rtk git commit -m "feat(rag): to_packed() on record and opaque-string-under-axis Ragged

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task S3: Ragged-axis `concatenate` (Rust kernel)

**Files:**
- Modify: `~/projects/SeqPro/src/ragged.rs` (add kernel), `~/projects/SeqPro/src/lib.rs` (register if needed)
- Modify: `~/projects/SeqPro/python/seqpro/rag/_ops.py` (Python wrapper `concatenate`)
- Modify: `~/projects/SeqPro/python/seqpro/rag/__init__.py` (export `concatenate`)
- Modify: `~/projects/SeqPro/skills/seqpro/SKILL.md` (document `concatenate`)
- Test: `~/projects/SeqPro/tests/rag/test_concatenate.py` (add)

**Interfaces:**
- Produces: `seqpro.rag.concatenate(rags: Sequence[Ragged], axis: int) -> Ragged` — concatenates Rageds along the ragged axis `axis` (negative allowed). Inputs share all leading fixed dims and the same number of groups; element order within each group is `rags[0]` then `rags[1]`… Works for numeric dtypes (int32/float32) needed by GVL. No Python loops; offset arithmetic + buffered copy in Rust (rayon).

- [ ] **Step 1: Write the failing test**

```python
# tests/rag/test_concatenate.py
import numpy as np
import seqpro.rag as r
from seqpro.rag import Ragged

def test_concatenate_ragged_axis_prepend_regular():
    # prepend a size-1 pad per group (the prepend_pad_itv use case)
    base = Ragged.from_offsets(np.array([10, 11, 12], np.int32), (2, None),
                               np.array([0, 2, 3], np.int64))   # [[10,11],[12]]
    pad = Ragged.from_offsets(np.array([-1, -1], np.int32), (2, None),
                              np.array([0, 1, 2], np.int64))      # [[-1],[-1]]
    out = r.concatenate([pad, base], axis=-1)
    assert out.to_ak().to_list() == [[-1, 10, 11], [-1, 12]]

def test_concatenate_matches_awkward_oracle():
    import awkward as ak
    a = Ragged.from_offsets(np.arange(5, dtype=np.float32), (2, None),
                            np.array([0, 3, 5], np.int64))
    b = Ragged.from_offsets(np.arange(5, 9, dtype=np.float32), (2, None),
                            np.array([0, 1, 4], np.int64))
    got = r.concatenate([a, b], axis=-1).to_ak().to_list()
    exp = ak.concatenate([a.to_ak(), b.to_ak()], axis=-1).to_list()
    assert got == exp
```

- [ ] **Step 2: Run to verify failure**

Run: `cd ~/projects/SeqPro && pixi run -e dev pytest tests/rag/test_concatenate.py -v`
Expected: FAIL with `AttributeError: module 'seqpro.rag' has no attribute 'concatenate'`.

- [ ] **Step 3: Implement the Rust kernel**

In `src/ragged.rs`, add a kernel (mirroring `_ragged_pack`/`_ragged_nested_pack`) that, given N (data, offsets) pairs sharing group count G, computes output offsets `out_off[g] = sum_i (off_i[g+1]-off_i[g])` cumulative, allocates the output data buffer, and copies each input group's slice into place (rayon over groups). Expose via PyO3 as `_ragged_concat`.

- [ ] **Step 4: Implement the Python wrapper + export**

```python
# python/seqpro/rag/_ops.py
def concatenate(rags, axis):
    """Concatenate Rageds along the ragged axis. See SKILL.md."""
    from ._core import Ragged
    if not rags:
        raise ValueError("concatenate requires at least one Ragged")
    rags = [r if isinstance(r, Ragged) else Ragged(r) for r in rags]
    # normalize axis, validate it is the ragged axis and leading dims match
    ref = rags[0]
    ax = axis % len(ref.shape)
    if ax != ref.rag_dim:
        raise ValueError(f"concatenate only supports the ragged axis ({ref.rag_dim}), got {axis}")
    packed = [x.to_packed() for x in rags]
    from seqpro.seqpro import _ragged_concat  # rust
    data, offsets = _ragged_concat([p.data for p in packed],
                                   [np.ascontiguousarray(p.offsets) for p in packed])
    return Ragged.from_offsets(data, ref.shape, offsets)
```

```python
# python/seqpro/rag/__init__.py  — add to imports and __all__
from ._ops import concatenate  # noqa: F401
# ... ensure "concatenate" is in __all__
```

- [ ] **Step 5: Document in SKILL.md**

Add a row to the `Ragged` ops table in `~/projects/SeqPro/skills/seqpro/SKILL.md`: `seqpro.rag.concatenate(rags, axis)` — concatenate Rageds along the ragged axis; offset-arithmetic Rust kernel; numeric dtypes.

- [ ] **Step 6: Run tests + build + full suite**

Run: `cd ~/projects/SeqPro && pixi run -e dev pytest tests/rag/test_concatenate.py -v && pixi run -e dev pytest tests/ -q`
Expected: new tests PASS (triggers maturin rebuild of the Rust ext); suite **544 passed**.

- [ ] **Step 7: Commit**

```bash
cd ~/projects/SeqPro && rtk git add src/ python/seqpro/rag/_ops.py python/seqpro/rag/__init__.py skills/seqpro/SKILL.md tests/rag/test_concatenate.py && rtk git commit -m "feat(rag): concatenate() along ragged axis (Rust kernel)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

> Task S4 (delete `_array.py`) is deferred to Phase 4 (Task G17), after GVL is awkward-free.

---

# Phase 1 (GVL) — the `RaggedVariants` wrapper

All remaining tasks are in `/Users/david/projects/GenVarLoader` on branch `rust-ragged-audit`.
Build/test via `pixi run -e dev <cmd>`.

### Task G1: Wrapper skeleton — construction, fields, shape, len, indexing

**Files:**
- Rewrite: `python/genvarloader/_dataset/_rag_variants.py`
- Test: `tests/unit/ragged/test_rag_variants.py` (existing; extend with `_core`-native cases)

**Interfaces:**
- Consumes: `seqpro.rag.Ragged` (`from_fields`, `from_offsets`, `.to_strings`, `.to_chars`, `.offsets`, `.data`, `.lengths`, `.shape`, indexing); `POS_TYPE`, `DOSAGE_TYPE` from `genoray._types`.
- Produces:
  - `class RaggedVariants` with `__init__(self, alt, start, ref=None, ilen=None, dosage=None, **fields)` where `alt`/`ref` are `Ragged` (either S1 char `(b,p,~v,~l)` or opaque-string `(b,p,~v)`) and `start`/`ilen`/`dosage`/`**fields` are numeric `Ragged` `(b,p,~v)`. Stores `self._rag: Ragged` (record, opaque-string alt/ref).
  - `classmethod from_record(cls, rag: Ragged) -> RaggedVariants` (wrap an existing record Ragged).
  - Properties: `.alt -> Ragged` (opaque string field), `.ref -> Ragged`, `.start -> Ragged[POS_TYPE]`, `.dosage`, `.shape -> tuple`, `.fields -> list[str]`, `.ilen -> Ragged[int32]` (stored or derived), `.end -> Ragged[POS_TYPE]`.
  - `__len__`, `__getitem__(idx) -> RaggedVariants`, `reshape(shape) -> RaggedVariants`, `squeeze(axis=None) -> RaggedVariants`.
  - Helper `_alt_chars(self, field="alt") -> Ragged` returning the S1 char view `(b,p,~v,~l)` of an allele field.

- [ ] **Step 1: Write failing tests for construction + access**

```python
# tests/unit/ragged/test_rag_variants.py (add; keep existing oracle tests)
import numpy as np
import seqpro.rag as r
from seqpro.rag import Ragged
from genvarloader import RaggedVariants

def _char_alt(var_off, char_off, chars):
    return Ragged.from_offsets(chars, (2, 1, None, None), [var_off, char_off])

def test_construct_from_char_and_numeric_fields():
    var_off = np.array([0, 2, 3], np.int64)            # b=2,p=1 -> 2 groups
    char_off = np.array([0, 2, 3, 6], np.int64)
    alt = _char_alt(var_off, char_off, np.frombuffer(b"ACGTTT", "S1").copy())
    start = Ragged.from_offsets(np.array([10, 20, 30], np.int32), (2, 1, None), var_off)
    ref = _char_alt(var_off, char_off, np.frombuffer(b"AAGTTT", "S1").copy())
    rv = RaggedVariants(alt=alt, start=start, ref=ref)
    assert rv.shape[0] == 2
    assert len(rv) == 2
    assert rv.alt.to_ak().to_list() == [[b"AC", b"G"], [b"TTT"]]
    assert rv.start.to_ak().to_list() == [[[10, 20]], [[30]]]
    # ilen derived from alt/ref char lengths
    assert rv.ilen.to_ak().to_list() == [[[0, 0]], [[0]]]

def test_getitem_returns_raggedvariants():
    var_off = np.array([0, 2, 3], np.int64)
    char_off = np.array([0, 2, 3, 6], np.int64)
    alt = _char_alt(var_off, char_off, np.frombuffer(b"ACGTTT", "S1").copy())
    start = Ragged.from_offsets(np.array([10, 20, 30], np.int32), (2, 1, None), var_off)
    rv = RaggedVariants(alt=alt, start=start, ilen=Ragged.from_offsets(
        np.zeros(3, np.int32), (2, 1, None), var_off))
    sub = rv[0]
    assert isinstance(sub, RaggedVariants)
    assert sub.alt.to_ak().to_list() == [[b"AC", b"G"]]
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest tests/unit/ragged/test_rag_variants.py -k "construct_from_char or getitem_returns" -v`
Expected: FAIL (new constructor/behavior not present).

- [ ] **Step 3: Implement the skeleton**

Replace the top of `_rag_variants.py`. Remove `import awkward as ak`, the `awkward.contents` imports, `RaggedVariant(ak.Record)`, and `ak.behavior[...] = RaggedVariants` (bottom of file). Implement:

```python
from __future__ import annotations
from typing import Any
import numpy as np
import seqpro.rag as spr
from seqpro.rag import Ragged
from genoray._types import DOSAGE_TYPE, POS_TYPE

_ALLELE_FIELDS = ("alt", "ref")

def _as_opaque(rag: Ragged) -> Ragged:
    """Normalize an allele field to opaque-string (b,p,~v). Accepts an S1 char
    (b,p,~v,~l) Ragged (collapse via to_strings) or an already-opaque Ragged."""
    return rag.to_strings() if not getattr(rag, "is_string", False) else rag

def _share_offsets(rag: Ragged, offsets) -> Ragged:
    """Rebuild `rag` onto the given (identical) variant-level offsets object so all
    record fields share it (Ragged.from_fields requires identity, not equality)."""
    if rag.offsets is offsets:
        return rag
    if getattr(rag, "is_string", False):
        chars = rag.to_chars()
        return Ragged.from_offsets(chars.data, rag.shape, offsets,
                                   str_offsets=chars.offsets).to_strings()
    return Ragged.from_offsets(rag.data, rag.shape, offsets)

class RaggedVariants:
    """Variable-length variants as a single record Ragged with shape
    (batch, ploidy, ~variants). `alt`/`ref` are opaque-string fields; `start` and
    optional `ilen`/`dosage`/extra fields are numeric. Guaranteed: `alt`, `start`,
    and one of `ref`/`ilen`."""

    __slots__ = ("_rag",)

    def __init__(self, alt, start, ref=None, ilen=None, dosage=None, **fields):
        if ref is None and ilen is None:
            raise ValueError("Must provide one of ref or ilen.")
        alt = _as_opaque(alt)
        off = alt.offsets
        rec: dict[str, Ragged] = {"alt": alt, "start": _share_offsets(start, off)}
        if ref is not None:
            rec["ref"] = _share_offsets(_as_opaque(ref), off)
        if ilen is not None:
            rec["ilen"] = _share_offsets(ilen, off)
        if dosage is not None:
            rec["dosage"] = _share_offsets(dosage, off)
        for k, v in fields.items():
            rec[k] = _share_offsets(v, off)
        self._rag = Ragged.from_fields(rec)

    @classmethod
    def from_record(cls, rag: Ragged) -> "RaggedVariants":
        obj = cls.__new__(cls)
        obj._rag = rag
        return obj

    @property
    def fields(self) -> list[str]:
        return self._rag.fields

    def _alt_chars(self, field: str = "alt") -> Ragged:
        return self._rag[field].to_chars()

    @property
    def alt(self) -> Ragged:
        return self._rag["alt"]

    @property
    def ref(self) -> Ragged:
        return self._rag["ref"]

    @property
    def start(self):
        return self._rag["start"]

    @property
    def dosage(self):
        return self._rag["dosage"]

    @property
    def shape(self):
        return self._rag.shape

    @property
    def ilen(self):
        if "ilen" in self.fields:
            return self._rag["ilen"]
        alt_len = self._alt_chars("alt").lengths
        ref_len = self._alt_chars("ref").lengths
        return Ragged.from_offsets((alt_len - ref_len).astype(np.int32),
                                   self._rag["start"].shape, self._rag["start"].offsets)

    @property
    def end(self):
        if "ref" in self.fields:
            reflen = Ragged.from_offsets(self._alt_chars("ref").lengths.astype(POS_TYPE),
                                         self.start.shape, self.start.offsets)
            return self.start + reflen
        ilen = self.ilen
        return self.start - np.clip(ilen, None, 0) + 1

    def __len__(self) -> int:
        return len(self._rag)

    def __getitem__(self, idx) -> "RaggedVariants":
        return RaggedVariants.from_record(self._rag[idx])

    def reshape(self, shape) -> "RaggedVariants":
        return RaggedVariants.from_record(self._rag.reshape(*shape)
                                          if isinstance(shape, tuple) else self._rag.reshape(shape))

    def squeeze(self, axis=None, **kw) -> "RaggedVariants":
        return self[0]
```

> Note: `__getitem__` must match `_indexing.py`'s expectations (Task G9 verifies). If `_core` multi-leading-axis indexing diverges from today's awkward semantics on a case the suite exercises, normalize in `__getitem__` (e.g. map to the equivalent record index) rather than changing call sites.

- [ ] **Step 4: Run the new tests**

Run: `pixi run -e dev pytest tests/unit/ragged/test_rag_variants.py -k "construct_from_char or getitem_returns" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_rag_variants.py tests/unit/ragged/test_rag_variants.py && rtk git commit -m "refactor(rag-variants): record-Ragged wrapper skeleton (construction, access, indexing)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task G2: `RaggedVariants.to_packed`

**Files:**
- Modify: `python/genvarloader/_dataset/_rag_variants.py`
- Test: `tests/unit/ragged/test_ragged_rc_packing.py` (existing; extend)

**Interfaces:**
- Consumes: seqpro record `to_packed` (Task S2).
- Produces: `RaggedVariants.to_packed(self) -> RaggedVariants` returning a packed copy (contiguous, zero-based buffers) with identical logical content.

- [ ] **Step 1: Write failing test**

```python
# tests/unit/ragged/test_ragged_rc_packing.py (append)
import numpy as np
from seqpro.rag import Ragged
from genvarloader import RaggedVariants

def test_to_packed_after_slice_roundtrips():
    var_off = np.array([0, 2, 3, 4], np.int64)   # 3 groups (b=3,p=1)
    char_off = np.array([0, 2, 3, 6, 7], np.int64)
    alt = Ragged.from_offsets(np.frombuffer(b"ACGTTTX", "S1").copy(),
                              (3, 1, None, None), [var_off, char_off])
    start = Ragged.from_offsets(np.array([1, 2, 3, 4], np.int32), (3, 1, None), var_off)
    rv = RaggedVariants(alt=alt, start=start,
                        ilen=Ragged.from_offsets(np.zeros(4, np.int32), (3, 1, None), var_off))
    sub = rv[np.array([2, 0])].to_packed()
    assert sub.alt.to_ak().to_list() == [[b"X"], [b"AC", b"G"]]
    assert sub.start.to_ak().to_list() == [[[4]], [[1, 2]]]
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest tests/unit/ragged/test_ragged_rc_packing.py::test_to_packed_after_slice_roundtrips -v`
Expected: FAIL (`to_packed` not implemented on wrapper).

- [ ] **Step 3: Implement**

```python
# in RaggedVariants
    def to_packed(self) -> "RaggedVariants":
        return RaggedVariants.from_record(self._rag.to_packed())
```

Delete the old `to_packed` body and its helpers that are now unused: `_is_canonical_alleles`, `_decompose_alleles`, `_pack_alleles` (top of file). (Their `_haps` counterparts `_alt_layout_parts`/`_build_allele_layout` are handled in Task G10.)

- [ ] **Step 4: Run test**

Run: `pixi run -e dev pytest tests/unit/ragged/test_ragged_rc_packing.py::test_to_packed_after_slice_roundtrips -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_rag_variants.py tests/unit/ragged/test_ragged_rc_packing.py && rtk git commit -m "refactor(rag-variants): to_packed via record Ragged; drop awkward packing helpers

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task G3: `RaggedVariants.rc_` (masked reverse-complement)

**Files:**
- Modify: `python/genvarloader/_dataset/_rag_variants.py`
- Reuse: `python/genvarloader/_ragged.py::reverse_complement_masked` (already `_core`-native) and `_COMP`
- Test: `tests/unit/ragged/test_ragged_rc_packing.py`

**Interfaces:**
- Consumes: `reverse_complement_masked(rag: Ragged[bytes], mask: NDArray[bool]) -> Ragged` (in `_ragged.py`).
- Produces: `RaggedVariants.rc_(self, to_rc: NDArray[bool] | None = None) -> RaggedVariants` — reverse-complements `alt` (and `ref` if present) for the masked batch rows; in place where the layout allows, else returns a packed new object. `to_rc` is one bool per batch row (axis 0); `None` means all.

- [ ] **Step 1: Write failing test (parity vs existing behavior)**

```python
# tests/unit/ragged/test_ragged_rc_packing.py (append)
import numpy as np
from seqpro.rag import Ragged
from genvarloader import RaggedVariants

def test_rc_all_complements_and_reverses():
    var_off = np.array([0, 1, 2], np.int64)   # 2 groups, 1 variant each
    char_off = np.array([0, 2, 5], np.int64)
    alt = Ragged.from_offsets(np.frombuffer(b"ACGTA", "S1").copy(),
                              (2, 1, None, None), [var_off, char_off])
    start = Ragged.from_offsets(np.array([0, 0], np.int32), (2, 1, None), var_off)
    rv = RaggedVariants(alt=alt, start=start,
                        ilen=Ragged.from_offsets(np.zeros(2, np.int32), (2, 1, None), var_off))
    out = rv.rc_(np.array([True, True]))
    assert out.alt.to_ak().to_list() == [[b"GT"], [b"TAC"]]   # AC->GT, GTA->TAC
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest tests/unit/ragged/test_ragged_rc_packing.py::test_rc_all_complements_and_reverses -v`
Expected: FAIL.

- [ ] **Step 3: Implement via flat allele-level R=1 view**

```python
# in RaggedVariants
    def rc_(self, to_rc=None):
        from .._ragged import reverse_complement_masked
        if to_rc is None:
            to_rc = np.ones(self.shape[0], np.bool_)
        elif not to_rc.any():
            return self
        rec = {}
        for f in self.fields:
            if f in _ALLELE_FIELDS and f in self.fields:
                chars = self._rag[f].to_chars().to_packed()   # (b,p,~v,~l) S1
                allele_off = np.asarray(chars.offsets)         # char-level boundaries
                n_alleles = allele_off.size - 1
                # per-allele mask: to_rc is per batch-row (axis 0); broadcast over
                # ploidy then variants (C order), matching reverse_complement_masked.
                view = Ragged.from_offsets(chars.data, (n_alleles, None), allele_off)
                reverse_complement_masked(view, to_rc)   # broadcasts internally
                rec[f] = view.to_strings_like(self._rag[f])  # rebuild opaque field
            else:
                rec[f] = self._rag[f]
        return RaggedVariants.from_record(Ragged.from_fields(rec))
```

> Implementation note: `reverse_complement_masked` already replicates a per-outer-row mask across inner fixed axes (see its docstring). The exact reconstruction of the opaque field from the RC'd char buffer must preserve the variant-level offsets; if `to_strings_like` is not available, rebuild via `Ragged.from_offsets(view.data, self._rag[f].to_chars().shape, variant_offsets, str_offsets=allele_off).to_strings()`. Gate correctness on the existing `tests/unit/ragged/test_ragged_rc_packing.py` and `tests/dataset/test_flat_mode_equivalence.py` oracles.

- [ ] **Step 4: Run RC tests**

Run: `pixi run -e dev pytest tests/unit/ragged/test_ragged_rc_packing.py -v`
Expected: PASS (new + existing).

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_rag_variants.py tests/unit/ragged/test_ragged_rc_packing.py && rtk git commit -m "refactor(rag-variants): rc_ via flat allele-level reverse_complement_masked

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task G4: `RaggedVariants.pad`

**Files:**
- Modify: `python/genvarloader/_dataset/_rag_variants.py`
- Consumes: `seqpro.rag.concatenate` (Task S3)
- Test: `tests/unit/ragged/test_rag_variants.py`

**Interfaces:**
- Produces: `RaggedVariants.pad(self, allele=b"N", ilen=0, start=-1, dosage=0.0, **pad_values) -> RaggedVariants` — ensures every group has ≥1 variant; groups that are already non-empty are unchanged; empty groups get one sentinel variant per field. Raises `ValueError` if a field lacks a pad value.

- [ ] **Step 1: Write failing test**

```python
# tests/unit/ragged/test_rag_variants.py (append)
import numpy as np
from seqpro.rag import Ragged
from genvarloader import RaggedVariants

def test_pad_fills_empty_groups_only():
    var_off = np.array([0, 2, 2, 3], np.int64)  # group1 empty
    char_off = np.array([0, 2, 3, 6], np.int64)
    alt = Ragged.from_offsets(np.frombuffer(b"ACGTTT", "S1").copy(),
                              (3, 1, None, None), [var_off, char_off])
    start = Ragged.from_offsets(np.array([1, 2, 3], np.int32), (3, 1, None), var_off)
    rv = RaggedVariants(alt=alt, start=start,
                        ilen=Ragged.from_offsets(np.zeros(3, np.int32), (3, 1, None), var_off))
    out = rv.pad()
    assert out.alt.to_ak().to_list() == [[b"AC", b"G"], [b"N"], [b"TTT"]]
    assert out.start.to_ak().to_list() == [[[1, 2]], [[-1]], [[3]]]
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest tests/unit/ragged/test_rag_variants.py::test_pad_fills_empty_groups_only -v`
Expected: FAIL.

- [ ] **Step 3: Implement (build a sentinel Ragged for empty groups, concatenate)**

Build, per field, a pad Ragged that has exactly one sentinel element for empty groups and zero elements for non-empty groups (lengths = `(self._rag["start"].lengths == 0)`), then `seqpro.rag.concatenate([base_field, pad_field], axis=-1)` and re-wrap. Use the per-field pad value; for allele (string) fields, build the sentinel via an S1 char Ragged of the encoded `allele` then `.to_strings()`. Validate all fields have a pad value:

```python
# in RaggedVariants
    def pad(self, allele=b"N", ilen=0, start=-1, dosage=0.0, **pad_values):
        if isinstance(allele, str):
            allele = allele.encode()
        pad_values = {"alt": allele, "ref": allele, "ilen": ilen,
                      "start": start, "dosage": dosage, **pad_values}
        missing = set(self.fields) - set(pad_values)
        if missing:
            raise ValueError(f"Missing pad values for fields: {missing}")
        empty = (self._rag["start"].lengths.reshape(-1) == 0)
        # ... build per-field pad Ragged (1 elem where empty, 0 elsewhere) and
        #     seqpro.rag.concatenate([self._rag[f], pad_f], axis=-1)
        # (see helper below)
        return RaggedVariants.from_record(Ragged.from_fields(out_fields))
```

> Provide a small module-level helper `_empty_group_pad(field_rag, value, empty_mask)` that returns a Ragged with one `value` per empty group (string-aware). Keep it loop-free: build offsets from `empty_mask.astype(int64)` and fill the data buffer with `value` repeated `empty.sum()` times (for strings, repeat the encoded bytes and set str_offsets accordingly).

- [ ] **Step 4: Run test**

Run: `pixi run -e dev pytest tests/unit/ragged/test_rag_variants.py::test_pad_fills_empty_groups_only -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_rag_variants.py tests/unit/ragged/test_rag_variants.py && rtk git commit -m "refactor(rag-variants): pad via seqpro.rag.concatenate (empty-group sentinels)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task G5: `RaggedVariants.to_nested_tensor_batch`

**Files:**
- Modify: `python/genvarloader/_dataset/_rag_variants.py`
- Test: existing torch tests under `tests/` (e.g. `tests/dataset/` torch paths) — gate on them; skipped where torch unavailable.

**Interfaces:**
- Produces: `RaggedVariants.to_nested_tensor_batch(self, device="cpu", tokenizer=None) -> dict[str, NestedTensor | int]` — same keys/semantics as today (`"alts"`/per-field nested tensors + `max_n_vars`/`max_alt_len`/`max_ref_len`), reading char-view `.data`/`.offsets` directly instead of walking awkward layouts.

- [ ] **Step 1: Write failing test (guarded by torch availability)**

```python
# tests/unit/ragged/test_rag_variants.py (append)
import numpy as np, pytest
from seqpro.rag import Ragged
from genvarloader import RaggedVariants
torch = pytest.importorskip("torch")

def test_to_nested_tensor_batch_shapes():
    var_off = np.array([0, 2, 3], np.int64)
    char_off = np.array([0, 2, 3, 6], np.int64)
    alt = Ragged.from_offsets(np.frombuffer(b"ACGTTT", "S1").copy(),
                              (2, 1, None, None), [var_off, char_off])
    start = Ragged.from_offsets(np.array([1, 2, 3], np.int32), (2, 1, None), var_off)
    rv = RaggedVariants(alt=alt, start=start,
                        ilen=Ragged.from_offsets(np.zeros(3, np.int32), (2, 1, None), var_off)
                        ).to_packed()
    out = rv.to_nested_tensor_batch()
    assert out["max_n_vars"] == 2
    assert out["max_alt_len"] == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest tests/unit/ragged/test_rag_variants.py::test_to_nested_tensor_batch_shapes -v`
Expected: FAIL.

- [ ] **Step 3: Implement using char-view buffers**

Reimplement using the record fields: for numeric fields, `data = field.to_packed().data`, `offsets = field.offsets` (variant-level) → `nt_jag`; for allele fields, `chars = field.to_chars().to_packed()`, allele-level `offsets = chars.offsets`, `data = chars.data` (tokenized via the `tokenizer` arg as today: `"seqpro"` → `sp.tokenize`, `None` → uint8, else callable). Compute `max_n_vars = int(self._rag["start"].lengths.max())`, `max_alt_len`/`max_ref_len = int(chars.lengths.max())`. Delete the old `_alleles_to_nested_tensor` awkward walker.

- [ ] **Step 4: Run test**

Run: `pixi run -e dev pytest tests/unit/ragged/test_rag_variants.py::test_to_nested_tensor_batch_shapes -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_rag_variants.py tests/unit/ragged/test_rag_variants.py && rtk git commit -m "refactor(rag-variants): nested-tensor batch from char-view buffers; drop awkward walker

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task G6: Finalize `_rag_variants.py` (remove all awkward)

**Files:**
- Modify: `python/genvarloader/_dataset/_rag_variants.py`
- Test: `tests/unit/ragged/test_rag_variants.py`, `tests/unit/ragged/test_ragged_rc_packing.py`

- [ ] **Step 1: Remove residual awkward**

Confirm the file no longer imports awkward and has no `ak.`/`from_ak`/`ak.behavior`/`RaggedVariant` references. Keep a tiny interop shim ONLY if a consumer still needs it (it should not after Phase 3): no `from_ak`. Run:

Run: `grep -nE "ak\.|awkward|from_ak|RaggedVariant\b|behavior" python/genvarloader/_dataset/_rag_variants.py`
Expected: no matches (empty output).

- [ ] **Step 2: Run the rag-variants unit tests**

Run: `pixi run -e dev pytest tests/unit/ragged/test_rag_variants.py tests/unit/ragged/test_ragged_rc_packing.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
rtk git add python/genvarloader/_dataset/_rag_variants.py && rtk git commit -m "refactor(rag-variants): remove awkward entirely from RaggedVariants

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

# Phase 2 (GVL) — shm + ragged ops

### Task G7: `_shm_layout.py` record-buffer serialization

**Files:**
- Modify: `python/genvarloader/_shm_layout.py` (`_write_rag_variants` ~138-209 and onward, `_read_rag_variants` ~546-594)
- Test: existing shm tests (search `tests/` for shm round-trip); add a focused round-trip test.

**Interfaces:**
- Consumes: `RaggedVariants` (record), `RaggedVariants.from_record`.
- Produces: `_write_rag_variants`/`_read_rag_variants` round-trip a `RaggedVariants` through the shared-memory buffer using `_core` buffers (`.data`/`.offsets`/`str_offsets`/`.shape`) — no awkward.

- [ ] **Step 1: Write failing round-trip test**

```python
# tests/unit/test_shm_layout_rag_variants.py (create)
import numpy as np
from seqpro.rag import Ragged
from genvarloader import RaggedVariants
from genvarloader._shm_layout import pack_to_shm, unpack_from_shm  # use the actual public entry points

def _rv():
    var_off = np.array([0, 2, 3], np.int64)
    char_off = np.array([0, 2, 3, 6], np.int64)
    alt = Ragged.from_offsets(np.frombuffer(b"ACGTTT", "S1").copy(),
                              (2, 1, None, None), [var_off, char_off])
    start = Ragged.from_offsets(np.array([1, 2, 3], np.int32), (2, 1, None), var_off)
    return RaggedVariants(alt=alt, start=start,
                          ilen=Ragged.from_offsets(np.zeros(3, np.int32), (2, 1, None), var_off))

def test_rag_variants_shm_roundtrip():
    rv = _rv()
    rt = unpack_from_shm(pack_to_shm(rv))   # adapt to real API surface
    assert rt.alt.to_ak().to_list() == rv.alt.to_ak().to_list()
    assert rt.start.to_ak().to_list() == rv.start.to_ak().to_list()
```

> Adjust the import/entry points to the file's actual public functions (read `_shm_layout.py`'s top-level `pack`/`unpack`/`to_shm` API; the dispatch is around lines 360-400 and 808-811).

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest tests/unit/test_shm_layout_rag_variants.py -v`
Expected: FAIL (writer/reader still awkward; constructor signature changed).

- [ ] **Step 3: Rewrite writer**

In `_write_rag_variants`, replace the `ak.to_layout` walk. For each field of `rv._rag`: numeric field → write `field.to_packed().offsets` (outer/variant offsets) + `field.data`; allele field → write variant offsets + char-level `str_offsets` + char `data`. Store `field_kind` (0 numeric, 1 alleles) and the leading fixed dims (batch, ploidy) in the descriptor as today. Use `rv._rag[name]` for access; get ploidy from `rv.shape[1]`.

- [ ] **Step 4: Rewrite reader**

In `_read_rag_variants`, replace the `ListOffsetArray`/`RegularArray`/`ak.zip` reconstruction. Rebuild each field as a `_core.Ragged`: numeric → `Ragged.from_offsets(leaf, (b, p, None), outer_offsets)`; allele → `Ragged.from_offsets(char_data, (b, p, None, None), [outer_offsets, inner_offsets]).to_strings()`. Then `return RaggedVariants(**{name: field for ...})` (or `from_record(Ragged.from_fields(...))`). Remove `from ._dataset._rag_variants import RaggedVariants`'s reliance on `from_ak`.

- [ ] **Step 5: Run round-trip + shm suite**

Run: `pixi run -e dev pytest tests/unit/test_shm_layout_rag_variants.py -v && pixi run -e dev pytest tests -k shm -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
rtk git add python/genvarloader/_shm_layout.py tests/unit/test_shm_layout_rag_variants.py && rtk git commit -m "refactor(shm): serialize RaggedVariants via _core buffers (no awkward)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task G8: `_ragged.py` — concat for `prepend_pad_itv`; drop awkward RC helpers

**Files:**
- Modify: `python/genvarloader/_ragged.py` (`prepend_pad_itv` ~136-168; `reverse_complement`/`_ak_comp_dna_helper` ~329-344; imports ~7-11)
- Test: existing interval tests (`tests/` interval/track paths); add a `prepend_pad_itv` parity test.

**Interfaces:**
- Consumes: `seqpro.rag.concatenate` (Task S3).
- Produces: `RaggedIntervals.prepend_pad_itv` unchanged signature/semantics, implemented via `seqpro.rag.concatenate`. `reverse_complement`/`_ak_comp_dna_helper` removed if unused elsewhere.

- [ ] **Step 1: Write failing parity test**

```python
# tests/unit/ragged/test_prepend_pad_itv.py (create)
import numpy as np
from seqpro.rag import Ragged
from genvarloader._ragged import RaggedIntervals

def test_prepend_pad_itv_prepends_one_per_group():
    def mk(vals, off):
        return Ragged.from_offsets(np.array(vals, np.int32), (1, 1, None),
                                   np.array(off, np.int64))
    ri = RaggedIntervals(mk([0, 5], [0, 2]), mk([5, 9], [0, 2]),
                         Ragged.from_offsets(np.array([1.0, 2.0], np.float32),
                                             (1, 1, None), np.array([0, 2], np.int64)))
    out = ri.prepend_pad_itv(start=-1, end=-1, value=0.0)
    assert out.starts.to_ak().to_list() == [[[-1, 0, 5]]]
    assert out.values.to_ak().to_list() == [[[0.0, 1.0, 2.0]]]
```

- [ ] **Step 2: Run to verify failure or current pass**

Run: `pixi run -e dev pytest tests/unit/ragged/test_prepend_pad_itv.py -v`
Expected: PASS today (awkward) — this test pins behavior before the swap. After Step 3 it must still PASS.

- [ ] **Step 3: Reimplement `prepend_pad_itv` via concatenate**

Replace the three `ak.concatenate([pad, self.X.to_ak()], axis=2)` blocks. Build each pad as a `Ragged` with one element per group, then `seqpro.rag.concatenate([pad_rag, self.starts], axis=-1)` etc.:

```python
# python/genvarloader/_ragged.py  (prepend_pad_itv)
import seqpro.rag as spr
b, t, *_ = self.values.shape
n = b * t
def _pad(value, dtype):
    return Ragged.from_offsets(np.full(n, value, dtype), (b, t, None),
                               np.arange(n + 1, dtype=np.int64))
new_starts = spr.concatenate([_pad(start, np.int32), self.starts], axis=-1)
new_ends   = spr.concatenate([_pad(end,   np.int32), self.ends],   axis=-1)
new_values = spr.concatenate([_pad(value, np.float32), self.values], axis=-1)
return RaggedIntervals(new_starts, new_ends, new_values)
```

- [ ] **Step 4: Remove dead awkward RC helpers**

Run: `grep -rn "reverse_complement\b\|_ak_comp_dna_helper" python/genvarloader` to confirm `reverse_complement` (the awkward one at `_ragged.py:340`) and `_ak_comp_dna_helper` have no remaining callers (the masked `reverse_complement_masked` stays). Delete them and the now-unused `import awkward as ak` / `import awkward.operations.str as ak_str` from `_ragged.py`. Keep `ak`-free.

- [ ] **Step 5: Run interval/track tests**

Run: `pixi run -e dev pytest tests/unit/ragged/test_prepend_pad_itv.py tests/dataset tests/unit -k "interval or track or pad" -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
rtk git add python/genvarloader/_ragged.py tests/unit/ragged/test_prepend_pad_itv.py && rtk git commit -m "refactor(ragged): prepend_pad_itv via seqpro.rag.concatenate; drop awkward RC helpers

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

# Phase 3 (GVL) — consumers

Each task swaps a file's awkward usage using the Recipe table, then gates on the relevant existing tests. Where a new constructor/type is involved, the swap is shown explicitly.

### Task G9: `splice_map` → `_core.Ragged` (`_indexing.py`, `_splice.py`, `_reference.py`, `_impl.py`)

**Files:**
- Modify: `python/genvarloader/_dataset/_indexing.py` (12 import; 328, 336-337, 382, 446, 451, 459, 464)
- Modify: `python/genvarloader/_dataset/_splice.py` (6 import; 175-176, 213-214, 249, 288-294)
- Modify: `python/genvarloader/_dataset/_reference.py` (8 import; 404)
- Modify: `python/genvarloader/_dataset/_impl.py` (8 import; 371-372)
- Test: spliced-output tests (`tests/dataset` + `tests/integration` splice paths)

**Interfaces:**
- Produces: `splice_map` / `full_splice_map` are `Ragged[intp]` (one ragged axis of region indices), not `ak.Array`. Consumers use `_core` ops.

- [ ] **Step 1: Pin current behavior**

Identify the splice test(s): `grep -rln "splice" tests/` and run the spliced-output suite to confirm green pre-change:
Run: `pixi run -e dev pytest tests -k splice -q`
Expected: PASS (baseline).

- [ ] **Step 2: Change `splice_map` construction + type**

In `_splice.py`, `splice_map` is built (line ~213 `.to_ak()`); make it a `Ragged` instead of converting to `ak.Array`. Update the `@dataclass` field annotations (175-176) from `ak.Array` to `Ragged`. Replace:
- `ak.count(sel, -1)` (289) → `sel.lengths` (R5)
- `ak.flatten(sel, -1).to_numpy().astype(np.intp)` (294) → `sel.to_packed().data.astype(np.intp)` (R4)
- `self.full_splice_map[abs_idx]` / `[row_idxs]` (249, 288) → `Ragged` indexing (works natively).

- [ ] **Step 3: Update `_indexing.py`**

- `splice_map: ak.Array` (328, 459, 464) → `Ragged`.
- `ak.max(splice_map, None)` / `ak.min(...)` (336-337) → `splice_map.to_packed().data.max()` / `.min()` (R1/R2).
- `ak.flatten(new_map.splice_map, None).to_numpy()` (382) → `new_map.splice_map.to_packed().data` (R3).
- `ak.count(r_idx, -1)` (446) → `r_idx.lengths` (R5); `ak.flatten(r_idx, -1).to_numpy()` (451) → `r_idx.to_packed().data` (R4).
- Remove `import awkward as ak` (12) if no longer used.

- [ ] **Step 4: Update `_reference.py` and `_impl.py`**

- `_reference.py:404` `ak.flatten(new_map.splice_map, None).to_numpy()` → `new_map.splice_map.to_packed().data` (R3); drop `import awkward` (8) if unused.
- `_impl.py:371-372` `ak.max(sm.splice_map, None)` / `ak.min(...)` → `.to_packed().data.max()/.min()` (R1/R2). (Note `_impl.py` keeps `import awkward as ak` until Task G14 if other sites remain; otherwise drop it here.)

- [ ] **Step 5: Run splice tests**

Run: `pixi run -e dev pytest tests -k splice -q && pixi run -e dev pytest tests/integration/dataset/test_query_filters.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
rtk git add python/genvarloader/_dataset/_indexing.py python/genvarloader/_dataset/_splice.py python/genvarloader/_dataset/_reference.py python/genvarloader/_dataset/_impl.py && rtk git commit -m "refactor(splice): splice_map as _core.Ragged; drop awkward aggregations

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task G10: `_haps.py` — allele layout helpers + AF-filter round-trip

**Files:**
- Modify: `python/genvarloader/_dataset/_haps.py` (`_build_allele_layout` 195-221, `_alt_layout_parts` 224-238, AF line 761, imports 23-27)
- Test: `tests/integration/dataset/test_query_filters.py` (AF path), `tests/dataset/test_flat_mode_equivalence.py`

**Interfaces:**
- Produces:
  - `_build_allele_layout(data, allele_offsets, group_offsets, ploidy) -> Ragged` returns an S1 char `(b, p, ~v, ~l)` `_core.Ragged` (was `ak.Array`). Callers pass it to `RaggedVariants(alt=...)` which collapses to opaque.
  - The AF filter (761) uses `_core` ops (no `.to_ak()`/`ak.to_regular`).
  - `_alt_layout_parts` removed (its only consumers were `_rag_variants.py` paths now deleted) OR retained returning numpy primitives from a `_core.Ragged` if still referenced.

- [ ] **Step 1: Identify `_build_allele_layout` consumers**

Run: `grep -rn "_build_allele_layout\|_alt_layout_parts" python/genvarloader`
Confirm consumers (the genotype gather that builds `alt`/`ref`). Each will now receive a `Ragged`.

- [ ] **Step 2: Reimplement `_build_allele_layout` to return a `Ragged`**

```python
# python/genvarloader/_dataset/_haps.py
from seqpro.rag import Ragged

def _build_allele_layout(data, allele_offsets, group_offsets, ploidy) -> Ragged:
    """Flat allele bytes + two offset levels -> (b, p, ~v, ~l) S1 Ragged."""
    buf = np.ascontiguousarray(data)
    if not buf.flags.writeable:
        buf = buf.copy()
    n_groups = group_offsets.size - 1
    b = n_groups // ploidy
    return Ragged.from_offsets(
        buf.view("S1"), (b, ploidy, None, None),
        [np.asarray(group_offsets, np.int64), np.asarray(allele_offsets, np.int64)],
    )
```

- [ ] **Step 3: Fix the AF-filter round-trip (line 761)**

Replace `genos = Ragged(ak.to_regular(genos[_keep].to_ak(), 1)).to_packed()` with a `_core`-native filter: index the `_core.Ragged` directly and pack — `genos = genos[_keep].to_packed()` (verify `_keep` indexing semantics match; `genos` is already a `_core.Ragged`). Remove the `ak.to_regular`/`to_ak` round-trip.

- [ ] **Step 4: Remove awkward imports**

Delete `import awkward as ak` (23) and the `awkward.contents`/`awkward.index` imports (26-27) once `_build_allele_layout`/`_alt_layout_parts` no longer use them. If `_alt_layout_parts` is now unreferenced, delete it; else reimplement it to read `.data`/`.offsets` off the `_core.Ragged`.

Run: `grep -nE "ak\.|awkward" python/genvarloader/_dataset/_haps.py`
Expected: no matches.

- [ ] **Step 5: Run AF + equivalence tests**

Run: `pixi run -e dev pytest tests/integration/dataset/test_query_filters.py tests/dataset/test_flat_mode_equivalence.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
rtk git add python/genvarloader/_dataset/_haps.py && rtk git commit -m "refactor(haps): allele layout returns _core.Ragged; AF filter without awkward

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task G11: `_flat_variants.py` — `to_ragged`/allele layout without awkward

**Files:**
- Modify: `python/genvarloader/_dataset/_flat_variants.py` (70-72, 100, 193-206 awkward-built `ak.Array(node)`; 340-345 `to_ragged`)
- Test: `tests/dataset/test_flat_variants.py`, `tests/dataset/test_flat_mode_equivalence.py`, `tests/dataset/test_flat_getitem_snapshot.py`

**Interfaces:**
- Produces: `_FlatVariants.to_ragged()` builds `RaggedVariants` from flat `_Flat` buffers via `_core.Ragged` + the new constructor (no `ak.Array(node)`); the two `ak.Array(node)` allele builders become `_core.Ragged` (reuse `_haps._build_allele_layout`).

- [ ] **Step 1: Pin behavior**

Run: `pixi run -e dev pytest tests/dataset/test_flat_variants.py tests/dataset/test_flat_getitem_snapshot.py -q`
Expected: PASS (baseline).

- [ ] **Step 2: Replace the two allele `ak.Array(node)` builders (100, 206)**

Both build a `(…, ~v, ~l)` allele awkward layout from flat buffers. Replace with `_haps._build_allele_layout(data, allele_offsets, group_offsets, ploidy)` (now returns a `Ragged`). Remove the local `import awkward as ak` + `awkward.contents`/`awkward.index` (70-72, 193-195).

- [ ] **Step 3: Update `to_ragged` (340-345)**

```python
# python/genvarloader/_dataset/_flat_variants.py  (_FlatVariants.to_ragged)
from ._rag_variants import RaggedVariants
kw = {name: flat.to_ragged() for name, flat in self.fields.items()}  # numeric -> Ragged
# build alt/ref allele Rageds via _build_allele_layout from the flat allele buffers
return RaggedVariants(**kw)   # alt/ref passed as Ragged char arrays
```

Adapt to the file's actual field layout (allele fields vs numeric `_Flat`s). The constructor accepts char-`Ragged` alt/ref and numeric `Ragged` fields.

- [ ] **Step 4: Remove awkward**

Run: `grep -nE "ak\.|awkward" python/genvarloader/_dataset/_flat_variants.py`
Expected: no matches.

- [ ] **Step 5: Run flat-variant tests**

Run: `pixi run -e dev pytest tests/dataset/test_flat_variants.py tests/dataset/test_flat_mode_equivalence.py tests/dataset/test_flat_getitem_snapshot.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
rtk git add python/genvarloader/_dataset/_flat_variants.py && rtk git commit -m "refactor(flat-variants): build RaggedVariants via _core.Ragged

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task G12: `_chunked.py` — type checks without awkward

**Files:**
- Modify: `python/genvarloader/_chunked.py` (119 docstring, 126 import, 136-137, 162-163)
- Test: chunked-iteration tests (`grep -rln "chunk" tests/`)

- [ ] **Step 1: Replace `isinstance(arr, ak.Array)` checks (R6)**

Both branches use `isinstance(arr, ak.Array)` to detect `RaggedVariants` (len/slicing). Replace with `isinstance(arr, RaggedVariants)`:

```python
# python/genvarloader/_chunked.py
from ._dataset._rag_variants import RaggedVariants   # at module top or local
# ...
if isinstance(arr, RaggedVariants):
    # len()/slicing work via the wrapper's __len__/__getitem__
```

Remove `import awkward as ak` (126) and update the docstring (119, 137, 163) to drop "ak.Array subclass".

- [ ] **Step 2: Run chunked tests**

Run: `pixi run -e dev pytest tests -k "chunk or dataloader or double_buffer" -q`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
rtk git add python/genvarloader/_chunked.py && rtk git commit -m "refactor(chunked): detect RaggedVariants by type, not ak.Array

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task G13: `_torch.py` — `to_nested_tensor` without awkward

**Files:**
- Modify: `python/genvarloader/_torch.py` (6 import, 287, 296)
- Test: torch tests (skipped where torch unavailable)

**Interfaces:**
- Produces: `to_nested_tensor(rag: Ragged) -> torch.Tensor` — accepts a `_core.Ragged` (drop the `ak.Array` union/branch).

- [ ] **Step 1: Replace the `ak.Array` branch**

`to_nested_tensor(rag: Ragged | ak.Array)` with `if isinstance(rag, ak.Array): rag = Ragged(rag)` → change signature to `rag: Ragged` and remove the branch (callers already pass `_core.Ragged`). Verify callers: `grep -rn "to_nested_tensor(" python/genvarloader`. Remove `import awkward as ak` (6).

- [ ] **Step 2: Run torch tests**

Run: `pixi run -e dev pytest tests -k "nested or torch" -q`
Expected: PASS (or skipped where torch absent).

- [ ] **Step 3: Commit**

```bash
rtk git add python/genvarloader/_torch.py && rtk git commit -m "refactor(torch): to_nested_tensor accepts _core.Ragged only

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task G14: `_write.py` — `ak.max`/`flatten`/`concatenate`

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py` (16 import, 900-901, 1006-1008)
- Test: write-path tests (`tests/dataset` write paths; plink2/PGEN may be skipped on macOS)

- [ ] **Step 1: Replace the per-region max (1006-1008)**

`v_idxs = ak.max(sp_genos.to_ak(), -1).to_numpy().max((1, 2))` reduces the innermost ragged axis (per (region,sample,ploidy) max), then max over (sample,ploidy) per region. Replace with a `_core`-native per-group max over `sp_genos` (a `_core.Ragged`) using offsets + `np.maximum.reduceat`, reshaped to `(regions, samples, ploidy)` then `.max((1,2))`:

```python
# python/genvarloader/_dataset/_write.py
g = sp_genos.to_packed()
flat, off = g.data, np.asarray(g.offsets)
# per-group max; empty groups -> sentinel handled as today (groups are non-empty here)
per_group = np.maximum.reduceat(flat, off[:-1])
v_idxs = per_group.reshape(g.shape[0], g.shape[1], g.shape[2]).max((1, 2))
```

Adapt the reshape dims to `sp_genos.shape` (regions, samples, ploidy). Keep the existing comment's intent (result must not alias `svar.genos.data`) — `reduceat` allocates a new array.

- [ ] **Step 2: Replace the concatenate+flatten (900-901)**

`ak.flatten(ak.concatenate([a.to_ak() for a in ls_sparse], -1), ...)` → `seqpro.rag.concatenate(ls_sparse, axis=-1).to_packed().data` (R7), preserving the subsequent flatten target (match the original flatten axis).

- [ ] **Step 3: Remove awkward import**

Remove `import awkward as ak` (16) once both sites are swapped.
Run: `grep -nE "ak\.|awkward" python/genvarloader/_dataset/_write.py`
Expected: no matches.

- [ ] **Step 4: Run write tests**

Run: `pixi run -e dev pytest tests/dataset -k "write or update" -q`
Expected: PASS (PGEN/plink2 cases skipped on macOS — fine).

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_write.py && rtk git commit -m "refactor(write): per-region max + concat without awkward

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

# Phase 4 — verify, delete awkward backend, docs

### Task G15: GVL awkward-free verification + comment/docstring cleanup

**Files:**
- Modify: any remaining files with stale awkward references in comments/docstrings: `_dataset/_tracks.py` (423, 467), `_flat.py` (127), `_dataset/_reconstruct.py`, `_dataset/_open.py`, `_dataset/_protocol.py`, `_dataset/_impl.py` (docstrings 855, 914), `_double_buffered_loader.py` (46), `_dataset/_query.py` comments.
- Test: full suite + lint.

- [ ] **Step 1: Production grep must be clean**

Run: `grep -rnE "ak\.|import awkward|from awkward|from_ak" python/genvarloader --include='*.py'`
Expected: **no matches** (zero awkward in production). If a match remains, it's a missed site — fix it before proceeding.

- [ ] **Step 2: Update stale comments/docstrings**

Reword comments that describe behavior in awkward terms (e.g. "ak.Array subclass", "ak.concatenate(axis=1) semantics", "awkward-backed Ragged/RaggedVariants" in `_impl.py:855`) to describe the `_core.Ragged` reality. These are doc-only; no behavior change.

- [ ] **Step 3: Full suite + lint**

Run: `pixi run -e dev pytest tests -q`
Expected: **800 passed, 0 failed**.
Run: `pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/`
Expected: clean (format makes no changes, or commit the formatting).
Run: `pixi run -e dev typecheck`
Expected: no new errors.

- [ ] **Step 4: Commit**

```bash
rtk git add -A python/genvarloader && rtk git commit -m "docs: scrub stale awkward references from comments; GVL is awkward-free in production

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task G16: Delete the seqpro awkward backend (`_array.py`)

**Files (in `~/projects/SeqPro`):**
- Delete: `python/seqpro/rag/_array.py`, `python/seqpro/rag/_gufuncs.py`
- Create: `python/seqpro/rag/_ak_interop.py` (relocated interop helpers)
- Modify: `python/seqpro/rag/_ingest.py`, `python/seqpro/rag/__init__.py`, `python/seqpro/rag/_ops.py`, `python/seqpro/rag/_core.py`
- Test: seqpro suite + all three consumer suites.

**Interfaces:**
- Produces: seqpro with no awkward backend; `Ragged` is `_core` only. Interop helpers (`unbox`, `RagParts`, `_parts_to_content`, `DTYPE_co`, `RDTYPE_co`) live in `_ak_interop.py`.

- [ ] **Step 1: Relocate interop helpers**

Move `unbox`, `RagParts`, `_parts_to_content`, `DTYPE_co`, `RDTYPE_co` (currently in `_array.py`, imported by `_ingest.py`/`__init__.py`) into a new `_ak_interop.py`; update imports in `_ingest.py`/`__init__.py`.

- [ ] **Step 2: Delete `_array.py` + `_gufuncs.py`; drop dead branches**

Delete both files. In `_ops.to_packed`, remove the `from ._array import Ragged as _ArrayRagged` fallback (329). In `_core.is_rag_dtype`, remove the `_ArrayRagged` branch.

- [ ] **Step 3: Run seqpro suite**

Run: `cd ~/projects/SeqPro && pixi run -e dev pytest tests/ -q`
Expected: **544 passed** (the 4 test files that were ported to the `_core` API during the audit now run only under `_core` and pass).

- [ ] **Step 4: Run all consumer suites against `_array`-free seqpro**

Run:
```bash
cd /Users/david/projects/GenVarLoader && pixi run -e dev pytest tests -q     # 800 passed
cd /Users/david/projects/genoray && pixi run test                            # 456 passed
cd /Users/david/projects/genvarformer && pixi run -e dev pytest tests -q     # CPU subset green
```
Expected: GVL 800, genoray 456, genvarformer CPU subset green. (genvarformer GPU/CUDA subset is run opportunistically on a Linux+CUDA host — not a blocker per spec §4.D.)

- [ ] **Step 5: Commit (seqpro)**

```bash
cd ~/projects/SeqPro && rtk git add -A && rtk git commit -m "refactor(rag)!: delete awkward _array backend; relocate interop to _ak_interop

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task G17: Docs — roadmap, skills, pixi repoint note

**Files:**
- Modify: `docs/roadmaps/rust-migration.md`
- Modify (if public behavior changed): `skills/genvarloader/SKILL.md`
- Modify: `pixi.toml` (both repos) — repoint pins before merge

- [ ] **Step 1: Update the rust-migration roadmap**

In `docs/roadmaps/rust-migration.md`, add a dated entry to the **Notes & decisions log** recording that GVL was migrated off awkward onto seqpro's Rust-backed `_core.Ragged` at the Python level (single record `Ragged` with opaque-string `alt`/`ref`; seqpro gained `concatenate`, record `to_ak`/`to_packed`; `_array.py` deleted). Tick the "Remove `awkward` from the foundation layer" bullet under **Phase 1** (line ~134) and the "drop `awkward` from these hot paths" clause under **Phase 2** (line ~145) as accomplished at the Python level, noting the Rust-crate rewrite of these kernels is still pending. Add the PR link when opened.

- [ ] **Step 2: Update the genvarloader skill if needed**

Re-read `skills/genvarloader/SKILL.md` against the public API. `RaggedVariants` stays in `__all__` with its documented `.alt`/`.ref`/`.start`/`.ilen`/`.end`/`.rc_`/`.pad`/`.to_nested_tensor_batch` surface. If the skill documents `RaggedVariants` as an awkward array or describes awkward-specific behavior, update it to "a record `seqpro.rag.Ragged` wrapper (opaque-string alleles)". Note the `==`/equality semantics now come from `_core.Ragged` (ufunc-based) rather than awkward.

- [ ] **Step 3: Repoint pixi pins (pre-merge)**

Before merging: in `GenVarLoader/pixi.toml` (91-92), `genoray/pixi.toml`, and `genvarformer/pixi.toml`, repoint `seqpro`/`genoray` from local editable `{ path = "../SeqPro" }` to the released/merged versions. Re-run each suite after repointing. (Do this last; keep local paths during development.)

- [ ] **Step 4: Commit**

```bash
rtk git add docs/roadmaps/rust-migration.md skills/genvarloader/SKILL.md && rtk git commit -m "docs: record awkward->_core.Ragged migration in roadmap + skill

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- §2 target (record Ragged + wrapper) → G1–G6. ✅
- §3 per-op mappings: construct/field/ilen/end → G1; rc_ → G3; to_packed → G2/S2; to_padded → existing `_ragged.to_padded` (unchanged, already `_core`); pad → G4/S3; nested-tensor → G5; shm → G7. ✅
- §4.A concat Rust kernel → S3; consumers → G8 (prepend_pad_itv), G4 (pad), G14 (write concat). ✅
- §4.B to_ak record fix → S1. ✅
- §4.C record to_packed → S2. ✅
- §4.D delete `_array.py` → G16. ✅
- §5 Phase 1/2/3 files: `_rag_variants` G1-6, `_shm_layout` G7, `_ragged` G8, `_haps` G10, `_indexing/_splice/_reference/_impl` G9+G9/G14, `_chunked` G12, `_flat_variants` G11, `_torch` G13, `_write` G14, `_flat`/`_tracks`/`_reconstruct` comments G15. ✅
- §6 gotchas: to_packed-after-to_chars (S2 removes the need), per-field oracle (tests), RC R=1 view (G3), multi-leading-axis indexing (G1 note + G9 gate). ✅
- §8 verification (800 / 544 / consumer suites) → G15, G16. ✅
- §9 docs (roadmap, skill, pixi repoint) → G17. ✅
- User's added ask (reflect in `rust-migration.md`) → G17 Step 1. ✅

**Placeholder scan:** consumer tasks reference exact file:line sites + Recipe codes; foundational tasks carry full code. The `pad`/`rc_`/`shm` tasks include "implementation notes" pointing at the precise helper to write and the oracle test that gates it — concrete, not "TODO". No "TBD"/"handle edge cases"/"similar to Task N".

**Type consistency:** `RaggedVariants.__init__(alt, start, ref, ilen, dosage, **fields)`, `.from_record`, `.alt/.ref/.start/.ilen/.end/.shape/.fields`, `.to_packed/.rc_/.pad/.to_nested_tensor_batch/.reshape/.squeeze/__len__/__getitem__` used consistently across G1–G14 and the consumers. `seqpro.rag.concatenate(rags, axis)` consistent across S3/G8/G14/G4. `_build_allele_layout(...) -> Ragged` consistent across G10/G11.
