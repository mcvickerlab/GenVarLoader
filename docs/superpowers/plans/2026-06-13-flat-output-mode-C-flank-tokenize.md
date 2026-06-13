# Flat output mode — C: flank fetch + tokenization + window mode — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move genvarformer's per-variant reference-flank fetch + byte→int tokenization (and the varenc ref/alt window assembly) into gvl as pure-numpy/numba flat token output, with a caller-supplied seqpro-style LUT.

**Architecture:** Extend sub-project A's flat variant decode (`get_variants_flat`). When a `flank_length` + token LUT are configured on `Haps`, compute `[flank5|flank3]` tokens as a ride-along `flank_tokens` field on `_FlatVariants`; when `with_seqs("variant-windows")` is active, instead emit a `_FlatVariantWindows` with two-level `ref_window`/`alt_window` token buffers (`ref_window` = one contiguous reference read; `alt_window` = flank5·alt·flank3 assembly). Reference reads reuse the existing `Reference.fetch`/`padded_slice` kernel (pads OOB with N); tokenization is a vectorized 256-LUT gather. Dedup of sample-invariant reads is a later, output-invariant optimization (Phase H), not required for correctness.

**Tech Stack:** Python, numpy, numba (`@nb.njit`), seqpro (`Ragged`, `DNA` alphabet), maturin/pixi. Tests via `pixi run -e dev pytest`.

---

## ⚠️ Dependency on sub-project A (read before starting)

This plan **builds on sub-project A** (flat variant decode), which is **not yet on `main`**. A lives on branch `feat/flat-output-mode-a0-a` (worktree `.claude/worktrees/flat-output-impl`). All symbol names below — `_FlatVariants`, `_FlatAlleles`, `get_variants_flat`, `Haps.__call__(..., flat=)`, `QueryView.flat_output`, the `_query.py` boundary at `_unwrap`/`to_ragged` — track A's current shape.

**Before Task 1:** rebase/branch this work onto the merged A (or onto `feat/flat-output-mode-a0-a` if A has not yet merged). If A's final API differs from the references here (e.g. `get_variants_flat` signature, `_FlatVariants.fields` layout), adjust the wiring tasks (Phase E–F) accordingly; the self-contained kernel/type/test tasks (Phase B–D) are unaffected.

Key A facts this plan relies on (verify they still hold):
- `get_variants_flat(haps, idx)` in `python/genvarloader/_dataset/_flat_variants.py:223` computes `v_idxs, row_offsets` via `_gather_v_idxs`, gathers `alt` via `_gather_alleles` into `_FlatAlleles(alt_data, alt_seq_off, row_offsets, shape)`, and returns `_FlatVariants(fields)`.
- `_FlatAlleles` (`_flat_variants.py:19`) = `(byte_data: uint8, seq_offsets: int64, var_offsets: int64, shape)`.
- `Haps.__call__(idx, r_idx, regions, ..., flat=False)` (`_haps.py:502`) dispatches the variants kind; `regions` is `(b,3)` = `(contig, start, end)`; `Haps.reference` is present.
- `Reference.fetch(contigs, starts, ends) -> Ragged[np.bytes_]` (`_reference.py:117`) pads OOB with `pad_char` (ord `N`); `padded_slice` in `_utils.py:14`.
- `with_settings` (`_impl.py:206`) and `with_seqs` (`_impl.py:485`); `_check_valid_state` (`_impl.py:361`); `_build_reconstructor` kind_map (`_reconstruct.py:279`, `"variants": RaggedVariants`).

---

## File Structure

- **Create** `python/genvarloader/_dataset/_flat_flanks.py` — LUT builder, the flank/window computation entry point, and the numba assembly kernel. One responsibility: turn gathered variant fields + reference into flank/window tokens.
- **Modify** `python/genvarloader/_dataset/_flat_variants.py` — add `flank_tokens` attr to `_FlatVariants`; add `_FlatWindow` (two-level token layout) and `_FlatVariantWindows`; call into `_flat_flanks` from `get_variants_flat`.
- **Modify** `python/genvarloader/_dataset/_haps.py` — add `flank_length` / `token_lut` / `token_dtype` fields to `Haps`; thread `regions` into the flat variants branch; route the windows kind.
- **Modify** `python/genvarloader/_dataset/_reconstruct.py` — register `"variant-windows"` kind → `_FlatVariantWindows` marker in the kind_map.
- **Modify** `python/genvarloader/_dataset/_impl.py` — `with_settings` flank kwargs + LUT build; `with_seqs` accept `"variant-windows"`; validation in `_check_valid_state`.
- **Modify** `python/genvarloader/_dataset/_query.py` — ensure `_FlatVariantWindows` passes through the flat boundary (unwrap/`to_ragged` handling, rc routing).
- **Modify** `python/genvarloader/__init__.py` — export `FlatVariantWindows` (and confirm `FlatVariants`/`FlatAlleles` exports from A).
- **Modify** `skills/genvarloader/SKILL.md` — document the new settings, kind, and types.
- **Tests** `tests/dataset/test_flat_flanks.py` (new) — oracle byte-identity, OOB, windows, validation, awkward-absence.

---

## Phase A — token LUT + settings

### Task 1: LUT builder

**Files:**
- Create: `python/genvarloader/_dataset/_flat_flanks.py`
- Test: `tests/dataset/test_flat_flanks.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/dataset/test_flat_flanks.py
import numpy as np
from genvarloader._dataset._flat_flanks import build_token_lut


def test_build_token_lut_dna():
    lut, dtype = build_token_lut(b"ACGT", unknown_token=4)
    assert lut.shape == (256,)
    assert dtype == np.uint8
    # alphabet bytes map to their index
    assert lut[ord("A")] == 0
    assert lut[ord("C")] == 1
    assert lut[ord("G")] == 2
    assert lut[ord("T")] == 3
    # everything else -> unknown_token
    assert lut[ord("N")] == 4
    assert lut[0] == 4
    # tokenizing via fancy index works
    seq = np.frombuffer(b"ACGTN", dtype=np.uint8)
    assert lut[seq].tolist() == [0, 1, 2, 3, 4]


def test_build_token_lut_dtype_promotes_to_int32():
    # max token id 300 doesn't fit in uint8 -> int32
    lut, dtype = build_token_lut(bytes(range(200)), unknown_token=300)
    assert dtype == np.int32
    assert lut.dtype == np.int32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_flat_flanks.py::test_build_token_lut_dna -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError: cannot import name 'build_token_lut'`.

- [ ] **Step 3: Write minimal implementation**

```python
# python/genvarloader/_dataset/_flat_flanks.py
"""Reference flank fetch + byte->int tokenization (sub-project C).

Produces flat token buffers from already-gathered variant fields + the reference
genome. No awkward on the hot path.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def build_token_lut(alphabet: bytes, unknown_token: int) -> tuple[NDArray, np.dtype]:
    """Build a 256-entry byte->token lookup table (seqpro-style).

    Every byte value in ``alphabet`` maps to its position; every other byte
    (including ``N`` and padded out-of-bounds positions) maps to ``unknown_token``.
    """
    max_token = max(len(alphabet) - 1, unknown_token)
    dtype = np.uint8 if max_token <= np.iinfo(np.uint8).max else np.int32
    lut = np.full(256, unknown_token, dtype=dtype)
    for i, b in enumerate(alphabet):
        lut[b] = i
    return lut, np.dtype(dtype)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/dataset/test_flat_flanks.py -v`
Expected: PASS (both LUT tests).

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_flat_flanks.py tests/dataset/test_flat_flanks.py
rtk git commit -m "feat(flat): byte->int token LUT builder for flank tokenization"
```

---

### Task 2: `with_settings` flank kwargs + `Haps` fields + LUT build

**Files:**
- Modify: `python/genvarloader/_dataset/_haps.py` (`Haps` dataclass fields)
- Modify: `python/genvarloader/_dataset/_impl.py:206` (`with_settings`)
- Test: `tests/dataset/test_flat_flanks.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/dataset/test_flat_flanks.py
import numpy as np


def test_with_settings_stores_flank_config(snap_dataset):
    # snap_dataset is the session phased VCF+reference dataset (see test_flat_getitem_snapshot.py)
    ds = (
        snap_dataset.with_seqs("variants")
        .with_settings(flank_length=5, token_alphabet=b"ACGT", unknown_token=4)
    )
    haps = ds._seqs
    assert haps.flank_length == 5
    assert haps.token_lut is not None
    assert haps.token_lut[ord("A")] == 0
    assert haps.token_dtype == np.uint8


def test_with_settings_flank_length_zero_disables(snap_dataset):
    ds = snap_dataset.with_seqs("variants").with_settings(
        flank_length=0, token_alphabet=b"ACGT", unknown_token=4
    )
    assert ds._seqs.flank_length == 0
```

> Reuse the `snap_dataset` fixture; import it via the conftest or copy the fixture import pattern from `tests/dataset/test_flat_getitem_snapshot.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_flat_flanks.py::test_with_settings_stores_flank_config -v`
Expected: FAIL with `AttributeError: 'Haps' object has no attribute 'flank_length'`.

- [ ] **Step 3: Add fields to `Haps`**

In `python/genvarloader/_dataset/_haps.py`, add to the `Haps` dataclass field block (near `min_af`, `max_af`, `var_fields`):

```python
    flank_length: int | None = None
    token_lut: NDArray | None = None
    token_dtype: np.dtype | None = None
```

- [ ] **Step 4: Thread kwargs through `with_settings`**

In `python/genvarloader/_dataset/_impl.py`, extend the `with_settings` signature (after `var_filter`):

```python
        flank_length: int | None = None,
        token_alphabet: bytes | None = None,
        unknown_token: int | None = None,
```

Add to the docstring Parameters and insert this block before the `# If any source state changed` rebuild (around line 348):

```python
        if (
            flank_length is not None
            or token_alphabet is not None
            or unknown_token is not None
        ):
            if not isinstance(self._seqs, Haps):
                raise ValueError(
                    "Flank settings require a dataset with genotypes (variants)."
                )
            haps = to_evolve.get("_seqs", self._seqs)
            new_flank_len = haps.flank_length if flank_length is None else flank_length
            lut, lut_dtype = haps.token_lut, haps.token_dtype
            if token_alphabet is not None or unknown_token is not None:
                if token_alphabet is None or unknown_token is None:
                    raise ValueError(
                        "token_alphabet and unknown_token must be set together."
                    )
                from ._flat_flanks import build_token_lut

                lut, lut_dtype = build_token_lut(token_alphabet, unknown_token)
            to_evolve["_seqs"] = replace(
                haps,
                flank_length=new_flank_len,
                token_lut=lut,
                token_dtype=lut_dtype,
            )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/dataset/test_flat_flanks.py::test_with_settings_stores_flank_config tests/dataset/test_flat_flanks.py::test_with_settings_flank_length_zero_disables -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
rtk git add python/genvarloader/_dataset/_haps.py python/genvarloader/_dataset/_impl.py tests/dataset/test_flat_flanks.py
rtk git commit -m "feat(flat): thread flank_length + token LUT settings onto Haps"
```

---

## Phase B — flank tokens (ride-along) computation

### Task 3: per-variant flank/window coordinates + `compute_flank_tokens` (flanks mode)

**Files:**
- Modify: `python/genvarloader/_dataset/_flat_flanks.py`
- Test: `tests/dataset/test_flat_flanks.py`

The coordinate rule (matches genvarformer `_read_flank_seq`, tokens.py:60-68):
`flank5 = [start-L, start)`, `end = start - min(ilen,0) + 1`, `flank3 = [end, end+L)`,
`ref_window = [start-L, end+L)`.

- [ ] **Step 1: Write the failing test (pure-numpy oracle)**

```python
# append to tests/dataset/test_flat_flanks.py
from genvarloader._dataset._flat_flanks import compute_flank_tokens


def _oracle_flank_tokens(reference, v_contigs, starts, ilens, flank_len, lut):
    """Independent reference: fetch [start-L,start) and [end,end+L) per variant,
    tokenize, lay out [flank5|flank3]."""
    ends = starts - np.minimum(ilens, 0) + 1
    f5 = reference.fetch(v_contigs, starts - flank_len, starts).data.view(np.uint8)
    f3 = reference.fetch(v_contigs, ends, ends + flank_len).data.view(np.uint8)
    f5 = f5.reshape(len(starts), flank_len)
    f3 = f3.reshape(len(starts), flank_len)
    flank_bytes = np.concatenate([f5, f3], axis=1)  # (n_var, 2L)
    return lut[flank_bytes]


def test_compute_flank_tokens_matches_oracle(snap_dataset):
    ds = (
        snap_dataset.with_seqs("variants")
        .with_output_format("flat")
        .with_settings(flank_length=5, token_alphabet=b"ACGT", unknown_token=4)
    )
    out = ds[[0, 1, 2], [0, 1, 2]]  # 2-D (region, sample) index
    assert out.flank_tokens is not None
    # reconstruct oracle from the same gathered v_idxs (see Task 5 wiring; until then
    # this test is enabled once compute_flank_tokens is invoked end-to-end).
```

> The end-to-end assertion lands in Task 7 (after wiring). For Task 3, unit-test `compute_flank_tokens` directly with hand-built inputs:

```python
def test_compute_flank_tokens_unit(snap_dataset):
    haps = snap_dataset.with_seqs("variants").with_settings(
        flank_length=3, token_alphabet=b"ACGT", unknown_token=4
    )._seqs
    ref = haps.reference
    lut = haps.token_lut
    # one (b=1, ploidy=1) group with two variants
    v_contigs = np.array([0, 0], dtype=np.int32)
    starts = np.array([10, 20], dtype=np.int32)
    ilens = np.array([0, -2], dtype=np.int32)  # SNP, 2bp deletion
    row_offsets = np.array([0, 2], dtype=np.int64)
    tokens, off = compute_flank_tokens(
        ref, v_contigs, starts, ilens, flank_len=3, lut=lut, row_offsets=row_offsets
    )
    expected = _oracle_flank_tokens(ref, v_contigs, starts, ilens, 3, lut)
    np.testing.assert_array_equal(tokens.reshape(-1, 6), expected)
    np.testing.assert_array_equal(off, row_offsets)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_flat_flanks.py::test_compute_flank_tokens_unit -v`
Expected: FAIL with `ImportError: cannot import name 'compute_flank_tokens'`.

- [ ] **Step 3: Implement `compute_flank_tokens`**

```python
# add to python/genvarloader/_dataset/_flat_flanks.py

def compute_flank_tokens(
    reference,
    v_contigs: NDArray[np.integer],   # (n_var,) contig id per variant
    starts: NDArray[np.integer],      # (n_var,)
    ilens: NDArray[np.integer],       # (n_var,)
    flank_len: int,
    lut: NDArray,
    row_offsets: NDArray[np.int64],   # (b*p + 1,) per-(instance,ploid) variant offsets
) -> tuple[NDArray, NDArray[np.int64]]:
    """Ride-along flank tokens: ``[flank5 | flank3]`` (2*flank_len tokens) per
    variant. Returns ``(token_data, offsets)`` where ``token_data`` is flat
    ``(n_var * 2 * flank_len,)`` and ``offsets == row_offsets`` (one row per variant,
    fixed inner dim 2*flank_len)."""
    starts = np.asarray(starts, np.int32)
    ilens = np.asarray(ilens, np.int32)
    ends = starts - np.minimum(ilens, 0) + 1
    f5 = reference.fetch(v_contigs, starts - flank_len, starts).data.view(np.uint8)
    f3 = reference.fetch(v_contigs, ends, ends + flank_len).data.view(np.uint8)
    n = starts.shape[0]
    f5 = f5.reshape(n, flank_len)
    f3 = f3.reshape(n, flank_len)
    flank_bytes = np.concatenate([f5, f3], axis=1)  # (n, 2L)
    tokens = lut[flank_bytes]  # vectorized 256-LUT gather -> lut.dtype
    return tokens.reshape(-1), np.asarray(row_offsets, np.int64)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/dataset/test_flat_flanks.py::test_compute_flank_tokens_unit -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_flat_flanks.py tests/dataset/test_flat_flanks.py
rtk git commit -m "feat(flat): compute ride-along flank tokens from reference"
```

---

## Phase C — window mode types + assembly

### Task 4: `_FlatWindow` two-level token type + `_FlatVariantWindows`

**Files:**
- Modify: `python/genvarloader/_dataset/_flat_variants.py`
- Test: `tests/dataset/test_flat_flanks.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/dataset/test_flat_flanks.py
from genvarloader._dataset._flat_variants import _FlatWindow, _FlatVariantWindows


def test_flat_window_to_ragged_roundtrip():
    # two groups (b*p=2), variant counts [2, 1]; window lens [3,4 | 2]
    token_data = np.arange(3 + 4 + 2, dtype=np.uint8)
    seq_offsets = np.array([0, 3, 7, 9], dtype=np.int64)   # per-variant
    var_offsets = np.array([0, 2, 3], dtype=np.int64)      # per group
    shape = (2, 1, None, None)
    w = _FlatWindow(token_data, seq_offsets, var_offsets, shape)
    rag = w.to_ragged()
    # element-identical content after wrapping
    np.testing.assert_array_equal(np.asarray(rag.data).view(np.uint8), token_data)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_flat_flanks.py::test_flat_window_to_ragged_roundtrip -v`
Expected: FAIL with `ImportError: cannot import name '_FlatWindow'`.

- [ ] **Step 3: Implement the types**

Add to `python/genvarloader/_dataset/_flat_variants.py` (model on the existing `_FlatAlleles` `to_ragged`/`reshape`/`squeeze`, but the `data` array carries tokens of arbitrary int dtype rather than `uint8`):

```python
@dataclass(slots=True)
class _FlatWindow:
    """Two-level flat token buffer for ref/alt windows, shape (b, p, ~v, ~win).
    Mirrors _FlatAlleles but `data` holds tokens (configured dtype), not bytes."""

    data: NDArray            # tokens (uint8 or int32), flat
    seq_offsets: NDArray[np.int64]   # per-variant window offsets, n_variants + 1
    var_offsets: NDArray[np.int64]   # per (instance, ploid) offsets, b*p + 1
    shape: tuple[int | None, ...]

    def to_ragged(self):
        # build the same two-level awkward layout _FlatAlleles uses, but over
        # `data` tokens instead of bytes. Copy the body of _FlatAlleles.to_ragged
        # and drop the `.view("S1")` byte cast (tokens are already int).
        ...

    def reshape(self, shape) -> "_FlatWindow":
        if isinstance(shape, int):
            shape = (shape,)
        return _FlatWindow(self.data, self.seq_offsets, self.var_offsets,
                           (*shape, None, None))

    def squeeze(self, axis: int | None = None) -> "_FlatWindow":
        # delegate to the same outer-dim squeeze logic as _FlatAlleles
        ...


@dataclass(slots=True)
class _FlatVariantWindows:
    """Window-mode variants output: scalar fields + ref/alt token windows.
    Raw alleles are intentionally absent (folded into the windows)."""

    fields: dict[str, Any]        # start / ilen / dosage / info -> _Flat
    ref_window: _FlatWindow
    alt_window: _FlatWindow

    @property
    def shape(self) -> tuple[int | None, ...]:
        return self.fields["start"].shape

    def to_ragged(self):
        # Return a lightweight container of ragged fields. Reuse RaggedVariants for
        # the scalar fields if it accepts them; ref_window/alt_window become Ragged.
        # (Final container type tracks consumer needs; see Task 8.)
        ...

    def reshape(self, shape) -> "_FlatVariantWindows":
        return _FlatVariantWindows(
            {k: v.reshape(shape) for k, v in self.fields.items()},
            self.ref_window.reshape(shape),
            self.alt_window.reshape(shape),
        )

    def squeeze(self, axis: int | None = None) -> "_FlatVariantWindows":
        return _FlatVariantWindows(
            {k: v.squeeze(axis) for k, v in self.fields.items()},
            self.ref_window.squeeze(axis),
            self.alt_window.squeeze(axis),
        )
```

Fill the `...` bodies by copying the corresponding `_FlatAlleles` method bodies (same file) and removing the byte-specific `.view("S1")` cast in `to_ragged`. Keep `reverse_masked` **off** `_FlatVariantWindows` for now (windows are reference-oriented, not RC'd — see spec §5; Task 6 handles the boundary).

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/dataset/test_flat_flanks.py::test_flat_window_to_ragged_roundtrip -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_flat_variants.py tests/dataset/test_flat_flanks.py
rtk git commit -m "feat(flat): _FlatWindow + _FlatVariantWindows two-level token types"
```

---

### Task 5: alt-window assembly kernel + `compute_windows`

**Files:**
- Modify: `python/genvarloader/_dataset/_flat_flanks.py`
- Test: `tests/dataset/test_flat_flanks.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/dataset/test_flat_flanks.py
from genvarloader._dataset._flat_flanks import compute_windows


def _oracle_windows(reference, v_contigs, starts, ilens, alt_data, alt_seq_off,
                    flank_len, lut):
    ends = starts - np.minimum(ilens, 0) + 1
    # ref_window: single contiguous read [start-L, end+L)
    rw = reference.fetch(v_contigs, starts - flank_len, ends + flank_len)
    ref_tok = lut[rw.data.view(np.uint8)]
    # alt_window: flank5 + alt + flank3
    f5 = reference.fetch(v_contigs, starts - flank_len, starts).data.view(np.uint8)
    f3 = reference.fetch(v_contigs, ends, ends + flank_len).data.view(np.uint8)
    f5 = f5.reshape(len(starts), flank_len)
    f3 = f3.reshape(len(starts), flank_len)
    alt_rows, alt_lens = [], np.diff(alt_seq_off)
    for i in range(len(starts)):
        a = alt_data[alt_seq_off[i]:alt_seq_off[i + 1]]
        alt_rows.append(np.concatenate([f5[i], a, f3[i]]))
    alt_tok = lut[np.concatenate(alt_rows)] if alt_rows else np.empty(0, lut.dtype)
    return ref_tok, np.asarray(rw.offsets), alt_tok


def test_compute_windows_unit(snap_dataset):
    haps = snap_dataset.with_seqs("variants").with_settings(
        flank_length=3, token_alphabet=b"ACGT", unknown_token=4
    )._seqs
    ref, lut = haps.reference, haps.token_lut
    v_contigs = np.array([0, 0], dtype=np.int32)
    starts = np.array([10, 20], dtype=np.int32)
    ilens = np.array([0, -2], dtype=np.int32)
    # alt alleles: "AC" and "T"
    alt_data = np.frombuffer(b"ACT", dtype=np.uint8).copy()
    alt_seq_off = np.array([0, 2, 3], dtype=np.int64)
    row_offsets = np.array([0, 2], dtype=np.int64)
    ref_w, alt_w = compute_windows(
        ref, v_contigs, starts, ilens, alt_data, alt_seq_off, 3, lut, row_offsets
    )
    e_ref_tok, e_ref_off, e_alt_tok = _oracle_windows(
        ref, v_contigs, starts, ilens, alt_data, alt_seq_off, 3, lut
    )
    np.testing.assert_array_equal(ref_w.data, e_ref_tok)
    np.testing.assert_array_equal(ref_w.seq_offsets, e_ref_off)
    np.testing.assert_array_equal(alt_w.data, e_alt_tok)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_flat_flanks.py::test_compute_windows_unit -v`
Expected: FAIL with `ImportError: cannot import name 'compute_windows'`.

- [ ] **Step 3: Implement the assembly kernel + `compute_windows`**

```python
# add to python/genvarloader/_dataset/_flat_flanks.py
import numba as nb
from ._flat_variants import _FlatWindow


@nb.njit(nogil=True, cache=True)  # pragma: no cover - njit
def _assemble_alt_windows(f5, f3, alt_data, alt_seq_off, flank_len):
    """Concatenate flank5 (fixed L) + alt (variable) + flank3 (fixed L) per variant
    into a flat byte buffer. f5/f3 are (n_var, L) row-major flat (n_var*L,)."""
    n = alt_seq_off.shape[0] - 1
    out_off = np.empty(n + 1, np.int64)
    out_off[0] = 0
    for i in range(n):
        alt_len = alt_seq_off[i + 1] - alt_seq_off[i]
        out_off[i + 1] = out_off[i] + 2 * flank_len + alt_len
    out = np.empty(out_off[n], np.uint8)
    for i in range(n):
        dst = out_off[i]
        for k in range(flank_len):
            out[dst] = f5[i * flank_len + k]
            dst += 1
        for k in range(alt_seq_off[i], alt_seq_off[i + 1]):
            out[dst] = alt_data[k]
            dst += 1
        for k in range(flank_len):
            out[dst] = f3[i * flank_len + k]
            dst += 1
    return out, out_off


def compute_windows(
    reference,
    v_contigs, starts, ilens,
    alt_data, alt_seq_off,
    flank_len, lut, row_offsets,
) -> tuple["_FlatWindow", "_FlatWindow"]:
    """ref_window = tokenized [start-L, end+L) (single contiguous read);
    alt_window  = tokenized flank5 . alt . flank3 (assembly)."""
    starts = np.asarray(starts, np.int32)
    ilens = np.asarray(ilens, np.int32)
    ends = starts - np.minimum(ilens, 0) + 1

    rw = reference.fetch(v_contigs, starts - flank_len, ends + flank_len)
    ref_tok = lut[rw.data.view(np.uint8)]
    ref_window = _FlatWindow(ref_tok, np.asarray(rw.offsets, np.int64),
                             np.asarray(row_offsets, np.int64), (None,))  # shape set by caller

    f5 = reference.fetch(v_contigs, starts - flank_len, starts).data.view(np.uint8)
    f3 = reference.fetch(v_contigs, ends, ends + flank_len).data.view(np.uint8)
    alt_bytes, alt_off = _assemble_alt_windows(
        np.ascontiguousarray(f5), np.ascontiguousarray(f3),
        np.asarray(alt_data, np.uint8), np.asarray(alt_seq_off, np.int64), flank_len,
    )
    alt_tok = lut[alt_bytes]
    alt_window = _FlatWindow(alt_tok, alt_off,
                             np.asarray(row_offsets, np.int64), (None,))
    return ref_window, alt_window
```

> The caller (Task 6) sets the real `shape` `(b, p, None, None)` on each `_FlatWindow` (the placeholder `(None,)` is overwritten).

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/dataset/test_flat_flanks.py::test_compute_windows_unit -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_flat_flanks.py tests/dataset/test_flat_flanks.py
rtk git commit -m "feat(flat): ref/alt window assembly + tokenization"
```

---

## Phase D — wire into `get_variants_flat`

### Task 6: extend `get_variants_flat` to attach flanks / emit windows

**Files:**
- Modify: `python/genvarloader/_dataset/_flat_variants.py:223` (`get_variants_flat`)
- Modify: `python/genvarloader/_dataset/_haps.py:514-522` (pass `regions`, route windows kind)
- Test: `tests/dataset/test_flat_flanks.py`

- [ ] **Step 1: Write the failing test (end-to-end ride-along)**

```python
# append to tests/dataset/test_flat_flanks.py
def test_flank_tokens_end_to_end_matches_oracle(snap_dataset):
    ds = (
        snap_dataset.with_seqs("variants")
        .with_output_format("flat")
        .with_settings(flank_length=5, token_alphabet=b"ACGT", unknown_token=4)
    )
    flat = ds[[0, 1, 2], [0, 1, 2]]
    assert flat.flank_tokens is not None
    # oracle: build from the SAME RaggedVariants the ragged path returns
    rag = (
        snap_dataset.with_seqs("variants")[[0, 1, 2], [0, 1, 2]]
    )
    # flank tokens are (b, p, ~v, 2L); compare against an independent fetch+tokenize
    # over rag.start / rag.ilen. (Helper mirrors genvarformer _read_flank_seq.)
    expected = _oracle_from_ragged(snap_dataset, rag, flank_len=5)
    np.testing.assert_array_equal(
        np.asarray(flat.flank_tokens.to_ragged().data).view(flat.flank_tokens.data.dtype),
        expected,
    )
```

Add the `_oracle_from_ragged` helper near the top of the test file:

```python
def _oracle_from_ragged(dataset, rag, flank_len):
    import seqpro as sp
    ref = dataset._seqs.reference
    lut, _ = build_token_lut(b"ACGT", 4)
    # per-variant contig: repeat region contig by ploidy then by variant counts
    # (mirror genvarformer tokens.py:542-547). Build from rag offsets.
    starts = np.asarray(rag.start.data)
    ilens = np.asarray(rag.ilen.data)
    contigs = _per_variant_contigs(dataset, rag)  # see helper below
    ends = starts - np.minimum(ilens, 0) + 1
    f5 = ref.fetch(contigs, starts - flank_len, starts).data.view(np.uint8).reshape(-1, flank_len)
    f3 = ref.fetch(contigs, ends, ends + flank_len).data.view(np.uint8).reshape(-1, flank_len)
    return lut[np.concatenate([f5, f3], axis=1)].reshape(-1)
```

> `_per_variant_contigs` repeats `regions[:,0]` by ploidy then by `rag.start` per-variant counts; copy the index math from `get_variants_flat`'s offset handling.

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_flat_flanks.py::test_flank_tokens_end_to_end_matches_oracle -v`
Expected: FAIL with `AttributeError: '_FlatVariants' object has no attribute 'flank_tokens'`.

- [ ] **Step 3a: Add `flank_tokens` attr to `_FlatVariants`**

In `python/genvarloader/_dataset/_flat_variants.py`, add to the `_FlatVariants` dataclass:

```python
    flank_tokens: Any = None  # _Flat | None  (ride-along, shape (b, p, ~v, 2L))
```

In `_FlatVariants.to_ragged`, after building `kw`, if `self.flank_tokens is not None` attach it (requires a `flank_tokens` optional field on `RaggedVariants` — add `flank_tokens: Ragged | None = None` to that dataclass in `_rag_variants.py`, defaulting None so existing call sites are unaffected):

```python
        rv = RaggedVariants(**kw)
        if self.flank_tokens is not None:
            rv.flank_tokens = self.flank_tokens.to_ragged()
        return rv
```

Also pass flank_tokens through `reshape`/`squeeze` (delegate like the dict fields).

- [ ] **Step 3b: Extend `get_variants_flat` signature + body**

Change the signature to accept `regions` and compute flanks/windows when configured:

```python
def get_variants_flat(haps: "Haps", idx, regions=None):
    ...
    # (existing body builds fields, v_idxs, row_offsets, alt_data/alt_seq_off)
    flat = _FlatVariants(fields)

    if haps.flank_length and haps.token_lut is not None and regions is not None:
        from ._flat_flanks import compute_flank_tokens, compute_windows
        from .._flat import _Flat

        L = haps.flank_length
        ploidy = genotypes.shape[-2]
        starts_v = np.asarray(haps.variants.start)[v_idxs]
        ilens_v = np.asarray(haps.variants.ilen)[v_idxs]
        group_contigs = np.repeat(regions[:, 0], ploidy)         # (b*p,)
        v_contigs = np.repeat(group_contigs, np.diff(row_offsets))  # (n_var,)

        if issubclass(haps.kind, _FlatVariantWindows):
            ref_w, alt_w = compute_windows(
                haps.reference, v_contigs, starts_v, ilens_v,
                alt_data, alt_seq_off, L, haps.token_lut, row_offsets,
            )
            wshape = (b, ploidy, None, None)
            ref_w.shape = wshape
            alt_w.shape = wshape
            # window scalar fields: keep start/ilen/dosage/info, drop alt/ref
            wfields = {k: v for k, v in fields.items() if k not in ("alt", "ref")}
            return _FlatVariantWindows(wfields, ref_w, alt_w)

        tok, off = compute_flank_tokens(
            haps.reference, v_contigs, starts_v, ilens_v, L, haps.token_lut, row_offsets,
        )
        flat.flank_tokens = _Flat.from_offsets(tok, (b, ploidy, None, 2 * L), off)

    return flat
```

> `alt_data` / `alt_seq_off` are the locals already produced by the existing `_gather_alleles(v_idxs, alt_bytes, alt_off)` call — hoist them into named locals if A inlined them.

- [ ] **Step 3c: Pass `regions` from `Haps.__call__`**

In `python/genvarloader/_dataset/_haps.py:519-522`, change the flat branch:

```python
            if flat:
                from ._flat_variants import get_variants_flat

                return cast(_H, get_variants_flat(self, idx, regions))
```

(Make the windows kind reach this branch: see Task 7's `issubclass(self.kind, (RaggedVariants, _FlatVariantWindows))` guard at line 514.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/dataset/test_flat_flanks.py::test_flank_tokens_end_to_end_matches_oracle -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_flat_variants.py python/genvarloader/_dataset/_haps.py python/genvarloader/_dataset/_rag_variants.py tests/dataset/test_flat_flanks.py
rtk git commit -m "feat(flat): attach flank_tokens / emit windows from get_variants_flat"
```

---

## Phase E — `with_seqs("variant-windows")` kind + boundary + validation

### Task 7: register the kind, route dispatch, validate

**Files:**
- Modify: `python/genvarloader/_dataset/_reconstruct.py:279` (kind_map)
- Modify: `python/genvarloader/_dataset/_haps.py:514` (dispatch guard) + `to_kind` (line 495)
- Modify: `python/genvarloader/_dataset/_impl.py:485` (`with_seqs` Literal) + `_check_valid_state`
- Modify: `python/genvarloader/_dataset/_query.py` (flat boundary handles `_FlatVariantWindows`)
- Test: `tests/dataset/test_flat_flanks.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/dataset/test_flat_flanks.py
def test_variant_windows_kind_end_to_end(snap_dataset):
    ds = (
        snap_dataset.with_output_format("flat")
        .with_settings(flank_length=4, token_alphabet=b"ACGT", unknown_token=4)
        .with_seqs("variant-windows")
    )
    out = ds[[0, 1], [0, 1]]
    assert out.ref_window is not None and out.alt_window is not None
    assert "alt" not in out.fields and "ref" not in out.fields


def test_variant_windows_requires_flank_settings(snap_dataset):
    import pytest
    with pytest.raises(ValueError, match="flank"):
        snap_dataset.with_seqs("variant-windows")  # no flank_length set
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_flat_flanks.py::test_variant_windows_kind_end_to_end -v`
Expected: FAIL — `with_seqs` rejects `"variant-windows"` (assert_never) or returns wrong type.

- [ ] **Step 3a: Register kind marker**

In `python/genvarloader/_dataset/_reconstruct.py`, import `_FlatVariantWindows` and add to the kind_map (line ~279):

```python
            "variants": RaggedVariants,
            "variant-windows": _FlatVariantWindows,
```

- [ ] **Step 3b: Dispatch + `to_kind`**

In `python/genvarloader/_dataset/_haps.py`:
- line 514: `if issubclass(self.kind, (RaggedVariants, _FlatVariantWindows)):`
- `to_kind` (line 496): treat `_FlatVariantWindows` like `RaggedVariants` for the "needs reference" check — windows **do** need a reference, so require it:

```python
    def to_kind(self, kind):
        needs_ref = kind not in (RaggedVariants,)  # windows + seqs need reference
        if needs_ref and self.reference is None:
            raise ValueError(f"Cannot return {kind.__name__}: no reference genome.")
        return cast(Haps[_NewH], replace(self, kind=kind))
```

- [ ] **Step 3c: `with_seqs` Literal + validation**

In `python/genvarloader/_dataset/_impl.py:485`, add `"variant-windows"` to the `kind` Literal and the `elif kind in ("haplotypes", "annotated", "variants")` branch (windows also need `Haps`). In `_check_valid_state` (line 361), add:

```python
        if self.sequence_type == "variant-windows":
            haps = self._seqs
            if not isinstance(haps, Haps) or not haps.flank_length or haps.token_lut is None:
                raise ValueError(
                    "with_seqs('variant-windows') requires flank_length, token_alphabet,"
                    " and unknown_token via with_settings(...)."
                )
```

> `sequence_type` is the public accessor for `_seqs_kind`; confirm its name in `_impl.py`.

- [ ] **Step 3d: Flat boundary passthrough**

In `python/genvarloader/_dataset/_query.py`, add `_FlatVariantWindows` to the `isinstance(o, (_Flat, _FlatAnnotatedHaps, _FlatVariants))` unwrap check (line ~139) and to the `_unwrap` `to_ragged` dispatch (lines ~361-370). For rc routing (`_rc_flat` at line ~350): windows are **not** reverse-complemented (spec §5) — skip rc for `_FlatVariantWindows` (route only `_FlatVariants` alt/ref through `reverse_masked`, leave windows untouched). Add a regression note in the test.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/dataset/test_flat_flanks.py::test_variant_windows_kind_end_to_end tests/dataset/test_flat_flanks.py::test_variant_windows_requires_flank_settings -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_reconstruct.py python/genvarloader/_dataset/_haps.py python/genvarloader/_dataset/_impl.py python/genvarloader/_dataset/_query.py tests/dataset/test_flat_flanks.py
rtk git commit -m "feat(flat): variant-windows kind, dispatch, validation, boundary"
```

---

## Phase F — public exports + acceptance tests

### Task 8: public type exports

**Files:**
- Modify: `python/genvarloader/__init__.py`
- Modify: `python/genvarloader/_dataset/_flat_variants.py` (public aliases)
- Test: `tests/dataset/test_flat_flanks.py`

- [ ] **Step 1: Write the failing test**

```python
def test_public_exports():
    import genvarloader as gvl
    assert hasattr(gvl, "FlatVariantWindows")
    assert hasattr(gvl, "FlatVariants")  # from A
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_flat_flanks.py::test_public_exports -v`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Add public aliases + `__all__`**

In `python/genvarloader/_dataset/_flat_variants.py`:

```python
FlatVariantWindows = _FlatVariantWindows  # public alias (keep underscored alias working)
```

In `python/genvarloader/__init__.py`, import and add `"FlatVariantWindows"` to `__all__` (and confirm `FlatVariants`/`FlatAlleles`/`FlatRagged` from A are present).

- [ ] **Step 4: Run to verify it passes**

Run: `pixi run -e dev pytest tests/dataset/test_flat_flanks.py::test_public_exports -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/__init__.py python/genvarloader/_dataset/_flat_variants.py tests/dataset/test_flat_flanks.py
rtk git commit -m "feat(flat): export FlatVariantWindows"
```

---

### Task 9: acceptance — index matrix + OOB + awkward-absence

**Files:**
- Test: `tests/dataset/test_flat_flanks.py`

- [ ] **Step 1: Write the acceptance tests**

```python
import pytest


@pytest.mark.parametrize("idx", [
    (0, 0),                      # scalar
    ([0, 1, 2], [0, 1, 2]),      # list / 2-D (region, sample)
    ([0, 0], [0, 1]),            # same region, two samples (dedup-relevant)
])
def test_flank_tokens_index_matrix(snap_dataset, idx):
    ds = (snap_dataset.with_seqs("variants").with_output_format("flat")
          .with_settings(flank_length=5, token_alphabet=b"ACGT", unknown_token=4))
    flat = ds[idx]
    rag = snap_dataset.with_seqs("variants")[idx]
    expected = _oracle_from_ragged(snap_dataset, rag, flank_len=5)
    got = np.asarray(flat.flank_tokens.to_ragged().data).reshape(-1)
    np.testing.assert_array_equal(got, expected)


def test_oob_flank_at_contig_start(snap_dataset):
    # region whose variant sits within flank_len of position 0 -> N -> unknown_token
    ds = (snap_dataset.with_seqs("variants").with_output_format("flat")
          .with_settings(flank_length=50, token_alphabet=b"ACGT", unknown_token=4))
    flat = ds[(0, 0)]
    toks = np.asarray(flat.flank_tokens.to_ragged().data)
    assert (toks == 4).any()  # some padded positions tokenized to unknown


def test_no_awkward_on_flank_hot_path(snap_dataset, monkeypatch):
    import awkward as ak
    calls = {"n": 0}
    orig = ak.highlevel.Array.__getitem__
    def spy(self, *a, **k):
        calls["n"] += 1
        return orig(self, *a, **k)
    monkeypatch.setattr(ak.highlevel.Array, "__getitem__", spy)
    ds = (snap_dataset.with_seqs("variants").with_output_format("flat")
          .with_settings(flank_length=5, token_alphabet=b"ACGT", unknown_token=4))
    calls["n"] = 0
    _ = ds[[0, 1, 2], [0, 1, 2]]
    assert calls["n"] == 0, "awkward __getitem__ called on flat flank hot path"
```

- [ ] **Step 2: Run them**

Run: `pixi run -e dev pytest tests/dataset/test_flat_flanks.py -k "index_matrix or oob or awkward" -v`
Expected: all PASS. If `test_no_awkward_on_flank_hot_path` fails, trace the awkward call into A's decode and replace with the flat path.

- [ ] **Step 3: Run the full suite (no regression)**

Run: `pixi run -e dev test`
Expected: PASS (existing ragged + `"variants"` paths unchanged).

- [ ] **Step 4: Commit**

```bash
rtk git add tests/dataset/test_flat_flanks.py
rtk git commit -m "test(flat): flank/window acceptance — index matrix, OOB, awkward-absence"
```

---

## Phase G — docs

### Task 10: update the genvarloader skill

**Files:**
- Modify: `skills/genvarloader/SKILL.md`

- [ ] **Step 1: Edit the skill**

Document, in the relevant sections (and re-check the "Common gotchas" / "Where to look next" table per CLAUDE.md):
- `with_settings(flank_length=, token_alphabet=, unknown_token=)` — seqpro-style LUT; `flank_length=0`/`None` disables; needs genotypes.
- `with_seqs("variant-windows")` — new kind → `FlatVariantWindows` (`ref_window`/`alt_window` token buffers; raw alleles dropped); requires flank settings + a reference.
- `FlatVariants.flank_tokens` — ride-along `[flank5|flank3]` tokens, shape `(b,p,~v,2L)`.
- Token dtype `uint8` (default) / `int32`; offsets int64; consumers cast for torch; windows/flanks are reference-oriented (not RC'd).

- [ ] **Step 2: Verify no broken references**

Run: `grep -n "with_output_format\|variant-windows\|flank_tokens\|FlatVariantWindows" skills/genvarloader/SKILL.md`
Expected: the new terms appear; surrounding pointers are accurate against this branch.

- [ ] **Step 3: Commit**

```bash
rtk git add skills/genvarloader/SKILL.md
rtk git commit -m "docs(skill): document flank settings, variant-windows kind, new types"
```

---

## Phase H — dedup optimization (benchmarked, output-invariant) — OPTIONAL

> Land only if the benchmark shows it beats the cache-cheap reference reads + tokenization. Output is byte-identical to Phase B–D, so all Phase F acceptance tests must still pass unchanged.

### Task 11: micro-benchmark the dedup decision

**Files:**
- Create: `benchmarks/bench_flank_dedup.py`

- [ ] **Step 1: Write the benchmark**

Build a super-batch where many `(region, sample)` pairs share regions (and thus variants). Time three paths over the gathered `v_idxs`: (a) per-occurrence read+tokenize (Phase B baseline), (b) `np.unique(v_idxs, return_inverse=True)` then read+tokenize unique + scatter, (c) the sorted-run linear-scan dedup (Task 12). Print wall-time per path for `flank_length` ∈ {5, 50, 250} and for both `flanks` and `windows` modes.

- [ ] **Step 2: Run and record**

Run: `pixi run -e dev python benchmarks/bench_flank_dedup.py`
Expected: a table. **Decision gate:** only proceed to Task 12 if (c) beats (a) for the target `flank_length`/mode; otherwise stop here and leave Phase B–D as the implementation. Record the numbers in the commit message.

- [ ] **Step 3: Commit**

```bash
rtk git add benchmarks/bench_flank_dedup.py
rtk git commit -m "bench(flat): flank dedup decision micro-benchmark"
```

### Task 12: sorted-run dedup (only if Task 11 says it wins)

**Files:**
- Modify: `python/genvarloader/_dataset/_flat_flanks.py`
- Test: `tests/dataset/test_flat_flanks.py`

- [ ] **Step 1: Write the dedup-invariance test**

```python
def test_dedup_invariant(snap_dataset, monkeypatch):
    # output identical with dedup forced on vs off
    base = (snap_dataset.with_seqs("variants").with_output_format("flat")
            .with_settings(flank_length=5, token_alphabet=b"ACGT", unknown_token=4))
    monkeypatch.setenv("GVL_FLANK_DEDUP", "0")
    a = base[[0, 0, 1], [0, 1, 1]].flank_tokens.to_ragged()
    monkeypatch.setenv("GVL_FLANK_DEDUP", "1")
    b = base[[0, 0, 1], [0, 1, 1]].flank_tokens.to_ragged()
    np.testing.assert_array_equal(np.asarray(a.data), np.asarray(b.data))
```

- [ ] **Step 2: Run to verify it fails** (dedup path not implemented).

Run: `pixi run -e dev pytest tests/dataset/test_flat_flanks.py::test_dedup_invariant -v`
Expected: FAIL.

- [ ] **Step 3: Implement sorted-run dedup**

Add a numba kernel that, exploiting per-`(instance,ploid)` sorted `v_idxs` runs (svar order), produces `unique_v_idxs` + an inverse index `occ -> unique_slot` via a k-way merge / linear scan (no global sort). Gate it behind the `GVL_FLANK_DEDUP` flag (default on once it wins). Read+tokenize unique variants once, scatter to occurrences. Keep `compute_flank_tokens`/`compute_windows` output identical.

- [ ] **Step 4: Run to verify it passes** + re-run all Phase F acceptance tests.

Run: `pixi run -e dev pytest tests/dataset/test_flat_flanks.py -v`
Expected: PASS (dedup-invariance + all prior tests).

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_flat_flanks.py tests/dataset/test_flat_flanks.py
rtk git commit -m "perf(flat): sorted-run dedup of sample-invariant flank reads"
```

---

## Phase I — genvarformer consumer thinning (separate repo, validation)

> Lands in `/root/genvarformer`, not gvl. Proves the win and exercises the API. Validate against genvarformer's existing batch-equality guardrail.

### Task 13: thin the genvarformer flank/window path

**Files (genvarformer):**
- Modify: `src/genvarformer/data/sources/tokens.py` (`_read_flank_seq` + `_fetch`)
- Modify: `src/genvarformer/data/sources/_helpers.py` (`_tokenizer`, `rag_to_nested`)

- [ ] **Step 1: Ride-along path** — configure the gvl dataset with
  `with_output_format("flat").with_settings(flank_length=L, token_alphabet=sp.DNA.alphabet, unknown_token=len(sp.DNA))`; consume `FlatVariants.flank_tokens` directly. Delete `_read_flank_seq` (tokens.py:53) and the `v_flank = _tokenizer(v_flank)` call (tokens.py:549).
- [ ] **Step 2: Window path** — for the varenc model, switch to `with_seqs("variant-windows")` and consume `ref_window`/`alt_window`; delete the downstream flank+allele concatenation.
- [ ] **Step 3: Run the batch-equality guardrail** — confirm byte-identical batches before/after. Record pass.
- [ ] **Step 4: Commit (in genvarformer)** with the gvl version bump it depends on.

---

## Self-Review

**Spec coverage:**
- §2.1 settings → Task 2 ✓; §2.2 `"variant-windows"` kind → Task 7 ✓; ride-along `flank_tokens` → Tasks 3/6 ✓.
- §2.3 types (`FlatVariantWindows`, two-level windows, `flank_tokens`) → Tasks 4/6/8 ✓.
- §2.4 skill → Task 10 ✓.
- §3 decode flow (coords, reads, tokenize, emit) → Tasks 3/5/6 ✓.
- §4 dedup (benchmarked, optional, sorted-run) → Tasks 11/12 ✓.
- §5 edge cases: OOB→N → Task 9 ✓; empty regions → covered by zero-variant rows (offsets handle it; add an explicit empty-region case to Task 9 if `snap_dataset` lacks one); rc_neg not RC'd → Task 7 step 3d ✓; validation → Task 7 ✓.
- §6 acceptance (byte-identity, dedup-invariance, awkward-absence) → Tasks 9/12 ✓.
- §7 consumer thinning → Task 13 ✓.

**Gap fixed inline:** §5 "empty regions emit empty windows" — add an explicit empty-region index case to `test_flank_tokens_index_matrix` (a region with no variants for the chosen sample) asserting a zero-length ragged row.

**Placeholder scan:** the `...` bodies in Task 4 are explicit "copy `_FlatAlleles.<method>` body, drop the `.view('S1')` cast" instructions, not vague TODOs. No "add error handling"-style placeholders.

**Type consistency:** `compute_flank_tokens(reference, v_contigs, starts, ilens, flank_len, lut, row_offsets)` and `compute_windows(reference, v_contigs, starts, ilens, alt_data, alt_seq_off, flank_len, lut, row_offsets)` are used identically in Tasks 3/5/6. `_FlatWindow(data, seq_offsets, var_offsets, shape)` consistent across Tasks 4/5/6. `_FlatVariantWindows(fields, ref_window, alt_window)` consistent across Tasks 4/6/7. `build_token_lut(alphabet, unknown_token) -> (lut, dtype)` consistent across Tasks 1/2.

---

## Execution Handoff

After this plan is approved, execute task-by-task. **Before Task 1, rebase onto sub-project A** (see the dependency note at the top).
