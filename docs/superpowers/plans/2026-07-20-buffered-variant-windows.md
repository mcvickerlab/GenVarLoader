# Buffered & Double-Buffered Dataloader Support for Variant-Window Output — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Per this repo's conventions, dispatch implementation tasks to **Sonnet or weaker** subagents (escalate to a stronger model only for a second-pass fix where the implementer failed), and use superpowers:dispatching-parallel-agents for the tasks marked **[parallel-ok]**.

**Goal:** Lift the `variant-windows` and `variants`-with-flank-tokens rejection in `gvl.Dataset`'s `mode="buffered"` and `mode="double_buffered"` torch transports, keeping byte-identical parity with the per-item (`mode=None`) path.

**Architecture:** Two independently-shippable PRs. **PR 1 (buffered)** is in-process only — it needs byte accounting for the new modes, instance-axis slicing of the flat window/flank types, and the guard removal. **PR 2 (double_buffered)** adds the cross-process serialization — a new shared-memory `kind=4` for `_FlatVariantWindows` (reusing `kind=2`'s FieldDescriptor format), a `flank_tokens` extension to `kind=2`, and producer-schema replay of the `VarWindowOpt`.

**Tech Stack:** Python (numpy, `struct`, `multiprocessing.shared_memory`), pytest, PyTorch `IterableDataset`. No Rust changes.

## Global Constraints

- **Target branch:** `main`. Work in the worktree `.claude/worktrees/buffered-vw` (branch `feat/buffered-variant-windows`).
- **Parity oracle:** byte-identical vs the `mode=None` per-item path (`ds[r, s]`) under the matching `with_seqs(...)` config. `variant-windows` has **no ragged form**, so its oracle is per-item flat output, never a flat-vs-ragged comparison.
- **Two output configs, both transports:** Config A = `with_seqs("variant-windows", VarWindowOpt(...))`; Config B = `with_seqs("variants")` + `with_settings(flank_length=...)` under `with_output_format("flat")`.
- **`ref`/`alt` are independent** (`VarWindowOpt.ref`/`.alt ∈ {"window","allele"}`); any subset of the 4 window slots (`ref_window`, `alt_window`, `ref`, `alt`) may be present. Every new path must handle arbitrary present-subsets.
- **No `src/` change** is expected. If one becomes necessary, run `pixi run -e dev maturin develop --release` before pytest (CLAUDE.md).
- **Run tests with** `pixi run -e dev pytest <path> -q`. Shared code (`_chunked.py`, `_impl.py`) touches many paths — before pushing, run the full unit tree: `pixi run -e dev pytest tests/unit -q`.
- **Lint/format/type before each commit:** `pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/ && pixi run -e dev typecheck`. Google-style docstrings on any new/edited docstring in `python/genvarloader/`.
- **prek hooks:** ensure installed before committing (`prek install`), since `.pre-commit-config.yaml` is present.

---

# PHASE 1 — PR 1: buffered (in-process)

Deliverable: `ds.to_dataloader(mode="buffered")` works for Config A and Config B, byte-identical to `mode=None`. Depends on nothing but `main`.

---

### Task 1.1: Byte accounting for variant-windows + flank_tokens  **[parallel-ok with 1.2]**

**Files:**
- Modify: `python/genvarloader/_dataset/_impl.py` — `_output_bytes_per_instance` (the `seq_kind` switch, ~lines 1370-1451)
- Test: `tests/unit/test_output_bytes_variant_windows.py` (create)

**Interfaces:**
- Consumes: `self._seqs` (a `Haps` with `.window_opt`, `.token_lut`, `.var_fields`, `._allele_bytes_sum`), `self.n_variants(regions, samples)`.
- Produces: `_output_bytes_per_instance(...)` returns a finite positive `int64` array for `sequence_type in {"variant-windows"}` and includes a `flank_tokens` term for `"variants"`.

**Background (read before writing):** the existing `"variants"` branch (`_impl.py:1406-1447`) is the template. It sums per-field bytes over `n_vars_total` and, for `alt`/`ref`, uses `haps_obj._allele_bytes_sum(ds_idx, f)`. The `else: raise AssertionError(f"unknown sequence_type {seq_kind!r}")` at ~1451 is what `variant-windows` currently hits. **Over-estimation is safe** (smaller chunks); **under-estimation is a bug** (double-buffered slot overflow, validated in Task 2.6). Compute exactly where the primitives allow.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_output_bytes_variant_windows.py
"""Byte-accounting must handle variant-windows and variants+flank_tokens."""
import numpy as np
import pytest
import genvarloader as gvl


def _vw_ds(ref="window", alt="window"):
    return (
        gvl.get_dummy_dataset()
        .with_tracks(False)
        .with_output_format("flat")
        .with_seqs(
            "variant-windows",
            gvl.VarWindowOpt(
                flank_length=3, token_alphabet=b"ACGT", unknown_token=4, ref=ref, alt=alt
            ),
        )
    )


@pytest.mark.parametrize("ref,alt", [("window", "window"), ("window", "allele"), ("allele", "allele")])
def test_variant_windows_bytes_positive(ref, alt):
    ds = _vw_ds(ref, alt)
    bpi = ds._output_bytes_per_instance(None, None)
    assert bpi.shape == ds.shape and bpi.dtype == np.int64
    assert (bpi >= 0).all() and bpi.sum() > 0
    # include_offsets must be >= payload-only (adds offset overhead).
    bpi_off = ds._output_bytes_per_instance(None, None, include_offsets=True)
    assert (bpi_off >= bpi).all() and bpi_off.sum() > bpi.sum()


def test_flank_tokens_adds_bytes():
    base = (
        gvl.get_dummy_dataset().with_seqs("variants").with_tracks(False).with_output_format("flat")
    )
    with_flank = base.with_settings(flank_length=3, token_alphabet=b"ACGT", unknown_token=0)
    b0 = base._output_bytes_per_instance(None, None).sum()
    b1 = with_flank._output_bytes_per_instance(None, None).sum()
    assert b1 > b0  # flank tokens are extra payload
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/unit/test_output_bytes_variant_windows.py -q`
Expected: FAIL — `AssertionError: unknown sequence_type 'variant-windows'` (windows) and `b1 == b0` (flank term missing).

- [ ] **Step 3: Add the variant-windows branch and the flank_tokens term**

In `_output_bytes_per_instance`, add a new `elif` **before** the final `else: raise AssertionError`:

```python
        elif seq_kind == "variant-windows":
            if not isinstance(self._seqs, Haps):
                raise AssertionError("variant-windows mode requires Haps")
            haps_obj = self._seqs
            opt = haps_obj.window_opt
            assert opt is not None, "variant-windows requires a VarWindowOpt"
            L = int(opt.flank_length)
            tok_itemsize = np.dtype(haps_obj.token_lut.dtype).itemsize
            n_vars = self.n_variants(regions, samples)  # (n_inst, ploidy)
            n_vars_flat = n_vars.reshape(-1, n_vars.shape[-1]).astype(np.int64)
            n_vars_total = n_vars_flat.sum(-1)
            ploidy = n_vars.shape[-1]
            # exact allele-byte sums per instance (summed over ploidy), same
            # primitive the "variants" branch uses for alt/ref.
            ref_alleles = haps_obj._allele_bytes_sum(ds_idx, "ref").reshape(-1, ploidy).sum(-1)
            alt_alleles = haps_obj._allele_bytes_sum(ds_idx, "alt").reshape(-1, ploidy).sum(-1)
            # token count per present window slot: window slots add 2L flank
            # tokens per variant; bare allele slots do not.
            ref_tokens = (ref_alleles + n_vars_total * 2 * L) if opt.ref == "window" else ref_alleles
            alt_tokens = (alt_alleles + n_vars_total * 2 * L) if opt.alt == "window" else alt_alleles
            total += (ref_tokens + alt_tokens) * tok_itemsize
            # scalar .fields (start/ilen/dosage/info) — alt/ref are NOT scalar
            # fields here (they became window slots).
            for f in haps_obj.var_fields:
                if f in ("alt", "ref"):
                    continue
                if f == "start":
                    total += n_vars_total * haps_obj.variants.start.dtype.itemsize
                elif f == "ilen":
                    total += n_vars_total * haps_obj.variants.ilen.dtype.itemsize
                elif f == "dosage":
                    if haps_obj.dosages is None:
                        continue
                    total += n_vars_total * haps_obj.dosages.data.dtype.itemsize
                else:
                    total += n_vars_total * haps_obj.variants.info[f].dtype.itemsize
            if include_offsets:
                n_scalar = sum(1 for f in haps_obj.var_fields if f not in ("alt", "ref"))
                n_window_slots = 2  # exactly one ref-slot + one alt-slot
                offset_total += OFF * ploidy * (n_scalar + n_window_slots)
                offset_total += OFF * n_vars_total * n_window_slots  # inner (per-variant) offsets
```

Then extend the `"variants"` branch (~1441-1447) to add a flank_tokens term. Immediately before its `if include_offsets:`:

```python
            # ride-along flank tokens (flat output only): 2L tokens per variant.
            if getattr(haps_obj, "flank_length", 0) and haps_obj.token_lut is not None:
                L = int(haps_obj.flank_length)
                ft_itemsize = np.dtype(haps_obj.token_lut.dtype).itemsize
                total += n_vars_total * 2 * L * ft_itemsize
                if include_offsets:
                    offset_total += OFF * ploidy  # flank_tokens outer offsets
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/unit/test_output_bytes_variant_windows.py -q`
Expected: PASS. (`tok_itemsize`/`_allele_bytes_sum`/`window_opt`/`token_lut` are real attributes; if `token_lut.dtype` raises, read the actual token dtype the builder uses in `_flat_variants.get_variants_flat` and adjust — the slot-fit test in Task 2.6 is the final arbiter of correctness.)

- [ ] **Step 5: Lint + commit**

```bash
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/ && pixi run -e dev typecheck
rtk git add python/genvarloader/_dataset/_impl.py tests/unit/test_output_bytes_variant_windows.py
rtk git commit -m "feat(dataloader): byte accounting for variant-windows and variants flank_tokens"
```

---

### Task 1.2: Instance-axis slicing of window / flank flat types  **[parallel-ok with 1.1]**

**Files:**
- Modify: `python/genvarloader/_dataset/_flat_variants.py` — add `__getitem__` to `_FlatWindow` (~line 190) and `_FlatVariantWindows` (~line 299); replace `_FlatVariants.__getitem__` (~line 425)
- Test: `tests/unit/test_flat_window_slicing.py` (create)

**Interfaces:**
- Produces: `_FlatWindow.__getitem__(slice) -> _FlatWindow`; `_FlatVariantWindows.__getitem__(slice) -> _FlatVariantWindows` and `__len__` semantics via `.shape[0]`; `_FlatVariants.__getitem__(slice)` that carries `flank_tokens` instead of raising.

**Background:** `_FlatWindow` has exactly the two-level offset structure of `_FlatAlleles` (`data`/`seq_offsets`/`var_offsets`/`shape`), so its slice mirrors `_FlatAlleles.__getitem__` (`_flat_variants.py:160-187`) verbatim (rename `byte_data`→`data`). `_FlatVariants.flank_tokens` is a `_Flat` whose leading axis is the instance axis, so it slices via `_Flat.__getitem__` (`_flat.py:80-102`).

- [ ] **Step 1: Write the failing test** (oracle = compare a slice of a real batch against per-item output, so it needs no hand-computed offsets)

```python
# tests/unit/test_flat_window_slicing.py
"""Instance-axis slicing of flat window / flank types matches per-item output."""
import numpy as np
import genvarloader as gvl


def _win_eq(a, b):
    """Compare two _FlatVariantWindows via to_ragged() awkward lists."""
    da, db = a.to_ragged(), b.to_ragged()
    assert set(da) == set(db), f"keys differ: {set(da)} vs {set(db)}"
    for k in da:
        assert da[k].to_ak().to_list() == db[k].to_ak().to_list(), f"{k} mismatch"


def test_flat_variant_windows_slice_matches_per_item():
    ds = (
        gvl.get_dummy_dataset()
        .with_tracks(False)
        .with_output_format("flat")
        .with_seqs(
            "variant-windows",
            gvl.VarWindowOpt(flank_length=2, token_alphabet=b"ACGT", unknown_token=4,
                             ref="window", alt="allele"),
        )
    )
    r = np.array([0, 0, 1], np.intp)
    s = np.array([0, 1, 0], np.intp)
    batch = ds[r, s]                      # one _FlatVariantWindows over 3 instances
    sliced = batch[1:3]                   # instances 1,2
    expected = ds[r[1:3], s[1:3]]
    _win_eq(sliced, expected)


def test_flat_variants_flank_tokens_slice_carries_tokens():
    ds = (
        gvl.get_dummy_dataset()
        .with_seqs("variants")
        .with_tracks(False)
        .with_settings(flank_length=2, token_alphabet=b"ACGT", unknown_token=0)
        .with_output_format("flat")
    )
    r = np.array([0, 0, 1], np.intp)
    s = np.array([0, 1, 0], np.intp)
    batch = ds[r, s]
    assert batch.flank_tokens is not None
    sliced = batch[1:3]                   # must NOT raise, must keep flank_tokens
    assert sliced.flank_tokens is not None
    exp = ds[r[1:3], s[1:3]]
    np.testing.assert_array_equal(
        sliced.flank_tokens.to_ragged().to_ak().to_list(),
        exp.flank_tokens.to_ragged().to_ak().to_list(),
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/unit/test_flat_window_slicing.py -q`
Expected: FAIL — `TypeError: '_FlatVariantWindows' object is not subscriptable` and `NotImplementedError` (flank_tokens slicing).

- [ ] **Step 3a: Add `_FlatWindow.__getitem__`** (mirror `_FlatAlleles.__getitem__`, `data` instead of `byte_data`)

```python
    def __getitem__(self, key) -> "_FlatWindow":
        """Slice the leading (instance) axis, rebasing both offset levels."""
        if not isinstance(key, slice):
            raise TypeError(f"_FlatWindow supports only instance-axis slicing, got {key!r}")
        n_inst = self.shape[0]
        if n_inst is None:
            raise ValueError("_FlatWindow.__getitem__: leading axis is the ragged axis")
        start, stop, step = key.indices(n_inst)
        if step != 1:
            raise ValueError("_FlatWindow slicing supports step=1 only")
        rows_per_inst = (len(self.var_offsets) - 1) // n_inst if n_inst else 0
        r0, r1 = start * rows_per_inst, stop * rows_per_inst
        v0, v1 = int(self.var_offsets[r0]), int(self.var_offsets[r1])
        new_var = np.ascontiguousarray(self.var_offsets[r0 : r1 + 1] - self.var_offsets[r0])
        new_seq = np.ascontiguousarray(self.seq_offsets[v0 : v1 + 1] - self.seq_offsets[v0])
        new_data = self.data[int(self.seq_offsets[v0]) : int(self.seq_offsets[v1])]
        new_shape = (stop - start,) + self.shape[1:]
        return _FlatWindow(new_data, new_seq, new_var, new_shape)
```

- [ ] **Step 3b: Add `_FlatVariantWindows.__getitem__`** (slice scalar fields + each present window slot)

```python
    def __getitem__(self, key) -> "_FlatVariantWindows":
        present = {n: w[key] for n, w in self._present().items()}
        return _FlatVariantWindows({k: v[key] for k, v in self.fields.items()}, **present)
```

- [ ] **Step 3c: Replace `_FlatVariants.__getitem__`** to carry `flank_tokens` (its leading axis is the instance axis, so `_Flat.__getitem__` slices it)

```python
    def __getitem__(self, key) -> "_FlatVariants":
        out = _FlatVariants({k: v[key] for k, v in self.fields.items()})
        if self.flank_tokens is not None:
            out.flank_tokens = self.flank_tokens[key]
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/unit/test_flat_window_slicing.py -q`
Expected: PASS. (If `flank_tokens[key]` raises inside `_Flat.__getitem__`, the flank_tokens memory layout does not put the instance axis first as assumed — inspect `len(batch.flank_tokens.offsets)` vs `b*p*2L+1`; if it is `b*p+1`, slice it with the two-level `_FlatAlleles` pattern from Step 3a instead. The per-item test above is the arbiter.)

- [ ] **Step 5: Lint + commit**

```bash
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/ && pixi run -e dev typecheck
rtk git add python/genvarloader/_dataset/_flat_variants.py tests/unit/test_flat_window_slicing.py
rtk git commit -m "feat(dataloader): instance-axis slicing for flat window and flank types"
```

---

### Task 1.3: Enable the buffered path + end-to-end parity + flip rejections

**Files:**
- Modify: `python/genvarloader/_chunked.py:126` (`_FLAT_TYPES`)
- Modify: `python/genvarloader/_torch.py:164-192` (narrow both guards to `double_buffered`)
- Test: `tests/unit/test_buffered_loader.py` (edit the two rejection tests; add positive parity tests)

**Interfaces:**
- Consumes: Task 1.1 (byte accounting), Task 1.2 (slicing).
- Produces: `ds.to_dataloader(mode="buffered")` yields Config A / Config B batches equal to `mode=None`.

- [ ] **Step 1: Write the failing end-to-end parity test** (append to `tests/unit/test_buffered_loader.py`; reuses `_rv_eq` already in that file)

```python
def _win_eq(a, b):
    da, db = a.to_ragged(), b.to_ragged()
    assert set(da) == set(db), f"keys differ: {set(da)} vs {set(db)}"
    for k in da:
        assert da[k].to_ak().to_list() == db[k].to_ak().to_list(), f"{k} mismatch"


def _iter_mode_none(ds, batch_size):
    """Per-item oracle: same sequential batches mode=None would yield."""
    n = int(np.prod(ds.full_shape))
    n = (n // batch_size) * batch_size  # drop_last=True
    r, s = np.unravel_index(np.arange(n), ds.shape)
    for i in range(0, n, batch_size):
        yield ds[r[i : i + batch_size], s[i : i + batch_size]]


@pytest.mark.parametrize("ref,alt", [("window", "window"), ("window", "allele"), ("allele", "allele")])
def test_buffered_variant_windows_matches_per_item(ref, alt):
    ds = (
        gvl.get_dummy_dataset()
        .with_tracks(False)
        .with_output_format("flat")
        .with_seqs(
            "variant-windows",
            gvl.VarWindowOpt(flank_length=2, token_alphabet=b"ACGT", unknown_token=4, ref=ref, alt=alt),
        )
    )
    bs = 2
    got = list(ds.to_dataloader(mode="buffered", batch_size=bs, shuffle=False,
                                drop_last=True, buffer_bytes=10 * 1024 * 1024))
    exp = list(_iter_mode_none(ds, bs))
    assert len(got) == len(exp)
    for g, e in zip(got, exp):
        _win_eq(g, e)


def test_buffered_variants_flank_tokens_matches_per_item():
    ds = (
        gvl.get_dummy_dataset().with_seqs("variants").with_tracks(False)
        .with_settings(flank_length=2, token_alphabet=b"ACGT", unknown_token=0)
        .with_output_format("flat")
    )
    bs = 2
    got = list(ds.to_dataloader(mode="buffered", batch_size=bs, shuffle=False,
                                drop_last=True, buffer_bytes=10 * 1024 * 1024))
    exp = list(_iter_mode_none(ds, bs))
    assert len(got) == len(exp)
    for g, e in zip(got, exp):
        np.testing.assert_array_equal(
            g.flank_tokens.to_ragged().to_ak().to_list(),
            e.flank_tokens.to_ragged().to_ak().to_list(),
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/unit/test_buffered_loader.py -q -k "variant_windows_matches or flank_tokens_matches"`
Expected: FAIL — `ValueError: mode='buffered' does not support 'variant-windows'` / `...does not support variants output carrying ride-along flank tokens`.

- [ ] **Step 3a: Add `_FlatVariantWindows` to `_FLAT_TYPES`** in `_chunked.py`

```python
    from ._flat import _Flat, _FlatAnnotatedHaps
    from ._dataset._flat_variants import _FlatVariants, _FlatVariantWindows

    _FLAT_TYPES = (_Flat, _FlatAnnotatedHaps, _FlatVariants, _FlatVariantWindows)
```

- [ ] **Step 3b: Narrow both guards** in `_torch.py:get_dataloader` from unconditional to `mode == "double_buffered"` only. Replace the two `if` blocks (`:164-192`):

```python
    # 'variant-windows' and flat variants+flank_tokens cannot yet ride the
    # double_buffered transport (the producer schema / shm format do not carry
    # the VarWindowOpt or the flank tokens). buffered runs in-process and does.
    if mode == "double_buffered" and getattr(dataset, "sequence_type", None) == "variant-windows":
        raise ValueError(
            "mode='double_buffered' does not support 'variant-windows' output yet: the "
            "producer schema/shared-memory format cannot carry the VarWindowOpt. Use "
            "mode='buffered' (in-process) or mode=None."
        )
    if (
        mode == "double_buffered"
        and getattr(dataset, "output_format", "ragged") == "flat"
        and getattr(dataset, "sequence_type", None) == "variants"
    ):
        _seqs = getattr(dataset, "_seqs", None)
        if getattr(_seqs, "flank_length", None) and getattr(_seqs, "token_lut", None) is not None:
            raise ValueError(
                "mode='double_buffered' with output_format='flat' does not support variants "
                "output carrying ride-along flank tokens yet; use mode='buffered' or mode=None."
            )
```

- [ ] **Step 3c: Flip the two buffered rejection tests.** In `tests/unit/test_buffered_loader.py`, change `test_flat_buffered_rejects_variant_windows` and `test_flat_buffered_rejects_variants_flank_tokens` so the `mode="buffered"` parametrization asserts success, and only `mode="double_buffered"` still asserts the `ValueError`. Simplest: change the parametrize to `["double_buffered"]` (buffered no longer rejects) and add a note that PR 2 removes the double_buffered arm. Keep `test_flat_buffered_plain_variants_still_works` unchanged.

```python
@pytest.mark.parametrize("mode", ["double_buffered"])  # buffered now supports both (PR1); PR2 drops these
def test_flat_buffered_rejects_variants_flank_tokens(mode):
    ...  # body unchanged

@pytest.mark.parametrize("mode", ["double_buffered"])
def test_flat_buffered_rejects_variant_windows(mode):
    ...  # body unchanged
```

- [ ] **Step 4: Run the buffered suite**

Run: `pixi run -e dev pytest tests/unit/test_buffered_loader.py -q`
Expected: PASS (all, including the flipped rejections and new parity tests).

- [ ] **Step 5: Lint, full unit tree (shared code touched), commit**

```bash
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/ && pixi run -e dev typecheck
pixi run -e dev pytest tests/unit -q
rtk git add python/genvarloader/_chunked.py python/genvarloader/_torch.py tests/unit/test_buffered_loader.py
rtk git commit -m "feat(dataloader): support variant-windows and flank tokens in mode='buffered'"
```

**PR 1 boundary:** open a draft PR into `main` at this point (buffered is independently shippable). Update docs in Task 1.4 before marking ready.

---

### Task 1.4: PR 1 docs

**Files:**
- Modify: `docs/source/dataset.md` and/or `docs/source/faq.md` (the `mode=`/output-mode compatibility notes), `skills/genvarloader/SKILL.md`

**Interfaces:** none (docs only).

- [ ] **Step 1: Update the buffered-transport compatibility note.** Find the doc section that lists which output modes work with `mode="buffered"`/`"double_buffered"` (grep `double_buffered` under `docs/` and `skills/`). State that `mode="buffered"` now supports `variant-windows` and flat `variants` with flank tokens; `mode="double_buffered"` support lands in PR 2. Verify no now-false claim remains.

- [ ] **Step 2: Verify `api.md` is unaffected**

Run: `pixi run -e dev python -c "import genvarloader as g; api=open('docs/source/api.md').read(); print('MISSING:', [n for n in g.__all__ if n not in api] or 'none')"`
Expected: `MISSING: none` (no new public symbol).

- [ ] **Step 3: Commit**

```bash
rtk git add docs/ skills/
rtk git commit -m "docs(dataloader): buffered transport now supports variant-windows and flank tokens"
```

---

# PHASE 2 — PR 2: double_buffered (cross-process serialization)

Deliverable: `ds.to_dataloader(mode="double_buffered")` works for Config A and Config B, byte-identical to `mode="buffered"` and `mode=None`. Depends on PR 1.

---

### Task 2.1: Producer-schema replay of VarWindowOpt + flank config  **[parallel-ok with 2.2, 2.3]**

**Files:**
- Modify: `python/genvarloader/_double_buffered_loader.py` — `_spawn_producer` schema build (~lines 199-217)
- Modify: `python/genvarloader/_producer.py` — `_apply_schema` (~lines 13-38)
- Test: `tests/unit/test_producer_schema.py` (create)

**Interfaces:**
- Produces: a schema dict carrying `window_opt` (a plain dict of `VarWindowOpt` primitives) and the flank-token settings; `_apply_schema` reconstructs `with_seqs("variant-windows", VarWindowOpt(...))` / `with_settings(flank_length=..., token_alphabet=..., unknown_token=...)`.

**Background:** `VarWindowOpt` (`_flat_variants.py:267-293`) is all primitives (`flank_length:int`, `token_alphabet:bytes`, `unknown_token:int`, `ref`, `alt`), trivially dict-serializable. The child rebuilds the token LUT inside `with_seqs`/`with_settings`, so no LUT array crosses the boundary.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_producer_schema.py
"""_apply_schema reconstructs variant-windows and flank configs in the child."""
import genvarloader as gvl
from genvarloader._producer import _apply_schema


def test_apply_schema_rebuilds_variant_windows():
    dummy = gvl.get_dummy_dataset().with_tracks(False)
    schema = {
        "with_seqs": "variant-windows",
        "output_format": "flat",
        "window_opt": {"flank_length": 3, "token_alphabet": b"ACGT",
                       "unknown_token": 4, "ref": "window", "alt": "allele"},
    }
    ds = _apply_schema(dummy, schema)
    assert ds.sequence_type == "variant-windows"
    assert ds._seqs.window_opt.flank_length == 3
    assert ds._seqs.window_opt.alt == "allele"


def test_apply_schema_rebuilds_flank_tokens():
    dummy = gvl.get_dummy_dataset().with_tracks(False)
    schema = {
        "with_seqs": "variants",
        "output_format": "flat",
        "flank_length": 2, "token_alphabet": b"ACGT", "unknown_token": 0,
    }
    ds = _apply_schema(dummy, schema)
    assert ds._seqs.flank_length == 2 and ds._seqs.token_lut is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/unit/test_producer_schema.py -q`
Expected: FAIL — `with_seqs('variant-windows') requires a VarWindowOpt` (schema replay passes `kind` only).

- [ ] **Step 3a: Emit the config in `_spawn_producer`.** After the existing `if isinstance(seqs, Haps):` block (`_double_buffered_loader.py:208-217`), add:

```python
            window_opt = getattr(seqs, "window_opt", None)
            if window_opt is not None:
                schema["window_opt"] = {
                    "flank_length": window_opt.flank_length,
                    "token_alphabet": window_opt.token_alphabet,
                    "unknown_token": window_opt.unknown_token,
                    "ref": window_opt.ref,
                    "alt": window_opt.alt,
                }
            elif getattr(seqs, "flank_length", None) and getattr(seqs, "token_lut", None) is not None:
                # plain-variants ride-along flank tokens (Config B)
                schema["flank_length"] = seqs.flank_length
                schema["token_alphabet"] = seqs.token_alphabet
                schema["unknown_token"] = seqs.unknown_token
```

(Verify the attribute names `token_alphabet`/`unknown_token` exist on the `Haps` for the flank path; if the flank config lives under different attribute names, read them from `with_settings`'s stored fields. Test `test_apply_schema_rebuilds_flank_tokens` pins the round-trip.)

- [ ] **Step 3b: Reconstruct in `_apply_schema`.** Replace the `with_seqs` replay line and extend the settings block:

```python
    if schema.get("with_seqs", "UNSET") != "UNSET":
        if schema.get("window_opt") is not None:
            from ._dataset._flat_variants import VarWindowOpt
            ds = ds.with_seqs(schema["with_seqs"], VarWindowOpt(**schema["window_opt"]))
        else:
            ds = ds.with_seqs(schema["with_seqs"])
    ...
    # inside the settings_kwargs block, add:
    if schema.get("flank_length") is not None:
        settings_kwargs["flank_length"] = schema["flank_length"]
        settings_kwargs["token_alphabet"] = schema["token_alphabet"]
        settings_kwargs["unknown_token"] = schema["unknown_token"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/unit/test_producer_schema.py -q`
Expected: PASS.

- [ ] **Step 5: Lint + commit**

```bash
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/ && pixi run -e dev typecheck
rtk git add python/genvarloader/_double_buffered_loader.py python/genvarloader/_producer.py tests/unit/test_producer_schema.py
rtk git commit -m "feat(dataloader): replay VarWindowOpt and flank config in the producer schema"
```

---

### Task 2.2: Shared-memory `kind=4` for `_FlatVariantWindows`  **[parallel-ok with 2.1]**

**Files:**
- Modify: `python/genvarloader/_shm_layout.py` — `write_chunk` dispatch, new `_write_flat_variant_windows`, new `_read_flat_variant_windows`, `_pack_descriptor` (+`kind==4`), `_unpack_one_descriptor` (+`kind==4`), `read_chunk` dispatch
- Test: `tests/unit/test_shm_variant_windows.py` (create)

**Interfaces:**
- Consumes: `_FlatVariantWindows` (`.fields` dict of `_Flat`; up to 4 `_FlatWindow` slots), `_FlatWindow` (`data`/`seq_offsets`/`var_offsets`/`shape`).
- Produces: `write_chunk([...fvw...], n) → read_chunk(buf, flat=True)` round-trips a `_FlatVariantWindows` byte-identically.

**Design (reuse the kind=2 FieldDescriptor format):** window slots serialize exactly like `kind=2` **allele** fields (`field_kind=1`: outer=`var_offsets`, inner=`seq_offsets`, data=tokens, `name`=slot name, leaf dtype via `_dtype_to_bytes`). Scalar `.fields` serialize like `kind=2` numeric fields (`field_kind=0`). On read, partition by name: names in `("ref_window","alt_window","ref","alt")` → `_FlatWindow(data, seq_off, var_off, (b, p, None, None))`; else → `_Flat(leaf, var_off, (b, p, None))`. The `_pack_descriptor`/`_unpack_one_descriptor` `kind==4` blocks are **byte-for-byte identical** to the existing `kind==2` blocks (same `<B4s7QB` FieldDescriptor loop) — the only difference is the `kind` byte and the reader's name-based partition.

- [ ] **Step 1: Write the failing round-trip test** (this pins the byte layout; hand-build a tiny `_FlatVariantWindows`)

```python
# tests/unit/test_shm_variant_windows.py
"""kind=4 round-trip for _FlatVariantWindows over the shm layout."""
import numpy as np
from genvarloader._shm_layout import write_chunk, read_chunk, HEADER_RESERVED
from genvarloader._flat import _Flat
from genvarloader._dataset._flat_variants import _FlatWindow, _FlatVariantWindows


def _make_fvw():
    # 2 instances, ploidy 1. start scalar field + ref_window + alt (bare) slots.
    start = _Flat(np.array([10, 20, 30], np.int32), np.array([0, 2, 3], np.int64), (2, 1, None))
    # ref_window: b*p=2 rows; var_offsets len 3; per-variant token runs via seq_offsets.
    rw = _FlatWindow(
        data=np.array([1, 2, 3, 4, 1, 2], np.uint8),
        seq_offsets=np.array([0, 3, 4, 6], np.int64),   # 3 variants
        var_offsets=np.array([0, 2, 3], np.int64),       # 2 rows -> 2,1 variants
        shape=(2, 1, None, None),
    )
    al = _FlatWindow(
        data=np.array([0, 1, 2, 3], np.uint8),
        seq_offsets=np.array([0, 1, 3, 4], np.int64),
        var_offsets=np.array([0, 2, 3], np.int64),
        shape=(2, 1, None, None),
    )
    return _FlatVariantWindows({"start": start}, ref_window=rw, alt=al)


def test_kind4_roundtrip():
    fvw = _make_fvw()
    buf = memoryview(bytearray(HEADER_RESERVED + (1 << 16)))
    write_chunk(buf, [fvw], n_instances=2)
    n, views = read_chunk(buf, copy=True, flat=True)
    assert n == 2 and len(views) == 1
    out = views[0]
    assert isinstance(out, _FlatVariantWindows)
    a, b = fvw.to_ragged(), out.to_ragged()
    assert set(a) == set(b)
    for k in a:
        assert a[k].to_ak().to_list() == b[k].to_ak().to_list(), f"{k} mismatch"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/unit/test_shm_variant_windows.py -q`
Expected: FAIL — `TypeError: write_chunk: unsupported array type <_FlatVariantWindows>`.

- [ ] **Step 3a: Dispatch + writer.** Import `_FlatVariantWindows` in `write_chunk`'s import block and add the dispatch branch (before the `_FlatVariants` branch):

```python
        if isinstance(a, _FlatVariantWindows):
            desc, cursor = _write_flat_variant_windows(buf, a, cursor)
        elif isinstance(a, _FlatVariants):
            ...
```

Add `_write_flat_variant_windows` (mirror `_write_flat_variants` exactly; window slots use the allele path with `data`/`seq_offsets`/`var_offsets`):

```python
def _write_flat_variant_windows(buf: memoryview, fvw, cursor: int) -> tuple[dict, int]:
    """Write a _FlatVariantWindows as a kind=4 block (kind=2 FieldDescriptor reuse)."""
    from ._dataset._flat_variants import _WINDOW_FIELD_NAMES

    field_descs: list[dict] = []

    def _emit_two_level(name, data, seq_off, var_off, regular_size):
        nonlocal cursor
        outer = np.ascontiguousarray(var_off, np.int64)
        inner = np.ascontiguousarray(seq_off, np.int64)
        leaf = np.ascontiguousarray(data)
        cursor = _align(cursor); outer_off = cursor
        np.frombuffer(buf, np.int64, outer.size, outer_off)[...] = outer
        cursor += outer.nbytes
        cursor = _align(cursor); inner_off = cursor
        np.frombuffer(buf, np.int64, inner.size, inner_off)[...] = inner
        cursor += inner.nbytes
        cursor = _align(cursor); data_off = cursor
        np.frombuffer(buf, leaf.dtype, leaf.size, data_off)[...] = leaf.ravel()
        cursor += leaf.nbytes
        field_descs.append({
            "field_kind": 1, "dtype_str": _dtype_to_bytes(leaf.dtype),
            "outer_offsets_offset": outer_off, "outer_offsets_nbytes": outer.nbytes,
            "inner_offsets_offset": inner_off, "inner_offsets_nbytes": inner.nbytes,
            "data_offset": data_off, "data_nbytes": leaf.nbytes,
            "regular_size": regular_size, "name": name.encode("utf-8"),
        })

    # scalar .fields first (numeric, field_kind=0) — mirror _write_flat_variants
    for name, f in fvw.fields.items():
        outer = np.ascontiguousarray(f.offsets, np.int64)
        leaf = np.ascontiguousarray(f.data)
        cursor = _align(cursor); outer_off = cursor
        np.frombuffer(buf, np.int64, outer.size, outer_off)[...] = outer
        cursor += outer.nbytes
        cursor = _align(cursor); data_off = cursor
        np.frombuffer(buf, leaf.dtype, leaf.size, data_off)[...] = leaf.ravel()
        cursor += leaf.nbytes
        field_descs.append({
            "field_kind": 0, "dtype_str": _dtype_to_bytes(leaf.dtype),
            "outer_offsets_offset": outer_off, "outer_offsets_nbytes": outer.nbytes,
            "inner_offsets_offset": 0, "inner_offsets_nbytes": 0,
            "data_offset": data_off, "data_nbytes": leaf.nbytes,
            "regular_size": _flat_ploidy(f.shape), "name": name.encode("utf-8"),
        })

    # present window slots (two-level, field_kind=1)
    for slot in _WINDOW_FIELD_NAMES:
        w = getattr(fvw, slot)
        if w is not None:
            _emit_two_level(slot, w.data, w.seq_offsets, w.var_offsets, _flat_ploidy(w.shape))

    return {
        "kind": 4, "dtype_str": b"\x00" * 4, "shape": [len(field_descs)],
        "data_offset": 0, "data_nbytes": 0, "offsets_offset": 0, "offsets_nbytes": 0,
        "inner_offsets_offset": 0, "inner_offsets_nbytes": 0, "name": b"",
        "_field_descs": field_descs,
    }, cursor
```

- [ ] **Step 3b: `_pack_descriptor` / `_unpack_one_descriptor`.** In `_pack_descriptor`, add `if kind == 4:` with a body **identical** to the existing `if kind == 2:` block (same `struct.pack("<B4s7QB", ...)` field loop). In `_unpack_one_descriptor`, add `if kind == 4:` identical to the `if kind == 2:` unpack block (same `<7Q` reads and cursor advances). Do not share by falling through — an explicit duplicate keeps the format self-documenting.

- [ ] **Step 3c: Reader + dispatch.** Add `_read_flat_variant_windows` (mirror `_read_flat_variants`, partitioning by name):

```python
def _read_flat_variant_windows(buf: memoryview, d: dict, copy: bool = True):
    from ._flat import _Flat
    from ._dataset._flat_variants import _FlatWindow, _FlatVariantWindows, _WINDOW_FIELD_NAMES

    fields: dict = {}
    windows: dict = {}
    for fd in d["_field_descs"]:
        name = fd["name"]
        leaf_dtype = _dtype_from_bytes(fd["dtype_str"])
        rs = fd["regular_size"]
        n_outer = fd["outer_offsets_nbytes"] // 8
        var_off = np.frombuffer(buf, np.int64, n_outer, fd["outer_offsets_offset"])
        leaf = np.frombuffer(buf, leaf_dtype, fd["data_nbytes"] // leaf_dtype.itemsize, fd["data_offset"])
        if copy:
            var_off, leaf = var_off.copy(), leaf.copy()
        n_bp = len(var_off) - 1
        b = n_bp // rs if rs else n_bp
        if name in _WINDOW_FIELD_NAMES:
            n_inner = fd["inner_offsets_nbytes"] // 8
            seq_off = np.frombuffer(buf, np.int64, n_inner, fd["inner_offsets_offset"])
            if copy:
                seq_off = seq_off.copy()
            windows[name] = _FlatWindow(leaf, seq_off, var_off, (b, rs, None, None))
        else:
            fields[name] = _Flat(leaf, var_off, (b, rs, None))
    return _FlatVariantWindows(fields, **windows)
```

Add the `read_chunk` dispatch branch (between `kind == 3` and `else`):

```python
        elif kind == 4:
            views.append(_read_flat_variant_windows(buf, d, copy=copy))
```

- [ ] **Step 4: Run round-trip test to verify it passes**

Run: `pixi run -e dev pytest tests/unit/test_shm_variant_windows.py -q`
Expected: PASS. If it fails on a byte-advance mismatch, the `kind==4` pack/unpack blocks disagree — re-check they mirror `kind==2` exactly.

- [ ] **Step 5: Lint + commit**

```bash
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/ && pixi run -e dev typecheck
rtk git add python/genvarloader/_shm_layout.py tests/unit/test_shm_variant_windows.py
rtk git commit -m "feat(dataloader): shm kind=4 serialization for _FlatVariantWindows"
```

---

### Task 2.3: Shared-memory `kind=2` flank_tokens extension  **[parallel-ok with 2.1, 2.2]**

**Files:**
- Modify: `python/genvarloader/_shm_layout.py` — `_write_flat_variants`, `_write_rag_variants`, `_pack_descriptor` (kind==2), `_unpack_one_descriptor` (kind==2), `_read_flat_variants`
- Test: `tests/unit/test_shm_flank_tokens.py` (create)

**Interfaces:**
- Produces: `write_chunk`/`read_chunk(flat=True)` round-trips a `_FlatVariants` whose `flank_tokens` is a `_Flat` of shape `(b, p, None, 2L)`.

**Design:** append an optional `flank_tokens` payload to the `kind=2` block. `flank_tokens` is a single-ragged `_Flat` but with a trailing fixed `2L` dim, so its full `shape` must be carried (the generic field path forces `(b, p, None)` and would drop `2L`). Encode after the FieldDescriptor list: a `u8 has_flank`; if 1, a shape (`u8 ndim` + `ndim × u64` dims with `None → 2**64-1`), `4s dtype_str`, and `4×u64` (`data_offset`, `data_nbytes`, `offsets_offset`, `offsets_nbytes`). `_write_rag_variants` always writes `has_flank=0` (RaggedVariants has no flank_tokens).

- [ ] **Step 1: Write the failing round-trip test**

```python
# tests/unit/test_shm_flank_tokens.py
"""kind=2 flank_tokens round-trip over the shm layout."""
import numpy as np
from genvarloader._shm_layout import write_chunk, read_chunk, HEADER_RESERVED
from genvarloader._flat import _Flat
from genvarloader._dataset._flat_variants import _FlatVariants


def test_kind2_flank_tokens_roundtrip():
    # 2 instances, ploidy 1, 2L=4. start scalar + flank_tokens (b,p,None,2L).
    start = _Flat(np.array([1, 2, 3], np.int32), np.array([0, 2, 3], np.int64), (2, 1, None))
    ft = _Flat(
        np.arange(3 * 4, dtype=np.uint8),               # 3 variants * 2L(=4) tokens
        np.array([0, 4, 8, 12], np.int64),              # n_rows = b*p*2L? verify vs builder
        (2, 1, None, 4),
    )
    fv = _FlatVariants({"start": start})
    fv.flank_tokens = ft
    buf = memoryview(bytearray(HEADER_RESERVED + (1 << 16)))
    write_chunk(buf, [fv], n_instances=2)
    n, views = read_chunk(buf, copy=True, flat=True)
    out = views[0]
    assert out.flank_tokens is not None
    assert out.flank_tokens.shape == (2, 1, None, 4)
    np.testing.assert_array_equal(np.asarray(out.flank_tokens.data), np.asarray(ft.data))
    np.testing.assert_array_equal(np.asarray(out.flank_tokens.offsets), np.asarray(ft.offsets))
```

(If the real `flank_tokens.offsets` length differs from this synthetic one, adjust the fixture to match a `_FlatVariants` produced by `ds[r,s]` on a dummy variants+flank dataset — the round-trip must equal whatever the constructor produces; the shape/offset preservation is what matters.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/unit/test_shm_flank_tokens.py -q`
Expected: FAIL — `out.flank_tokens is None` (dropped by the current kind=2 writer).

- [ ] **Step 3a: Writers.** In `_write_flat_variants`, after the field loop and before the `return`, serialize flank_tokens if present and attach a `_flank` descriptor; else `_flank=None`:

```python
    flank = None
    if fv.flank_tokens is not None:
        ft = fv.flank_tokens
        data = np.ascontiguousarray(ft.data)
        off = np.ascontiguousarray(ft.offsets, np.int64)
        cursor = _align(cursor); data_off = cursor
        np.frombuffer(buf, data.dtype, data.size, data_off)[...] = data.ravel()
        cursor += data.nbytes
        cursor = _align(cursor); off_off = cursor
        np.frombuffer(buf, np.int64, off.size, off_off)[...] = off
        cursor += off.nbytes
        flank = {
            "shape": list(ft.shape), "dtype_str": _dtype_to_bytes(data.dtype),
            "data_offset": data_off, "data_nbytes": data.nbytes,
            "offsets_offset": off_off, "offsets_nbytes": off.nbytes,
        }
    # add to the returned dict:
    #   "_flank": flank,
```

In `_write_rag_variants`'s returned dict add `"_flank": None`.

- [ ] **Step 3b: pack/unpack.** In `_pack_descriptor`'s `if kind == 2:` block, after the field loop, append:

```python
        flank = d.get("_flank")
        out += struct.pack("<B", 1 if flank else 0)
        if flank:
            out += struct.pack("<B", len(flank["shape"]))
            for dim in flank["shape"]:
                out += struct.pack("<Q", (2**64 - 1) if dim is None else int(dim))
            out += struct.pack("<4s4Q", flank["dtype_str"], flank["data_offset"],
                               flank["data_nbytes"], flank["offsets_offset"], flank["offsets_nbytes"])
```

In `_unpack_one_descriptor`'s `if kind == 2:` block, after the field loop, append the symmetric read (advancing `cursor` identically), storing `d["_flank"]` (mapping `2**64-1 → None` in the shape).

- [ ] **Step 3c: Reader.** In `_read_flat_variants`, before `return _FlatVariants(fields)`:

```python
    fv = _FlatVariants(fields)
    flank = d.get("_flank")
    if flank:
        leaf_dtype = _dtype_from_bytes(flank["dtype_str"])
        data = np.frombuffer(buf, leaf_dtype, flank["data_nbytes"] // leaf_dtype.itemsize, flank["data_offset"])
        off = np.frombuffer(buf, np.int64, flank["offsets_nbytes"] // 8, flank["offsets_offset"])
        if copy:
            data, off = data.copy(), off.copy()
        fv.flank_tokens = _Flat(data, off, tuple(flank["shape"]))
    return fv
```

(`_read_rag_variants` ignores `_flank`; `has_flank` is always 0 for RaggedVariants.)

- [ ] **Step 4: Run round-trip test to verify it passes**

Run: `pixi run -e dev pytest tests/unit/test_shm_flank_tokens.py -q`
Expected: PASS.

- [ ] **Step 5: Lint + commit**

```bash
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/ && pixi run -e dev typecheck
rtk git add python/genvarloader/_shm_layout.py tests/unit/test_shm_flank_tokens.py
rtk git commit -m "feat(dataloader): shm kind=2 carries _FlatVariants.flank_tokens"
```

---

### Task 2.4: Consumer reshape for `_FlatVariantWindows`

**Files:**
- Modify: `python/genvarloader/_double_buffered_loader.py` — `_reshape_ragged_for_chunk` (~lines 44-103)
- Test: `tests/unit/test_double_buffered_loader.py` (add a `_reshape_ragged_for_chunk` unit test mirroring `test_reshape_ragged_for_chunk_leaves_raggedvariants_untouched`)

**Interfaces:**
- Consumes: Task 2.2's `_read_flat_variant_windows` output.
- Produces: `_reshape_ragged_for_chunk([fvw], n_instances)` returns a `_FlatVariantWindows` whose per-slot ploidy axis is correct (the shm reader builds `(b, rs, None, None)` directly from `regular_size`, so this is mostly a passthrough guard that must NOT mangle the type).

- [ ] **Step 1: Write the failing test**

```python
def test_reshape_ragged_for_chunk_passes_variant_windows():
    import numpy as np
    from genvarloader._flat import _Flat
    from genvarloader._dataset._flat_variants import _FlatWindow, _FlatVariantWindows
    from genvarloader._double_buffered_loader import _reshape_ragged_for_chunk

    start = _Flat(np.array([1, 2], np.int32), np.array([0, 1, 2], np.int64), (2, 1, None))
    rw = _FlatWindow(np.arange(4, dtype=np.uint8), np.array([0, 2, 4], np.int64),
                     np.array([0, 1, 2], np.int64), (2, 1, None, None))
    fvw = _FlatVariantWindows({"start": start}, ref_window=rw)
    out = _reshape_ragged_for_chunk([fvw], n_instances=2)[0]
    assert isinstance(out, _FlatVariantWindows)
    assert out.ref_window is not None and out.ref_window.shape[0] == 2
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `pixi run -e dev pytest tests/unit/test_double_buffered_loader.py::test_reshape_ragged_for_chunk_passes_variant_windows -q`
Expected: FAIL if `_reshape_one` mangles `_FlatVariantWindows` (it falls through to the generic branch); PASS only once handled.

- [ ] **Step 3: Add a `_FlatVariantWindows` passthrough** in `_reshape_ragged_for_chunk`'s per-array loop (before the generic branches), since the shm reader already produced correct `(b, rs, None, None)` shapes:

```python
    from ._dataset._flat_variants import _FlatVariantWindows
    ...
    for arr in views:
        if isinstance(arr, _FlatVariantWindows):
            result.append(arr)          # ploidy axis already correct from the reader
            continue
        if isinstance(arr, RaggedAnnotatedHaps):
            ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/unit/test_double_buffered_loader.py::test_reshape_ragged_for_chunk_passes_variant_windows -q`
Expected: PASS.

- [ ] **Step 5: Lint + commit**

```bash
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/ && pixi run -e dev typecheck
rtk git add python/genvarloader/_double_buffered_loader.py tests/unit/test_double_buffered_loader.py
rtk git commit -m "feat(dataloader): pass _FlatVariantWindows through the double-buffered consumer reshape"
```

---

### Task 2.5: Enable double_buffered + end-to-end parity + flip rejections

**Files:**
- Modify: `python/genvarloader/_torch.py` — remove the two `double_buffered` guards added in Task 1.3
- Test: `tests/unit/test_double_buffered_loader.py` (add parity tests; remove the flipped rejection tests entirely)

**Interfaces:**
- Consumes: Tasks 2.1-2.4.
- Produces: `ds.to_dataloader(mode="double_buffered")` yields Config A / Config B batches equal to `mode="buffered"`.

- [ ] **Step 1: Write the failing parity test** (file-backed; mirror `test_double_buffered_iter_matches_buffered`, reuse `_rv_eq`/`_win_eq`)

```python
@pytest.mark.slow
@pytest.mark.parametrize("ref,alt", [("window", "window"), ("window", "allele")])
def test_double_buffered_variant_windows_matches_buffered(file_backed_ds, ref, alt):
    ds = (
        file_backed_ds.with_tracks(False).with_output_format("flat")
        .with_seqs("variant-windows",
                   gvl.VarWindowOpt(flank_length=2, token_alphabet=b"ACGT", unknown_token=4,
                                    ref=ref, alt=alt))
    )
    common = dict(batch_size=2, shuffle=False, drop_last=True, buffer_bytes=4 * 1024 * 1024)
    buf = list(ds.to_dataloader(mode="buffered", **common))
    db = list(ds.to_dataloader(mode="double_buffered", copy=True, **common))
    assert len(db) == len(buf)
    for b, d in zip(buf, db):
        da, dbb = b.to_ragged(), d.to_ragged()
        assert set(da) == set(dbb)
        for k in da:
            assert da[k].to_ak().to_list() == dbb[k].to_ak().to_list()


@pytest.mark.slow
def test_double_buffered_variants_flank_tokens_matches_buffered(file_backed_ds):
    ds = (
        file_backed_ds.with_seqs("variants").with_tracks(False)
        .with_settings(flank_length=2, token_alphabet=b"ACGT", unknown_token=0)
        .with_output_format("flat")
    )
    common = dict(batch_size=2, shuffle=False, drop_last=True, buffer_bytes=4 * 1024 * 1024)
    buf = list(ds.to_dataloader(mode="buffered", **common))
    db = list(ds.to_dataloader(mode="double_buffered", copy=True, **common))
    assert len(db) == len(buf)
    for b, d in zip(buf, db):
        np.testing.assert_array_equal(
            b.flank_tokens.to_ragged().to_ak().to_list(),
            d.flank_tokens.to_ragged().to_ak().to_list(),
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/unit/test_double_buffered_loader.py -q -k "variant_windows_matches or flank_tokens_matches" --no-header -p no:cacheprovider -m slow`
Expected: FAIL — `ValueError: mode='double_buffered' does not support 'variant-windows'`.

- [ ] **Step 3a: Remove both `double_buffered` guards** from `_torch.py` (the two blocks added in Task 1.3). The transports now support every output config.

- [ ] **Step 3b: Delete the now-obsolete rejection tests.** Remove `test_flat_buffered_rejects_variant_windows` and `test_flat_buffered_rejects_variants_flank_tokens` from `tests/unit/test_buffered_loader.py` (their `double_buffered` arm no longer rejects). Keep `test_flat_buffered_plain_variants_still_works`.

- [ ] **Step 4: Run the double-buffered suite (incl. slow)**

Run: `pixi run -e dev pytest tests/unit/test_double_buffered_loader.py -q -m slow`
Expected: PASS (new parity tests + the existing double-buffered suite).

- [ ] **Step 5: Lint + commit**

```bash
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/ && pixi run -e dev typecheck
rtk git add python/genvarloader/_torch.py tests/unit/test_buffered_loader.py tests/unit/test_double_buffered_loader.py
rtk git commit -m "feat(dataloader): support variant-windows and flank tokens in mode='double_buffered'"
```

---

### Task 2.6: Slot-fit regression + PR 2 docs + full tree

**Files:**
- Test: `tests/unit/test_double_buffered_loader.py` (add a slot-fit test mirroring `test_double_buffered_variants_offset_overflow_regression`)
- Modify: `docs/source/dataset.md`/`faq.md`, `skills/genvarloader/SKILL.md`

**Interfaces:** validates Task 1.1's byte accounting is not an under-estimate for the new modes (an under-estimate → `ProducerError: buffer is smaller than requested size`).

- [ ] **Step 1: Write the slot-fit test** (small `buffer_bytes` forces tight slots; if byte accounting undersizes, the producer overflows)

```python
@pytest.mark.slow
def test_double_buffered_variant_windows_slot_fits(file_backed_ds):
    ds = (
        file_backed_ds.with_tracks(False).with_output_format("flat")
        .with_seqs("variant-windows",
                   gvl.VarWindowOpt(flank_length=4, token_alphabet=b"ACGT", unknown_token=4))
    )
    common = dict(batch_size=4, shuffle=False, drop_last=True, buffer_bytes=1 << 20)
    # Must not raise ProducerError (buffer too small) — byte accounting must not undersize.
    db = list(ds.to_dataloader(mode="double_buffered", copy=True, **common))
    buf = list(ds.to_dataloader(mode="buffered", **common))
    assert len(db) == len(buf)
```

- [ ] **Step 2: Run it**

Run: `pixi run -e dev pytest tests/unit/test_double_buffered_loader.py::test_double_buffered_variant_windows_slot_fits -q -m slow`
Expected: PASS. If it raises `ProducerError (...buffer is smaller...)`, the Task 1.1 windows byte accounting under-estimates — increase it (add the missing term) until this passes.

- [ ] **Step 3: Docs.** Update the `mode=`/output-mode compatibility notes (`docs/source/*.md`, `skills/genvarloader/SKILL.md`) to state that **both** `buffered` and `double_buffered` now support `variant-windows` and flat `variants` with flank tokens. Remove the PR-1-era "double_buffered lands later" wording.

- [ ] **Step 4: Full unit tree + api check + commit**

```bash
pixi run -e dev pytest tests/unit -q
pixi run -e dev python -c "import genvarloader as g; api=open('docs/source/api.md').read(); print('MISSING:', [n for n in g.__all__ if n not in api] or 'none')"
rtk git add tests/unit/test_double_buffered_loader.py docs/ skills/
rtk git commit -m "test(dataloader): double-buffered slot-fit regression + docs for variant-windows support"
```

---

## Self-Review

**Spec coverage** (against `docs/superpowers/specs/2026-07-20-buffered-variant-windows-design.md`):
- Gap 1 (byte accounting) → Task 1.1 + validated by Task 2.6. ✅
- Gap 2 (slicing) → Task 1.2 + `_FLAT_TYPES` in Task 1.3. ✅
- Gap 3 (shm serialize) → Task 2.2 (kind=4) + Task 2.3 (flank_tokens) + Task 2.4 (reshape). ✅
- Gap 4 (producer schema) → Task 2.1. ✅
- Gap 5 (guards) → narrowed in Task 1.3, removed in Task 2.5. ✅
- Both configs (A variant-windows, B variants+flank) → every task covers both. ✅
- Both transports → Phase 1 buffered, Phase 2 double_buffered. ✅
- Parity oracle = `mode=None` per-item (Task 1.3) + transport parity (Task 2.5). ✅
- shm `kind=4` decision (reuse kind=2 FieldDescriptor) → Task 2.2. ✅
- Docs/skill → Task 1.4 + Task 2.6. ✅

**Known verification points flagged inline for the implementer** (TDD arbiters, not placeholders): token dtype source for byte accounting (1.1 Step 4 / 2.6), `flank_tokens` memory layout for slicing (1.2 Step 4), `flank_tokens` real offsets length for the synthetic shm fixture (2.3 Step 1), and `Haps` flank attribute names for the schema (2.1 Step 3a). Each has a pinning test that fails loudly if the assumption is wrong.

**Type consistency:** `_FlatWindow(data, seq_offsets, var_offsets, shape)`, `_Flat(data, offsets, shape)`, `_FlatAlleles(byte_data, seq_offsets, var_offsets, shape)`, `_FlatVariants(fields)` + `.flank_tokens`, `_FlatVariantWindows(fields, **window_slots)`, `VarWindowOpt(flank_length, token_alphabet, unknown_token, ref, alt)` — used consistently across tasks. Descriptor dict keys (`_field_descs`, `_flank`, `field_kind`, `regular_size`) match the verbatim `_shm_layout.py` names.
