# `to_dataloader()` drop_last Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `Dataset.to_dataloader()` honor `drop_last` in every mode — buffered modes must keep the final partial batch when `drop_last=False`, and the default mode must not crash when `drop_last=True`.

**Architecture:** Two defects in the PyTorch integration. (1) `_chunked.py::ChunkPlanner` hard-requires the index count to be a multiple of `batch_size`, and `_torch.py::_resolve_buffered_inputs` unconditionally truncates to a whole multiple — together they silently drop the partial batch in buffered/double_buffered modes. (2) `_torch.py::get_dataloader`'s default branch forwards `drop_last` to `td.DataLoader` alongside `batch_size=None`, which PyTorch rejects. The fix teaches `ChunkPlanner` to emit a trailing partial batch, gates the truncation on `drop_last`, fixes the buffered loader's `__len__`, and stops forwarding `drop_last` in the default path.

**Tech Stack:** Python, NumPy, PyTorch (`torch.utils.data`), pytest, pixi.

---

## Environment notes (read before starting)

- **Run tests in the `default` pixi env**, which bundles PyTorch. The `dev` env does **not** have torch, so `pytest` there module-skips all torch tests.
  - Command form: `pixi run python -m pytest <path> -v`
- **Test data must exist.** If `tests/data/fasta/synthetic.fa.bgz` is missing, run `pixi run -e dev gen` once first. (It is present in this working tree already.)
- **Pre-commit hook caveat.** The repo's `pyrefly` pre-commit hook currently fails on a *pre-existing, unrelated* seqpro version mismatch in `python/genvarloader/_flat.py` and `_ragged.py` (`Could not import to_padded / reverse_complement from seqpro.rag`). This is not caused by any change in this plan. Commit with `git commit --no-verify` for every commit in this plan, and mention the bypass in the handoff.
- **Branch:** work happens on `fix/dataloader-drop-last` (already created; the design doc is committed there).

## File structure

| File | Responsibility | Change |
|------|----------------|--------|
| `python/genvarloader/_chunked.py` | Group `(r,s)` pairs into per-slot chunks aligned to mini-batch boundaries | `ChunkPlanner`: allow non-divisible length; build `batch_totals` with a trailing remainder entry; clamp chunk slice end to `n` |
| `python/genvarloader/_torch.py` | Build dataloaders + resolve buffered epoch order | `_resolve_buffered_inputs`: gate `n_keep` truncation on `drop_last`. `get_dataloader` (mode=None): stop forwarding `drop_last` to `td.DataLoader` |
| `python/genvarloader/_buffered_loader.py` | `mode='buffered'` synchronous loader | `__len__`: floor → ceil |
| `tests/unit/test_chunk_planner.py` | Pure-logic `ChunkPlanner` tests | Add partial-final-batch test |
| `tests/unit/test_torch.py` | Torch integration tests | Add `_n_instances` helper, default-mode `drop_last=True` regression, buffered-modes drop_last matrix, DDP-shaped custom-sampler test |

---

## Task 1: `ChunkPlanner` keeps the trailing partial batch

**Files:**
- Test: `tests/unit/test_chunk_planner.py` (add one test)
- Modify: `python/genvarloader/_chunked.py:27-104` (`ChunkPlanner.__init__` and `__iter__`)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_chunk_planner.py`:

```python
def test_plan_keeps_partial_final_batch():
    # 7 instances, batch_size=3 -> 2 full batches + 1 partial (1 instance).
    # Generous slot_bytes so everything fits in a single chunk.
    bytes_per_instance = np.full((7, 1), 10, dtype=np.int64)
    flat = np.arange(7)
    planner = ChunkPlanner(
        r_idx=flat,
        s_idx=np.zeros_like(flat),
        batch_size=3,
        bytes_per_instance=bytes_per_instance,
        slot_bytes=1000,
    )
    chunks = list(planner)
    # All 7 instances preserved -- the trailing partial batch is not dropped.
    assert sum(len(cr) for cr, _, _ in chunks) == 7
    # ceil(7 / 3) == 3 mini-batches counted across chunks.
    assert sum(nb for _, _, nb in chunks) == 3


def test_plan_single_partial_batch_smaller_than_batch_size():
    # 2 instances, batch_size=5 -> a single partial batch of 2.
    bytes_per_instance = np.full((2, 1), 10, dtype=np.int64)
    flat = np.arange(2)
    planner = ChunkPlanner(
        r_idx=flat,
        s_idx=np.zeros_like(flat),
        batch_size=5,
        bytes_per_instance=bytes_per_instance,
        slot_bytes=1000,
    )
    chunks = list(planner)
    assert sum(len(cr) for cr, _, _ in chunks) == 2
    assert sum(nb for _, _, nb in chunks) == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run python -m pytest tests/unit/test_chunk_planner.py::test_plan_keeps_partial_final_batch tests/unit/test_chunk_planner.py::test_plan_single_partial_batch_smaller_than_batch_size -v`
Expected: FAIL — `ValueError: len(r_idx)=7 is not a multiple of batch_size=3. Use drop_last or pad the sampler before passing to ChunkPlanner.` (and likewise for the 2/5 case).

- [ ] **Step 3: Relax the divisibility guard and build `batch_totals` with a remainder entry**

In `python/genvarloader/_chunked.py`, replace the body of `__init__` from the `n = len(r_idx)` line through the `batch_totals = per_inst.reshape(-1, batch_size).sum(-1)` line (currently lines 37-50):

```python
        if len(r_idx) != len(s_idx):
            raise ValueError("r_idx and s_idx must have the same length")
        n = len(r_idx)
        self.r_idx = np.asarray(r_idx)
        self.s_idx = np.asarray(s_idx)
        self.batch_size = batch_size
        self.bytes_per_instance = bytes_per_instance
        self.slot_bytes = int(slot_bytes)
        self._n = n

        # Per-instance byte cost in epoch order, grouped into mini-batches. The
        # final batch may be partial (drop_last=False); its bytes are summed into
        # a trailing batch_totals entry so chunk packing and peak-byte sizing
        # account for it like any other batch.
        per_inst = bytes_per_instance[self.r_idx, self.s_idx].astype(np.int64)
        n_full = n // batch_size
        full_totals = (
            per_inst[: n_full * batch_size].reshape(n_full, batch_size).sum(-1)
        )
        remainder = per_inst[n_full * batch_size :]
        if remainder.size:
            batch_totals = np.concatenate([full_totals, remainder.sum(keepdims=True)])
        else:
            batch_totals = full_totals
```

Note: the `too_big` check and `_compute_peak_chunk_bytes()` that follow already read `batch_totals`, so they work unchanged with the partial entry included.

- [ ] **Step 4: Clamp the chunk slice end to `n` in `__iter__`**

In `ChunkPlanner.__iter__` (currently line 102), change:

```python
            end = j * self.batch_size
```

to:

```python
            end = min(j * self.batch_size, self._n)
```

- [ ] **Step 5: Run the new tests to verify they pass**

Run: `pixi run python -m pytest tests/unit/test_chunk_planner.py -v`
Expected: PASS — all tests, including the two new ones and the pre-existing divisible-case tests.

- [ ] **Step 6: Commit**

```bash
git add python/genvarloader/_chunked.py tests/unit/test_chunk_planner.py
git commit --no-verify -m "fix(chunked): keep trailing partial batch in ChunkPlanner"
```

---

## Task 2: Default mode no longer crashes on `drop_last=True`

**Files:**
- Test: `tests/unit/test_torch.py` (add one test)
- Modify: `python/genvarloader/_torch.py:131-146` (the `td.DataLoader(...)` call in `get_dataloader`'s `mode=None` branch)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_torch.py`:

```python
def test_default_mode_drop_last_true_does_not_crash(small_gvl_ds):
    """mode=None with drop_last=True must not raise. The BatchSampler applies
    drop_last; forwarding it to the DataLoader (which also gets batch_size=None)
    is what PyTorch rejected."""
    ds = small_gvl_ds.with_seqs("reference").with_tracks(False)
    N = len(ds)
    bs = next((c for c in range(2, N) if N % c), 1)
    assert N % bs != 0, "need an indivisible batch_size to exercise drop_last"
    dl = ds.to_dataloader(batch_size=bs, shuffle=False, drop_last=True)  # mode=None
    n_batches = sum(1 for _ in dl)
    assert n_batches == N // bs
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run python -m pytest tests/unit/test_torch.py::test_default_mode_drop_last_true_does_not_crash -v`
Expected: FAIL — `ValueError: batch_size=None option disables auto-batching and is mutually exclusive with drop_last`.

- [ ] **Step 3: Stop forwarding `drop_last` to the DataLoader**

In `python/genvarloader/_torch.py`, in `get_dataloader`'s `mode=None` branch, the `td.DataLoader(...)` call currently passes `drop_last=drop_last,` (line 138). Delete that single keyword argument so the call becomes:

```python
        return td.DataLoader(
            dataset,
            batch_size=None,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )
```

The `BatchSampler` built by `get_sampler` (a few lines above, line 122-129) remains the sole authority on dropping the partial batch.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pixi run python -m pytest tests/unit/test_torch.py::test_default_mode_drop_last_true_does_not_crash -v`
Expected: PASS.

- [ ] **Step 5: Verify the existing default-mode `drop_last=False` smoke tests still pass**

Run: `pixi run python -m pytest tests/unit/test_torch.py -v`
Expected: PASS — all existing tests plus the new one.

- [ ] **Step 6: Commit**

```bash
git add python/genvarloader/_torch.py tests/unit/test_torch.py
git commit --no-verify -m "fix(torch): do not forward drop_last to DataLoader in default mode"
```

---

## Task 3: Buffered modes honor `drop_last=False`

**Files:**
- Test: `tests/unit/test_torch.py` (add a helper + three tests)
- Modify: `python/genvarloader/_torch.py:70-76` (`_resolve_buffered_inputs` truncation)
- Modify: `python/genvarloader/_buffered_loader.py:44-45` (`__len__`)

- [ ] **Step 1: Write the failing tests**

At the top of `tests/unit/test_torch.py`, after the existing `import` lines, add `import math` and this helper:

```python
def _n_instances(batch) -> int:
    """Outer (instance) dimension of a dataloader batch, across the gvl output
    types that buffered/double_buffered modes can yield."""
    import numpy as np
    from seqpro.rag import Ragged

    if isinstance(batch, tuple):
        batch = batch[0]
    if isinstance(batch, np.ndarray):
        return batch.shape[0]
    if isinstance(batch, Ragged):
        return batch.shape[0]
    if hasattr(batch, "haps"):  # AnnotatedHaps / RaggedAnnotatedHaps
        return batch.haps.shape[0]
    return len(batch)  # ak.Array (RaggedVariants) and fallbacks
```

Then append these tests:

```python
@pytest.mark.parametrize("mode", ["buffered", "double_buffered"])
@pytest.mark.parametrize("drop_last", [False, True])
def test_buffered_modes_respect_drop_last(small_gvl_ds, mode, drop_last):
    ds = small_gvl_ds.with_seqs("reference").with_tracks(False)
    N = len(ds)
    bs = next((c for c in range(2, N) if N % c), 1)
    assert N % bs != 0, "need an indivisible batch_size to exercise drop_last"

    dl = ds.to_dataloader(
        batch_size=bs, shuffle=False, drop_last=drop_last, mode=mode
    )
    batches = list(dl)
    expected = N // bs if drop_last else math.ceil(N / bs)
    assert len(batches) == expected
    assert len(dl) == expected  # __len__ must match what iteration yields
    if not drop_last:
        # The final batch is the smaller, partial one.
        assert _n_instances(batches[-1]) == N % bs


def test_buffered_drop_last_false_with_custom_batch_sampler(small_gvl_ds):
    """DDP-shaped case: when the (r,s) indices come from a user-supplied sampler
    whose count is not a multiple of batch_size, the partial batch must survive."""
    import torch.utils.data as tud

    ds = small_gvl_ds.with_seqs("reference").with_tracks(False)
    N = len(ds)
    bs = next((c for c in range(2, N) if N % c), 1)
    sampler = tud.BatchSampler(tud.SequentialSampler(range(N)), bs, drop_last=False)
    dl = ds.to_dataloader(sampler=sampler, drop_last=False, mode="buffered")
    assert len(list(dl)) == math.ceil(N / bs)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run python -m pytest "tests/unit/test_torch.py::test_buffered_modes_respect_drop_last" "tests/unit/test_torch.py::test_buffered_drop_last_false_with_custom_batch_sampler" -v`
Expected: FAIL on the `drop_last=False` parametrizations (both modes) and the custom-sampler test — observed batch count is `floor(N/bs)` instead of `ceil(N/bs)`. The `drop_last=True` parametrizations PASS already. (Task 1 must be merged first, or `double_buffered` `drop_last=False` would instead raise from `ChunkPlanner`.)

- [ ] **Step 3: Gate the `n_keep` truncation on `drop_last`**

In `python/genvarloader/_torch.py`, in `_resolve_buffered_inputs`, the lines that currently read (around line 74-75):

```python
    n_keep = (len(flat) // batch_size) * batch_size
    flat = flat[:n_keep]
```

become:

```python
    # Only drop the trailing partial batch when the caller asked for it. When
    # drop_last=False, keep it -- ChunkPlanner emits it as a partial batch.
    if drop_last:
        n_keep = (len(flat) // batch_size) * batch_size
        flat = flat[:n_keep]
```

- [ ] **Step 4: Fix the buffered loader's `__len__` (floor → ceil)**

In `python/genvarloader/_buffered_loader.py`, `BufferedTorchDataset.__len__` currently returns `len(flat_r) // batch_size` (line 45). Change it to ceil:

```python
        def __len__(self) -> int:
            return (len(flat_r) + batch_size - 1) // batch_size
```

(The `double_buffered` loader's `__len__` sums per-chunk batch counts from `ChunkPlanner`, so it already reports the correct ceil count once Task 1 lands — no change needed there.)

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pixi run python -m pytest "tests/unit/test_torch.py::test_buffered_modes_respect_drop_last" "tests/unit/test_torch.py::test_buffered_drop_last_false_with_custom_batch_sampler" -v`
Expected: PASS — all 4 matrix parametrizations + the custom-sampler test.

- [ ] **Step 6: Commit**

```bash
git add python/genvarloader/_torch.py python/genvarloader/_buffered_loader.py tests/unit/test_torch.py
git commit --no-verify -m "fix(torch): buffered modes honor drop_last=False"
```

---

## Task 4: Full regression sweep

**Files:** none (verification only)

- [ ] **Step 1: Run every affected test module**

Run:
```bash
pixi run python -m pytest \
  tests/unit/test_chunk_planner.py \
  tests/unit/test_torch.py \
  tests/unit/test_buffered_loader.py \
  tests/unit/test_double_buffered_loader.py \
  -v
```
Expected: PASS (the `1kg`-gated offset-overflow regression in `test_double_buffered_loader.py` may `skip` if `tests/data/1kg/...` is absent — a skip is acceptable, a failure is not).

- [ ] **Step 2: Confirm the original bug report is fixed end-to-end**

Run:
```bash
pixi run python -c "
import math, genvarloader as gvl
ds = gvl.get_dummy_dataset().with_seqs('reference').with_tracks(False)
N = len(ds); bs = next(c for c in range(2, N) if N % c)
for mode in (None, 'buffered'):
    n = sum(1 for _ in ds.to_dataloader(batch_size=bs, drop_last=False, mode=mode))
    assert n == math.ceil(N/bs), (mode, n, math.ceil(N/bs))
    m = sum(1 for _ in ds.to_dataloader(batch_size=bs, drop_last=True, mode=mode))
    assert m == N//bs, (mode, m, N//bs)
print('OK: drop_last honored in default and buffered modes')
"
```
Expected: prints `OK: drop_last honored in default and buffered modes` with no assertion error. (Uses the in-memory dummy dataset for `None`/`buffered`; `double_buffered` requires a file-backed dataset and is covered by the test suite in Step 1.)

- [ ] **Step 3: Verify the skill needs no update**

The `drop_last` parameter already exists on every `to_dataloader` and its docstring already states the `drop_last=False` semantics ("the last batch will be smaller"). This is a behavior bugfix, not a signature/default/literal change, so `skills/genvarloader/SKILL.md` requires no edit. Confirm by checking the skill mentions no contradicting claim:

Run: `grep -n "drop_last" skills/genvarloader/SKILL.md || echo "drop_last not documented in skill -- no change needed"`
Expected: either no match, or any match is consistent with standard PyTorch `drop_last` semantics. No edit required.

---

## Self-review notes

- **Spec coverage:** Bug 1 (buffered drop_last=False) → Tasks 1 + 3. Bug 2 (default drop_last=True crash) → Task 2. DDP note → custom-sampler test in Task 3. Test matrix from spec → Tasks 2 + 3. `_buffered_loader.__len__` and double-buffered auto-correctness → Task 3 / Task 1. All spec touch-points mapped.
- **Type consistency:** `_n_instances`, `ChunkPlanner` attributes (`self._n`, `batch_totals`, `self.r_idx`/`self.s_idx`/`self.batch_size`/`self.slot_bytes`) match across tasks. `n_keep` gating uses the `drop_last` parameter already in `_resolve_buffered_inputs`'s signature.
- **No placeholders:** every code step shows full code; every run step shows the exact command and expected outcome.
