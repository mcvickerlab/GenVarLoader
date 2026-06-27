# Phase 5 W5 — Consolidation: golden-snapshot parity, delete numba, add rayon

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the numba-oracle parity suites to on-disk golden fixtures, delete the entire numba backend (registry, kernels, `GVL_BACKEND`), and add `rayon` batch parallelism to the rust read-path kernels — gated byte-identical throughout.

**Architecture:** Three strictly-ordered stages in one PR (`phase-5-w5` → `rust-migration`), with clean commit boundaries. **Stage A (snapshot)** must run while numba still exists: it captures rust output to committed `.npz` goldens, cross-checked against the numba oracle at generation time, and rewrites every parity test to assert `rust == golden` (importing rust callables *directly*, never via `_dispatch`). **Stage B (delete)** removes all numba now that the parity suite no longer needs it. **Stage C (rayon)** parallelizes the kernels, gated `serial == parallel` byte-identical against the frozen goldens.

**Tech Stack:** Rust (ndarray, PyO3, rayon), Python (numpy, hypothesis for *generation only*), maturin, pytest.

## Global Constraints

- **Branch:** `phase-5-w5`, already cut off `rust-migration @ efb87ea` (W2/W3/W4 merged). Working dir is the main repo (not a worktree).
- **Byte-identical parity is the landing gate.** Stage A's goldens are the frozen oracle; every later change must keep `rust == golden`.
- **Generate goldens from rust, cross-checked against numba.** At generation time (numba present), golden := rust output, and the generator asserts `numba == rust` before saving. This makes the frozen point provably equal to the oracle.
- **Committed parity tests must NOT import `_dispatch`.** Replay imports rust callables directly from the extension/production wrappers, so Stage B's dispatch deletion does not touch the test suite.
- **maturin rebuild before pytest:** after ANY `src/` edit run `pixi run -e dev maturin develop --release` before pytest, or the stale `.so` is imported. (`cargo test` compiles from source and is exempt.)
- **All pytest invocations need** `--basetemp=$(pwd)/.pytest_tmp` (os.link Errno 18 on Carter).
- **Conventional commits** with trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Use `rtk` prefix on git commands. No squash.
- **Rayon gating:** each parallelized kernel takes a `parallel: bool` (computed Python-side via `should_parallelize(...)`); the `else` serial branch stays as the byte-identity reference; thread count comes from rayon's global pool via `RAYON_NUM_THREADS`. Follow the existing `get_reference` idiom in `src/reference/mod.rs:56-120` exactly — `split_at_mut` chain → `Vec<&mut [_]>` → `into_par_iter()`. **Do NOT** put raw `*mut` pointers into a rayon closure (not `Send`; won't compile / unsound to force).
- **Three commit boundaries** inside the one PR: `snapshot…`, `delete numba…`, `rayon…` (each stage's tasks roll up into its boundary; intermediate task commits are fine).

---

## File Structure

**Stage A — new files:**
- `tests/parity/_golden.py` — snapshot/replay infrastructure: deterministic example collection, object-array `.npz` save/load, `RUST_KERNELS` name→callable table, replay-assert helpers mirroring the 4 `_harness.py` shapes.
- `tests/parity/generate_goldens.py` — regeneration driver (run manually while numba present; commits `.npz`). A per-kernel registry table drives it.
- `tests/parity/golden/*.npz` — committed frozen fixtures (one per kernel/test).
- `tests/parity/test_import_no_numba.py` — (added Stage B) import-guard.

**Stage A — modified:** every `tests/parity/test_*_parity.py` (convert from cross-backend to golden replay); `tests/parity/_harness.py` (helpers gain golden-replay variants or are superseded by `_golden.py`).

**Stage B — modified:** `python/genvarloader/_dispatch.py` (deleted); the 6 production modules with `get(name)(...)` call sites and `register()` blocks (`_reference.py`, `_intervals.py`, `_genotypes.py`, `_flat_variants.py`, `_rag_variants.py`, `_reconstruct.py`); the backend-conditional branch sites (`_query.py`, `_haps.py`, `_reconstruct.py`, `_tracks.py`, `_reference.py`); the 11 `import numba` files; `_threads.py`, `_ragged.py`, `__init__.py`; `pyproject.toml`, `pixi.toml`.

**Stage C — modified:** `src/reconstruct/mod.rs`, `src/tracks/mod.rs`, `src/genotypes/mod.rs`, `src/intervals.rs`, plus the FFI wrappers in `src/ffi/mod.rs` that gain a `parallel` arg, and the Python callers that pass it; `python/genvarloader/_threads.py` (RAYON_NUM_THREADS); `docs/roadmaps/rust-migration.md`.

---

# STAGE A — Golden snapshot (numba still present)

### Task A1: Golden infrastructure (`_golden.py`)

**Files:**
- Create: `tests/parity/_golden.py`
- Create: `tests/parity/golden/.gitkeep`
- Test: `tests/parity/test_golden_infra.py`

**Interfaces:**
- Produces:
  - `GOLDEN_DIR: Path` — `Path(__file__).parent / "golden"`.
  - `collect_examples(strategy, n: int) -> list` — deterministic draw of `n` examples from a hypothesis strategy (no DB, derandomized).
  - `save_golden(name: str, cases: list) -> None` — write `GOLDEN_DIR/{name}.npz` as a single object array `cases` (allow_pickle).
  - `load_golden(name: str) -> list` — read it back.
  - `RUST_KERNELS: dict[str, Callable]` — kernel-name → rust callable, imported directly (verified against each `register(..., rust=…)` in production).
  - `replay_return(name, cases)`, `replay_tuple(name, cases)`, `replay_inplace(name, cases, out_factory, out_index)`, `replay_dict(name, cases)` — load-free replay helpers taking pre-loaded `cases`, each asserting `rust(*inputs)` byte-identical to the stored golden (dtype + shape + values), mirroring the 4 `_harness.py` shapes.

- [ ] **Step 1: Write the failing test**

```python
# tests/parity/test_golden_infra.py
"""Self-tests for the golden snapshot/replay infrastructure."""
from __future__ import annotations

import numpy as np
from hypothesis import strategies as st

from tests.parity import _golden


def test_collect_examples_deterministic():
    s = st.integers(0, 1_000_000)
    a = _golden.collect_examples(s, 20)
    b = _golden.collect_examples(s, 20)
    assert a == b
    assert len(a) == 20


def test_save_load_roundtrip_mixed(tmp_path, monkeypatch):
    monkeypatch.setattr(_golden, "GOLDEN_DIR", tmp_path)
    cases = [
        ((np.arange(3, dtype=np.int32), None, 5), np.arange(3, dtype=np.int32) * 2),
        ((np.zeros(0, np.uint8),), np.zeros(0, np.uint8)),
    ]
    _golden.save_golden("demo", cases)
    back = _golden.load_golden("demo")
    assert len(back) == 2
    np.testing.assert_array_equal(back[0][0][0], cases[0][0][0])
    assert back[0][0][1] is None
    assert back[0][0][2] == 5


def test_rust_kernels_table_callable():
    # Every registered name resolves to a real callable imported directly.
    assert _golden.RUST_KERNELS, "RUST_KERNELS is empty"
    for name, fn in _golden.RUST_KERNELS.items():
        assert callable(fn), f"{name} -> {fn!r} not callable"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/parity/test_golden_infra.py -q --basetemp=$(pwd)/.pytest_tmp`
Expected: FAIL — `ModuleNotFoundError: tests.parity._golden`.

- [ ] **Step 3: Write `_golden.py`**

```python
# tests/parity/_golden.py
"""Frozen-golden snapshot + replay for the parity suite.

Goldens are generated from the RUST implementation and cross-checked against
the numba oracle at generation time (see generate_goldens.py). Replay imports
rust callables DIRECTLY — never via _dispatch — so these tests survive the
numba/dispatch deletion in Stage B.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from hypothesis import HealthCheck, Phase, given, settings

GOLDEN_DIR = Path(__file__).parent / "golden"


def collect_examples(strategy, n: int) -> list:
    """Deterministically draw ``n`` examples from a hypothesis strategy.

    Derandomized + no database + generate-only phase ⇒ stable across runs for a
    fixed hypothesis version. Inputs are frozen INTO the golden, so the replay
    test never re-runs hypothesis.
    """
    out: list = []

    @settings(
        max_examples=n,
        derandomize=True,
        database=None,
        phases=[Phase.generate],
        suppress_health_check=list(HealthCheck),
        deadline=None,
    )
    @given(strategy)
    def _collect(ex):
        if len(out) < n:
            out.append(ex)

    _collect()
    return out


def save_golden(name: str, cases: list) -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GOLDEN_DIR / f"{name}.npz", cases=np.array(cases, dtype=object))


def load_golden(name: str) -> list:
    data = np.load(GOLDEN_DIR / f"{name}.npz", allow_pickle=True)
    return list(data["cases"])


# --- direct rust-callable table -------------------------------------------------
# Each entry MUST equal the `rust=` argument of the matching register(...) call in
# production. Verify each against the dispatch map before trusting it.
def _build_rust_kernels() -> dict[str, Callable]:
    from genvarloader import genvarloader as _ext  # compiled extension

    table: dict[str, Callable] = {
        "intervals_to_tracks": _ext.intervals_to_tracks,
        "tracks_to_intervals": _ext.tracks_to_intervals,
        "get_diffs_sparse": _ext.get_diffs_sparse,
        "choose_exonic_variants": _ext.choose_exonic_variants,
        "gather_alleles": _ext.gather_alleles,
        "gather_rows_i32": _ext.gather_rows_i32,
        "gather_rows_f32": _ext.gather_rows_f32,
        "compact_keep_i32": _ext.compact_keep_i32,
        "compact_keep_f32": _ext.compact_keep_f32,
        "fill_empty_scalar_i32": _ext.fill_empty_scalar_i32,
        "fill_empty_scalar_f32": _ext.fill_empty_scalar_f32,
        "fill_empty_fixed_i32": _ext.fill_empty_fixed_i32,
        "fill_empty_fixed_f32": _ext.fill_empty_fixed_f32,
        "fill_empty_seq_u8": _ext.fill_empty_seq_u8,
        "fill_empty_seq_i32": _ext.fill_empty_seq_i32,
        "get_reference": _ext.get_reference,
        "reconstruct_haplotypes_from_sparse": _ext.reconstruct_haplotypes_from_sparse,
        "shift_and_realign_tracks_sparse": _ext.shift_and_realign_tracks_sparse,
        "rc_alleles": _ext.rc_alleles,
    }
    # NOTE: kernels whose `rust=` is a PYTHON WRAPPER (not a bare extension fn) —
    # e.g. assemble_variant_buffers (u8/i32 dtype dispatch). Add those by importing
    # the SAME wrapper the registration used; ground-truth against the register() call.
    return table


RUST_KERNELS: dict[str, Callable] = _build_rust_kernels()


def _eq(name: str, i: int, got, exp) -> None:
    got = np.asarray(got)
    exp = np.asarray(exp)
    assert got.dtype == exp.dtype, f"{name}[{i}]: dtype {got.dtype} != {exp.dtype}"
    assert got.shape == exp.shape, f"{name}[{i}]: shape {got.shape} != {exp.shape}"
    np.testing.assert_array_equal(got, exp, err_msg=f"{name}[{i}] value mismatch")


def replay_return(name: str, cases: list) -> None:
    fn = RUST_KERNELS[name]
    for ci, (inputs, golden) in enumerate(cases):
        _eq(f"{name}#{ci}", 0, fn(*inputs), golden)


def replay_tuple(name: str, cases: list) -> None:
    fn = RUST_KERNELS[name]
    for ci, (inputs, golden) in enumerate(cases):
        got = fn(*inputs)
        got = got if isinstance(got, tuple) else (got,)
        gold = golden if isinstance(golden, tuple) else (golden,)
        assert len(got) == len(gold), f"{name}#{ci}: tuple len {len(got)} != {len(gold)}"
        for j, (a, b) in enumerate(zip(got, gold)):
            _eq(f"{name}#{ci}", j, a, b)


def replay_inplace(name: str, cases: list, out_factory: Callable, out_index: int) -> None:
    fn = RUST_KERNELS[name]
    for ci, (inputs, golden) in enumerate(cases):
        out = out_factory(inputs)
        args = list(inputs)
        args.insert(out_index, out)
        fn(*args)
        _eq(f"{name}#{ci}", 0, out, golden)


def replay_dict(name: str, cases: list) -> None:
    fn = RUST_KERNELS[name]
    for ci, (inputs, golden) in enumerate(cases):
        got = fn(*inputs)
        assert set(got) == set(golden), f"{name}#{ci}: keys {set(got)} != {set(golden)}"
        for k in sorted(golden):
            _eq(f"{name}#{ci}:{k}.data", 0, np.asarray(got[k][0]), np.asarray(golden[k][0]))
            _eq(f"{name}#{ci}:{k}.off", 1,
                np.asarray(got[k][1], np.int64), np.asarray(golden[k][1], np.int64))
```

Note: `replay_inplace`'s `out_factory` takes `inputs` (so it can size the out buffer from `total_out` carried in the frozen case — the in-place strategies return `(total_out, inputs)`).

- [ ] **Step 4: Run the self-test**

Run: `pixi run -e dev pytest tests/parity/test_golden_infra.py -q --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS (3 tests). If `RUST_KERNELS` raises on a missing extension symbol, ground-truth that symbol's name against `src/lib.rs` and the matching `register()` call.

- [ ] **Step 5: Commit**

```bash
rtk git add tests/parity/_golden.py tests/parity/test_golden_infra.py tests/parity/golden/.gitkeep
rtk git commit -m "test(parity): golden snapshot/replay infrastructure (Phase 5 W5)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task A2: Golden generator + freeze kernel-level goldens

**Files:**
- Create: `tests/parity/generate_goldens.py`
- Create: `tests/parity/golden/<kernel>.npz` (committed artifacts)
- Test: regeneration is the test (the generator asserts numba==rust per case).

**Interfaces:**
- Consumes: `_golden.{collect_examples,save_golden,RUST_KERNELS}`, `strategies.*`, `genvarloader._dispatch.backends` (numba oracle — generation-time only).
- Produces: one `.npz` per kernel-level test, plus an `output_adapter` per kernel that normalizes `(numba_out, rust_out)` to comparable form and produces the stored golden.

**Kernel registry table (drives the generator).** Each row: kernel name, strategy factory, output shape (`return`/`tuple`/`inplace`/`dict`), N examples. Ground-truth the strategy names against `tests/parity/strategies.py` and each kernel's argument count against its existing `test_*_parity.py`.

| Golden name | Strategy | Shape | N |
|---|---|---|---|
| `intervals_to_tracks` | `intervals_to_tracks_inputs()` | inplace (out_index per existing test) | 200 |
| `get_diffs_sparse` | `get_diffs_sparse_inputs()` | tuple | 200 |
| `choose_exonic_variants` | `choose_exonic_variants_inputs()` | tuple | 200 |
| `gather_rows_i32` | `gather_rows_inputs(np.int32)` | tuple | 100 |
| `gather_rows_f32` | `gather_rows_inputs(np.float32)` | tuple | 100 |
| `gather_alleles` | `gather_alleles_inputs()` | tuple | 100 |
| `compact_keep_i32` | `compact_keep_inputs(np.int32)` | tuple | 100 |
| `compact_keep_f32` | `compact_keep_inputs(np.float32)` | tuple | 100 |
| `fill_empty_scalar_i32` | `fill_empty_scalar_inputs(np.int32)` | tuple | 100 |
| `fill_empty_scalar_f32` | `fill_empty_scalar_inputs(np.float32)` | tuple | 100 |
| `fill_empty_fixed_i32` | `fill_empty_fixed_inputs(np.int32)` | tuple | 100 |
| `fill_empty_fixed_f32` | `fill_empty_fixed_inputs(np.float32)` | tuple | 100 |
| `fill_empty_seq_u8` | `fill_empty_seq_inputs(np.uint8)` | tuple | 100 |
| `fill_empty_seq_i32` | `fill_empty_seq_inputs(np.int32)` | tuple | 100 |
| `tracks_to_intervals` | `tracks_to_intervals_inputs()` | tuple | 200 |
| `get_reference` | `get_reference_inputs()` | return | 200 |
| `shift_and_realign_tracks_sparse` | `shift_and_realign_tracks_inputs()` | inplace (out_index 0; case carries `total_out`) | 200 |
| `reconstruct_haplotypes_from_sparse` | `reconstruct_haplotypes_inputs()` | inplace (out_index 0; case carries `total_out`) | 200 |

(`rc_alleles`, `assemble_variant_buffers`, and the PRNG functions are handled in A4/A5 — non-standard shapes/fixtures.)

- [ ] **Step 1: Write `generate_goldens.py`**

```python
# tests/parity/generate_goldens.py
"""Regenerate frozen golden fixtures for the parity suite.

RUN MANUALLY while numba is still installed (Stage A):
    pixi run -e dev python -m tests.parity.generate_goldens

For each kernel: draw N deterministic examples, compute the golden from RUST,
and assert the numba oracle agrees BEFORE saving. After numba deletion this
script still regenerates from rust (the numba cross-check is skipped if the
backend is gone).
"""
from __future__ import annotations

import numpy as np

from genvarloader import _dispatch
from tests.parity import _golden, strategies

# (name, strategy, shape, n, extra) — see plan table. `inplace` carries an
# out_factory/out_index; the strategy returns (total_out, inputs) for those.
RETURN, TUPLE, INPLACE = "return", "tuple", "inplace"

SPEC = [
    ("get_diffs_sparse", strategies.get_diffs_sparse_inputs(), TUPLE, 200, None),
    ("get_reference", strategies.get_reference_inputs(), RETURN, 200, None),
    # ... fill in remaining rows from the plan table ...
]

# in-place kernels: strategy yields (total_out, inputs); out inserted at index 0.
INPLACE_SPEC = [
    ("intervals_to_tracks", strategies.intervals_to_tracks_inputs(), 200,
     lambda inp: np.zeros(int(inp[-1][-1]), np.float32), 7),  # out_index per existing test
    ("shift_and_realign_tracks_sparse", strategies.shift_and_realign_tracks_inputs(), 200,
     lambda total_out: np.zeros(total_out, np.float32), 0),
    ("reconstruct_haplotypes_from_sparse", strategies.reconstruct_haplotypes_inputs(), 200,
     lambda total_out: np.zeros(total_out, np.uint8), 0),
]


def _normalize(out):
    if isinstance(out, tuple):
        return tuple(np.asarray(x) for x in out)
    if isinstance(out, dict):
        return {k: (np.asarray(v[0]), np.asarray(v[1])) for k, v in out.items()}
    return np.asarray(out)


def _assert_oracle(name, a, b):
    # numba (a) vs rust (b) — both already normalized
    if isinstance(a, tuple):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y, err_msg=f"{name} oracle mismatch")
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for k in a:
            np.testing.assert_array_equal(a[k][0], b[k][0])
            np.testing.assert_array_equal(np.asarray(a[k][1], np.int64),
                                          np.asarray(b[k][1], np.int64))
    else:
        np.testing.assert_array_equal(a, b, err_msg=f"{name} oracle mismatch")


def _have_numba(name):
    try:
        _dispatch.backends(name)
        return True
    except Exception:
        return False


def gen_value_kernels():
    for name, strat, shape, n, _ in SPEC:
        examples = _golden.collect_examples(strat, n)
        rust = _golden.RUST_KERNELS[name]
        nb = _dispatch.backends(name)[0] if _have_numba(name) else None
        cases = []
        for inp in examples:
            r = _normalize(rust(*inp))
            if nb is not None:
                _assert_oracle(name, _normalize(nb(*inp)), r)
            cases.append((inp, r))
        _golden.save_golden(name, cases)
        print(f"  {name}: {len(cases)} cases")


def gen_inplace_kernels():
    for name, strat, n, out_factory, out_index in INPLACE_SPEC:
        examples = _golden.collect_examples(strat, n)
        rust = _golden.RUST_KERNELS[name]
        nb = _dispatch.backends(name)[0] if _have_numba(name) else None
        cases = []
        for ex in examples:
            # strategy returns (total_out, inputs) for shift/reconstruct;
            # intervals_to_tracks returns the inputs tuple directly.
            if isinstance(ex, tuple) and len(ex) == 2 and np.isscalar(ex[0]):
                total_out, inputs = ex
                of = lambda _inp, t=total_out: out_factory(t)
            else:
                inputs = ex
                of = out_factory
            out_r = of(inputs)
            args = list(inputs); args.insert(out_index, out_r); rust(*args)
            if nb is not None:
                out_n = of(inputs)
                an = list(inputs); an.insert(out_index, out_n); nb(*an)
                np.testing.assert_array_equal(out_n, out_r, err_msg=f"{name} oracle")
            cases.append((inputs, np.asarray(out_r)))
        _golden.save_golden(name, cases)
        print(f"  {name}: {len(cases)} cases")


if __name__ == "__main__":
    print("Generating value-kernel goldens...")
    gen_value_kernels()
    print("Generating in-place-kernel goldens...")
    gen_inplace_kernels()
    print("Done.")
```

Fill in the full `SPEC` list from the plan table. Ground-truth `intervals_to_tracks`'s `out_index` and out dtype/shape against its existing `test_intervals_to_tracks_parity.py` (it uses `assert_inplace_kernel_parity`).

- [ ] **Step 2: Generate the goldens**

Run: `pixi run -e dev python -m tests.parity.generate_goldens`
Expected: prints each kernel's case count; **no oracle-mismatch assertion**. If a mismatch fires, that is a real numba/rust divergence on a generated input — STOP and investigate per the numba-oracle-bug policy (check whether numba is the buggy one) before freezing.

- [ ] **Step 3: Verify the goldens are non-trivial**

Run: `pixi run -e dev python -c "from tests.parity import _golden; import numpy as np; c=_golden.load_golden('get_reference'); print(len(c), np.asarray(c[0][1]).shape)"`
Expected: 200 and a non-empty shape.

- [ ] **Step 4: Commit (goldens + generator)**

```bash
rtk git add tests/parity/generate_goldens.py tests/parity/golden/*.npz
rtk git commit -m "test(parity): freeze kernel-level golden fixtures (Phase 5 W5)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task A3: Convert kernel-level parity tests to golden replay

**Files:**
- Modify: all kernel-level `tests/parity/test_*_parity.py` (the ~14 using `_dispatch.backends` via `_harness`).
- Test: the converted tests themselves.

**Interfaces:**
- Consumes: `_golden.{load_golden, replay_return, replay_tuple, replay_inplace, replay_dict}`.

**Conversion pattern (apply to every kernel-level test).** Replace the `@given(strategy)` + `assert_kernel_parity*` body with a one-shot golden replay. Example — `test_get_diffs_sparse_parity.py`:

- [ ] **Step 1: Rewrite one test as the reference conversion**

```python
# tests/parity/test_get_diffs_sparse_parity.py
"""get_diffs_sparse: rust vs frozen golden (oracle frozen Phase 5 W5)."""
from __future__ import annotations

import pytest

from tests.parity import _golden

pytestmark = pytest.mark.parity


def test_get_diffs_sparse_golden():
    cases = _golden.load_golden("get_diffs_sparse")
    assert cases, "empty golden"
    _golden.replay_tuple("get_diffs_sparse", cases)
```

- [ ] **Step 2: Run it (rust backend)**

Run: `pixi run -e dev pytest tests/parity/test_get_diffs_sparse_parity.py -q --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS.

- [ ] **Step 3: Convert the remaining kernel-level tests** following the same pattern, choosing the matching replay helper:
  - `replay_tuple`: get_diffs_sparse, choose_exonic_variants, gather_rows (i32/f32), gather_alleles, compact_keep (i32/f32), fill_empty_scalar/fixed/seq (all dtype variants), tracks_to_intervals.
  - `replay_return`: get_reference.
  - `replay_inplace`: intervals_to_tracks (out_index/out_factory from its old test), shift_and_realign_tracks_sparse, reconstruct_haplotypes_from_sparse.
  - For multi-dtype files (e.g. `test_flat_variants_parity.py` covering many fill/gather kernels), one `test_<kernel>_golden()` per golden name.
  - Delete the now-unused `@given`, `strategies` imports, and `_harness`/`_dispatch` imports from each converted file.

- [ ] **Step 4: Run all converted kernel-level tests (rust)**

Run: `pixi run -e dev pytest tests/parity -q --basetemp=$(pwd)/.pytest_tmp -k "golden"`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add tests/parity/
rtk git commit -m "test(parity): replay kernel-level parity against frozen goldens (Phase 5 W5)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task A4: Snapshot + convert dataset-level (`GVL_BACKEND`-flip) tests

**Files:**
- Modify: `generate_goldens.py` (add dataset-golden generation), `_golden.py` (add `save/load` for Ragged-shaped outputs if needed).
- Modify: `test_dataset_parity.py`, `test_haplotypes_dataset_parity.py`, `test_spliced_haplotypes_parity.py`, `test_annotated_spliced_haplotypes_parity.py`, `test_fused_haps_parity.py`, `test_fused_tracks_parity.py`, `test_reference_dataset_parity.py`, `test_reference_fetch_parity.py`, `test_variants_dataset_parity.py` (all `GVL_BACKEND`-flip tests).
- Create: `tests/parity/golden/ds_*.npz`.

**Conversion pattern.** Each test currently: builds a deterministic dataset (session fixtures `phased_svar_gvl`, `build_*` seeded) → reads `ds[r,s]` under numba and rust → compares. Convert to: snapshot the agreed output's constituent arrays to `.npz` (generated while numba present, cross-checked) → test reads `ds[r,s]` under rust only → compares against golden. **Keep the spy guards** (they prove the rust kernel fires; still valid). **Delete** the `monkeypatch.setenv("GVL_BACKEND", ...)` flips and the numba read.

- [ ] **Step 1: Add a dataset-output serializer to `_golden.py`**

```python
def flatten_output(out):
    """Serialize a dataset __getitem__ result to a dict of arrays for golden storage.

    Handles Ragged (.data/.offsets), RaggedAnnotatedHaps (.haps/.var_idxs/.ref_coords),
    plain ndarray, and tuples thereof. Returns a JSON-able structure of np arrays.
    """
    import numpy as np
    from seqpro.rag import Ragged
    from genvarloader._ragged import RaggedAnnotatedHaps

    if isinstance(out, RaggedAnnotatedHaps):
        return {"kind": "annot",
                "haps": (np.asarray(out.haps.data), np.asarray(out.haps.offsets, np.int64)),
                "var_idxs": (np.asarray(out.var_idxs.data), np.asarray(out.var_idxs.offsets, np.int64)),
                "ref_coords": (np.asarray(out.ref_coords.data), np.asarray(out.ref_coords.offsets, np.int64))}
    if isinstance(out, Ragged):
        return {"kind": "ragged",
                "data": np.asarray(out.data), "offsets": np.asarray(out.offsets, np.int64)}
    if isinstance(out, tuple):
        return {"kind": "tuple", "items": [flatten_output(o) for o in out]}
    return {"kind": "array", "data": np.asarray(out)}


def assert_output_matches_golden(out, golden) -> None:
    """Assert a fresh dataset output equals a flattened golden (byte-identical)."""
    got = flatten_output(out)
    assert got["kind"] == golden["kind"], f"kind {got['kind']} != {golden['kind']}"
    # ... recursively compare arrays via _eq ... (mirror flatten_output structure)
```

(Implement the recursive comparison in `assert_output_matches_golden` mirroring `flatten_output`'s branches.)

- [ ] **Step 2: Add dataset-golden generation to `generate_goldens.py`**

For each dataset test, build the same fixture/dataset the test uses, read `ds[r,s]` under **numba** and **rust** (env flip — generation time only), assert equal, then `save_golden("ds_<testname>", flatten_output(rust_out))`. Use a `gen_dataset_goldens()` function driven by a small table of `(golden_name, build_fn, index)`.

- [ ] **Step 3: Convert one dataset test as the reference** — `test_haplotypes_dataset_parity.py`:

```python
def test_haplotypes_mode_dataset_golden(phased_svar_gvl, reference, monkeypatch):
    ds = gvl.Dataset.open(phased_svar_gvl, reference=reference).with_seqs("haplotypes")
    # spy guard stays — proves the fused rust kernel fires
    orig = _haps_mod.reconstruct_haplotypes_fused
    calls = {"n": 0}
    def _spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)
    monkeypatch.setattr(_haps_mod, "reconstruct_haplotypes_fused", _spy)

    out_rust = ds[:, :]
    assert calls["n"] > 0, "fused rust kernel never fired — vacuous"
    # non-triviality + golden compare
    _golden.assert_output_matches_golden(out_rust, _golden.load_flat_golden("ds_haplotypes_mode"))
```

(`load_flat_golden` = `load_golden` returning the single flattened dict; add a thin variant or store as a 1-element `cases` list.)

- [ ] **Step 4: Regenerate dataset goldens + run**

```bash
pixi run -e dev python -m tests.parity.generate_goldens
pixi run -e dev maturin develop --release   # only if src changed (it didn't here)
pixi run -e dev pytest tests/parity -q --basetemp=$(pwd)/.pytest_tmp
```
Expected: all PASS on rust.

- [ ] **Step 5: Convert remaining dataset tests + commit** (same pattern; keep each spy guard; drop the env flips).

```bash
rtk git add tests/parity/
rtk git commit -m "test(parity): replay dataset-level parity against frozen goldens (Phase 5 W5)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task A5: Snapshot + convert PRNG direct-import tests; Stage-A gate

**Files:**
- Modify: `test_prng_parity.py`, `test_rc_alleles_parity.py`, `test_assemble_variant_buffers_parity.py`.
- Create: `tests/parity/golden/prng_*.npz`, `rc_alleles.npz`, `assemble_variant_buffers.npz`.

- [ ] **Step 1: Freeze PRNG tables.** In `generate_goldens.py`, add a `gen_prng()` that builds a table of `(input → numba _xorshift64/_hash4 output)` over a deterministic input list, asserts the rust `_debug_*` equals it, and saves. Convert `test_prng_parity.py` to load the table and assert rust `_debug_xorshift64`/`_hash4` == frozen output (no numba import).

- [ ] **Step 2: Freeze `rc_alleles` + `assemble_variant_buffers`.** These use bespoke strategies/fixed arrays (see their existing tests). Add generation entries (rust golden + numba cross-check) and convert the tests to replay. For `assemble_variant_buffers` (dict-returning, dtype-dispatched wrapper), add its rust wrapper to `RUST_KERNELS` and use `replay_dict`.

- [ ] **Step 3: Regenerate everything + full parity suite gate**

```bash
pixi run -e dev python -m tests.parity.generate_goldens
pixi run -e dev pytest tests/parity -q --basetemp=$(pwd)/.pytest_tmp
```
Expected: entire `tests/parity` green on the default rust backend.

- [ ] **Step 4: Prove no committed parity test imports `_dispatch`**

Run: `rtk grep -rn "_dispatch\|GVL_BACKEND\|_harness" tests/parity/test_*.py`
Expected: **no matches** in committed test files (allowed only in `generate_goldens.py`). Fix any stragglers.

- [ ] **Step 5: Cross-check goldens still equal numba one final time** (the generator already asserts this; re-run to confirm clean), then commit the snapshot stage boundary.

```bash
rtk git add tests/parity/
rtk git commit -m "test(parity): freeze PRNG/rc_alleles/assemble goldens; Stage-A snapshot complete (Phase 5 W5)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

# STAGE B — Delete numba

> Goldens now guard rust independently of numba. Safe to delete.

### Task B1: Replace dispatched call sites with direct rust; delete the registry

**Files:**
- Delete: `python/genvarloader/_dispatch.py`
- Modify: `_reference.py`, `_intervals.py`, `_genotypes.py`, `_flat_variants.py`, `_rag_variants.py`, `_reconstruct.py` (22 `get(name)(...)` call sites + 20 `register()` blocks).

**Interfaces:**
- Consumes: the dispatch map (kernel name → rust symbol) from the W5 investigation. Each `get("name")(args)` becomes a direct call to the rust callable that `register(name, rust=…)` named.

- [ ] **Step 1:** For each of the 22 call sites, replace `get("kernel")(args)` with the direct rust callable (already imported at module scope as `_<kernel>_rust` or `from ..genvarloader import <kernel>`). Delete the paired `register(...)` block. Use the dispatch investigation's "replace-with-rust-symbol" column as the authority; verify each rust symbol is already imported in that module (it is — both backends were imported for registration).
- [ ] **Step 2:** Delete `python/genvarloader/_dispatch.py` and every `from .._dispatch import ...` / `import genvarloader._dispatch` line (including the `# noqa: F401 — triggers register(...)` import lines in any remaining non-parity modules). ALSO delete the now-dead test infra that depended on `_dispatch`: `tests/parity/_harness.py` (the old cross-backend assert helpers — fully superseded by `_golden.py`) and `tests/parity/test_harness_tuple.py` (its meta-test, the only remaining `_harness` consumer). Confirm no other file imports `_harness` before deleting.
- [ ] **Step 2b (test-infra spy rewrite — REQUIRED, else dataset goldens go vacuous):** `tests/parity/_golden.py::make_kernel_spy` currently spies by MUTATING the dispatch registry (`_disp.register(name, rust=spy, …)`). Once Step 1 makes call sites direct, registry mutation intercepts nothing — the spy never fires and the dataset tests' `assert calls["n"] > 0` guards fail. Rewrite `make_kernel_spy` to monkeypatch the DIRECT rust symbol at its production call site (the module-level name the converted call site now uses — e.g. `_genotypes.reconstruct_haplotypes_from_sparse`, `_tracks.shift_and_realign_tracks_sparse`, etc.), mirroring how the fused-path spies already monkeypatch `_haps_mod.reconstruct_*_fused`. It must remain a counting wrapper returning a `restore()`. Remove the function-local `from genvarloader import _dispatch` import. Verify each converted dataset test's spy still fires (`calls["n"] > 0`) after the rewrite.
- [ ] **Step 3: Rebuild + run the read-path tests**

```bash
pixi run -e dev maturin develop --release
pixi run -e dev pytest tests/parity tests/dataset tests/unit -q --basetemp=$(pwd)/.pytest_tmp
```
Expected: PASS (goldens + dataset/unit). A `KeyError: no kernel registered` or `ModuleNotFoundError: _dispatch` means a missed call site — fix it.
- [ ] **Step 4: Commit.**

---

### Task B2: Collapse backend-conditional branches; delete `GVL_BACKEND`

**Files:**
- Modify: `_query.py` (delete `_active_backend()` + the two `if _active_backend()=="numba"` RC post-pass branches — keep the rust in-kernel-RC behavior), `_haps.py` (4 `if _backend=="rust"` fused-vs-composed forks → keep fused), `_reconstruct.py` (2 forks → keep fused), `_reference.py` (3 backend branches → keep rust: always call `get_reference` with the 7-arg rust signature incl. `to_rc`; drop the numba post-pass), `_tracks.py` (2 `if ...=="rust"` RC post-pass branches → now unconditional).

**Critical:** the RC accounting must stay byte-identical. On rust, RC is folded in-kernel; the deleted numba branches were the *external* post-pass. Removing the `=="numba"` branch and keeping the rust path is correct **only if** the rust path already RC's in-kernel — which the W3/earlier work established. The goldens enforce this.

- [ ] **Step 1:** Delete `_active_backend()` and every `os.environ.get("GVL_BACKEND")` / `== "numba"` / `== "rust"` branch, keeping the rust arm inline. For `_reference.py:get_reference()`, drop the 6-vs-7-arg conditional — always pass `to_rc`.
- [ ] **Step 2: Rebuild + run the full read path + the strand/RC-heavy goldens**

```bash
pixi run -e dev maturin develop --release
pixi run -e dev pytest tests/parity tests/dataset tests/unit -q --basetemp=$(pwd)/.pytest_tmp
```
Expected: PASS — especially the spliced/annotated/strand-mixed dataset goldens (the RC-sensitive ones).
- [ ] **Step 3: Commit.**

---

### Task B3: Delete numba kernels + imports; refactor `_threads.py` and `_ragged.py`

**Files:**
- Modify (delete `@njit`/`@nb.vectorize` bodies + `import numba`): `_flat_variants.py`, `_genotypes.py`, `_intervals.py`, `_reference.py`, `_tracks.py`, `_flat.py`, `_flat_flanks.py`, `_dataset/_utils.py`, `_variants/_sitesonly.py`, `_ragged.py`, `_threads.py` (28 njit + 1 vectorize total).
- Refactor: `_threads.py` (OS thread detection, no numba), `_ragged.py` (keep `_COMP`, drop `@nb.vectorize` on `ufunc_comp_dna`), `__init__.py` (rename/adjust the `cap_numba_threads()` call).

- [ ] **Step 1: Refactor `_threads.py`** to drop numba:

```python
# python/genvarloader/_threads.py
from __future__ import annotations
import os

_MIN_BYTES_PER_THREAD = 1 << 20  # 1 MiB
_NUM_THREADS: int | None = None


def _detect_cpus() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))  # respects cgroup cpuset (Linux)
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def _resolve_num_threads() -> int:
    env = os.environ.get("GVL_NUM_THREADS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return _detect_cpus()


def cap_threads() -> int:
    """Resolve worker count once and pin rayon's pool via RAYON_NUM_THREADS.

    Must run before the first rust parallel call (rayon reads RAYON_NUM_THREADS
    at global-pool init). Idempotent.
    """
    global _NUM_THREADS
    if _NUM_THREADS is None:
        _NUM_THREADS = _resolve_num_threads()
        os.environ.setdefault("RAYON_NUM_THREADS", str(_NUM_THREADS))
    return _NUM_THREADS


def num_threads() -> int:
    return cap_threads()


def should_parallelize(total_bytes: int) -> bool:
    return total_bytes >= num_threads() * _MIN_BYTES_PER_THREAD
```

Update `__init__.py`: replace the `cap_numba_threads()` call with `cap_threads()` (keep it at import so `RAYON_NUM_THREADS` is set before any read). Update `_reference.py`'s `should_parallelize` import if the call signature changed (it didn't).

- [ ] **Step 2: `_ragged.py`** — remove the `@nb.vectorize` decorator and the `import numba as nb`. Keep `_COMP`. If `ufunc_comp_dna` is still referenced, replace it with a plain numpy LUT apply (`_COMP[arr]`); if unused after numba deletion, delete it. Ground-truth its usages first.

- [ ] **Step 2b (PRODUCTION numba fallbacks — REPLACE with numpy, do NOT delete):** Four wrappers in `_flat_variants.py` route int32/float32 to typed rust cores but fall back to a numba kernel for **arbitrary dtypes** (custom VCF FORMAT fields, issue #231 — "values are never silently down-cast"): `_gather_rows` → `_gather_rows_numba`, `_compact_keep` → `_compact_keep_numba`, `_fill_empty_scalar` → `_fill_empty_scalar_numba`, `_fill_empty_fixed` → `_fill_empty_fixed_numba`. These are **live production paths**, NOT dead code — deleting them regresses #231. Replace each `_*_numba` fallback with a pure-numpy, dtype-preserving implementation (these are simple ragged ops: per-row gather by `geno_offset_idx`/offsets; compact by boolean `keep` mask per row; fill empty rows with a dummy/scalar). Keep the i32/f32 rust fast paths. **Gate:** the 4 dtype-regression tests in `test_flat_variants_parity.py` (`test_gather_rows_dtype_regression`, `test_compact_keep_dtype_regression`, `test_fill_empty_scalar_dtype_regression`, `test_fill_empty_fixed_dtype_regression`, which exercise int16/int64) must still pass — they are the numpy replacements' correctness gate. (`test_fill_empty_seq_dtype_regression` already uses int32 → rust; unaffected.) Do this BEFORE Step 3's blanket deletion so the fallbacks have replacements.

- [ ] **Step 3:** Delete every remaining `@nb.njit` body and `import numba`/`import numba as nb` across the 9 kernel modules — **except the 4 production fallbacks handled in Step 2b** (those are now numpy, no `@njit`). For helper njit functions only used by other njit functions (e.g. `reconstruct_haplotype_from_sparse`, `_xorshift64`, `_hash4`, `padded_slice`, `_get_reference_row`), delete them too — rust owns these paths now. Verify nothing non-numba still imports them (grep each symbol).

- [ ] **Step 4: Rebuild + full tree**

```bash
pixi run -e dev maturin develop --release
pixi run -e dev pytest tests -q --basetemp=$(pwd)/.pytest_tmp
pixi run -e dev ruff check python/ tests/
pixi run -e dev typecheck
```
Expected: full tree green; no `import numba` remains (`rtk grep -rn "import numba\|@nb\.\|@numba\.\|nb.prange" python/` → no matches).
- [ ] **Step 5: Commit.**

---

### Task B4: Drop numba/llvmlite deps; import-guard; Stage-B gate

**Files:**
- Modify: `pyproject.toml` (remove `numba>=…`; remove `@nb.njit`/`@numba.njit` coverage exclusions; remove the `parity: byte-identical numba-vs-rust` marker description if it names numba), `pixi.toml` (remove `numba = "==0.59.1"` from the py310 feature and any other env).
- Create: `tests/parity/test_import_no_numba.py`.

**RELAXED GUARD (user decision 2026-06-27):** `import genvarloader` still pulls numba+llvmlite transitively via seqpro 0.20.0 (eager numba import in seqpro itself), which genvarloader cannot control. So the guard asserts genvarloader's OWN source is numba-free (achievable + verified), NOT the whole import graph. A seqpro follow-up issue tracks the eager import (it blocks the full W6 RSS drop).

- [ ] **Step 1: Write the own-code import-guard test**

```python
# tests/parity/test_import_no_numba.py
"""genvarloader's OWN modules must not import numba (Phase 5 W5).

NOTE: `import genvarloader` may still pull numba transitively via seqpro
(seqpro 0.20.0 eagerly imports numba). That is outside genvarloader's control;
this guard asserts genvarloader's own source is numba-free. See the seqpro
follow-up issue for the transitive import and the W6 RSS impact.
"""
from __future__ import annotations

import pathlib

import genvarloader


def test_genvarloader_own_code_imports_no_numba():
    pkg_dir = pathlib.Path(genvarloader.__file__).parent
    offenders: list[str] = []
    for py in pkg_dir.rglob("*.py"):
        for ln, line in enumerate(py.read_text().splitlines(), 1):
            s = line.strip()
            if s.startswith("import numba") or s.startswith("from numba"):
                offenders.append(f"{py.relative_to(pkg_dir)}:{ln}: {s}")
    assert not offenders, "genvarloader modules import numba:\n" + "\n".join(offenders)
```

- [ ] **Step 2: Run it (expect PASS — B3 already removed all numba from genvarloader), then drop genvarloader's DIRECT numba dep**

Run: `pixi run -e dev pytest tests/parity/test_import_no_numba.py -q --basetemp=$(pwd)/.pytest_tmp` → PASS.
Then remove genvarloader's OWN `numba` dependency from `pyproject.toml` and `pixi.toml` (genvarloader no longer uses it directly). NOTE: numba will likely remain INSTALLED in the env because seqpro depends on it — that is expected and fine; the own-code guard does not require numba to be absent from the environment. Re-solve (`pixi install`) and confirm the env still builds. Do NOT remove numba if doing so breaks the seqpro dependency solve — if seqpro pins numba, just remove genvarloader's direct declaration and leave the transitive one.

- [ ] **Step 3: Full tree + guard gate**

```bash
pixi run -e dev maturin develop --release
pixi run -e dev pytest tests -q --basetemp=$(pwd)/.pytest_tmp
pixi run -e dev cargo test --release
```
Expected: full tree green; import-guard PASS; cargo green.
- [ ] **Step 4: Commit the delete-numba stage boundary.**

```bash
rtk git commit -am "feat: delete numba backend — rust-only read path (Phase 5 W5)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

# STAGE C — Rayon batch parallelism

> Each kernel gains a `parallel: bool`; the serial branch is the byte-identity reference. Gate every kernel: `serial == parallel` and both `== golden`.

### Task C1: Parallelize `reconstruct_haplotypes_from_sparse`

**Files:**
- Modify: `src/reconstruct/mod.rs` (the `for k in 0..n_work` loop, lines 312-388), `src/ffi/mod.rs` (the FFI wrappers that call it — add a `parallel` arg, thread it through the 4 fused entries), the Python callers in `_haps.py`/`_reconstruct.py`/`_genotypes.py` (pass `should_parallelize(total_out_bytes)`).
- Test: `tests/parity/test_rayon_equivalence.py` (new) — serial vs parallel byte-identity over the frozen goldens.

**Interfaces:**
- The core fn gains `parallel: bool`. Use the `get_reference` idiom: pre-carve the three output buffers (`out`, optional `annot_v_idxs`, optional `annot_ref_pos`) into disjoint per-`k` chunks via `split_at_mut` chains, then `chunks.into_par_iter().enumerate().for_each(...)`. **Do not** move raw `*mut` pointers into the closure — carve `&mut [_]` slices (which are `Send`).

- [ ] **Step 1: Write the failing rayon-equivalence test**

```python
# tests/parity/test_rayon_equivalence.py
"""Serial vs parallel rust output must be byte-identical (and == golden)."""
from __future__ import annotations
import numpy as np
import pytest
from tests.parity import _golden

pytestmark = pytest.mark.parity


def test_reconstruct_haplotypes_serial_eq_parallel():
    cases = _golden.load_golden("reconstruct_haplotypes_from_sparse")
    fn = _golden.RUST_KERNELS["reconstruct_haplotypes_from_sparse"]
    for ci, (inputs, golden) in enumerate(cases):
        outs = {}
        for parallel in (False, True):
            out = np.zeros(golden.shape, golden.dtype)
            args = list(inputs)
            args.insert(0, out)
            fn(*args, parallel=parallel)  # signature gains keyword `parallel`
            outs[parallel] = out
        np.testing.assert_array_equal(outs[False], outs[True], err_msg=f"case {ci}")
        np.testing.assert_array_equal(outs[True], golden, err_msg=f"case {ci} vs golden")
```

(If the FFI signature passes `parallel` positionally, adjust the call. Decide the FFI arg convention and keep it consistent across kernels.)

- [ ] **Step 2: Run — expect FAIL** (`parallel` kwarg not accepted yet).
- [ ] **Step 3: Implement** the `parallel` branch in `reconstruct_haplotypes_from_sparse` (chunk-carve the 3 buffers, `into_par_iter`), thread `parallel` through `src/ffi/mod.rs` (the bare entry + the 4 fused entries that wrap the core), and pass `should_parallelize(...)` from the Python callers. `use rayon::prelude::*;` is already imported in `reference/mod.rs`; add it to `reconstruct/mod.rs`.
- [ ] **Step 4: Rebuild + run** the new test + the reconstruct golden + the haps dataset goldens.

```bash
pixi run -e dev maturin develop --release
pixi run -e dev cargo test --release reconstruct
pixi run -e dev pytest tests/parity -q --basetemp=$(pwd)/.pytest_tmp
```
Expected: PASS (serial==parallel==golden).
- [ ] **Step 5: Commit.**

---

### Task C2: Parallelize the track kernels

**Files:**
- Modify: `src/tracks/mod.rs` (`shift_and_realign_tracks_sparse` outer `for query` loop at 470; `tracks_to_intervals` Pass 1 @569 and Pass 2 @615 — parallelize each pass, keep the sequential cumsum between), `src/ffi/mod.rs` (+ `intervals_and_realign_track_fused`), Python callers (`_reconstruct.py`, `_intervals.py`).
- Test: extend `test_rayon_equivalence.py` with `shift_and_realign_tracks_sparse` and `tracks_to_intervals`.

- [ ] **Step 1:** Add serial-vs-parallel cases for both kernels (load their goldens, run `parallel` False/True, assert equal + == golden).
- [ ] **Step 2:** Implement `parallel` in each, using the chunk-carve idiom (outer-query parallelism). For `tracks_to_intervals`, parallelize Pass 1 and Pass 2 independently; the cumsum stays serial.
- [ ] **Step 3: Rebuild + run** the new cases + track goldens + `cargo test --release tracks`.
- [ ] **Step 4: Commit.**

---

### Task C3: Parallelize `get_diffs_sparse` + `intervals_to_tracks`

**Files:**
- Modify: `src/genotypes/mod.rs` (`get_diffs_sparse` outer `for query` @27), `src/intervals.rs` (`intervals_to_tracks` `for query` @45), FFI + Python callers.
- Test: extend `test_rayon_equivalence.py`.

- [ ] **Step 1–4:** Same recipe: add serial-vs-parallel golden cases, implement `parallel` (outer-query par; `get_diffs_sparse` writes disjoint `diffs[[query,hap]]` cells — carve per-query or use a parallel row iterator over the 2D array), rebuild, run goldens + `cargo test --release`, commit.

(`get_reference` is already parallel — no work.)

---

### Task C4: Roadmap + Stage-C gate

**Files:**
- Modify: `docs/roadmaps/rust-migration.md` (tick W5/W6/W7 tasks; add a dated Notes entry: numba deleted, golden snapshot scheme, rayon kernels; set Phase 5 marker — leave 🚧 until PR6/W8-W9 measure-and-merge; record PR placeholder for backfill).

- [ ] **Step 1: Full-tree final gate**

```bash
pixi run -e dev maturin develop --release
pixi run -e dev pytest tests -q --basetemp=$(pwd)/.pytest_tmp
pixi run -e dev cargo test --release
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format --check python/ tests/
pixi run -e dev typecheck
pixi run -e dev cargo clippy --release
```
Expected: all green; import-guard green; serial==parallel across all kernels.
- [ ] **Step 2:** Update the roadmap; commit the rayon stage boundary.

```bash
rtk git commit -am "perf(rust): rayon batch parallelism, gated byte-identical (Phase 5 W5)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

- **Spec coverage:** (a) golden snapshot → Tasks A1–A5 (infra, generate, convert all 3 mechanisms, gate, no-`_dispatch` proof). (b) delete numba → B1–B4 (dispatch, conditionals, kernels+imports, deps+import-guard). (c) rayon → C1–C4 (reconstruct, tracks, diffs/intervals, gate). The "neither numba nor llvmlite imported" assertion is B4. The `parallel:bool`+`RAYON_NUM_THREADS` gating is C1 + B3's `_threads.py`.
- **Placeholder scan:** the per-kernel `SPEC` list in A2 and the "convert remaining tests" steps are data-driven repetitions of a fully-shown pattern (DRY), not placeholders — each names the exact strategy, shape, and replay helper. The rust kernel bodies in Stage C are referenced by file:line with the canonical `get_reference` idiom shown verbatim, rather than transcribed (they are 80+ lines and would go stale).
- **Type consistency:** `RUST_KERNELS` (name→callable), `collect_examples`/`save_golden`/`load_golden`, and the four `replay_*` helpers are defined in A1 and consumed unchanged in A3–A5 and C1–C3. `should_parallelize`/`cap_threads`/`num_threads` defined in B3 and consumed in C1–C3. `parallel: bool` FFI convention chosen in C1 and reused in C2–C3.
- **Risks flagged for the controller:** (1) `RUST_KERNELS` has a few Python-wrapper kernels (`assemble_variant_buffers`, possibly `get_reference`/`shift_and_realign_tracks`/`reconstruct_haplotypes_from_sparse`) whose `rust=` is not a bare extension symbol — the implementer must ground-truth each against its `register()` call. (2) `collect_examples` determinism depends on the pinned hypothesis version; goldens are regenerated only intentionally. (3) Stage B's RC-branch collapse is the parity-critical step — the strand/spliced/annotated dataset goldens are its gate. (4) Rayon `Send`: carve `&mut [_]` slices, never raw `*mut` in the closure.
