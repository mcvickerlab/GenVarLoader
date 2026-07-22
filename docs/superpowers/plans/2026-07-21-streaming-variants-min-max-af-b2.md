# Streaming variants `min_af`/`max_af` (Wave B PR-B2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `min_af`/`max_af` allele-frequency filtering to streaming `StreamingDataset.with_seqs("variants")` — full support on the SVAR1 backend, byte-identical to the written oracle; VCF/PGEN reproduce the written `RuntimeError` guard.

**Architecture:** AF filtering is effectively SVAR-only on the written path (`gvl.write` never computes AF; VCF needs a pre-built `info=["AF"]` index, PGEN has no INFO). So SVAR1 gains a real filter (an optional global `afs` table folded into `Svar1Backend::generate_variants`' existing per-variant inline keep), while VCF/PGEN just raise the same AF-missing guard the written path raises. Everything downstream of the keep (`assemble_variants_window`, FFI, `RaggedVariants` packing) is unchanged.

**Tech Stack:** Rust (PyO3/ndarray) for the SVAR1 kernel; Python for the `StreamingDataset` surface + guards; genoray `SparseVar.cache_afs()` for AF in test data; pytest + cargo test.

**Design spec:** `docs/superpowers/specs/2026-07-21-streaming-variants-min-max-af-b2-design.md`

## Global Constraints

- **Target branch:** `streaming` (not `main`) — streaming-coordination rules, `CLAUDE.md`.
- **Byte-identical parity** with the written `Dataset` variants path at `jitter=0` is the correctness gate for SVAR1.
- **Guard/error surfaces must match the written path exactly** — same exception types and messages: AF-missing → `RuntimeError("Either this dataset is not backed by an SVAR file, or the SVAR file has not had AFs cached yet.")` (`_haps.py:334-340`); AF + non-variants output → `NotImplementedError` (`_haps.py:707-709`).
- **Inline-fold** the AF keep (no separate compact pass) — byte-identical because AF keep is a pure per-variant predicate.
- **No new public exported symbol** — only two new keyword args on `StreamingDataset.with_settings`. No `api.md`/`__all__` change.
- **Rebuild Rust before Python tests:** after any `src/` edit run `pixi run -e dev maturin develop --release`, else pytest imports the stale extension (`CLAUDE.md`).
- **`cargo test` needs libpython on the link path:** `LD_LIBRARY_PATH=$PWD/.pixi/envs/dev/lib` (memory: cargo-test-libpython-ldpath). Prefer `pixi run -e dev cargo test` which sets the env via activation.
- **Commit style:** conventional commits; end messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Docs-only commits may use `--no-verify` (pre-commit hooks run the full slow suite); code commits should pass hooks.
- **VCF/PGEN scope caveat:** PR-B2 raises the AF-missing guard for *all* VCF/PGEN AF requests. The pre-built-AF-`.gvi` VCF edge (where the written path would filter) is **#319**, not this PR.

---

### Task 1: `StreamingDataset.with_settings(min_af, max_af)` params + storage

**Files:**
- Modify: `python/genvarloader/_dataset/_streaming.py` (class fields ~142-150; `with_settings` ~1009-1053; the `with_settings`/copy propagation list ~282-288)
- Test: `tests/dataset/test_streaming_settings.py` (create, or add to the nearest existing streaming-settings test module if one exists — check `tests/dataset/` first)

**Interfaces:**
- Produces: `StreamingDataset._min_af: float | None` and `._max_af: float | None` (default `None`); `StreamingDataset.with_settings(*, jitter=None, rng=None, deterministic=None, min_af=None, max_af=None) -> StreamingDataset`.

- [ ] **Step 1: Write the failing test**

```python
# tests/dataset/test_streaming_settings.py
import numpy as np
import genvarloader as gvl


def test_with_settings_min_max_af_stored(streaming_case):
    regions, reference, variants, _written = streaming_case("svar1")
    sds = gvl.StreamingDataset(regions, reference=reference, variants=variants)
    assert sds._min_af is None and sds._max_af is None

    out = sds.with_settings(min_af=0.1, max_af=0.9)
    # new instance carries the settings; original is unchanged (immutability)
    assert out._min_af == 0.1 and out._max_af == 0.9
    assert sds._min_af is None and sds._max_af is None
    # unrelated settings preserved through the copy
    assert out._jitter == sds._jitter
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_settings.py::test_with_settings_min_max_af_stored -v`
Expected: FAIL — `AttributeError: 'StreamingDataset' object has no attribute '_min_af'`.

- [ ] **Step 3: Add the class fields**

In `_streaming.py`, next to the existing settings fields (after `_deterministic: bool = True`, ~line 150):

```python
    _min_af: "float | None" = None
    _max_af: "float | None" = None
```

- [ ] **Step 4: Propagate through the internal copy list**

In the `__init__` block that copies settings fields (~282-288), add `"_min_af"` and `"_max_af"` to the list of attribute names copied from a source instance so `with_settings`/subset views preserve them:

```python
        for _name in (
            "_seq_kind",
            "_out_len",   # keep whatever names already exist in this tuple
            "_jitter",
            "_rng",
            "_deterministic",
            "_min_af",
            "_max_af",
        ):
```

(Only add the two new names; keep the existing entries exactly as they are.)

- [ ] **Step 5: Extend `with_settings`**

In `with_settings` (~1009), add the two keyword params and the setattr wiring, mirroring the `jitter` handling:

```python
    def with_settings(
        self,
        *,
        jitter: "int | None" = None,
        rng: "int | np.random.Generator | None" = None,
        deterministic: "bool | None" = None,
        min_af: "float | None" = None,
        max_af: "float | None" = None,
    ) -> "StreamingDataset":
        ...
        out = copy.copy(self)
        if jitter is not None:
            if jitter < 0:
                raise ValueError(f"jitter must be non-negative, got {jitter}.")
            object.__setattr__(out, "_jitter", int(jitter))
        if rng is not None:
            object.__setattr__(out, "_rng", rng)
        if deterministic is not None:
            object.__setattr__(out, "_deterministic", bool(deterministic))
        if min_af is not None:
            object.__setattr__(out, "_min_af", float(min_af))
        if max_af is not None:
            object.__setattr__(out, "_max_af", float(max_af))
        return out
```

Also add `min_af`/`max_af` to the docstring `Parameters` block (one line each: "Inclusive allele-frequency lower/upper bound for `with_seqs('variants')` output; requires a cached `AF` (SVAR + `cache_afs()`), else raises at iterate time. Matches `Dataset.with_settings`.").

- [ ] **Step 6: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_settings.py::test_with_settings_min_max_af_stored -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add python/genvarloader/_dataset/_streaming.py tests/dataset/test_streaming_settings.py
git commit -m "feat(streaming): with_settings(min_af, max_af) params + storage (PR-B2, #317)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Output-mode guard — `NotImplementedError` for AF + non-variants output

**Files:**
- Modify: `python/genvarloader/_dataset/_streaming.py` (`_iter_batches` fail-fast guard block, ~464-500)
- Test: `tests/dataset/test_streaming_settings.py`

**Interfaces:**
- Consumes: `self._min_af`/`self._max_af` (Task 1), `_variants = self._seq_kind is RaggedVariants` (already computed at `_streaming.py:464`).

- [ ] **Step 1: Write the failing test**

```python
def test_af_filter_rejects_non_variants_output(streaming_case):
    regions, reference, variants, _written = streaming_case("svar1")
    sds = (
        gvl.StreamingDataset(regions, reference=reference, variants=variants)
        .with_seqs("haplotypes")
        .with_settings(min_af=0.1)
    )
    with pytest.raises(NotImplementedError, match="AF"):
        next(iter(sds.to_iter(batch_size=4)))
```

(Add `import pytest` to the test module.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_settings.py::test_af_filter_rejects_non_variants_output -v`
Expected: FAIL — no exception raised (or wrong type).

- [ ] **Step 3: Add the guard**

In `_iter_batches`, immediately after `_variants = self._seq_kind is RaggedVariants` (~line 464) and before the SVAR2 fail-fast block:

```python
            # PR-B2 (#317): AF filtering is a variants-output-only feature on the
            # written path (`_haps.py:707-709` raises NotImplementedError for
            # haplotype/annotated). Mirror that so streaming and written agree.
            _af_filter = self._min_af is not None or self._max_af is not None
            if _af_filter and not _variants:
                raise NotImplementedError(
                    "min_af/max_af filtering is only supported for "
                    'with_seqs("variants") output (matching the written Dataset, '
                    "which raises for haplotype/annotated output)."
                )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_settings.py::test_af_filter_rejects_non_variants_output -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/genvarloader/_dataset/_streaming.py tests/dataset/test_streaming_settings.py
git commit -m "feat(streaming): guard min_af/max_af to variants output only (PR-B2, #317)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Rust — `Svar1Backend` AF table + keep-fold + `#[new]` params

**Files:**
- Modify: `src/ffi/stream_engine.rs` — `Svar1Backend` struct (`:109-141`); `generate_variants` (`:257-315`); `Svar1StreamEngine::build` (`:331-368`); `#[new]` pyclass constructor + `#[pyo3(signature=...)]` (`:394-432`)
- Test: `src/ffi/stream_engine.rs` (add a `#[cfg(test)] mod` unit test, or extend an existing one in the file)

**Interfaces:**
- Produces (Rust-facing): `Svar1Backend` fields `afs: Option<Array1<f32>>`, `min_af: Option<f32>`, `max_af: Option<f32>`; `Svar1StreamEngine::new` gains trailing pyO3 params `afs: Option<PyReadonlyArray1<f32>>`, `min_af: Option<f32>`, `max_af: Option<f32>` (all default `None`). Python (Task 4) passes these.

- [ ] **Step 1: Write the failing Rust unit test**

Add to the test module in `src/ffi/stream_engine.rs`. Construct a minimal `Svar1Backend` directly (mirror the fields an existing test in this file builds; if none, build the struct literal) with a hand-made global variant table and `afs`, then assert `generate_variants` composes region + AF keeps as AND. Example shape:

```rust
    #[test]
    fn svar1_generate_variants_af_and_region_compose_as_and() {
        // 3 global variants at POS 10, 20, 30 (0-based starts 9,19,29), all SNPs (ilen 0).
        // AF = [0.05, 0.5, 0.95]. Region [15, 25) keeps only variant #2 by position.
        // With min_af=0.1, max_af=0.9: variant #2 (AF 0.5) survives both;
        //   variant #1 (AF .05) fails AF, variant #3 (AF .95) fails AF AND region.
        let backend = make_test_backend(
            /* v_starts */ &[9, 19, 29],
            /* ilens */ &[0, 0, 0],
            /* alt bytes+offsets for "A","C","G" */ ..,
            /* afs */ Some(&[0.05f32, 0.5, 0.95]),
            /* min_af */ Some(0.1),
            /* max_af */ Some(0.9),
        );
        // one region [15,25), one sample, ploidy 1, CSR carrying all 3 globals in the hap
        let batch = backend.generate_variants(0, &make_window_all_three(), 0, 1).unwrap();
        assert_eq!(batch.start.as_slice().unwrap(), &[19]); // only variant #2's start
        assert_eq!(batch.ilen.as_slice().unwrap(), &[0]);
        assert_eq!(batch.row_offsets.as_slice().unwrap(), &[0, 1]);
    }
```

If the file has no existing `Svar1Backend` unit-construction helper, write `make_test_backend`/`make_window_all_three` as local test helpers building the struct literal + a `FilledWindow` with `o_starts=[0]`, `o_stops=[3]`, and a store stub yielding `geno_v_idxs()==[0,1,2]`. (Match how the existing `#[test]` fns in this file stub the store — read them first; reuse their helper if present.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev cargo test svar1_generate_variants_af_and_region_compose_as_and 2>&1 | tail -20`
Expected: FAIL to compile — `afs`/`min_af`/`max_af` fields don't exist yet.

- [ ] **Step 3: Add the fields to `Svar1Backend`**

In the struct (`:109-141`), alongside the global tables:

```rust
    v_starts: Array1<i32>,
    ilens: Array1<i32>,
    alt_alleles: Array1<u8>,
    alt_offsets: Array1<i64>,
    /// PR-B2 (#317): optional per-variant global allele frequency, parallel to
    /// `v_starts`. `Some` only when the source `.svar` had AF cached
    /// (`SparseVar.cache_afs`) AND AF filtering was requested. Consumed by
    /// `generate_variants`' inline keep. `None` => no AF filter applied here.
    afs: Option<Array1<f32>>,
    min_af: Option<f32>,
    max_af: Option<f32>,
```

- [ ] **Step 4: Fold the AF keep into `generate_variants`**

In `generate_variants` (`:257-315`), replace the inner keep body:

```rust
            for o in os..oe {
                let gvi = geno[o];
                let v_start = self.v_starts[gvi as usize] as i64;
                let v_ilen = self.ilens[gvi as usize] as i64;
                let v_end = v_start - v_ilen.min(0) + 1;
                let region_keep = v_start < r_e as i64 && v_end > r_s as i64;
                let af_keep = match &self.afs {
                    Some(afs) => {
                        let af = afs[gvi as usize];
                        self.min_af.map_or(true, |m| af >= m)
                            && self.max_af.map_or(true, |m| af <= m)
                    }
                    None => true,
                };
                if region_keep && af_keep {
                    kept.push(gvi);
                }
            }
```

- [ ] **Step 5: Thread through `Svar1StreamEngine::build`**

In `build` (`:331-368`), add params and set the fields. Add to the signature after `variants: bool`:

```rust
        afs: Option<Array1<f32>>,
        min_af: Option<f32>,
        max_af: Option<f32>,
```

and in the `Svar1Backend { ... }` literal:

```rust
            variants,
            afs,
            min_af,
            max_af,
```

- [ ] **Step 6: Thread through the `#[new]` pyclass constructor**

In the `#[new] fn new(...)` (`:394-432`): add trailing pyO3 params and to the `#[pyo3(signature = (...))]` list (with defaults so old callers are unaffected):

```rust
    #[pyo3(signature = (
        // ... existing args unchanged ...
        annotated=false,
        variants=false,
        afs=None,
        min_af=None,
        max_af=None,
    ))]
    fn new(
        // ... existing params ...
        annotated: bool,
        variants: bool,
        afs: Option<PyReadonlyArray1<f32>>,
        min_af: Option<f32>,
        max_af: Option<f32>,
    ) -> PyResult<Self> {
        // near where the other PyReadonlyArray1 args are converted to owned Array1:
        let afs = afs.map(|a| a.as_array().to_owned());
        // ... existing body ...
        // pass afs, min_af, max_af into Svar1StreamEngine::build(...)
    }
```

Pass `afs, min_af, max_af` at the `Svar1StreamEngine::build(...)` call inside `new`.

- [ ] **Step 7: Update the two other `build` call sites (Rust tests / `new_rs`)**

`Svar1StreamEngine::build` is also reachable from `new_rs`/existing tests. Grep and fix every call to pass the three new args (existing callers pass `None, None, None`):

```bash
grep -n "Svar1StreamEngine::build(\|\.build(" src/ffi/stream_engine.rs
```

Add `, None, None, None` (afs/min_af/max_af) to each pre-existing call so they compile unchanged in behavior.

- [ ] **Step 8: Run the unit test to verify it passes**

Run: `pixi run -e dev cargo test svar1_generate_variants_af_and_region_compose_as_and 2>&1 | tail -20`
Expected: PASS.

- [ ] **Step 9: Run the full Rust suite to confirm no regressions**

Run: `pixi run -e dev cargo test 2>&1 | tail -20`
Expected: all pass (existing `generate_variants` parity tests still green — `afs=None` is a no-op).

- [ ] **Step 10: Commit**

```bash
git add src/ffi/stream_engine.rs
git commit -m "feat(streaming): SVAR1 generate_variants folds min_af/max_af into keep (PR-B2, #317)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: SVAR1 `_afs` loading + `has_cached_af` + `build_engine` threading

**Files:**
- Modify: `python/genvarloader/_dataset/_streaming.py` — `_Svar1Backend.__init__` (`:1148-1293`, esp. after `idx = sv.index.sort("index")` at `:1200`); `_Svar1Backend.build_engine` (`:1295`); the `_iter_batches` `build_engine(...)` call site (`:567`)
- Test: `tests/dataset/test_streaming_settings.py`

**Interfaces:**
- Consumes: Rust `Svar1StreamEngine(..., afs=, min_af=, max_af=)` (Task 3), `self._min_af`/`_max_af` (Task 1).
- Produces: `_Svar1Backend._afs: NDArray[np.float32] | None`; `_Svar1Backend.has_cached_af: bool`; extended `build_engine(..., min_af=None, max_af=None)` shared signature.

- [ ] **Step 1: Write the failing test (AF actually filters on a cached SVAR1)**

```python
import shutil
from genoray import SparseVar


def _af_cached_svar(streaming_case, tmp_path):
    """Copy the svar1 case's .svar into tmp, cache AFs, return (regions, ref, svar, af_array)."""
    regions, reference, variants, _ = streaming_case("svar1")
    dst = tmp_path / "af.svar"
    shutil.copytree(variants, dst)
    sv = SparseVar(str(dst))
    sv.cache_afs()
    af = sv.index.sort("index")["AF"].to_numpy()
    return regions, reference, str(dst), af


def test_svar1_min_af_actually_filters(streaming_case, tmp_path):
    regions, reference, svar, af = _af_cached_svar(streaming_case, tmp_path)
    thr = float(np.median(af))  # guarantees some in, some out

    sds = gvl.StreamingDataset(regions, reference=reference, variants=svar)
    base = sds.with_seqs("variants")
    filt = base.with_settings(min_af=thr)

    def total(ds):
        n = 0
        for data, r_idx, _s in ds.to_iter(batch_size=4):
            for k in range(len(r_idx)):
                for h in range(ds.ploidy):
                    n += np.atleast_1d(np.asarray(data[k].alt[h])).shape[0]
        return n

    n_base, n_filt = total(base), total(filt)
    assert n_base > 0
    assert n_filt < n_base  # the filter removed variants
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_settings.py::test_svar1_min_af_actually_filters -v`
Expected: FAIL — filter is ignored, `n_filt == n_base` (or `TypeError`/`AttributeError` from the not-yet-wired path).

- [ ] **Step 3: Load `_afs` in `_Svar1Backend.__init__`**

After `idx = sv.index.sort("index")` (~`:1200`):

```python
        # PR-B2 (#317): optional cached allele frequency, parallel to _v_starts
        # (global variant order). Present iff the .svar had AFs cached
        # (genoray SparseVar.cache_afs). Read here so build_engine can hand it to
        # the Rust engine; None keeps the engine's AF filter a no-op.
        if "AF" in idx.collect_schema().names():
            object.__setattr__(
                self, "_afs", idx["AF"].to_numpy().astype(np.float32, copy=False)
            )
        else:
            object.__setattr__(self, "_afs", None)
```

(Use `object.__setattr__` only if `_Svar1Backend` is a frozen dataclass — match the surrounding assignments; if it uses plain `self._afs = ...`, do that instead. Read the surrounding `__init__` lines to see which pattern applies.)

- [ ] **Step 4: Add the `has_cached_af` property**

On `_Svar1Backend`:

```python
    @property
    def has_cached_af(self) -> bool:
        return self._afs is not None
```

- [ ] **Step 5: Extend `_Svar1Backend.build_engine`**

Add `min_af=None, max_af=None` to the signature and pass AF through to the `Svar1StreamEngine(...)` constructor. Only pass `afs` when filtering is actually requested (keeps the engine no-op otherwise):

```python
    def build_engine(self, jobs, batch_size, output_length,
                     annotated=False, variants=False,
                     min_af=None, max_af=None):
        from ..genvarloader import Svar1StreamEngine
        ...
        _af_arg = self._afs if (min_af is not None or max_af is not None) else None
        engine = Svar1StreamEngine(
            # ... all existing positional/keyword args unchanged ...
            annotated,
            variants,
            afs=_af_arg,
            min_af=min_af,
            max_af=max_af,
        )
        return engine
```

- [ ] **Step 6: Pass `min_af`/`max_af` at the `_iter_batches` call site**

At the `engine = backend.build_engine(engine_jobs, batch_size, _out_len, _annotated, _variants)` call (`:567`):

```python
                engine = backend.build_engine(
                    engine_jobs, batch_size, _out_len, _annotated, _variants,
                    min_af=self._min_af, max_af=self._max_af,
                )
```

- [ ] **Step 7: Rebuild the Rust extension**

Run: `pixi run -e dev maturin develop --release 2>&1 | tail -5`
Expected: builds and installs (Task 3's Rust must be compiled into the extension before pytest).

- [ ] **Step 8: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_settings.py::test_svar1_min_af_actually_filters -v`
Expected: PASS (`n_filt < n_base`).

- [ ] **Step 9: Commit**

```bash
git add python/genvarloader/_dataset/_streaming.py tests/dataset/test_streaming_settings.py
git commit -m "feat(streaming): SVAR1 loads cached AF + threads min_af/max_af to engine (PR-B2, #317)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: AF-missing guard (`RuntimeError` parity) across all backends

**Files:**
- Modify: `python/genvarloader/_dataset/_streaming.py` — add `has_cached_af` to `_VcfBackend`/`_PgenBackend` (return `False`); AF-missing guard in `_iter_batches` (after the Task-2 output-mode guard); extend `_VcfBackend.build_engine`/`_PgenBackend.build_engine` signatures to accept+ignore `min_af`/`max_af`
- Test: `tests/dataset/test_streaming_settings.py`

**Interfaces:**
- Consumes: `backend.has_cached_af` (Task 4 for SVAR1; this task for VCF/PGEN), `self._min_af`/`_max_af`.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("backend", ["svar1", "vcf", "pgen"])
def test_af_missing_guard_raises(streaming_case, backend):
    # svar1 here is the UN-cached fixture (no cache_afs) -> no AF, like vcf/pgen
    regions, reference, variants, _written = streaming_case(backend)
    sds = (
        gvl.StreamingDataset(regions, reference=reference, variants=variants)
        .with_seqs("variants")
        .with_settings(min_af=0.1)
    )
    with pytest.raises(RuntimeError, match="AFs cached"):
        next(iter(sds.to_iter(batch_size=4)))


def test_af_present_svar1_does_not_raise(streaming_case, tmp_path):
    regions, reference, svar, _af = _af_cached_svar(streaming_case, tmp_path)
    sds = (
        gvl.StreamingDataset(regions, reference=reference, variants=svar)
        .with_seqs("variants")
        .with_settings(min_af=0.1)
    )
    # cached AF -> iterates fine
    _ = next(iter(sds.to_iter(batch_size=4)))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_settings.py -k "af_missing_guard or af_present_svar1" -v`
Expected: `test_af_missing_guard_raises` FAILs (no RuntimeError, or wrong error from build_engine getting an unexpected kwarg for VCF/PGEN).

- [ ] **Step 3: Add `has_cached_af` to the record backends**

On both `_VcfBackend` and `_PgenBackend`:

```python
    @property
    def has_cached_af(self) -> bool:
        # PR-B2 (#317): streaming VCF/PGEN cannot supply AF (gvl.write never
        # computes it; PGEN has no INFO). AF filtering for VCF source INFO is #319.
        return False
```

- [ ] **Step 4: Accept-and-ignore `min_af`/`max_af` in the record `build_engine`s**

Extend `_VcfBackend.build_engine` (`:1858`) and `_PgenBackend.build_engine` (`:1999`) signatures so the uniform call site (Task 4 Step 6) type-checks. They ignore the values (the guard below prevents reaching here with AF filtering active):

```python
    def build_engine(self, jobs, batch_size, output_length,
                     annotated=False, variants=False,
                     min_af=None, max_af=None):
        # min_af/max_af unused: VCF/PGEN AF is guarded out in _iter_batches (#319).
        ...  # body unchanged
```

- [ ] **Step 5: Add the AF-missing guard in `_iter_batches`**

Right after the Task-2 output-mode guard, once `backend` is known (the record-backend branch has `backend` in scope; place the guard where both `_af_filter` and `backend` are available — e.g. immediately before the `build_engine` call at `:567`, applying to the record backends). Use the exact written message:

```python
                # PR-B2 (#317): AF-missing guard, byte-for-byte the written path's
                # (_haps.py:334-340). SVAR1 without cache_afs, and all VCF/PGEN,
                # report has_cached_af == False.
                if _af_filter and not backend.has_cached_af:
                    raise RuntimeError(
                        "Either this dataset is not backed by an SVAR file, or the "
                        "SVAR file has not had AFs cached yet."
                    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_settings.py -k "af_missing_guard or af_present_svar1" -v`
Expected: all PASS (svar1-uncached/vcf/pgen raise `RuntimeError`; cached svar1 iterates).

- [ ] **Step 7: Commit**

```bash
git add python/genvarloader/_dataset/_streaming.py tests/dataset/test_streaming_settings.py
git commit -m "feat(streaming): AF-missing RuntimeError guard parity (svar1/vcf/pgen) (PR-B2, #317)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Byte-identical SVAR1 AF parity vs the written oracle

**Files:**
- Modify: `tests/dataset/test_streaming_variants_parity.py` (add an AF-parity test reusing `_assert_variants_cell_matches`)
- Test: same file

**Interfaces:**
- Consumes: `genoray.SparseVar.cache_afs()`, `Dataset.with_settings(min_af=, max_af=)`, `StreamingDataset.with_settings(min_af=, max_af=)`, `_assert_variants_cell_matches` (already in the module).

- [ ] **Step 1: Write the failing parity test**

```python
import shutil
from genoray import SparseVar


@pytest.mark.parametrize(
    "min_af,max_af",
    [(0.3, None), (None, 0.7), (0.2, 0.8)],
)
def test_streaming_svar1_af_matches_written(streaming_case, tmp_path, min_af, max_af):
    """SVAR1 min_af/max_af streaming output is byte-identical to the written oracle."""
    regions, reference, variants, _ = streaming_case("svar1")

    # Cache AFs into a copy of the .svar, then write a fresh oracle from it so the
    # written variants.arrow carries the same AF column the streaming path reads.
    svar = tmp_path / "af.svar"
    shutil.copytree(variants, svar)
    SparseVar(str(svar)).cache_afs()

    out = tmp_path / "ds"
    gvl.write(out, regions, variants=str(svar), overwrite=True)
    written = gvl.Dataset.open(out, reference=reference)

    ds = written.with_seqs("variants").with_settings(min_af=min_af, max_af=max_af)
    sds = (
        gvl.StreamingDataset(regions, reference=reference, variants=str(svar))
        .with_seqs("variants")
        .with_settings(min_af=min_af, max_af=max_af)
    )

    seen, total = set(), 0
    for data, r_idx, s_idx in sds.to_iter(batch_size=4):
        for k in range(len(r_idx)):
            r, s = int(r_idx[k]), int(s_idx[k])
            total += _assert_variants_cell_matches(data[k], ds[r, s], sds.ploidy)
            seen.add((r, s))
    assert seen == {(r, s) for r in range(ds.shape[0]) for s in range(ds.shape[1])}
    # Non-vacuous: at least one variant survived, and the filter removed some vs
    # unfiltered (guards against a threshold that keeps/drops everything).
    unfiltered = written.with_seqs("variants")
    n_unf = sum(
        np.atleast_1d(np.asarray(unfiltered[r, s].alt[h])).shape[0]
        for r in range(ds.shape[0]) for s in range(ds.shape[1])
        for h in range(sds.ploidy)
    )
    assert 0 < total < n_unf
```

If `write`/`Dataset.open` on an AF-cached `.svar` does not surface AF in `variants.info` (verify during implementation via `gvl.Dataset.open(out,...)`'s `Haps.variants.info` keys), fall back to opening the written dataset and confirming `"AF"` is among `_Variants.available_info_fields(out / "variants.arrow")`; if the written link path drops AF, adjust the oracle to read the `.svar` directly. (This is the one integration point to confirm early — see spec §5.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_variants_parity.py::test_streaming_svar1_af_matches_written -v`
Expected: FAIL only if a wiring bug remains; if Tasks 1-5 are correct it may already PASS — in that case tighten the thresholds so `0 < total < n_unf` genuinely exercises a partial filter, and treat a green run as the deliverable.

- [ ] **Step 3: Make it pass**

No new production code expected (Tasks 1-5 implement the behavior). If parity fails, debug against the written oracle: compare `ds[r,s].start`/`.alt`/`.ilen` to the streamed cell for the first mismatching `(r,s)`; the likely culprits are (a) AF dtype/order mismatch (`_afs` must be global order, `float32`), or (b) the written oracle not carrying AF (fix the fixture per Step 1's note).

- [ ] **Step 4: Run the full variants-parity module**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_variants_parity.py -v`
Expected: all PASS (PR-B1's no-filter parity + the new AF parity).

- [ ] **Step 5: Commit**

```bash
git add tests/dataset/test_streaming_variants_parity.py
git commit -m "test(streaming): byte-identical SVAR1 min_af/max_af parity vs written (PR-B2, #317)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Docs + roadmap update

**Files:**
- Modify: `skills/genvarloader/SKILL.md`; `docs/source/dataset.md`; `docs/source/faq.md`; `docs/roadmaps/streaming-dataset.md`

**Interfaces:** none (docs only).

- [ ] **Step 1: SKILL.md** — in the `StreamingDataset.with_settings` coverage, note `min_af`/`max_af` are now accepted for `with_seqs("variants")` on SVAR1 sources with cached AFs (`SparseVar.cache_afs()`); VCF/PGEN raise the same AF-missing `RuntimeError` as the written path. Match the written `Dataset.with_settings` wording.

- [ ] **Step 2: dataset.md / faq.md** — add a short streaming AF-filtering note: SVAR-only, requires `cache_afs()`, variants output only; cross-reference the written path's identical guard.

- [ ] **Step 3: roadmap** — in `docs/roadmaps/streaming-dataset.md`, tick PR-B2, update the Wave B status line (PR-B2 done; PR-B3/PR-B4 remain), and link this design + #317 + #319 (VCF live-INFO AF follow-up). Record the SVAR-only scope decision.

- [ ] **Step 4: Verify api.md needs no change**

Run: `pixi run -e dev python -c "import re,genvarloader as g; api=open('docs/source/api.md').read(); print('MISSING:', [n for n in g.__all__ if n not in api] or 'none')"`
Expected: `MISSING: none` (no new exported symbol).

- [ ] **Step 5: Commit**

```bash
git add skills/genvarloader/SKILL.md docs/source/dataset.md docs/source/faq.md docs/roadmaps/streaming-dataset.md
git commit -m "docs(streaming): document SVAR1 min_af/max_af variants filtering (PR-B2, #317)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: Full-suite verification + PR

**Files:** none (verification + PR).

- [ ] **Step 1: Rebuild + full Python tree**

Run: `pixi run -e dev maturin develop --release && pixi run -e dev pytest tests -q 2>&1 | tail -25`
Expected: all pass (scoped runs skip `tests/unit/` — run the whole tree per `CLAUDE.md`).

- [ ] **Step 2: Full Rust suite**

Run: `pixi run -e dev cargo test 2>&1 | tail -15`
Expected: all pass.

- [ ] **Step 3: Lint + typecheck + docstring style**

Run:
```bash
pixi run -e dev ruff check python/ tests/
pixi run -e dev ruff format --check python/ tests/
pixi run -e dev typecheck
pixi run -e dev python scripts/docstring_style.py --check python/genvarloader
```
Expected: clean.

- [ ] **Step 4: Push + open draft PR into `streaming`**

```bash
git push -u origin spec/streaming-b2-minmaxaf
gh pr create --draft --base streaming \
  --title "feat(streaming): variants min_af/max_af — Wave B PR-B2 (#317)" \
  --body "$(cat <<'EOF'
Streaming `with_seqs("variants")` `min_af`/`max_af` filtering (**Wave B PR-B2**, #317).

- **SVAR1: full support.** Optional global `afs` table on `Svar1Backend`, folded into `generate_variants`' existing per-variant inline keep; byte-identical to the written oracle at `jitter=0` (parity via `SparseVar.cache_afs()`).
- **VCF/PGEN: guard-parity.** Reproduce the written `RuntimeError` AF-missing guard (AF is SVAR-only on the written path today — `gvl.write` never computes AF; PGEN has no INFO). VCF live-INFO AF is deferred to #319.
- **Guards mirror the written path exactly:** AF + non-variants output → `NotImplementedError`; AF without cached AF → `RuntimeError`.

Closes #317. Relates to #304, #319. Design: `docs/superpowers/specs/2026-07-21-streaming-variants-min-max-af-b2-design.md`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 5: Add the PR to the StreamingDataset project board and set #317 → review**

```bash
gh project item-add 1 --owner mcvickerlab --url <PR_URL>
```

(Move #317 to the board's in-review/Done state per the team's convention once the PR is up.)

## Self-Review

**1. Spec coverage:**
- SVAR1 full support → Tasks 3-4, 6. ✓
- VCF/PGEN guard-parity → Task 5. ✓
- `with_settings(min_af, max_af)` surface → Task 1. ✓
- Output-mode guard (NotImplementedError) → Task 2. ✓
- AF-missing guard (RuntimeError) → Task 5. ✓
- Inline-fold keep → Task 3. ✓
- Byte-identical parity + non-vacuity → Task 6. ✓
- Rust unit (AND composition) → Task 3. ✓
- Docs (SKILL/dataset/faq/roadmap) + api.md no-op → Task 7. ✓
- #319 follow-up referenced → Tasks 5, 7, 8. ✓

**2. Placeholder scan:** Rust test helpers in Task 3 Step 1 (`make_test_backend`/`make_window_all_three`) are described as "build the struct literal / match existing test stubs" — this is a real instruction (read the file's existing `#[test]` helpers first) rather than hidden work, because the exact stub shape depends on how the file's current tests construct a store; flagged explicitly. Task 6 Step 1's AF-surfacing fallback is a genuine early-verify integration point (spec §5), not a TODO.

**3. Type consistency:** `_afs` is `NDArray[np.float32] | None` (Python) ↔ `afs: Option<Array1<f32>>` (Rust) ↔ `afs: Option<PyReadonlyArray1<f32>>` (`#[new]`), consistent across Tasks 3-4. `has_cached_af: bool` defined identically on all three backends (Tasks 4-5). `build_engine(..., min_af=None, max_af=None)` shared signature across all three backends (Tasks 4-5) and the single call site (Task 4 Step 6). `min_af`/`max_af` are `float | None` end to end.
