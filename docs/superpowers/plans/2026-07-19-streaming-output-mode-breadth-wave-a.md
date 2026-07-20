# Streaming Output-Mode Breadth — Wave A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Per this project's CLAUDE.md, use **Sonnet or weaker** models for implementation (reserve stronger models for second-pass fixes only).

**Goal:** Add `with_len`, read-time jitter, and `with_seqs("annotated")` to the write-free `StreamingDataset` on the merged SVAR1/VCF/PGEN backends, byte-identical to `gvl.write()` + `Dataset[r, s]` under matching settings (jitter=0).

**Architecture:** Both stream engines (`Svar1StreamEngine`, `RecordStreamEngine`) reconstruct via one shared Rust core `generate_batch_core` (`src/ffi/mod.rs:981-1081`), currently ragged-only, annotation-less, no `output_length`. Wave A extends that core once (length + optional annotation + optional global-variant-base offset), threads a stored per-engine config through both engine constructors and their `generate` call sites, and adds the Python config surface + jitter (read-window widening + per-region rng translate, Python-side). #300 is resolved as doc corrections + genuine same-POS fixtures (the ordering already matches by construction at the pinned genoray rev).

**Tech Stack:** Python (frozen dataclass API, numpy), Rust (PyO3/`maturin`, `ndarray`), genoray git-dep (unchanged — no rev bump), pytest + `vcfixture` fixtures.

## Global Constraints

- **Parity oracle:** byte-identical vs `gvl.write()` + `Dataset.open()[r, s]` under matching `with_len`/`with_seqs`/`deterministic`, at **jitter=0**. This is the standing contract (`docs/archive/roadmaps/rust-migration.md`).
- **Rebuild Rust before Python tests:** after ANY edit under `src/`, run `pixi run -e dev maturin develop --release` before `pytest`, or pytest imports the stale extension (CLAUDE.md).
- **Run both test dirs when touching shared code:** `pixi run -e dev pytest tests/dataset tests/unit -q`. Before the final push, run the full tree `pixi run -e dev pytest tests -q`.
- **No genoray change / no rev bump** (Development Notes, CLAUDE.md). Wave A is entirely gvl-side.
- **Target branch:** `streaming` (work on `spec/277-output-mode-wave-a`). Draft PR into `streaming`, not `main`.
- **`to_iter` always yields `Ragged`.** No `with_output_format`; `with_len("variable")` is NOT accepted (only `int | "ragged"`).
- **rng parameter name is `rng`** (`int | np.random.Generator | None`), matching `Dataset.with_settings` — not `seed`.
- **Public-API + docs gates:** any new public knob updates `docs/source/{api,faq,dataset}.md`, `README.md` where relevant, and `skills/genvarloader/SKILL.md` (CLAUDE.md). Update `docs/roadmaps/streaming-dataset.md` and the StreamingDataset project board.
- **Lint/format/type:** `pixi run -e dev ruff check python/ tests/`, `ruff format python/ tests/`, `pixi run -e dev typecheck`; `cargo clippy` clean for Rust. Install prek hooks before committing.
- **Commit style:** conventional commits (commitizen). Co-author trailer per environment rules.

---

## File Structure

**Python (`python/genvarloader/_dataset/_streaming.py`)** — the single file holding `StreamingDataset` + the three backends. Wave A adds:
- New frozen fields: `_seq_kind`, `_output_length`, `_jitter`, `_rng`, `_deterministic`.
- New methods: `with_len`, `with_settings`; extended `with_seqs`.
- `_iter_batches` / backend `build_engine` / `generate_batch` gain `output_length`, annotated-mode, and jitter-widened regions.

**Rust:**
- `src/ffi/mod.rs` — extend `generate_batch_core` (length + annotation + global-base); the SVAR1 `svar1_generate_batch` wrapper.
- `src/ffi/stream_engine.rs` — `Svar1StreamEngine`: stored config (`output_length`, `annotated`), `next_batch` return, `Svar1Backend::generate`.
- `src/record_stream/engine.rs` — `RecordStreamEngine`: same stored config + `RecordBackend::generate`.
- `src/record_stream/transpose.rs` — `DecodedWindow` gains a `var_base: i64` field (global variant base).
- `src/record_stream/pgen.rs` — store the already-computed `var_start` onto the window.
- `src/record_stream/vcf.rs` — compute + store the window's global variant base.

**Tests:**
- `tests/dataset/test_streaming_with_len.py` (new) — fixed/ragged length parity, 3 backends.
- `tests/dataset/test_streaming_jitter.py` (new) — jitter property tests.
- `tests/dataset/test_streaming_annotated_parity.py` (new) — annotated parity, 3 backends.
- `tests/dataset/test_streaming_vcf_parity.py` + `conftest.py` (modify) — #300 doc fixes + new same-POS fixtures.

**Docs:** `docs/source/{api,faq,dataset}.md`, `README.md`, `skills/genvarloader/SKILL.md`, `docs/roadmaps/streaming-dataset.md`.

## Task ordering & parallelism (honest)

`with_len` and `annotated` both funnel through `generate_batch_core` + both engine call sites, so they are **not** cleanly parallel. Critical path: **T1 → T2 → T4 → T5 → T6**. **T3 (jitter)** is largely Python-side and can run in parallel with T4 **only with awareness that both edit `_streaming.py`** — prefer landing T2, then T3 and T4 sequentially (or T3∥T4 with a rebase discipline). Use subagent-driven-development; dispatch T3∥T4 only if the executor can serialize the `_streaming.py` merge.

---

### Task 1: Python config surface (foundation, no behavior change)

Adds the knobs and frozen fields; defaults exactly preserve today's output. No Rust. This unblocks every later task by defining the fields they read.

**Files:**
- Modify: `python/genvarloader/_dataset/_streaming.py` (dataclass fields ~48-127; `with_seqs` 493-503; add `with_len`/`with_settings`; `__init__` 129-…)
- Test: `tests/dataset/test_streaming_config.py` (new)

**Interfaces:**
- Consumes: existing `StreamingDataset` frozen dataclass, `copy.copy`.
- Produces (later tasks rely on these exact names/types):
  - Fields: `_seq_kind: type` (default `RaggedSeqs`), `_output_length: int | Literal["ragged"]` (default `"ragged"`), `_jitter: int` (default `0`), `_rng: int | np.random.Generator | None` (default `None`), `_deterministic: bool` (default `True`).
  - `with_len(self, length: int | Literal["ragged"]) -> StreamingDataset`
  - `with_settings(self, *, jitter: int | None = None, rng: int | np.random.Generator | None = None, deterministic: bool | None = None) -> StreamingDataset`
  - `with_seqs(self, kind: Literal["haplotypes", "annotated"]) -> StreamingDataset`

- [ ] **Step 1: Write the failing test**

Create `tests/dataset/test_streaming_config.py`:

```python
import numpy as np
import pytest

import genvarloader as gvl
from genvarloader._ragged import RaggedSeqs
from genvarloader._dataset._streaming import StreamingDataset


def _tiny_sds() -> StreamingDataset:
    # Internal/test construction path (no real variant source): inject a
    # reconstruct callback so we exercise the config surface only.
    regions = np.array([[0, 0, 10], [0, 10, 20]], dtype=np.int32)
    return StreamingDataset(
        regions,
        contigs=["chr1"],
        n_samples=2,
        ploidy=2,
        _reconstruct_window=lambda r, s: None,
    )


def test_defaults_preserve_current_behavior():
    sds = _tiny_sds()
    assert sds._seq_kind is RaggedSeqs
    assert sds._output_length == "ragged"
    assert sds._jitter == 0
    assert sds._rng is None
    assert sds._deterministic is True


def test_with_len_sets_output_length_and_copies():
    sds = _tiny_sds()
    out = sds.with_len(200)
    assert out is not sds
    assert out._output_length == 200
    assert sds._output_length == "ragged"  # original unchanged
    assert sds.with_len("ragged")._output_length == "ragged"


def test_with_len_rejects_variable_and_nonpositive():
    sds = _tiny_sds()
    with pytest.raises((ValueError, NotImplementedError)):
        sds.with_len("variable")  # no streaming analog
    with pytest.raises(ValueError):
        sds.with_len(0)


def test_with_settings_sets_jitter_rng_deterministic():
    sds = _tiny_sds().with_settings(jitter=4, rng=0, deterministic=False)
    assert sds._jitter == 4
    assert sds._rng == 0
    assert sds._deterministic is False


def test_with_seqs_accepts_annotated_rejects_variants():
    sds = _tiny_sds()
    from genvarloader._ragged import RaggedAnnotatedHaps

    assert sds.with_seqs("annotated")._seq_kind is RaggedAnnotatedHaps
    assert sds.with_seqs("haplotypes")._seq_kind is RaggedSeqs
    with pytest.raises(NotImplementedError):
        sds.with_seqs("variants")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_config.py -q`
Expected: FAIL (`AttributeError: _seq_kind` / `with_len` not defined).

- [ ] **Step 3: Add the frozen fields**

In the dataclass body (after `_prefetch_strategy`, ~line 127) add:

```python
    # --- Wave A (#277) output-mode config; defaults preserve pre-Wave-A behavior ---
    # Sequence output kind: RaggedSeqs (haplotypes) | RaggedAnnotatedHaps (annotated).
    _seq_kind: type = RaggedSeqs
    # Output length: "ragged" (per-hap actual length) or a fixed int >= 1.
    _output_length: "int | str" = "ragged"
    # Read-time jitter (0 = deterministic, byte-parity gate).
    _jitter: int = 0
    # rng seed/Generator for jitter + fixed-length shifts (matches Dataset.with_settings).
    _rng: "int | np.random.Generator | None" = None
    # Deterministic: disables random within-window shifts for fixed-length output.
    _deterministic: bool = True
```

Add imports at the top with the other `_ragged` imports:

```python
from .._ragged import RaggedSeqs, RaggedAnnotatedHaps
```

(Confirm `RaggedSeqs` import path against the existing import block; `_reconstruct.py` uses `from .._ragged import RaggedSeqs, RaggedAnnotatedHaps`.)

- [ ] **Step 4: Wire `__init__` to set the new fields from defaults**

`StreamingDataset.__init__` uses `object.__setattr__` (frozen dataclass). After the existing field assignments, initialize the new fields from their dataclass defaults so construction paths (public + injected) both populate them:

```python
        for _name in ("_seq_kind", "_output_length", "_jitter", "_rng", "_deterministic"):
            object.__setattr__(
                self, _name, type(self).__dataclass_fields__[_name].default
            )
```

Then wire the existing `jitter` constructor kwarg to `_jitter` (replacing the `NotImplementedError` guard at ~line 171 only in Task 3; for Task 1 keep the guard but also set `_jitter=0`). For Task 1, leave the `jitter != 0` guard intact.

- [ ] **Step 5: Implement `with_len`, `with_settings`, extend `with_seqs`**

Replace the existing `with_seqs` (493-503) and add the two new methods:

```python
    def with_seqs(
        self, kind: Literal["haplotypes", "annotated"]
    ) -> "StreamingDataset":
        """Select the sequence output kind. ``"haplotypes"`` (default) or
        ``"annotated"`` (:class:`AnnotatedHaps` -- haplotypes plus per-position
        variant indices and reference coordinates). ``"variants"`` /
        ``"variant-windows"`` / ``"reference"`` are Wave B / later plans."""
        kind_map = {"haplotypes": RaggedSeqs, "annotated": RaggedAnnotatedHaps}
        if kind not in kind_map:
            raise NotImplementedError(
                f"StreamingDataset.with_seqs({kind!r}) is not implemented; "
                'only "haplotypes" and "annotated" are supported in Wave A. '
                '"variants"/"variant-windows" are Wave B (#304); "reference" is later.'
            )
        out = copy.copy(self)
        object.__setattr__(out, "_seq_kind", kind_map[kind])
        return out

    def with_len(self, length: "int | Literal['ragged']") -> "StreamingDataset":
        """Set haplotype/annotated output length. ``"ragged"`` (default) yields
        per-hap actual length; a fixed ``int >= 1`` yields exactly that many bases
        per hap. Unlike :meth:`Dataset.with_len`, ``"variable"`` is not accepted:
        :meth:`to_iter` always yields ``Ragged`` (there is no ArrayDataset analog),
        so pad the ragged output yourself for a dense array."""
        if length == "variable":
            raise NotImplementedError(
                'StreamingDataset.with_len("variable") is not supported; to_iter '
                'always yields Ragged. Use with_len(int) or with_len("ragged").'
            )
        if length != "ragged":
            if not isinstance(length, (int, np.integer)) or int(length) < 1:
                raise ValueError(
                    f"with_len(length) must be a positive int or 'ragged', got {length!r}."
                )
            length = int(length)
        out = copy.copy(self)
        object.__setattr__(out, "_output_length", length)
        return out

    def with_settings(
        self,
        *,
        jitter: "int | None" = None,
        rng: "int | np.random.Generator | None" = None,
        deterministic: "bool | None" = None,
    ) -> "StreamingDataset":
        """Modify jitter / rng / determinism, returning a new dataset. Mirrors the
        relevant subset of :meth:`Dataset.with_settings` (same parameter names).
        ``jitter>0`` is a documented, reproducible augmentation, NOT byte-parity
        with a written ``Dataset`` (see :meth:`to_iter`)."""
        out = copy.copy(self)
        if jitter is not None:
            if jitter < 0:
                raise ValueError(f"jitter must be non-negative, got {jitter}.")
            object.__setattr__(out, "_jitter", int(jitter))
        if rng is not None:
            object.__setattr__(out, "_rng", rng)
        if deterministic is not None:
            object.__setattr__(out, "_deterministic", bool(deterministic))
        return out
```

Ensure `Literal` is imported (it is — used by the current `with_seqs`).

- [ ] **Step 6: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_config.py -q`
Expected: PASS (all 5).

- [ ] **Step 7: Verify no regression on existing streaming tests**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_parity.py tests/dataset/test_streaming_scale.py tests/unit -q`
Expected: PASS (defaults unchanged → identical output).

- [ ] **Step 8: Lint + commit**

```bash
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/
git add python/genvarloader/_dataset/_streaming.py tests/dataset/test_streaming_config.py
git commit -m "feat(streaming): with_len/with_settings/with_seqs(annotated) config surface (#277)"
```

---

### Task 2: Fixed-length output through the shared core + both engines

Threads `output_length` (and per-hap `shifts`) through `generate_batch_core`, both engine constructors, both `*Backend::generate` call sites, and the Python `build_engine`/`generate_batch`. After this task, `sds.with_len(L)` yields byte-parity fixed-length haplotypes for all three backends.

**Files:**
- Modify: `src/ffi/mod.rs` (`generate_batch_core` 981-1081; `svar1_generate_batch` 1090-1149)
- Modify: `src/ffi/stream_engine.rs` (`Svar1StreamEngine::new` 277-311; `Svar1Backend::generate` 161-209)
- Modify: `src/record_stream/engine.rs` (`RecordStreamEngine::new` 272-296; `RecordBackend::generate` 140-206)
- Modify: `python/genvarloader/_dataset/_streaming.py` (all three `build_engine`; `_Svar1Backend.generate_batch`; `_iter_batches` engine/readahead output-offset shaping)
- Test: `tests/dataset/test_streaming_with_len.py` (new)

**Interfaces:**
- Consumes: Task 1 fields (`_output_length`, `_deterministic`).
- Produces:
  - `generate_batch_core(..., output_length: i64, shifts: Option<ArrayView2<i32>>, ...)` — `output_length=-1` ragged (current behavior), `>=0` fixed.
  - Engine constructors gain a trailing `output_length: i64` parameter (after `batch_size`); stored on the engine and passed to `generate`.
  - Python: `build_engine(jobs, batch_size, output_length)` and `generate_batch(..., output_length)`.

**Reference:** the length logic to mirror is in `reconstruct_haplotypes_fused` (`src/ffi/mod.rs:804-810`): `output_length >= 0 ? output_length : (ref_len + diff).max(0)`. The current ragged-only computation to REPLACE is `generate_batch_core` `src/ffi/mod.rs:1041-1052`.

- [ ] **Step 1: Write the failing parity test**

Create `tests/dataset/test_streaming_with_len.py`. Use the existing streaming parity fixtures/helpers (see `tests/dataset/test_streaming_parity.py` and `test_streaming_vcf_parity.py` for the fixture names and the `gvl.write`→`Dataset.open` oracle pattern; reuse the SVAR1, VCF, and PGEN fixtures those files use):

```python
import numpy as np
import pytest

import genvarloader as gvl

# Parametrize over the three merged backends via the same fixtures the existing
# parity tests use. Replace the fixture names below with the actual ones in
# conftest.py (e.g. the svar1/vcf/pgen streaming fixtures used by
# test_streaming_parity.py / test_streaming_vcf_parity.py).
BACKENDS = ["svar1", "vcf", "pgen"]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("length", [16, 32])
def test_fixed_length_matches_written(streaming_case, backend, length):
    # streaming_case: a fixture returning (regions, reference, variants_path,
    # written_dataset) for `backend`. See conftest.py.
    regions, reference, variants, written = streaming_case(backend)

    ds = written.with_len(length).with_seqs("haplotypes")
    sds = (
        gvl.StreamingDataset(regions, reference=reference, variants=variants)
        .with_len(length)
        .with_seqs("haplotypes")
    )

    for data, r_idx, s_idx in sds.to_iter(batch_size=4):
        got = data  # Ragged[S1], shape (batch, ploidy, length)
        for row in range(len(r_idx)):
            exp = ds[int(r_idx[row]), int(s_idx[row])]  # written oracle
            np.testing.assert_array_equal(
                got[row].to_padded(b"N"), np.asarray(exp).reshape(sds.ploidy, -1)
            )
            # fixed length: every hap is exactly `length`
            assert got[row].to_padded(b"N").shape[-1] == length
```

> Note to implementer: the exact oracle-comparison shape mirrors `test_streaming_parity.py::test_streaming_matches_written_all_cells`. Copy that test's comparison helper rather than reinventing it; only add the `with_len(length)` + fixed-length assertion.

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_with_len.py -q`
Expected: FAIL — `with_len(L)` currently has no effect on streaming output (still ragged actual length), so the fixed-length shape assertion fails.

- [ ] **Step 3: Extend `generate_batch_core` (Rust) with `output_length` + optional `shifts`**

In `src/ffi/mod.rs`, change `generate_batch_core`'s signature to add (before `parallel: bool`):

```rust
    output_length: i64,               // -1 = ragged, >=0 = fixed
    shifts: Option<ndarray::ArrayView2<i32>>,  // None => zeros (deterministic)
```

Replace the ragged-only offset loop (`src/ffi/mod.rs:1041-1052`) with length-aware offsets mirroring `reconstruct_haplotypes_fused`:

```rust
    let mut out_offsets_vec: Array1<i64> = Array1::zeros(n_work + 1);
    {
        let mut acc: i64 = 0;
        for k in 0..n_work {
            let query = k / ploidy;
            let hap = k % ploidy;
            let per: i64 = if output_length >= 0 {
                output_length
            } else {
                let ref_len = (regions_arr[[query, 2]] - regions_arr[[query, 1]]) as i64;
                let diff = diffs[[query, hap]] as i64;
                (ref_len + diff).max(0)
            };
            acc += per;
            out_offsets_vec[k + 1] = acc;
        }
    }
```

Replace the hardwired `shifts_arr` construction (currently zeros) so it uses the passed `shifts` when `Some`, else zeros. Locate the existing `shifts_arr` local in `generate_batch_core` and set:

```rust
    let shifts_owned;
    let shifts_view = match shifts {
        Some(s) => s,
        None => {
            shifts_owned = Array2::<i32>::zeros((n_work / ploidy, ploidy));
            shifts_owned.view()
        }
    };
```

and pass `shifts_view` into `reconstruct_haplotypes_from_sparse` in place of the current zeros arg. (Keep the `None, None` annotation args for now — annotation is Task 4.)

- [ ] **Step 4: Update `svar1_generate_batch` wrapper + SVAR1 engine + record engine call sites**

`svar1_generate_batch` (`src/ffi/mod.rs:1090`): add `output_length: i64` param (before `parallel`) and forward it (with `shifts=None`) to `generate_batch_core`.

`Svar1Backend::generate` (`src/ffi/stream_engine.rs:194-208`): add `self.output_length` and `None` to the `generate_batch_core` call. Add an `output_length: i64` field to the backend/core-config struct set in `Svar1StreamEngine::new`, and a trailing `output_length: i64` constructor param (after `batch_size`, extend the `#[pyo3(signature = ...)]` list).

`RecordBackend::generate` (`src/record_stream/engine.rs:191-205`): identical change — pass `self.output_length` + `None`; add the field + constructor param to `RecordStreamEngine::new` (`engine.rs:272-296`).

- [ ] **Step 5: Thread `output_length` through the Python backends**

In `_streaming.py`, change all three `build_engine` signatures to `build_engine(self, jobs, batch_size, output_length)` and pass `output_length` as the new trailing engine constructor arg. Change `_Svar1Backend.generate_batch` to accept `output_length` and forward it to `svar1_generate_batch`. In `_iter_batches`, resolve the effective length once:

```python
        _out_len = -1 if self._output_length == "ragged" else int(self._output_length)
```

and pass `_out_len` to `build_engine(..., _out_len)` (engine branch) and to `generate_batch(..., _out_len)` (readahead branch). The Ragged output shaping in `_iter_batches` is unchanged — offsets from the engine already encode the fixed length.

- [ ] **Step 6: Rebuild Rust and run the parity test**

Run: `pixi run -e dev maturin develop --release`
Then: `pixi run -e dev pytest tests/dataset/test_streaming_with_len.py -q`
Expected: PASS for all 3 backends × {16, 32}.

- [ ] **Step 7: Regression + ragged still default**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_parity.py tests/dataset/test_streaming_scale.py -q`
Expected: PASS (ragged default → `output_length=-1` → identical to before).

- [ ] **Step 8: Clippy, lint, commit**

```bash
cargo clippy --manifest-path Cargo.toml 2>&1 | tail -5
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/
git add src/ python/genvarloader/_dataset/_streaming.py tests/dataset/test_streaming_with_len.py
git commit -m "feat(streaming): fixed-length output (with_len) through both stream engines (#277)"
```

---

### Task 3: Read-time jitter

Widens the read window by ±jitter (contig-clamped) and translates each region's query by a per-region rng draw. Non-deterministic fixed-length also draws per-hap `shifts`. Python-side; reuses Task 2's `shifts` param. Property-tested (not byte-parity).

**Files:**
- Modify: `python/genvarloader/_dataset/_streaming.py` (`__init__` jitter guard removal; `_iter_batches` region widening + translation + rng; pass `shifts` to `generate_batch`/engine)
- Modify: `src/ffi/stream_engine.rs` / `src/record_stream/engine.rs` **only if** per-region jitter offset must be applied inside the engine (see design note below); prefer Python-side region translation to avoid Rust changes.
- Test: `tests/dataset/test_streaming_jitter.py` (new)

**Interfaces:**
- Consumes: Task 1 (`_jitter`, `_rng`, `_deterministic`), Task 2 (`output_length`, engine `shifts` path).
- Produces: jittered regions are computed in Python and passed as the region bounds the engine/`generate_batch` already consume — no new Rust API if translation happens before the regions cross into Rust.

**Design note (implementer):** The engine constructors take `job_region_starts`/`job_region_ends` (u32) per window. Jitter is a pure translation of those bounds plus a wider *read*. Because the SVAR1 engine resolves variant ranges from `v_starts_c/v_ends_c` per contig and the record engines decode by region bounds, translating the region bounds in Python (and letting the read cover the widened span) keeps jitter entirely Python-side. Validate `output_length <= region_len` at `with_len`/`with_settings` composition time (a jittered fixed window must fit the base region).

- [ ] **Step 1: Write failing property tests**

Create `tests/dataset/test_streaming_jitter.py`:

```python
import numpy as np
import pytest

import genvarloader as gvl


@pytest.mark.parametrize("backend", ["svar1", "vcf", "pgen"])
def test_jitter_reproducible_same_rng(streaming_case, backend):
    regions, reference, variants, _ = streaming_case(backend)
    mk = lambda: gvl.StreamingDataset(
        regions, reference=reference, variants=variants
    ).with_settings(jitter=8, rng=0, deterministic=False)
    a = [np.asarray(d.to_padded(b"N")) for d, *_ in mk().to_iter(batch_size=4)]
    b = [np.asarray(d.to_padded(b"N")) for d, *_ in mk().to_iter(batch_size=4)]
    assert len(a) == len(b)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("backend", ["svar1", "vcf", "pgen"])
def test_jitter_different_rng_differs(streaming_case, backend):
    regions, reference, variants, _ = streaming_case(backend)
    mk = lambda seed: gvl.StreamingDataset(
        regions, reference=reference, variants=variants
    ).with_settings(jitter=8, rng=seed, deterministic=False)
    a = np.concatenate(
        [np.asarray(d.to_padded(b"N")).ravel() for d, *_ in mk(0).to_iter(4)]
    )
    b = np.concatenate(
        [np.asarray(d.to_padded(b"N")).ravel() for d, *_ in mk(1).to_iter(4)]
    )
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("backend", ["svar1", "vcf", "pgen"])
def test_jitter_fixed_length_output_shape(streaming_case, backend):
    regions, reference, variants, _ = streaming_case(backend)
    sds = (
        gvl.StreamingDataset(regions, reference=reference, variants=variants)
        .with_len(16)
        .with_settings(jitter=8, rng=0, deterministic=False)
    )
    for data, *_ in sds.to_iter(batch_size=4):
        assert data.to_padded(b"N").shape[-1] == 16
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_jitter.py -q`
Expected: FAIL — construction with `jitter>0` currently raises `NotImplementedError` (`_streaming.py:171`).

- [ ] **Step 3: Remove the jitter guard; wire jitter into `__init__`**

Delete the `if jitter != 0: raise NotImplementedError(...)` block (`_streaming.py:171-175`) and set `object.__setattr__(self, "_jitter", int(jitter))` in the public path.

- [ ] **Step 4: Implement region widening + per-region translation in `_iter_batches`**

Add a helper that, given the run's `Generator` and a window's `r_idx`, returns translated region bounds and (for non-deterministic fixed output) per-hap `shifts`:

```python
    def _rng_gen(self) -> np.random.Generator:
        return np.random.default_rng(self._rng)
```

In `_iter_batches`, when `self._jitter > 0`, draw `jitter_off` per region **in sweep order** from a single `Generator` created once per `to_iter` call, translate the `self._regions[r_idx, 1:3]` bounds by the per-region offset (clamped to `[0, contig_len]` via `self._ref` contig lengths), and use the translated bounds when building `engine_jobs` / calling `generate_batch`. For non-deterministic fixed-length output, compute `shifts` (per the spec: `max_shift = diffs.clip(min=0) + (lengths - output_length).clip(min=0)`) and pass through the engine `shifts` path.

> Implementer: keep the `Generator` creation OUTSIDE the plan loop so draws are deterministic in sweep order; document the "per-region draw in sweep order" contract in the `to_iter` docstring. If the engine path cannot accept per-window `shifts` without a constructor change, apply jitter (translation) via region bounds (no Rust change) and gate `deterministic=False` fixed-length `shifts` behind the readahead/`generate_batch` path first, filing engine-`shifts` support as a follow-up if needed. Prefer: translation-only jitter with `deterministic=True` fixed length works with zero Rust change; document any deferral.

- [ ] **Step 5: Rebuild (if Rust touched) + run jitter tests**

Run: `pixi run -e dev maturin develop --release` (only if `src/` changed)
Then: `pixi run -e dev pytest tests/dataset/test_streaming_jitter.py -q`
Expected: PASS.

- [ ] **Step 6: Confirm jitter=0 still byte-parity**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_parity.py tests/dataset/test_streaming_with_len.py -q`
Expected: PASS (jitter defaults to 0 → no translation).

- [ ] **Step 7: Document the rng contract + commit**

Add the rng-contract paragraph to the `to_iter` and `with_settings` docstrings (per-region draw in sweep order; reproducible per `rng`; translation-only; NOT byte-parity with `Dataset`). Then:

```bash
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/
git add python/genvarloader/_dataset/_streaming.py tests/dataset/test_streaming_jitter.py src/ 2>/dev/null
git commit -m "feat(streaming): read-time jitter with documented rng contract (#277)"
```

---

### Task 4: Annotated output (`AnnotatedHaps`) with global `var_idxs`

Adds annotated reconstruction to both engines: emit `annot_v_idxs` + `annot_ref_pos`, return a 4-tuple, and (for VCF/PGEN) map window-local column indices to dataset-global ids via a window `var_base`.

**Files:**
- Modify: `src/ffi/mod.rs` (`generate_batch_core`: add optional annotation output + `var_base: i64`)
- Modify: `src/record_stream/transpose.rs` (`DecodedWindow` + `var_base: i64` field; set it in `fill_decoded_window` or its callers)
- Modify: `src/record_stream/pgen.rs` (store the computed `var_start` onto the window)
- Modify: `src/record_stream/vcf.rs` (compute + store the window's global variant base)
- Modify: `src/ffi/stream_engine.rs`, `src/record_stream/engine.rs` (annotated `next_batch` path / mode flag; 4-tuple return)
- Modify: `python/genvarloader/_dataset/_streaming.py` (`_iter_batches` builds `AnnotatedHaps` when `_seq_kind is RaggedAnnotatedHaps`)
- Test: `tests/dataset/test_streaming_annotated_parity.py` (new)

**Interfaces:**
- Consumes: Task 2 (`generate_batch_core` with `output_length`/`shifts`), Task 1 (`_seq_kind`).
- Produces:
  - `generate_batch_core(..., annotated: bool, var_base: i64, ...)` returning, when `annotated`, `(data, annot_v, annot_pos, offsets)`; else `(data, offsets)`.
  - Engines gain an `annotated: bool` config field; annotated `next_batch` returns the 4-tuple (or a separate `next_batch_annotated`).
  - `DecodedWindow.var_base: i64` = global variant index of window column 0.
  - Python yields `AnnotatedHaps` (via `RaggedAnnotatedHaps`) batches.

**Grounding facts (from extraction):** the core writes `variant as i32` (= `v_idxs[v]`) into `annot_v_idxs` (`src/reconstruct/mod.rs:296-305`). SVAR1 `geno_v_idxs` are already global → free. Record windows' `geno_v_idxs` are window-local → add `var_base` so the emitted id is `var_base + local`. PGEN's `var_start` is computed at `pgen.rs:571-593` but discarded; VCF has none.

- [ ] **Step 1: Write the failing annotated parity test**

Create `tests/dataset/test_streaming_annotated_parity.py`:

```python
import numpy as np
import pytest

import genvarloader as gvl


@pytest.mark.parametrize("backend", ["svar1", "vcf", "pgen"])
def test_annotated_matches_written(streaming_case, backend):
    regions, reference, variants, written = streaming_case(backend)
    ds = written.with_seqs("annotated")
    sds = gvl.StreamingDataset(
        regions, reference=reference, variants=variants
    ).with_seqs("annotated")

    for data, r_idx, s_idx in sds.to_iter(batch_size=4):
        # data is AnnotatedHaps-shaped ragged: haps + var_idxs + ref_coords
        for row in range(len(r_idx)):
            exp = ds[int(r_idx[row]), int(s_idx[row])]  # AnnotatedHaps
            got_haps = data.haps[row].to_padded(b"N")
            got_vidx = data.var_idxs[row].to_padded(-1)
            got_pos = data.ref_coords[row].to_padded(-1)
            np.testing.assert_array_equal(got_haps, np.asarray(exp.haps).reshape(got_haps.shape))
            np.testing.assert_array_equal(got_vidx, np.asarray(exp.var_idxs).reshape(got_vidx.shape))
            np.testing.assert_array_equal(got_pos, np.asarray(exp.ref_coords).reshape(got_pos.shape))
```

> Implementer: the exact `AnnotatedHaps` accessor shape (`.haps`/`.var_idxs`/`.ref_coords` and how a streaming annotated batch is packed) is defined by how Task-4 Step 6 packs the Python output; align the test to that. Model the oracle comparison on `test_streaming_parity.py`.

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_annotated_parity.py -q`
Expected: FAIL — `with_seqs("annotated")` sets `_seq_kind` but `_iter_batches` still produces haplotype-only output.

- [ ] **Step 3: Add annotation output + `var_base` to `generate_batch_core`**

In `src/ffi/mod.rs`, add params `annotated: bool` and `var_base: i64` to `generate_batch_core`. When `annotated`, allocate `annot_v: Array1<i32>` and `annot_pos: Array1<i32>` sized like `out_data`, pass `Some(annot_v.view_mut())`, `Some(annot_pos.view_mut())` to `reconstruct_haplotypes_from_sparse` (in place of the current `None, None`), and, when `var_base != 0`, add `var_base` to every non-negative entry of `annot_v` after reconstruction:

```rust
    if annotated && var_base != 0 {
        for x in annot_v.iter_mut() {
            if *x >= 0 { *x += var_base as i32; }
        }
    }
```

Return type becomes an enum/tuple: have `generate_batch_core` return `(Array1<u8>, Option<Array1<i32>>, Option<Array1<i32>>, Array1<i64>)`; ragged callers ignore the `None`s. (Alternatively split into `generate_batch_core` and `generate_batch_core_annotated` sharing a private helper — implementer's choice; keep one reconstruct call path.)

- [ ] **Step 4: Plumb `var_base` onto record windows**

`DecodedWindow` (`src/record_stream/transpose.rs:29-37`): add `pub var_base: i64,` (default 0). In `PgenWindowFiller::fill` (`src/record_stream/pgen.rs`), set `slot.var_base = var_start as i64;` (the value computed at `pgen.rs:571-593`, currently discarded). In `VcfWindowFiller::fill` (`src/record_stream/vcf.rs`), compute the window's global variant base = count of contig-order variants before the window's first decoded atom (use the same `.gvi`/position index the range read already consults) and set `slot.var_base`. SVAR1 needs no `var_base` (its `geno_v_idxs` are already global; pass `var_base=0`).

- [ ] **Step 5: Add annotated mode to both engines**

Add `annotated: bool` config to `Svar1StreamEngine` and `RecordStreamEngine` (constructor param after `output_length`). In `*Backend::generate`, call the annotated core when `annotated`, returning the 4 arrays; expose via `next_batch` returning a 4-tuple when annotated (or add `next_batch_annotated`). Record backend passes `slot.var_base` as `var_base`; SVAR1 passes `0`.

- [ ] **Step 6: Build `AnnotatedHaps` batches in Python**

In `_iter_batches`, when `self._seq_kind is RaggedAnnotatedHaps`, request annotated output from the engine/`generate_batch`, and assemble a `RaggedAnnotatedHaps` (haps S1 + var_idxs i32 + ref_coords i32) using the shared `out_offsets`. Pass `annotated=True` and `_out_len` to `build_engine`. Model the packing on the written path's `_FlatAnnotatedHaps`→`RaggedAnnotatedHaps` (`_haps.py:664-668`) but for the streaming ragged batch.

- [ ] **Step 7: Rebuild + run annotated parity (all 3 backends)**

Run: `pixi run -e dev maturin develop --release`
Then: `pixi run -e dev pytest tests/dataset/test_streaming_annotated_parity.py -q`
Expected: PASS for svar1/vcf/pgen (haps + var_idxs + ref_coords byte-identical).

- [ ] **Step 8: Regression + clippy + commit**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_parity.py tests/dataset/test_streaming_with_len.py tests/dataset/test_streaming_jitter.py -q`
Then:

```bash
cargo clippy --manifest-path Cargo.toml 2>&1 | tail -5
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/
git add src/ python/genvarloader/_dataset/_streaming.py tests/dataset/test_streaming_annotated_parity.py
git commit -m "feat(streaming): annotated output (AnnotatedHaps) with global var_idxs (#277, #300)"
```

---

### Task 5: #300 — correct docs + lock the file-order invariant with genuine fixtures

The streaming/write orderings match by construction (both file-order at the pinned rev). This task corrects the wrong "lexicographic" docstrings and adds fixtures where file order ≠ lexicographic order, so the invariant is tested genuinely, not coincidentally.

**Files:**
- Modify: `tests/dataset/conftest.py` (`vcf_snp_ins_del_multi` docstring 494-505 + PGEN twin 680-681; add new fixtures)
- Modify: `tests/dataset/test_streaming_vcf_parity.py` (module caveat 21-27; add same-POS annotated-var_idxs assertions)
- Create: VCF fixture data under `tests/data/streaming/` (non-lexicographic same-POS + pre-split triallelic)

**Interfaces:**
- Consumes: Task 4 (annotated `var_idxs` for VCF/PGEN).
- Produces: fixtures `vcf_same_pos_nonlex` (two rows, file order `A>T` then `A>G`) and `vcf_same_pos_triallelic` (three rows same POS, non-lexicographic file order), plus their PGEN twins (via the existing plink2 conversion pattern in conftest).

- [ ] **Step 1: Write the failing invariant test**

Add to `tests/dataset/test_streaming_vcf_parity.py`:

```python
@pytest.mark.parametrize("fixture", ["vcf_same_pos_nonlex", "vcf_same_pos_triallelic"])
def test_same_pos_var_idxs_file_order(request, fixture):
    # Annotated var_idxs must match the written .gvi (file-order) ids even when
    # file order != lexicographic ALT order — proving the agreement is by
    # construction (file-order), not coincidental lexicographic match. Resolves #300.
    regions, reference, variants, written = request.getfixturevalue(fixture)
    ds = written.with_seqs("annotated")
    sds = gvl.StreamingDataset(
        regions, reference=reference, variants=variants
    ).with_seqs("annotated")
    for data, r_idx, s_idx in sds.to_iter(batch_size=4):
        for row in range(len(r_idx)):
            exp = ds[int(r_idx[row]), int(s_idx[row])]
            np.testing.assert_array_equal(
                data.var_idxs[row].to_padded(-1),
                np.asarray(exp.var_idxs).reshape(data.var_idxs[row].to_padded(-1).shape),
            )
```

- [ ] **Step 2: Create the fixtures**

Add `vcf_same_pos_nonlex` and `vcf_same_pos_triallelic` to `tests/dataset/conftest.py`, mirroring `vcf_snp_ins_del_multi` (469-543) but with same-POS rows in **non-lexicographic file order** (e.g. pos=150 rows `A>T` then `A>G`; triallelic pos=150 `A>T`, `A>C`, `A>G`), pre-split biallelic (`bcftools norm -m -`), bgzipped + tabixed. Add PGEN twins via the existing `plink2 --bcf … --make-pgen --allow-extra-chr --output-chr chrM` pattern (conftest 667-705).

- [ ] **Step 3: Run to verify failure (or coincidental pass)**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_vcf_parity.py -k same_pos -q`
Expected: FAIL if Task 4's VCF `var_base` is wrong for non-lexicographic order; PASS proves the file-order invariant holds. (If it passes immediately, that is the invariant confirmed — keep the test as the guard.)

- [ ] **Step 4: Correct the wrong docstrings/comments**

Rewrite the `vcf_snp_ins_del_multi` docstring (`conftest.py:494-505`), its PGEN twin note (680-681), and the `test_streaming_vcf_parity.py` module caveat (21-27) to state the TRUE invariant: the streaming `ChunkAssembler` tie-breaks same-POS atoms by **file order** (`seq = record_seq<<32 | atom_ix`), the written oracle orders by `.gvi` **file-row order**, and `gvl.write` accepts only pre-split biallelic input, so the two agree **by construction** (one atom per record ⇒ `(pos, record-order)` both sides) — not coincidentally by lexicographic match. Reference #300.

- [ ] **Step 5: Run + commit**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_vcf_parity.py -q`
Expected: PASS (existing + new same-POS invariant).

```bash
git add tests/dataset/conftest.py tests/dataset/test_streaming_vcf_parity.py tests/data/streaming/
git commit -m "test(streaming): lock file-order same-POS var_idxs invariant; fix #300 docs (#277, #300)"
```

---

### Task 6: Docs, skill, roadmap, board

**Files:**
- Modify: `docs/source/faq.md`, `docs/source/dataset.md` (streaming section), `docs/source/api.md` (if any new `__all__` symbol — none expected; new methods are on existing `StreamingDataset`), `README.md` (streaming feature bullet if it enumerates output modes)
- Modify: `skills/genvarloader/SKILL.md` (StreamingDataset section: `with_len`, `with_settings`/jitter, `with_seqs("annotated")`)
- Modify: `docs/roadmaps/streaming-dataset.md` (tick #277 Wave A; link this spec/plan/PR; note Wave B #304)

- [ ] **Step 1: Update the skill**

Document the three new knobs on `StreamingDataset` in `skills/genvarloader/SKILL.md`: `with_len(int | "ragged")` (note: no `"variable"`), `with_settings(jitter=, rng=, deterministic=)` with the jitter rng-contract caveat (reproducible augmentation, not byte-parity), and `with_seqs("annotated")` → `AnnotatedHaps`. State that `variants`/`variant-windows`/`min_af`/`max_af`/`var_fields` are Wave B (#304).

- [ ] **Step 2: Update prose docs**

Add the same to `docs/source/faq.md` (a Q on streaming output modes) and `docs/source/dataset.md` (streaming subsection). Verify api.md sync:

```bash
python -c "import re,genvarloader as g; api=open('docs/source/api.md').read(); print('MISSING:', [n for n in g.__all__ if n not in api] or 'none')"
```
Expected: `MISSING: none`.

- [ ] **Step 3: Update the roadmap**

In `docs/roadmaps/streaming-dataset.md`, mark #277 Wave A, add a row linking `2026-07-19-streaming-output-mode-breadth-wave-a-design.md` + this plan + the PR, and note Wave B is #304.

- [ ] **Step 4: Full-tree test + commit**

Run: `pixi run -e dev pytest tests -q` (full tree, per CLAUDE.md before pushing shared-code changes)
Expected: PASS.

```bash
git add docs/ skills/ README.md 2>/dev/null
git commit -m "docs(streaming): document Wave A output-mode knobs + roadmap (#277)"
```

- [ ] **Step 5: Open the draft PR into `streaming`**

```bash
git push -u origin spec/277-output-mode-wave-a
gh pr create --draft --base streaming --repo mcvickerlab/GenVarLoader \
  --title "streaming: output-mode breadth Wave A — with_len, jitter, annotated (#277)" \
  --body "Implements #277 Wave A (spec: docs/superpowers/specs/2026-07-19-streaming-output-mode-breadth-wave-a-design.md). Resolves #300 (file-order invariant). Wave B (variants surface) is #304."
```
Add the PR to the StreamingDataset project board.

---

## Self-Review

**Spec coverage:**
- `with_len` → T1 (surface) + T2 (behavior). ✓
- jitter → T1 (surface) + T3 (behavior + rng contract). ✓
- `with_seqs("annotated")` → T1 (surface) + T4 (behavior, all 3 backends, global var_idxs). ✓
- `ref_coords`/`var_idxs` parity → T4. ✓
- #300 resolution (doc fix + genuine fixtures + var_idxs parity) → T5. ✓
- Config surface mirrors written Dataset, drops `"variable"`, `rng` naming → T1. ✓
- jitter=0 byte-parity gate preserved → asserted in T2/T3/T4 regression steps. ✓
- Docs + skill + roadmap/board → T6. ✓
- No genoray change / no rev bump → nothing in any task touches `Cargo.toml`. ✓

**Placeholder scan:** Test-fixture names (`streaming_case`, backend fixtures) are flagged as "align to conftest.py" rather than invented — the implementer must bind them to the real fixtures used by `test_streaming_parity.py`/`test_streaming_vcf_parity.py`. This is a deliberate pointer, not a placeholder for logic; the comparison logic is fully specified. The two Rust "implementer's choice" points (annotated as 4-tuple return vs. separate method; jitter engine-`shifts` vs. Python translation) are genuine design latitude with a stated default, not undecided logic.

**Type consistency:** `output_length` is `i64` (`-1` ragged) end-to-end (T2/T4). `var_base` is `i64` on `DecodedWindow`, added as `i32` to `annot_v` (T4). `_output_length` Python is `int | "ragged"`, converted to `_out_len: int` (`-1` for ragged) at the `_iter_batches` boundary (T2). `_rng` matches `Dataset.with_settings`'s `rng`. `_seq_kind` is `RaggedSeqs | RaggedAnnotatedHaps` throughout.
