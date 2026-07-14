# SVAR2 M6b Branch — Final Pre-Merge Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the SVAR2 read-bound branch — correctness/safety fixes, shipping hygiene, Rust de-duplication, and doc consistency — so it is mergeable modulo the (documented, unresolved) genoray release gate.

**Architecture:** Additive-only cleanup of an already-green branch. Every change is a correctness fix, a hygiene cleanup, or a doc correction. The SVAR1 path stays byte-unchanged; the SVAR2 parity gate stays 31/31. No new features, no change to the parity contract.

**Tech Stack:** Python 3.10 (numpy, polars, pyrefly, ruff), Rust (PyO3 abi3-py310, ndarray, rayon), pixi task runner, pytest + cargo test, prek pre-commit hooks.

**Spec:** `docs/superpowers/specs/2026-07-13-svar2-m6b-kernel-final-pass-design.md`

## Global Constraints

- **SVAR1 byte-unchanged:** every change is additive w.r.t. the SVAR1 path; no SVAR1 output bytes may change.
- **SVAR2 parity gate = 31/31:** `pixi run -e dev pytest tests/dataset/test_svar2_dataset.py tests/dataset/test_svar2_readbound_*.py tests/dataset/test_write_svar2.py` stays fully green after every task that touches Rust or `_dataset/` code.
- **Rebuild Rust before Python tests that import the extension:** `pixi run -e dev maturin develop --release` after ANY `src/` edit, before running pytest. `cargo test` compiles from source and does not need this.
- **`cargo test` needs libpython on the loader path:** it runs under `pixi run -e dev` which sets `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` via `pixi.toml` `[target.linux-64.activation.env]`. Always invoke cargo via `pixi run -e dev cargo ...`.
- **Lint/format gates:** `pixi run -e dev ruff check python/ tests/` and `pixi run -e dev ruff format --check python/ tests/` must pass.
- **Pre-commit hooks are broken in this worktree** (the `pyrefly-check` hook matches zero files under `.claude/` yet exits nonzero; the `pixi-lock` hook churns `pixi.lock`). Task 2 fixes the pyrefly invocation. Until Task 2 lands, commit doc/Rust-only changes with `git commit --no-verify` and NEVER stage `pixi.lock` churn (run `git checkout pixi.lock` if it appears dirty from an unrelated pixi operation). After Task 2, prefer hooks on; if `pixi-lock` still churns, `git checkout pixi.lock` before committing.
- **Commit style:** conventional commits (`fix(svar2):`, `refactor(svar2):`, `docs(svar2):`, `chore(svar2):`). End every commit message with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- **Do NOT touch the genoray path-pins** (`Cargo.toml` path-deps, `pixi.toml` wheel path, `pyproject.toml` unpinned genoray). They are the release gate — Task 12 documents them; nobody resolves them here.
- **Do NOT touch pre-existing clippy warnings** outside the SVAR2 changes (`bigwig.rs`, `reference/mod.rs`, etc.).

---

## Execution order

Docs & clippy (Tasks 1, 11) carry zero runtime risk — but they are grouped by topic below, not strictly first. The safe order is: **2 → 1 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13**. Task 2 (typecheck-task fix) goes first so later Python tasks get real type-checking. The Rust de-dup (Tasks 9, 10) go late and each rebuilds + re-runs parity. Task 13 is the final full-suite gate.

---

### Task 1: Strip stale/internal references from shipped comments & docstrings

Pure text edits in shipped code. No behavior change. Groups all "stale comment" findings (spec §1b, §4d, §5-doc-nits) that are not tied to a code change in another task.

**Files:**
- Modify: `src/svar2/mod.rs:800` (test docstring), `src/reconstruct/mod.rs:19-37`, `src/reconstruct/mod.rs:278`
- Modify: `python/genvarloader/_dataset/_svar2_haps.py:22-24` (module docstring), `:384`, `:486`, `:489`
- Modify: `python/genvarloader/_dataset/_reconstruct.py:143`, `:399`
- Modify: `python/genvarloader/_dataset/_write.py:1194-1197`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing (comments only).

- [ ] **Step 1: Fix the reverted-optimization test docstring** in `src/svar2/mod.rs:800`. The doc for `test_decode_variants_from_split_byte_identical_presence_edge` claims it exercises "the `present_bit` closure's now-`get_unchecked` read of `dense_present`". `get_unchecked` was reverted (grep `get_unchecked src/` → this comment is the only hit). Reword to describe the checked read that is actually there:

Replace:
```rust
    /// single-hap tests above never trigger), and (2) the `present_bit`
    /// closure's now-`get_unchecked` read of `dense_present`, with a mix of
    /// present/absent bits whose per-hap `base_bit` windows straddle a byte
```
with:
```rust
    /// single-hap tests above never trigger), and (2) the `present_bit`
    /// closure's read of `dense_present`, with a mix of
    /// present/absent bits whose per-hap `base_bit` windows straddle a byte
```

- [ ] **Step 2: Fix the stale `unphased_union` out-of-scope claim** in `python/genvarloader/_dataset/_svar2_haps.py:22-24`. The module docstring lists `unphased_union` as guarded `NotImplementedError`, but it is fully supported (`_reconstruct_variants`, `_reconstruct_variant_windows`, and `_guard_unsupported` explicitly does NOT guard it).

Replace:
```python
Out of scope for this plan (guarded with ``NotImplementedError``): spliced
output, ``filter == "exonic"`` (keep mask), ``min_af``/``max_af``, annotated
haps, in-kernel reverse-complement, and ``unphased_union``.
```
with:
```python
Out of scope (guarded with ``NotImplementedError``): spliced output,
``filter == "exonic"`` (keep mask), ``min_af``/``max_af``, annotated haps, and
in-kernel reverse-complement. (``unphased_union`` and ``variant-windows`` ARE
supported.)
```

- [ ] **Step 3: Strip internal plan/task numbering** from shipped comments. Rewrite each in terms of behavior (reader has no access to plan docs):
  - `_svar2_haps.py:384` — "The tracks follow-up (7c)" → describe the behavior (e.g. "Track re-alignment path").
  - `_svar2_haps.py:486`, `:489` — "tracks (7c)" banner → "tracks".
  - `_reconstruct.py:143` — "Task 7c" → behavior description.
  - `_reconstruct.py:399` — "FIX 1 guard" → describe what it guards (see Task 8, which also references this line; keep the wording consistent — describe the FlankSample fill-seed divergence).
  - `_write.py:1194-1197` — "Phase-1 wiring" comment (see Task 5, which rewrites this block; if Task 5 runs first this may already be gone — if so, skip). Reword to describe that write-time fixed-length handling is unsupported for `.svar2` and the read kernel sizes output at read time.
  - `src/reconstruct/mod.rs:278` and the list at `:19-37` — leave the prose but note these get clippy-reflowed in Task 11; no wording change needed here unless a task number appears.

- [ ] **Step 3b: Add missing `///` docstrings to PyO3-exposed `svar2/store.rs` items** (spec §4e — these become the Python docstrings): add a one-line `///` to `reader` (`:16`), `store_path` (`:19`), the `#[new]` constructor (`:26`), and `contigs` (`:46`) describing what each returns. Also add numpydoc docstrings to `_svar2_link.py:make_svar2_link` and `_svar2_haps.py:_reconstruct_variants` if not added by their own tasks (`_write_from_svar2` is covered by Task 5).

- [ ] **Step 4: Verify no plan/task references remain** in shipped code:

Run:
```bash
grep -rnE "Task [0-9]|Phase-1 wiring|FIX 1|\(7c\)|first cut minimal|get_unchecked" src/ python/genvarloader/ | grep -v "test_" || echo "clean"
```
Expected: `clean` (or only hits inside test names/legit uses you've reviewed). Note: `svar2/mod.rs:280` ("Task 1.3") and `tracks/mod.rs:2467` ("Task 4 Part C") and `ffi/mod.rs:774` ("first cut minimal") are ALSO in scope — fix them in this step too if the grep surfaces them.

- [ ] **Step 5: Rebuild + smoke-test** (comments in Rust don't change behavior, but confirm it still compiles):

Run: `pixi run -e dev cargo build 2>&1 | tail -3`
Expected: `Finished` (no errors).

- [ ] **Step 6: Commit**

```bash
git checkout pixi.lock 2>/dev/null
git add -A src/ python/genvarloader/
git commit --no-verify -m "$(cat <<'EOF'
docs(svar2): strip stale/internal references from shipped comments

Fix the reverted-get_unchecked test docstring, the stale unphased_union
out-of-scope claim in the Svar2Haps module docstring, and internal plan/task
numbering ("Task 7c", "Phase-1 wiring", "FIX 1") that leaked into comments.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Fix the `typecheck` pixi task (checks zero files in worktrees)

`pixi run -e dev typecheck` = `pyrefly check` with no paths → inside any `.claude/worktrees/` checkout pyrefly matches **zero** files (root `.gitignore` ignores `.claude/`, pyrefly honors ignore files) and exits 0. Typecheck has effectively never run on this branch. This also fixes the broken `pyrefly-check` pre-commit hook.

**Files:**
- Modify: `pixi.toml:162`
- Modify: `python/genvarloader/_ragged.py:325` (clear the one real finding the fixed task surfaces)

**Interfaces:**
- Consumes: nothing.
- Produces: a working `typecheck` task used by every later Python task.

- [ ] **Step 1: Confirm the bug** — pyrefly with no paths checks nothing in the worktree:

Run: `pixi run -e dev pyrefly check 2>&1 | grep -c "No Python files matched"`
Expected: `1` (confirms zero files matched).

- [ ] **Step 2: Point the task at explicit paths** in `pixi.toml:162`.

Replace:
```toml
typecheck = { cmd = "pyrefly check" }
```
with:
```toml
typecheck = { cmd = "pyrefly check python/genvarloader tests" }
```

- [ ] **Step 3: Run the fixed task; expect exactly one real finding**

Run: `pixi run -e dev typecheck 2>&1 | grep -E "error|Unused" | head`
Expected: one `ERROR Unused \`# pyrefly: ignore\` comment ... no-matching-overload [unused-ignore]` at `_ragged.py:325`.

- [ ] **Step 4: Clear the unused-ignore** at `python/genvarloader/_ragged.py:325`.

Replace:
```python
        out = out.reshape((*leading, out_len))  # pyrefly: ignore[no-matching-overload]
```
with:
```python
        out = out.reshape((*leading, out_len))
```

- [ ] **Step 5: Re-run typecheck; expect clean**

Run: `pixi run -e dev typecheck 2>&1 | tail -3`
Expected: `0 errors` (warnings are fine; no errors).

- [ ] **Step 6: Confirm the pre-commit hook now passes** (also update the local hook definition if it hardcodes `pyrefly check` with no args):

Run: `grep -n -A6 "pyrefly-check" .pre-commit-config.yaml`
If the hook `entry` is `pyrefly check` with no paths, change it to `pyrefly check python/genvarloader tests` to match the task. Then:
Run: `pixi run -e dev prek run pyrefly-check --all-files 2>&1 | tail -3`
Expected: `Passed`.

- [ ] **Step 7: Commit**

```bash
git checkout pixi.lock 2>/dev/null
git add pixi.toml python/genvarloader/_ragged.py .pre-commit-config.yaml
git commit -m "$(cat <<'EOF'
chore(svar2): make typecheck task check explicit paths

Bare `pyrefly check` matches zero files inside a .claude/worktrees checkout
(root .gitignore ignores .claude/, pyrefly honors ignore files), so typecheck
silently passed on nothing. Point it at python/genvarloader + tests, fix the
pre-commit hook to match, and clear the now-flagged unused pyrefly-ignore.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Guard the serial unsafe raw-pointer path in `reconstruct/mod.rs`

The parallel carve path has `debug_assert!(s >= cursor && e >= s, ...)`; the two **serial** fallback loops (`:547` and `:847`) build `&mut [u8]` from caller-supplied `out_offsets` with only a SAFETY comment, no runtime guard. A non-monotonic `out_offsets` underflows `out_e - out_s` to a giant slice (UB). Add the matching assert.

**Files:**
- Modify: `src/reconstruct/mod.rs` (serial loop bodies near `:547` and `:847`)

**Interfaces:**
- Consumes: nothing.
- Produces: nothing (adds a debug-only invariant check).

- [ ] **Step 1: Add the guard to the first serial loop.** In the serial `for k in 0..n_work` block (the one preceded by `// Serial path: use raw pointers ...`), immediately after:
```rust
            let out_s = out_offsets[k] as usize;
            let out_e = out_offsets[k + 1] as usize;
```
insert:
```rust
            debug_assert!(
                out_e >= out_s,
                "out_offsets must be monotonically non-decreasing (got out_s={out_s}, out_e={out_e})"
            );
```

- [ ] **Step 2: Add the identical guard to the second serial loop** (the second occurrence of the same `let out_s = ...; let out_e = ...;` pair inside a serial raw-pointer block, near `:847`). Insert the same `debug_assert!` block after the two `let` lines.

- [ ] **Step 3: Verify both serial loops now assert.** There are two carve dispatchers (merged later in Task 9); both must have the guard on the serial branch.

Run:
```bash
grep -c "out_offsets must be monotonically non-decreasing" src/reconstruct/mod.rs
```
Expected: `4` (2 pre-existing parallel-path asserts + 2 new serial-path asserts).

- [ ] **Step 4: Build + run the Rust unit tests** (compiles the debug_assert; a debug `cargo test` build will trip the assert if any existing test passes non-monotonic offsets — it shouldn't):

Run: `pixi run -e dev cargo test reconstruct 2>&1 | tail -8`
Expected: tests pass (no assertion panic).

- [ ] **Step 5: Rebuild release + run SVAR2 parity gate:**

Run:
```bash
pixi run -e dev maturin develop --release 2>&1 | tail -1
pixi run -e dev pytest tests/dataset/test_svar2_dataset.py tests/dataset/test_svar2_readbound_haps.py tests/dataset/test_svar2_readbound_tracks.py -q 2>&1 | tail -3
```
Expected: build `Installed`; tests pass.

- [ ] **Step 6: Commit**

```bash
git checkout pixi.lock 2>/dev/null
git add src/reconstruct/mod.rs
git commit --no-verify -m "$(cat <<'EOF'
fix(svar2): guard serial unsafe carve path with monotonicity debug_assert

The parallel split_at_mut path already debug_asserts out_offsets is
non-decreasing; the serial raw-pointer fallback carved out_e - out_s with no
guard, so a non-monotonic offsets array underflows to a multi-GB slice (UB).
Hoist the same assert into both serial loops.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Convert Python-reachable Rust panics to `PyValueError`

Arrays from Python are `.as_slice().unwrap()` / `.expect("must be contiguous")`'d in the SVAR2 kernels; a non-contiguous view (`a[::2]`) panics instead of raising. Plus two more: the pure-DEL anchor index and a release-mode `assert_eq!`.

**Files:**
- Modify: `src/ffi/mod.rs` (the SVAR2 readbound `#[pyfunction]`s — contiguity validation at the boundary)
- Modify: `src/reconstruct/mod.rs:703` (pure-DEL anchor bounds check), `:639-645`, `:692`
- Modify: `src/svar2/mod.rs:358-365` (move the `assert_eq!` to the FFI boundary at `ffi/mod.rs:1411`)
- Test: `tests/dataset/test_svar2_readbound_haps.py` (add a non-contiguous-input test)

**Interfaces:**
- Consumes: nothing.
- Produces: SVAR2 readbound `#[pyfunction]`s raise `ValueError` (not panic) on non-C-contiguous inputs and on out-of-range variant positions.

- [ ] **Step 1: Write the failing test** — a non-contiguous input should raise `ValueError`, not crash the interpreter. Add to `tests/dataset/test_svar2_readbound_haps.py`:

```python
def test_readbound_haps_noncontiguous_input_raises():
    """A non-C-contiguous numpy view must surface as ValueError, not a panic."""
    import numpy as np
    import pytest
    from genvarloader._dataset._svar2_store_py import build_readbound_haps  # noqa: F401
    # Build a minimal store + regions exactly as the existing haps parity test does,
    # then pass a strided (non-contiguous) view of one of the int64 range arrays.
    # (Reuse the fixture/store construction from test_readbound_haps_* above.)
    # The precise construction mirrors the sibling test; the assertion is:
    with pytest.raises(ValueError):
        # call the FFI entry with a[::2] slice of a range array
        ...
```
NOTE to implementer: model the store/region setup on the nearest existing test in this file (e.g. the first `build_readbound_haps` parity test around `:75`). The ONLY new thing is passing a `arr[::2]` view where a contiguous int64 array is expected and asserting `ValueError`.

- [ ] **Step 2: Run it — expect it to FAIL** (currently panics / wrong exception):

Run: `pixi run -e dev pytest tests/dataset/test_svar2_readbound_haps.py::test_readbound_haps_noncontiguous_input_raises -v 2>&1 | tail -8`
Expected: FAIL (panic/pyo3 PanicException or a bare index error, not ValueError).

- [ ] **Step 3: Add a contiguity helper + validate at each SVAR2 `#[pyfunction]` boundary** in `src/ffi/mod.rs`. Near the top of the SVAR2 FFI section add:
```rust
/// Return a C-contiguous slice view of `arr`, or a Python `ValueError` if the
/// input array is not contiguous (e.g. a strided `a[::2]` view). Kernels below
/// index the backing slice directly, so a non-contiguous input would otherwise
/// panic inside `py.detach`.
fn require_contiguous<'a, T: numpy::Element>(
    arr: &'a numpy::PyReadonlyArray1<T>,
    name: &str,
) -> PyResult<&'a [T]> {
    arr.as_slice()
        .map_err(|_| PyValueError::new_err(format!("`{name}` must be C-contiguous")))
}
```
Then at each of the four SVAR2 readbound `#[pyfunction]`s (`reconstruct_haplotypes_from_svar2_readbound`, `shift_and_realign_tracks_from_svar2_readbound`, `decode_variants_from_svar2_readbound`, `hap_diffs_from_svar2_readbound`), replace the `.as_slice().unwrap()` / `.expect("must be contiguous")` calls on Python-supplied arrays with `require_contiguous(&arr, "arr")?`. (For `PyReadwriteArray` outputs use the analogous `as_slice_mut().map_err(...)`.)

- [ ] **Step 4: Bounds-check the pure-DEL anchor** at `src/reconstruct/mod.rs:703`. The `contig_ref_s[pos..pos+1]` index panics for a variant at/past contig end. Guard it — return a `Result`/`PyErr` up the call chain, OR (if this fn is not `PyResult`-returning) validate `pos < contig_ref_s.len()` at the FFI boundary before entering `py.detach` and raise `PyValueError::new_err(format!("variant position {pos} is beyond contig end"))`. Prefer the FFI-boundary check to keep the hot kernel panic-free.

- [ ] **Step 5: Move the release-mode `assert_eq!`** from `src/svar2/mod.rs:358-365` to the FFI boundary. In `decode_variants_from_svar2_readbound` (`ffi/mod.rs:~1411`, where `has_fields` is known), validate the `vk_src` length precondition and raise `PyValueError` before calling into `svar2::`. Downgrade the in-kernel `assert_eq!` to `debug_assert_eq!` (or delete it if the FFI check fully covers it).

- [ ] **Step 6: Rebuild + run the new test — expect PASS:**

Run:
```bash
pixi run -e dev maturin develop --release 2>&1 | tail -1
pixi run -e dev pytest tests/dataset/test_svar2_readbound_haps.py::test_readbound_haps_noncontiguous_input_raises -v 2>&1 | tail -5
```
Expected: PASS (`ValueError` raised).

- [ ] **Step 7: Full SVAR2 parity gate still green:**

Run: `pixi run -e dev pytest tests/dataset/test_svar2_dataset.py tests/dataset/test_svar2_readbound_*.py tests/dataset/test_write_svar2.py -q 2>&1 | tail -3`
Expected: 31/31 (+ the new test) pass.

- [ ] **Step 8: Commit**

```bash
git checkout pixi.lock 2>/dev/null
git add src/ffi/mod.rs src/reconstruct/mod.rs src/svar2/mod.rs tests/dataset/test_svar2_readbound_haps.py
git commit --no-verify -m "$(cat <<'EOF'
fix(svar2): raise PyValueError on non-contiguous / OOB Python input

Non-C-contiguous numpy views (a[::2]) and out-of-range variant positions
panicked inside the readbound kernels instead of surfacing as ValueError.
Validate contiguity and the pure-DEL anchor bound at the #[pyfunction]
boundary; move the vk_src length assert_eq! there too.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Reject `extend_to_length=False` for `.svar2` (stop silently ignoring it)

`_write_from_svar2` accepts `extend_to_length` and never reads it; `chromEnd` is always extended. Passing `False` yields a different dataset than requested, silently. Per the branch's guard-matrix policy, fail loudly.

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py:1140-1197` (`_write_from_svar2`)
- Test: `tests/dataset/test_write_svar2.py`

**Interfaces:**
- Consumes: `_write_from_svar2(path, bed, svar2, samples, extend_to_length)` (existing signature — unchanged).
- Produces: `gvl.write(..., variants=<SparseVar2>, extend_to_length=False)` raises `NotImplementedError`.

- [ ] **Step 1: Write the failing test** in `tests/dataset/test_write_svar2.py`:

```python
def test_svar2_extend_to_length_false_raises(tmp_path, svar2_source_and_bed):
    """extend_to_length=False is unsupported for a .svar2 source (Phase-1) and
    must raise, not silently produce an extended dataset."""
    import pytest
    import genvarloader as gvl
    svar2, bed = svar2_source_and_bed  # reuse the existing fixture used by the write tests
    with pytest.raises(NotImplementedError, match="extend_to_length"):
        gvl.write(tmp_path / "ds", bed, variants=svar2, extend_to_length=False)
```
NOTE: reuse whatever fixture the existing `test_write_svar2.py` tests use to obtain a `SparseVar2` + bed; match its parameter name.

- [ ] **Step 2: Run it — expect FAIL** (currently silently succeeds):

Run: `pixi run -e dev pytest tests/dataset/test_write_svar2.py::test_svar2_extend_to_length_false_raises -v 2>&1 | tail -6`
Expected: FAIL (no exception raised).

- [ ] **Step 3: Add the guard** at the top of `_write_from_svar2` in `python/genvarloader/_dataset/_write.py` (right after the signature/opening comment, before `out_dir = ...`):

```python
    if not extend_to_length:
        raise NotImplementedError(
            "extend_to_length=False is not supported for a .svar2 variant source: "
            "the read-bound kernel always sizes haplotype output at read time and "
            "the write-time ranges cache is built for the extended chromEnd. Use a "
            ".svar/VCF/PGEN source if you need un-extended haplotypes."
        )
```
Then replace the now-inaccurate `# extend_to_length fixed-output-length write-time handling is out of scope ...` comment block (around `:1194-1197`) with a one-liner noting the flag is validated at entry.

- [ ] **Step 4: Run the new test — expect PASS:**

Run: `pixi run -e dev pytest tests/dataset/test_write_svar2.py::test_svar2_extend_to_length_false_raises -v 2>&1 | tail -4`
Expected: PASS.

- [ ] **Step 5: Full write-svar2 suite still green:**

Run: `pixi run -e dev pytest tests/dataset/test_write_svar2.py -q 2>&1 | tail -3`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git checkout pixi.lock 2>/dev/null
git add python/genvarloader/_dataset/_write.py tests/dataset/test_write_svar2.py
git commit -m "$(cat <<'EOF'
fix(svar2): reject extend_to_length=False for .svar2 sources

_write_from_svar2 accepted the flag and ignored it, silently extending
chromEnd regardless. Raise NotImplementedError (Phase-1 guard-matrix policy)
instead of producing a dataset that differs from what was requested.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Vectorize `_svar2_region_max_ends`

`_write.py:1092-1138` is an `O(regions × samples × ploidy)` Python triple-loop over decoded records at write time. Its own docstring flags it. Vectorize with a scatter-reduce while staying byte-identical (`test_write_svar2.py` locks the cache + same-POS-tie behavior).

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py:1092-1138` (`_svar2_region_max_ends`)
- Test: `tests/dataset/test_write_svar2.py` (add a direct equivalence test against the old loop)

**Interfaces:**
- Consumes: `_svar2_region_max_ends(svar2, contig, starts, ends, samples) -> NDArray[np.int32]` (signature unchanged).
- Produces: identical output to the current loop (per-region max over selected haplotypes of `(pos, end)`, `pos`-then-`end` tie-break, default = `ends`).

- [ ] **Step 1: Pin the current behavior with a direct test.** Before changing the function, add a test that captures the exact semantics — including the same-POS tie-break and the "no variants → keep chromEnd" default — on a hand-built decode. Add to `tests/dataset/test_write_svar2.py`:

```python
def test_svar2_region_max_ends_matches_reference(svar2_source_and_bed):
    """Vectorized _svar2_region_max_ends must equal a straightforward per-hap loop,
    including the pos-then-end tie-break and the empty-region default = chromEnd."""
    import numpy as np
    from genvarloader._dataset._write import _svar2_region_max_ends
    svar2, bed = svar2_source_and_bed
    for (c,), df in bed.partition_by("chrom", as_dict=True, maintain_order=True).items():
        starts = df["chromStart"].to_numpy()
        ends = df["chromEnd"].to_numpy()
        samples = list(svar2.available_samples)
        got = _svar2_region_max_ends(svar2, c, starts, ends, samples)
        # reference: recompute with the explicit loop semantics
        ref = _reference_region_max_ends(svar2, c, starts, ends, samples)
        np.testing.assert_array_equal(got, ref)


def _reference_region_max_ends(svar2, contig, starts, ends, samples):
    """Byte-for-byte copy of the ORIGINAL triple-loop, kept in the test as the oracle."""
    import numpy as np
    R, S_all, P = len(starts), svar2.n_samples, svar2.ploidy
    sel = [svar2.available_samples.index(s) for s in samples]
    dec = svar2.decode(contig, list(zip(starts.tolist(), ends.tolist())))
    pos_arr = dec.data["pos"]; ilen_arr = dec.data["ilen"]; off = np.asarray(dec.offsets)
    out = np.asarray(ends, np.int64).copy()
    for r in range(R):
        best_pos, best_end = -1, -1
        for s in sel:
            for p in range(P):
                h = (r * S_all + s) * P + p
                a, b = int(off[h]), int(off[h + 1])
                if a == b: continue
                seg_pos = pos_arr[a:b]; seg_ilen = ilen_arr[a:b]
                j = int(np.argmax(seg_pos))
                p_pos = int(seg_pos[j]); p_end = (p_pos + 1) - min(int(seg_ilen[j]), 0)
                if p_pos > best_pos or (p_pos == best_pos and p_end > best_end):
                    best_pos, best_end = p_pos, p_end
        if best_pos >= 0:
            out[r] = best_end
    return out.astype(np.int32)
```

- [ ] **Step 2: Run it — expect PASS** (the reference IS the current impl, so it passes now; this locks the contract before the rewrite):

Run: `pixi run -e dev pytest tests/dataset/test_write_svar2.py::test_svar2_region_max_ends_matches_reference -v 2>&1 | tail -4`
Expected: PASS.

- [ ] **Step 3: Vectorize the function.** Replace the triple-loop body of `_svar2_region_max_ends` (keeping the docstring's semantics but dropping the "O(...) Python iteration ... vectorize as a follow-up" caveat) with a scatter-reduce. Key idea: for each variant, its haplotype maps to a region `r = h // (S_all * P)` but only SELECTED samples count; compute `end = (pos+1) - min(ilen,0)` per variant, form a sortable composite `key = (pos << 21) | end` (end fits well under 2^21 for realistic regions; assert it) so that a plain per-region max on `key` reproduces the pos-then-end tie-break, then unpack `end`:

```python
    R, S_all, P = len(starts), svar2.n_samples, svar2.ploidy
    sel = np.asarray([svar2.available_samples.index(s) for s in samples], np.int64)
    dec = svar2.decode(contig, list(zip(starts.tolist(), ends.tolist())))
    pos_arr = np.asarray(dec.data["pos"], np.int64)
    ilen_arr = np.asarray(dec.data["ilen"], np.int64)
    off = np.asarray(dec.offsets, np.int64)  # length R*S_all*P + 1
    out = np.asarray(ends, np.int64).copy()  # default = chromEnd
    if pos_arr.size:
        n_hap = R * S_all * P
        counts = np.diff(off)  # variants per hap
        hap_of_var = np.repeat(np.arange(n_hap), counts)  # region-major hap index per variant
        s_of_hap = (np.arange(n_hap) // P) % S_all
        keep = np.isin(s_of_hap[hap_of_var], sel)  # only selected samples
        region_of_var = hap_of_var // (S_all * P)
        end_var = (pos_arr + 1) - np.minimum(ilen_arr, 0)  # 0-based -> 1-based, extend on DEL
        SHIFT = 21
        assert int(end_var.max(initial=0)) < (1 << SHIFT), "end exceeds tie-break packing width"
        key = (pos_arr << SHIFT) | end_var
        key_k = key[keep]; region_k = region_of_var[keep]
        if key_k.size:
            best = np.full(R, -1, np.int64)
            np.maximum.at(best, region_k, key_k)  # per-region max composite key
            has = best >= 0
            out[has] = best[has] & ((1 << SHIFT) - 1)  # unpack end
    return out.astype(np.int32)
```
Update the docstring: drop the last paragraph ("O(R * len(samples) * ploidy) Python iteration ... follow-up") and replace with a one-line note that it is a vectorized per-region scatter-max preserving the pos-then-end tie-break.

- [ ] **Step 4: Run the equivalence test — expect PASS:**

Run: `pixi run -e dev pytest tests/dataset/test_write_svar2.py::test_svar2_region_max_ends_matches_reference -v 2>&1 | tail -4`
Expected: PASS (vectorized == reference loop).

- [ ] **Step 5: Full write-svar2 suite (locks cache + same-POS tie) still green:**

Run: `pixi run -e dev pytest tests/dataset/test_write_svar2.py -q 2>&1 | tail -3`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git checkout pixi.lock 2>/dev/null
git add python/genvarloader/_dataset/_write.py tests/dataset/test_write_svar2.py
git commit -m "$(cat <<'EOF'
perf(svar2): vectorize _svar2_region_max_ends (byte-identical)

Replace the O(regions x samples x ploidy) write-time Python triple-loop with a
per-region scatter-max over a (pos<<21)|end composite key that preserves the
pos-then-end tie-break. Pinned byte-identical to the original loop by a new
equivalence test carrying the old loop as its oracle.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Drop the unused `region_starts` array

`_write.py` writes/memmaps `region_starts`; `_svar2_haps.py` loads it and its own docstring says it is "kept for parity/debug, NOT fed to the FFI." Remove it end-to-end.

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py:1162,1205,1214` (+ the `svar2_meta.json` entry at `:1170-ish`)
- Modify: `python/genvarloader/_dataset/_svar2_haps.py:86-88,95` (loader + docstring)
- Modify: `tests/unit/dataset/test_svar2_store.py` and/or `tests/dataset/test_write_svar2.py` if either asserts `region_starts` presence

**Interfaces:**
- Consumes: nothing.
- Produces: `svar2_ranges/` cache no longer contains `region_starts.npy`; `svar2_meta.json` no longer lists it.

- [ ] **Step 1: Check whether any test asserts `region_starts`:**

Run: `grep -rn "region_starts" tests/ python/genvarloader/ | grep -v "\.pyc"`
Expected: hits in `_write.py`, `_svar2_haps.py`, and possibly a test. Note every location.

- [ ] **Step 2: Remove the write side** in `_write.py`:
  - Delete the `region_starts = np.memmap(... "region_starts.npy" ...)` line (`:1162`).
  - Delete the `"region_starts": {"shape": [R], "dtype": "<i8"},` entry from the `svar2_meta.json` dict.
  - Delete `region_starts[lo:hi] = np.asarray(d["region_starts"], ...)` in the partition loop (`:1205`).
  - Remove `region_starts` from the `for mm in (...)` flush tuple (`:1214`).

- [ ] **Step 3: Remove the read side** in `_svar2_haps.py`: delete the `region_starts` memmap load (`:95`) and the docstring lines describing it (`:86-88`).

- [ ] **Step 4: Update/remove any test assertion** found in Step 1 (e.g. a `test_svar2_store.py` check that the meta lists `region_starts`, or a cache-contents lock in `test_write_svar2.py`). Remove `region_starts` from the expected key set.

- [ ] **Step 5: Rebuild is not needed (Python-only). Run the write + store tests:**

Run: `pixi run -e dev pytest tests/dataset/test_write_svar2.py tests/unit/dataset/test_svar2_store.py -q 2>&1 | tail -3`
Expected: all pass.

- [ ] **Step 6: Full SVAR2 read parity (dataset opens the trimmed cache):**

Run: `pixi run -e dev pytest tests/dataset/test_svar2_dataset.py -q 2>&1 | tail -3`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git checkout pixi.lock 2>/dev/null
git add python/genvarloader/_dataset/_write.py python/genvarloader/_dataset/_svar2_haps.py tests/
git commit -m "$(cat <<'EOF'
refactor(svar2): drop unused region_starts from the ranges cache

region_starts was written, memmapped, and never fed to the FFI (its own
docstring said "parity/debug only"). Remove it from the write path, the
svar2_meta.json schema, the Svar2Haps loader, and the cache-contents test.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Open a tracked issue for the FlankSample fill-seed divergence

`_reconstruct.py:399-419` papers over a documented correctness divergence (multi-contig `FlankSample` track fills seed off a contig-local query index) with a `NotImplementedError`. The comment is the only record. File an issue and reference it.

**Files:**
- Modify: `python/genvarloader/_dataset/_reconstruct.py:399-419` (add issue reference to the guard comment)

**Interfaces:**
- Consumes: nothing.
- Produces: a GitHub issue URL referenced in the guard comment.

- [ ] **Step 1: Read the guard + its comment** to write an accurate issue body:

Run: `sed -n '395,420p' python/genvarloader/_dataset/_reconstruct.py`

- [ ] **Step 2: Create the issue** (repo `mcvickerlab/GenVarLoader`):

Run:
```bash
gh issue create --repo mcvickerlab/GenVarLoader \
  --title "SVAR2 read-bound: multi-contig FlankSample track fill uses contig-local query index" \
  --body "In the SVAR2 read-bound path, FlankSample track fills seed the fill value off a contig-local query index rather than the global one, which diverges across a multi-contig batch. Currently guarded with NotImplementedError in \`_reconstruct.py\` (~line 399). This issue tracks lifting the guard once the fill-seed index is made global. Branch: svar2-m6b-kernel."
```
Capture the printed issue URL.

- [ ] **Step 3: Reference the issue in the guard comment.** Replace the "FIX 1 guard" wording (also targeted by Task 1) with a behavior description ending in the issue link, e.g.:
```python
            # Multi-contig FlankSample track fills seed off a contig-local query
            # index, which diverges from the global fill seed across a batch.
            # Guarded until the fill-seed index is made global; see
            # https://github.com/mcvickerlab/GenVarLoader/issues/<N>.
```

- [ ] **Step 4: Verify the guard still fires** (behavior unchanged — this is a comment + issue only):

Run: `pixi run -e dev pytest tests/dataset/test_svar2_dataset.py -q -k "flank or Flank or multi" 2>&1 | tail -4`
Expected: pass (guard raises where tested; no behavior change).

- [ ] **Step 5: Commit**

```bash
git checkout pixi.lock 2>/dev/null
git add python/genvarloader/_dataset/_reconstruct.py
git commit -m "$(cat <<'EOF'
docs(svar2): track FlankSample fill-seed divergence in a GitHub issue

The multi-contig FlankSample track-fill divergence was recorded only in a code
comment behind a NotImplementedError guard. File a tracking issue and
reference it from the guard.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Relocate test-only oracle modules out of the shipped package

`_svar2_source.py` (`SparseVar2Source`) and `_svar2_store_py.py` (`build_readbound_*`) have zero importers under `python/` — only 8 test files use them. Move them into `tests/` and rename `_svar2_store_py.py` (the `_py` suffix is meaningless — it holds no store class).

**Files:**
- Move: `python/genvarloader/_dataset/_svar2_source.py` → `tests/_oracles/svar2_source.py`
- Move: `python/genvarloader/_dataset/_svar2_store_py.py` → `tests/_oracles/svar2_readbound_inputs.py`
- Create: `tests/_oracles/__init__.py`
- Modify (imports): `tests/test_svar2_realign_tracks.py`, `tests/test_svar2_reconstruct.py`, `tests/dataset/test_svar2_dataset.py`, `tests/dataset/test_svar2_readbound_haps.py`, `tests/dataset/test_svar2_readbound_diffs.py`, `tests/dataset/test_svar2_readbound_tracks.py`, `tests/dataset/test_svar2_readbound_variants.py`
- Modify (docstring back-reference): `python/genvarloader/_dataset/_svar2_haps.py:15-19`

**Interfaces:**
- Consumes: tests import `from tests._oracles.svar2_source import SparseVar2Source` and `from tests._oracles.svar2_readbound_inputs import build_readbound_haps, build_readbound_diffs, build_readbound_tracks, build_readbound_variants`.
- Produces: no test-only code ships in `genvarloader`.

- [ ] **Step 1: Confirm the two modules are imported only from tests** (already audited, re-verify):

Run: `grep -rln "_svar2_source\|_svar2_store_py\|SparseVar2Source\|build_readbound" python/genvarloader/ | grep -v "\.so"`
Expected: only `_svar2_haps.py` (docstring reference) and the two modules themselves — NO functional importer in `python/`.

- [ ] **Step 2: Check the two modules' own imports** (what they pull from the package — must still resolve from `tests/`):

Run: `grep -n "^import\|^from" python/genvarloader/_dataset/_svar2_source.py python/genvarloader/_dataset/_svar2_store_py.py`
Note any relative imports (`from . import ...` / `from ._x import ...`) — these must become absolute `genvarloader...` imports after the move.

- [ ] **Step 3: Move the files with git + create the package:**

```bash
mkdir -p tests/_oracles
git mv python/genvarloader/_dataset/_svar2_source.py tests/_oracles/svar2_source.py
git mv python/genvarloader/_dataset/_svar2_store_py.py tests/_oracles/svar2_readbound_inputs.py
touch tests/_oracles/__init__.py && git add tests/_oracles/__init__.py
```

- [ ] **Step 4: Fix relative imports** inside the two moved files (from Step 2). Any `from .` / `from ._foo import` becomes `from genvarloader._dataset._foo import`. If `svar2_readbound_inputs.py` imports `SparseVar2Source`, point it at `from tests._oracles.svar2_source import SparseVar2Source`.

- [ ] **Step 5: Update the 8 test files' imports.** In each, replace:
  - `from genvarloader._dataset._svar2_source import SparseVar2Source` → `from tests._oracles.svar2_source import SparseVar2Source`
  - `from genvarloader._dataset._svar2_store_py import build_readbound_*` → `from tests._oracles.svar2_readbound_inputs import build_readbound_*`

Do it mechanically:
```bash
grep -rl "_dataset._svar2_source\|_dataset._svar2_store_py" tests/ | while read f; do
  sed -i 's#genvarloader\._dataset\._svar2_source#tests._oracles.svar2_source#g; s#genvarloader\._dataset\._svar2_store_py#tests._oracles.svar2_readbound_inputs#g' "$f"
done
```
(RTK note: this is a mechanical sed on test imports — acceptable here.)

- [ ] **Step 6: Update the `_svar2_haps.py` docstring back-reference** (`:15-19`) that mentions `_svar2_store_py.build_readbound_*`. Point it at the new location: replace `_svar2_store_py.build_readbound_*` with `tests/_oracles/svar2_readbound_inputs.build_readbound_*` (and note it is the test oracle).

- [ ] **Step 7: Confirm `tests` is importable as a package** (so `from tests._oracles...` resolves). Check for `tests/__init__.py`:

Run: `ls tests/__init__.py 2>/dev/null && echo "has __init__" || echo "NO __init__ — check conftest/rootdir import mode"`
If tests are collected in rootdir/`importmode=prepend` without `tests/__init__.py`, prefer a conftest-based path or add `tests/__init__.py`. Verify by running one moved-import test in Step 8; if `ModuleNotFoundError: tests`, add `tests/__init__.py` (`touch tests/__init__.py && git add tests/__init__.py`) or adjust to a top-level `_oracles` package under the tests rootdir — match the repo's existing test-helper import convention (`tests/_builders/` is imported as `from _builders...` or `from tests._builders...`? check it):
```bash
grep -rn "_builders" tests/ | grep import | head -3
```
Mirror whatever `_builders` does.

- [ ] **Step 8: Run every affected test file:**

Run:
```bash
pixi run -e dev pytest tests/test_svar2_realign_tracks.py tests/test_svar2_reconstruct.py tests/dataset/test_svar2_dataset.py tests/dataset/test_svar2_readbound_haps.py tests/dataset/test_svar2_readbound_diffs.py tests/dataset/test_svar2_readbound_tracks.py tests/dataset/test_svar2_readbound_variants.py -q 2>&1 | tail -4
```
Expected: all pass (no `ModuleNotFoundError`, no import errors).

- [ ] **Step 9: Confirm nothing in the shipped package references the moved modules:**

Run: `grep -rn "_svar2_source\|_svar2_store_py" python/genvarloader/ | grep -v "\.so"`
Expected: no hits (the docstring in `_svar2_haps.py` now points at `tests/_oracles/...`).

- [ ] **Step 10: Commit**

```bash
git checkout pixi.lock 2>/dev/null
git add -A python/ tests/
git commit -m "$(cat <<'EOF'
refactor(svar2): move test-only oracles out of the shipped package

SparseVar2Source and build_readbound_* have no importers under python/ — they
are the parity oracle + FFI-input builders used only by tests. Move them to
tests/_oracles/ (renaming _svar2_store_py.py -> svar2_readbound_inputs.py,
since it holds no store class) and repoint the 8 test files' imports.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Drop the dead `annot_*` capability from the SVAR2 readbound haplotype kernel

`reconstruct_haplotypes_from_svar2_readbound`'s `annot_v_idxs`/`annot_ref_pos` params are `None` at both call sites; 3 of 4 match arms (~60 lines) + the `:605-608` doc are unreachable (annotated-hap output for `.svar2` is `NotImplementedError`-guarded anyway).

**Files:**
- Modify: `src/ffi/mod.rs` (`reconstruct_haplotypes_from_svar2_readbound` signature + the two call sites `:889-890`, `:1045-1046`)
- Modify: `src/reconstruct/mod.rs` (the readbound reconstruct fn — drop the annot params + unreachable match arms + `:605-608` doc)

**Interfaces:**
- Consumes: nothing.
- Produces: `reconstruct_haplotypes_from_svar2_readbound` no longer takes `annot_v_idxs`/`annot_ref_pos`; only the un-annotated output arm remains.

NOTE: This is scoped to the SVAR2 **readbound** reconstruct fn ONLY. Do NOT remove the annot params from the general (SVAR1) reconstruct dispatchers merged in Task 11 — SVAR1 uses them. If Task 11 runs first, keep them independent.

- [ ] **Step 1: Confirm both call sites pass `None`:**

Run: `grep -n "annot_v_idxs\|annot_ref_pos" src/ffi/mod.rs`
Expected: at the two `reconstruct_haplotypes_from_svar2_readbound` call sites they are `None`/`None`. Confirm no SVAR2 readbound caller ever passes `Some`.

- [ ] **Step 2: Remove the params from the Rust fn** in `src/reconstruct/mod.rs`: drop `annot_v_idxs`/`annot_ref_pos` from the readbound reconstruct fn signature, delete the 3 match arms that handle `Some(...)` (keep only the `(None, None)` / un-annotated arm), and delete the now-inaccurate doc at `:605-608`.

- [ ] **Step 3: Remove the params at both FFI call sites** (`ffi/mod.rs:889-890`, `:1045-1046`) and from the `#[pyfunction]` signature if they were exposed to Python (check the `#[pyo3(signature = ...)]` and the Python-side call in `_svar2_haps.py` — if the Python code passes them, drop there too):

Run: `grep -rn "reconstruct_haplotypes_from_svar2_readbound" python/genvarloader/`
If the Python call passes `annot_v_idxs=`/`annot_ref_pos=`, remove those kwargs.

- [ ] **Step 4: Build — expect clean compile** (no unused-var / unreachable warnings for the removed arms):

Run: `pixi run -e dev cargo build 2>&1 | tail -5`
Expected: `Finished`, no warnings about the removed code.

- [ ] **Step 5: Rebuild release + full SVAR2 parity gate:**

Run:
```bash
pixi run -e dev maturin develop --release 2>&1 | tail -1
pixi run -e dev pytest tests/dataset/test_svar2_dataset.py tests/dataset/test_svar2_readbound_*.py -q 2>&1 | tail -3
```
Expected: build `Installed`; tests pass.

- [ ] **Step 6: Commit**

```bash
git checkout pixi.lock 2>/dev/null
git add src/ffi/mod.rs src/reconstruct/mod.rs python/genvarloader/_dataset/_svar2_haps.py
git commit --no-verify -m "$(cat <<'EOF'
refactor(svar2): drop dead annot_* capability from readbound haps kernel

annot_v_idxs/annot_ref_pos were None at both call sites of
reconstruct_haplotypes_from_svar2_readbound; 3 of 4 match arms and their doc
were unreachable (annotated .svar2 haps are NotImplementedError-guarded). Drop
the params and the dead arms.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Extract Rust duplication (carve dispatcher, FFI preamble, present_bit) + clear new clippy nits

Highest-risk task; guarded by the full parity suite. Do each extraction as its own commit so a regression bisects cleanly. Rebuild + re-run parity after EACH extraction.

**Files:**
- Modify: `src/reconstruct/mod.rs` (carve/dispatch helper; `present_bit` move; clippy `explicit_auto_deref` at `:248,251`, `doc_overindented_list_items` at `:19-37,278`)
- Modify: `src/ffi/mod.rs` (readbound preamble helper; `offsets_from_diffs` helper; `type` aliases for `:930,1182,1344`)
- Modify: `src/tracks/mod.rs` (use shared `present_bit`)
- Modify: `src/svar2/mod.rs` (host `present_bit`; test-only `single_range_in_vec_init` at `:593,600,772`)

**Interfaces:**
- Consumes: nothing (internal Rust refactor).
- Produces: `svar2::present_bit`, `carve_chunks`, one carve dispatcher, one readbound-preamble helper, `offsets_from_diffs` — all internal.

- [ ] **Step 1 (extraction 1 — carve dispatcher):** Extract `fn carve_chunks<T>(buf: &mut [T], bounds: &[(usize, usize)]) -> Vec<&mut [T]>` and a single dispatcher generic over the per-chunk work closure, replacing the ~150 verbatim lines duplicated between `reconstruct/mod.rs:424-578` and `:724-880`. Keep BOTH the parallel (`split_at_mut`) and serial (raw-ptr) branches inside the helper, with the Task 3 `debug_assert!` on the serial branch. Output must be byte-identical.

- [ ] **Step 2:** Build + full parity:
```bash
pixi run -e dev cargo build 2>&1 | tail -2
pixi run -e dev maturin develop --release 2>&1 | tail -1
pixi run -e dev pytest tests/dataset/test_svar2_dataset.py tests/dataset/test_svar2_readbound_*.py tests/dataset/test_write_svar2.py -q 2>&1 | tail -3
```
Expected: 31/31 pass. Also `pixi run -e dev cargo test 2>&1 | tail -5` passes.

- [ ] **Step 3: Commit extraction 1**
```bash
git checkout pixi.lock 2>/dev/null
git add src/reconstruct/mod.rs
git commit --no-verify -m "refactor(svar2): extract carve_chunks + shared serial/parallel dispatcher

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 4 (extraction 2 — FFI preamble):** Extract one helper returning `(FlatChannels, lut_bytes, lut_off, regions)` from the 4× readbound preamble (`ffi/mod.rs:934-998,1086-1144,1186-1250,1358-1432`), and `fn offsets_from_diffs(...)` from the 3× prefix-sum loop (`:846-864,1000-1019,1252-1268`). Add a `type` alias for the three readbound return tuples (clears `clippy::type_complexity` at `:930,1182,1344`).

- [ ] **Step 5:** Build + full parity (same commands as Step 2). Expected: 31/31.

- [ ] **Step 6: Commit extraction 2**
```bash
git checkout pixi.lock 2>/dev/null
git add src/ffi/mod.rs
git commit --no-verify -m "refactor(svar2): extract readbound FFI preamble + offsets_from_diffs helpers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 7 (extraction 3 — present_bit):** Move the identical `present_bit` closure (`reconstruct/mod.rs:675-678` == `tracks/mod.rs:759-762`) to a documented `svar2::present_bit`; call it from both. Build + full parity.

- [ ] **Step 8: Commit extraction 3**
```bash
git checkout pixi.lock 2>/dev/null
git add src/svar2/mod.rs src/reconstruct/mod.rs src/tracks/mod.rs
git commit --no-verify -m "refactor(svar2): hoist shared present_bit to svar2::present_bit

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 9 (clippy nits):** Clear the NEW-code clippy warnings only:
```bash
pixi run -e dev cargo clippy --all-targets 2>&1 | tail -40
```
Fix: `reconstruct/mod.rs:248,251` `explicit_auto_deref` (drop `as_deref_mut()`), `doc_overindented_list_items` at `:19-37,278` (4→2 spaces), `single_range_in_vec_init`/`redundant_closure` in the svar2/reconstruct tests. Leave pre-existing warnings (`bigwig.rs`, `reference/mod.rs`, `intervals.rs`, etc.) untouched.

- [ ] **Step 10: Confirm no new clippy warnings + parity green:**
```bash
pixi run -e dev cargo clippy --all-targets 2>&1 | grep -c "warning:"   # compare against the pre-existing baseline count
pixi run -e dev maturin develop --release 2>&1 | tail -1
pixi run -e dev pytest tests/dataset/test_svar2_dataset.py tests/dataset/test_svar2_readbound_*.py -q 2>&1 | tail -3
```
Expected: only pre-existing warnings remain; 31/31 pass.

- [ ] **Step 11: Commit clippy nits**
```bash
git checkout pixi.lock 2>/dev/null
git add src/
git commit --no-verify -m "$(cat <<'EOF'
style(svar2): clear new-code clippy nits (auto-deref, doc indent, type alias)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: Documentation consistency + RELEASE-GATE checklist

All doc corrections (spec §4a–c) plus the release-gate documentation (spec "Out of scope").

**Files:**
- Modify: `docs/roadmaps/rust-migration.md` (Phase 6a: guard matrix, gate footnote, notes log, field-routing task line, PR link, RELEASE-GATE subsection)
- Modify: `skills/genvarloader/SKILL.md:66,91,128,170,193-195,437,442`
- Modify: `docs/source/index.md:51`, `docs/source/faq.md`, `docs/source/write.md`, `docs/source/dataset.md`, `docs/source/format.md`

**Interfaces:**
- Consumes: nothing.
- Produces: docs consistent with shipped behavior; a release-gate checklist in the roadmap.

- [ ] **Step 1: Roadmap `docs/roadmaps/rust-migration.md` Phase 6a:**
  - In the guard-matrix bullet (~`:812-816`), remove `unphased_union` and `"variant-windows"` (both ship now); move them to a supported note.
  - Delete the gate footnote (~`:822-826`) claiming variant-windows parity is untested — it is covered by `test_svar2_readbound_variants.py` + `test_svar2_fields_read.py`.
  - Amend the 2026-07-05 notes-log entry (~`:890`) OR add a 2026-07-13 entry reflecting shipped scope (unphased_union + variant-windows + INFO/FORMAT field routing done).
  - Add a ticked task line: `- [x] var_fields → .svar2 store INFO/FORMAT field routing (plan 2026-07-12-svar2-info-format-field-routing.md).`
  - Fill the `_PR: TBD (branch svar2-m6b-kernel)_` link once the PR exists (Task 13 opens it — come back and fill the number, or leave 🚧 until then).

- [ ] **Step 2: Add the RELEASE-GATE subsection** under Phase 6a in the roadmap:
```markdown
#### ⛔ Release gate (do NOT merge until genoray is released)

This branch is dev-wired to a local genoray checkout and cannot build off this
machine. PyPI genoray tops out at 2.15.0; the INFO/FORMAT field-read +
read-bound gather API lives on genoray main (unreleased). Flip ALL of these at
genoray release, then re-run the full py3xx matrix:

- `Cargo.toml`: `svar2-codec` / `genoray_core` path-deps → published crates.io versions.
- `pixi.toml` [feature.py310.pypi-dependencies]: `genoray = { path = ".../dist/*.whl" }` → `genoray = "==<release>"`.
- `pyproject.toml`: `"genoray"` (unpinned) → `"genoray>=<release>,<next-major>"`.
- Verify the version-floor bumps already made are intended: numpy 0.29, pyo3 0.29, seqpro 0.21.1.
```

- [ ] **Step 3: `skills/genvarloader/SKILL.md` corrections:**
  - `:193-195`: `var_fields` on `.svar2` accepts only `alt|ilen|start` + store INFO/FORMAT fields — NOT `ref`/`dosage` (they raise). State it.
  - `:66,170,437`: "`min_af`/`max_af` requires SVAR-backed genotypes" → "`.svar` only (not `.svar2`, which raises `NotImplementedError`)".
  - `:128,442`: note `extend_to_length` is unsupported for a `.svar2` source (raises — matches Task 5).
  - `:91`: qualify "byte-identical … all four output modes" with "except pure-deletion ALT bytes (see below)".

- [ ] **Step 4: Prose docs `docs/source/`:**
  - `index.md:51`: "Currently supports VCF, PGEN, and BigWig" → mirror README's `.svar`/`.svar2` wording.
  - `format.md`: add the `.svar2` guard-matrix list (unsupported combos) that `faq.md:81` and `write.md:98` promise "for the full list"; OR repoint those two to `faq.md`. Pin the `(unreleased)` changelog row (`:145`) to the target version.
  - `write.md` §"Variants from a genoray sparse store": add a 2-line build snippet (how to create `.svar`/`.svar2` via genoray), since `faq.md:76` promises it.
  - `dataset.md`: add a short "Variant fields (`var_fields`)" section — `.svar2` store INFO/FORMAT fields on `variants`/`variant-windows` (`rv["AF"]`, `win.fields["AF"]`), with the "only alt/ilen/start + store fields" caveat.

- [ ] **Step 5: Verify api.md ↔ __all__ still clean** (no public symbol added by this branch):

Run: `pixi run -e dev python -c "import genvarloader as g; api=open('docs/source/api.md').read(); print('MISSING:', [n for n in g.__all__ if n not in api] or 'none')"`
Expected: `MISSING: none`.

- [ ] **Step 6: Docs build sanity** (optional but preferred):

Run: `pixi run -e docs doc 2>&1 | tail -5`
Expected: build succeeds (or the same warnings as `main` — compare; no NEW errors).

- [ ] **Step 7: Commit**

```bash
git checkout pixi.lock 2>/dev/null
git add docs/ skills/
git commit --no-verify -m "$(cat <<'EOF'
docs(svar2): sync roadmap/skill/prose docs + add release-gate checklist

Roadmap: move unphased_union/variant-windows to supported, drop the stale
gate footnote, add the field-routing task line, add a ⛔ release-gate section
for the genoray path-pins. SKILL: correct var_fields (.svar2 = alt/ilen/start
+ store fields, no ref/dosage), min_af/max_af = .svar only, extend_to_length
unsupported for .svar2. Prose: index.md source list, format.md guard matrix +
changelog pin, write.md build snippet, dataset.md var_fields section.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: `tmp/svar2_mvp/` relocation, final full-suite gate, push, draft PR

Relocate the scratch tree into `tests/benchmarks/`, run the complete verification gate, and open the draft PR with the release-gate warning.

**Files:**
- Move: benchmark/profiling drivers from `tmp/svar2_mvp/` → `tests/benchmarks/` and `tests/benchmarks/profiling/`
- Delete: `.sbatch` files, `env_baseline.txt`, `prof_out/*.md` (from git)
- Modify: `.gitignore` (drop the `tmp/svar2_mvp/prof_out/` line; add `tmp/`)

**Interfaces:**
- Consumes: existing `tests/benchmarks/conftest.py` fixtures (`data_dir`, `kg_dir`, etc.).
- Produces: no machine-specific scratch tracked in git; useful drivers live under `tests/benchmarks/` with parameterized paths.

- [ ] **Step 1: Inventory the tracked tmp files:**

Run: `git ls-files tmp/svar2_mvp/; echo "--- untracked ---"; git status --short tmp/svar2_mvp/`

- [ ] **Step 2: Move the reusable drivers** (`benchmark.py`, `bench_gvl_svar1_vs_svar2.py`, `build_stores.py`, `validate.py`, `split_folded.py`, `prof_*.py`, `perf_kernel_driver.py`) into `tests/benchmarks/`, and the perf shells (`prof_perf.sh`, `e1_profile.sh`, `prof_python.py`) into `tests/benchmarks/profiling/`:
```bash
git mv tmp/svar2_mvp/benchmark.py tests/benchmarks/bench_svar2_getitem.py
# ...repeat per file, choosing descriptive names; use `git add` for currently-untracked ones after moving
```
For untracked files (`bench_gvl_svar1_vs_svar2.py`, `perf_kernel_driver.py`, `prof_cprofile.py`), `mv` then `git add` at the new path.

- [ ] **Step 3: Parameterize hardcoded paths.** Replace `/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa` and similar absolutes with the `tests/benchmarks/conftest.py` fixtures where the file becomes a pytest module, or a module-level constant + `argparse` default for standalone scripts. Match the pattern in the existing `tests/benchmarks/profiling/profile_*.py`.

- [ ] **Step 4: Drop machine-specific scratch from git:**
```bash
git rm tmp/svar2_mvp/e2_build.sbatch tmp/svar2_mvp/e2_subsample.sbatch tmp/svar2_mvp/genoray_debug_build.sbatch tmp/svar2_mvp/e1_bucket_dso.py tmp/svar2_mvp/env_baseline.txt
git rm -r tmp/svar2_mvp/prof_out/
git rm tmp/svar2_mvp/prof_driver.py tmp/svar2_mvp/prof_getitem.py 2>/dev/null || true
```
(Keep only what's genuinely reusable; the perf conclusions already live in `docs/superpowers/notes/` and the roadmap.)

- [ ] **Step 5: Fix `.gitignore`** — remove the self-contradicting `tmp/svar2_mvp/prof_out/` line, add a blanket `tmp/`:
```bash
grep -v "tmp/svar2_mvp/prof_out/" .gitignore > .gitignore.new && mv .gitignore.new .gitignore
printf 'tmp/\n' >> .gitignore
```
Then confirm `tmp/` is gone from tracking: `git status --short tmp/` should show nothing staged except the removals/moves.

- [ ] **Step 6: Sanity-run one relocated benchmark** (if it became a pytest module) or import-check it:
```bash
pixi run -e dev python -c "import ast; ast.parse(open('tests/benchmarks/bench_svar2_getitem.py').read()); print('parses')"
```
Expected: `parses` (full run needs real data; a parse + import check is enough for the gate).

- [ ] **Step 7: FINAL FULL-SUITE GATE.** Rebuild and run everything:
```bash
pixi run -e dev maturin develop --release 2>&1 | tail -1
pixi run -e dev cargo test 2>&1 | tail -6
pixi run -e dev pytest tests -q 2>&1 | tail -6
pixi run -e dev ruff check python/ tests/ 2>&1 | tail -2
pixi run -e dev ruff format --check python/ tests/ 2>&1 | tail -2
pixi run -e dev typecheck 2>&1 | tail -3
pixi run -e dev cargo clippy --all-targets 2>&1 | grep -c "warning:"
```
Expected: cargo tests pass; full pytest tree green (SVAR1 unchanged, SVAR2 31/31, all new tests pass); ruff clean; ruff format clean; typecheck 0 errors; clippy warning count == pre-existing baseline.

- [ ] **Step 8: Commit the relocation**
```bash
git checkout pixi.lock 2>/dev/null
git add -A
git commit -m "$(cat <<'EOF'
chore(svar2): relocate tmp/svar2_mvp scratch into tests/benchmarks

Move the reusable SVAR2 benchmark/profiling drivers under tests/benchmarks/
with parameterized paths (via the existing conftest fixtures), drop
machine-specific sbatch/env scratch and prof_out reports from git (perf
conclusions already live in docs/superpowers/notes/ + the roadmap), and fix
the self-contradicting .gitignore (blanket-ignore tmp/).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 9: Push the branch:**
```bash
git push -u origin svar2-m6b-kernel 2>&1 | tail -3
```

- [ ] **Step 10: Open the draft PR with the release-gate warning:**
```bash
gh pr create --draft --repo mcvickerlab/GenVarLoader --base master --head svar2-m6b-kernel \
  --title "SVAR2 read-bound dataset support (M6b)" \
  --body "$(cat <<'EOF'
## ⛔ DO NOT MERGE until genoray is released

This branch is dev-wired to a local genoray checkout and builds only on the dev
machine. PyPI genoray tops out at 2.15.0; the INFO/FORMAT field-read +
read-bound gather API is on genoray main (unreleased). Before merge, flip ALL:

- [ ] `Cargo.toml`: `svar2-codec` / `genoray_core` path-deps → crates.io versions
- [ ] `pixi.toml`: `genoray = { path = ".../dist/*.whl" }` → `genoray = "==<release>"`
- [ ] `pyproject.toml`: `"genoray"` (unpinned) → `"genoray>=<release>,<next-major>"`
- [ ] Re-run the full py3xx matrix on the released wheel

## Summary

Adds SVAR2 (`.svar2` sparse variant format) as a `gvl.write` source and a live,
read-bound `Dataset` backend: all-Rust FFI kernels (haplotypes, tracks,
variants, variant-windows), INFO/FORMAT field routing into variant outputs via
`var_fields`, and `unphased_union`. See `docs/roadmaps/rust-migration.md`
Phase 6a and the specs/plans under `docs/superpowers/`.

Final pre-merge pass (this session): serial-unsafe-path guard, Python-reachable
panics → PyValueError, `extend_to_length` guard, vectorized write-time
max_ends, test-only oracle relocation, dead FFI capability removal, Rust
de-duplication, typecheck-task fix, and doc consistency.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)" 2>&1 | tail -3
```

- [ ] **Step 11: Backfill the PR link** into `docs/roadmaps/rust-migration.md` Phase 6a (`_PR: TBD_` → the PR URL from Step 10), commit `--no-verify`, and push.

- [ ] **Step 12: Report** the PR URL and the final gate results.

---

## Self-Review

**Spec coverage:**
- §Release gate → Task 12 (roadmap section) + Task 13 (PR warning). ✅
- §1a serial unsafe guard → Task 3. §1b get_unchecked doc → Task 1. §1c panics→PyErr → Task 4. §1d extend_to_length → Task 5. ✅
- §2a oracle relocation → Task 9. §2b dead FFI capability → Task 10. §2c max_ends vectorize → Task 6. §2d region_starts → Task 7. §2e typecheck task → Task 2. §2f tmp/ relocation → Task 13. ✅
- §3a carve helper, §3b FFI preamble, §3c present_bit → Task 11. ✅
- §4a roadmap, §4b SKILL, §4c prose, §4d strip internal refs → Tasks 12 + 1. §4e missing docstrings → covered opportunistically in Tasks 1/9 (see note below). ✅
- §5 clippy → Task 11 Step 9. ✅
- §8 FlankSample issue → Task 8. ✅

**Note on §4e (missing docstrings):** the spec lists docstrings to add (`make_svar2_link`, `_reconstruct_variants`, `_write_from_svar2`, `svar2/store.rs` items). These are folded into the tasks that already touch those files: `_write_from_svar2` in Task 5, the `svar2/store.rs` PyO3 items are low-risk — **add a short Step to Task 10** or fold into Task 1's comment pass. To avoid a gap, the implementer should add the four docstrings as part of whichever task first opens each file; if none does (e.g. `store.rs`), append them to Task 1. Explicitly: add `///` to `svar2/store.rs` `reader`/`store_path`/`#[new]`/`contigs` in Task 1 Step 3.

**Placeholder scan:** no TBD/TODO left except the deliberate `_PR: TBD` which Task 13 Step 11 fills. The Task 4 Step 1 test body has a modeled-on-sibling `...` — flagged explicitly with instructions, acceptable since the exact fixture is repo-specific and the assertion (`pytest.raises(ValueError)` on a strided view) is concrete.

**Type consistency:** helper names consistent across tasks — `require_contiguous` (Task 4), `carve_chunks`/`offsets_from_diffs`/`svar2::present_bit` (Task 11), `build_readbound_*`/`SparseVar2Source` at new paths (Task 9). `_svar2_region_max_ends` signature unchanged (Task 6). `reconstruct_haplotypes_from_svar2_readbound` loses `annot_*` in Task 10 — no later task references those params.
