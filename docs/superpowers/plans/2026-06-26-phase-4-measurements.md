# Phase 4 Close-Out: Perf + RSS Measurements

**Date:** 2026-06-26
**Machine:** Carter HPC (AMD EPYC 7543, linux-64)
**Corpus:** chr22_geuv (5 samples, 165 e-gene regions)
**Measured-at code HEAD:** 32132c9 (test(bench): realistic chr22_geuv write/update perf driver)
**Build:** `maturin develop --release` (abi3, CPython 3.10)
**NUMBA_NUM_THREADS=1** (single-threaded control)

---

## write() — wall-clock (median of 3)

| Run | wall |
|-----|------|
| 1   | 1.959s |
| 2   | 1.911s |
| 3   | 1.934s |

**Median: 1.934s**

## write() — peak RSS (memray)

Peak memory usage: **3.520 GB**

---

## update() — wall-clock (median of 3)

| Run | wall |
|-----|------|
| 1   | 0.091s |
| 2   | 0.081s |
| 3   | 0.081s |

**Median: 0.081s** (track=read-depth-2, samples=5)

## update() — peak RSS (memray)

Peak memory usage: **3.519 GB**

> **Caveat:** run_update() writes the base dataset (untimed gvl.write) and then runs the timed gvl.update in the SAME process. This memray process-peak is therefore dominated by the base-dataset write (≈ the write() peak above), NOT the marginal cost of update(). The update WALL (0.081s) IS correctly isolated to the gvl.update call; update's peak RSS in isolation is not measured by this single-process driver.

---

## Full-tree parity gate

### Rust backend (default)
```
984 passed, 21 skipped, 4 xfailed, 1 warning in 277.23s (0:04:37)
```
Result: **PASS** (0 failures)

### Numba backend (GVL_BACKEND=numba)
```
984 passed, 21 skipped, 4 xfailed, 1 warning in 254.08s (0:04:14)
```
Result: **PASS** (0 failures). @slow tests run by default in this repo (no -m "not slow" addopts, no --runslow skip hook). The pre-existing flaky test tests/unit/test_double_buffered_loader.py::test_shm_cleanup_after_close (intermittent /dev/shm gvl- segment leak on the numba backend; rust always passes) did NOT fail this run — not a regression.

---

## Write-path parity (tests/parity)

```
77 passed, 1 skipped in 79.77s (0:01:19)
```
Result: **PASS**

---

## cargo-test + lint + typecheck

| Check | Result |
|-------|--------|
| `cargo test --release` | PASS (107 + 4 + 0 = 111 tests; pre-existing `unused variable: n_contigs` warning noted, not a regression) |
| `ruff check python/ tests/` | PASS (all checks passed) |
| `ruff format --check python/ tests/` | PASS (after auto-format of _write.py) |
| `pyrefly check` | PASS (0 errors, 37 suppressed, 392 warnings) |

---

## Notes

- Test infrastructure: added `__init__.py` to `tests/unit/`, `tests/unit/dataset/`,
  `tests/integration/`, `tests/integration/dataset/` to fix collection collision between
  two same-named `test_write.py` files (committed separately as fix commit f92e386).
- `maturin develop --release` produced abi3 wheel `genvarloader-0.35.0-cp310-abi3-linux_x86_64.whl`.
- memray output files written to worktree root (w.bin, u.bin) to avoid cross-device EXDEV.
