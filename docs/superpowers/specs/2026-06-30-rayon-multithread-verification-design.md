# Rayon multithread verification harness + thread-cap fix

**Date:** 2026-06-30
**Issue:** [#263](https://github.com/mcvickerlab/GenVarLoader/issues/263) — Nondeterministic rayon deadlock when iterating `Dataset` from concurrent spawn worker processes (v0.36.0)
**Status:** design approved, awaiting spec review

## Problem

`genvarloader==0.36.0` introduced rayon parallelism into the Rust load path. In production, iterating a `Dataset` concurrently from multiple `spawn`-ed worker processes **intermittently hard-deadlocks** (threads parked on a futex, load ≈ 1.0, 0% GPU). It is nondeterministic and aggravated by CPU oversubscription. Stable only at `tasks_per_gpu=1`.

The parallel paths are **not exercised under realistic multithreaded load by the test suite**:

- The end-to-end read tests never cross the `should_parallelize(total_bytes)` byte gate on the small test corpus, so the parallel branch is never taken through the public API.
- `tests/parity/test_rayon_equivalence.py` *does* force `parallel=True` at the FFI boundary, but on tiny golden inputs — so rayon runs effectively single-threaded, with no concurrency and no contention. It proves *logical* serial==parallel equivalence, not *deadlock freedom under load*.

## Key finding from the code audit

**gvl's entire Rust load path is lock-free.** A search across `src/**/*.rs` finds **zero** `Mutex`, `RwLock`, `OnceCell`, `lazy_static`, or `Lazy`, and no nested rayon `install`/`broadcast`:

- `src/intervals.rs`, `src/reconstruct*` and the getitem kernels build **disjoint** per-query output slices via the `split_at_mut` cursor idiom, then `into_par_iter().for_each(...)`. No shared mutable state, no locks.
- `src/bigwig.rs::write_track` uses a `thread_local!` reader cache (not a shared lock) and an unconditional `par_iter` — and is on the **write** path, not the reported read/iterate path.

Consequence: issue #263's "audit for a lock-across-parallel-region / nested-install / condvar-in-worker" almost certainly comes up **empty inside gvl's own code**. Combined with #263's evidence (spawn → independent per-process pools, nondeterminism, `N×16` threads on a ~15-core cgroup, stable only when unsubscribed), the dominant driver is **CPU oversubscription** that `cap_threads()` fails to prevent, plus the GIL being held across every rayon region.

Every rayon call today runs **with the GIL held** — there is no `py.allow_threads` anywhere in `src/`.

## Goals

1. Make the genuinely-multithreaded paths runnable and verified on the normal corpus (the "disable size thresholding" ask), permanently.
2. Fix the two concrete `cap_threads()` bugs that cause oversubscription (#263 ask #2).
3. Release the GIL around the rayon FFI calls, defensively.
4. Ship a multiprocess stress reproducer that either reproduces the hang (proving it is gvl-triggerable) or provides evidence the cause was oversubscription (now fixed).

Non-goals: rewriting the getitem dispatch, changing the on-disk format, or touching the write-path `par_iter`.

## Design

Four workstreams in one bundled PR. #4 (root-cause fix beyond `allow_threads`) is **contingent** on the harness reproducing a hang.

### 1. Force-parallel override — `python/genvarloader/_threads.py`

Single chokepoint. Add a `GVL_FORCE_PARALLEL` env var read by `should_parallelize()`:

```python
def should_parallelize(total_bytes: int) -> bool:
    if _force_parallel():          # GVL_FORCE_PARALLEL truthy → always parallel
        return True
    return total_bytes >= num_threads() * _MIN_BYTES_PER_THREAD
```

- Truthy = `{"1", "true", "yes", "on"}` (case-insensitive), matching common env conventions.
- Permanent, documented mechanism (not a throwaway monkeypatch). It is what the regression test flips, and a support lever for reproducing customer issues.
- Symmetry with a force-serial toggle is **out of scope** (YAGNI); the size gate already yields serial by default on small inputs.

### 2. `cap_threads()` correctness fix — `python/genvarloader/_threads.py`

Two bugs, both named in #263:

**(a) `setdefault` → overwrite.** An ambient `RAYON_NUM_THREADS=16` (base image) currently wins in spawn workers, so `cap_threads()` never caps. Change to assign directly:

```python
os.environ["RAYON_NUM_THREADS"] = str(_NUM_THREADS)   # GVL's resolved count wins
```

GVL's resolved count wins; users steer explicitly via `GVL_NUM_THREADS`. Still runs before the first rust parallel call, so it takes effect at global-pool init. (Overwriting *after* pool init is a no-op — rayon reads the env var once — so the existing "must run first" contract is unchanged and still documented.)

**(b) CFS-quota-aware CPU detection.** `_detect_cpus()` uses `sched_getaffinity`, which ignores a CFS *quota* (reports 16 on a 15.3-core budget). Add cgroup quota parsing:

- cgroup v2: read `/sys/fs/cgroup/cpu.max` → `"<quota> <period>"`; if quota is not `"max"`, `quota_cpus = ceil(quota / period)`.
- cgroup v1 fallback: `/sys/fs/cgroup/cpu/cpu.cfs_quota_us` and `cpu.cfs_period_us`; quota `> 0` ⇒ `ceil(quota / period)`.
- Effective count = `max(1, min(affinity_count, quota_cpus))` when a quota is present, else the affinity count.
- All file reads are wrapped so a missing/unreadable cgroup file (non-Linux, cgroup-less) falls back to affinity detection. No hard dependency on cgroup layout.

Extract the cgroup-quota probe into a small helper (`_cgroup_cpu_quota() -> int | None`) so it is unit-testable by pointing it at fixture files / monkeypatched readers.

### 3. GIL release around rayon FFI — `src/ffi/mod.rs`

Wrap each parallel-capable FFI call's Rust core in `py.allow_threads(|| ...)` so the GIL is dropped for the duration of the rayon region. Applies to the parallel entry points: `reconstruct_haplotypes_from_sparse`, `reconstruct_haplotypes_fused`, `shift_and_realign_tracks_sparse`, `get_diffs_sparse`, `intervals_to_tracks`, `tracks_to_intervals`.

Correctness constraints (must hold, verified by the parity suite staying green):

- The rayon closures touch **no Python** — confirmed by the lock-free audit; they operate purely on `ndarray` views. Safe to run without the GIL.
- `PyReadonlyArray` / `PyReadwriteArray` borrows are resolved to `ndarray` views **before** entering `allow_threads`, and no Python code runs inside, so numpy buffers are not mutated underneath us. Standard PyO3 pattern.
- Functions that currently take `py: Python` already have the token; those that don't gain a `py: Python` parameter (PyO3 injects it, signature-transparent to Python callers).

This removes "GIL held across the parallel region" as a variable and lets other Python threads progress during the rust compute — a defensive fix independent of whether the harness reproduces a hang.

### 4. Multiprocess stress reproducer + forced-parallel equivalence — `tests/`

**(a) Stress reproducer** (`tests/integration/test_rayon_stress.py`, marked `slow`). Mirrors #263:

- `multiprocessing.get_context("spawn")` + `concurrent.futures.ProcessPoolExecutor` with N workers.
- Each worker opens its own `Dataset` and iterates it many times.
- Run with `GVL_FORCE_PARALLEL=1` and a per-worker `RAYON_NUM_THREADS` chosen so that `n_workers × threads` comfortably exceeds the available cores — reproducing the #263 oversubscription condition that widens the race window.
- Wrap `future.result(timeout=...)` in a wall-clock timeout. **Hang → `TimeoutError` → test fails = deadlock reproduced.** Clean completion across repeated launches = evidence the cause was oversubscription.
- Parameterize a couple of `(n_workers, launches)` points; keep total runtime bounded so it fits the `slow` tier.

**(b) Forced-parallel equivalence** (unit/integration, default tier). With `GVL_FORCE_PARALLEL=1`, assert `dataset[...]` output is byte-identical to the serial result (`GVL_FORCE_PARALLEL` unset / small input) across the output modes (haplotypes, tracks, annotated, variants). This reaches what the tiny-golden parity test cannot: real end-to-end parallel dispatch through the public API on the actual corpus.

**(c) `cap_threads` unit tests** (`tests/unit/test_threads.py`, extend). Cover: `GVL_FORCE_PARALLEL` truthy/falsy parsing; `RAYON_NUM_THREADS` is **overwritten** not preserved; `_cgroup_cpu_quota()` parses v2 `cpu.max` (quota and `"max"`), v1 quota/period, and returns `None` on missing files; effective count = `min(affinity, quota)`.

### 5. Contingent: root-cause fix beyond `allow_threads`

Only if the stress harness still reproduces a hang **after** the cap fix and `allow_threads`. Driven by systematic-debugging on the reproducer. Suspect order, given the lock-free audit: (a) a dependency's rayon usage on the shared global pool (e.g. bigtools during track reads); (b) rayon global-pool init races under contention. Left as a spec-level placeholder — no speculative code without harness evidence.

## Verification

Rust rebuilt (`maturin develop --release`) before any Python test that imports the extension, per the repo's stale-binary caveat.

- `pixi run -e dev cargo-test` — Rust unit tests (compiles from source).
- `pixi run -e dev pytest tests/parity/test_rayon_equivalence.py -q` — serial==parallel==golden stays byte-identical after `allow_threads`.
- `pixi run -e dev pytest tests/unit/test_threads.py -q` — cap/quota/force-parallel unit tests.
- `pixi run -e dev pytest tests/integration/test_rayon_stress.py -q` — stress reproducer (slow tier; run explicitly).
- Full tree `pixi run -e dev pytest tests -q` + `ruff check python/ tests/` + `typecheck` before push (shared-code change touches `_threads.py`).

## Risks & mitigations

- **`allow_threads` correctness** — mitigated by the lock-free audit (closures touch no Python) and the parity suite remaining byte-identical. If any closure is later found to touch Python, that call stays GIL-held.
- **Overwriting a user's `RAYON_NUM_THREADS`** — intentional per #263; `GVL_NUM_THREADS` is the documented user knob. Documented in the `_threads.py` module docstring.
- **Stress test flakiness / runtime** — bounded launches, `slow` marker, generous timeout; the test asserts *completion*, not timing, so it is not perf-sensitive.
- **Harness fails to reproduce** — that is itself a result: it makes oversubscription the leading explanation and the cap fix the resolution. Recorded in the PR.

## Public API / skill impact

`GVL_FORCE_PARALLEL` and the `RAYON_NUM_THREADS`-overwrite behavior are env-level knobs, not `__all__` symbols — no `skills/genvarloader/SKILL.md` public-API change. Document both env vars in the `_threads.py` module docstring; mention `GVL_NUM_THREADS` / `GVL_FORCE_PARALLEL` in user-facing docs if a threading section exists.
