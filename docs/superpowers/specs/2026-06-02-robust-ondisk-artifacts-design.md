# Robust on-disk artifacts: atomic creation + validation — Design

**Status:** Approved design with all implementation decisions resolved; ready for implementation planning.

**Goal:** Make GenVarLoader's generated on-disk artifacts safe under concurrency and resilient to format drift. Two coupled problems, one initiative:

1. **Concurrency (closes [#21]):** Parallel jobs that share an input (classically one reference FASTA across a SLURM array) race to build the same `.gvlfa` cache and corrupt it. The same race can affect `gvl.write` dataset directories.
2. **Validation hardening:** Files written into a `*.gvl/` dataset directory by `gvl.write` are write-once-assume-valid — no real format-version gate and no integrity check on read. A future on-disk layout change would silently load old datasets as wrong data; an interrupted write leaves an undetected partial dataset.

This extends the robust-sidecar idiom already established by the `.gvlfa` FASTA cache (PR #206) and the existing `SvarLink` (`_svar_link.py`): fingerprint/version + graceful resolution + safe creation.

[#21]: https://github.com/mcvickerlab/GenVarLoader/issues/21

---

## Background & decisions

The corruption in #21 stems from a **shared mutable target**: today `_fasta_cache.build()` does `np.memmap(gvlfa_dir / "sequence.bin", mode="w+")` and writes into the live path; `gvl.write()` writes its many files directly into the destination directory. Concurrent builders interleave writes, and readers can observe partially-written artifacts.

Two ways to make creation safe were considered:

- **Advisory locks** (`flock`/`fcntl`) make other jobs wait. On network filesystems (NFS, Lustre, GPFS) — exactly where #21's parallel jobs run — advisory locks are unreliable: `flock` is often local-only across nodes, POSIX locks depend on a frequently-misconfigured network lock manager, and Lustre needs a specific mount option. The failure mode is silent (looks locked, isn't).
- **Atomic build-to-temp-then-rename** makes the *publish* indivisible. Each job builds into its own private sibling temp directory, then `os.replace`s it into place. `rename(2)` is a single metadata operation the server serializes (atomic even on NFS, within one filesystem); there is no shared target during the build, so writes cannot interleave. "Last writer wins" is harmless because every builder produces byte-identical valid content.

**Decisions (from brainstorming):**

| # | Decision |
|---|---|
| Concurrency primitive | **Atomic rename for correctness** (works on any filesystem) **+ advisory lock as a best-effort optimization** (avoid N redundant builds). The lock is never load-bearing for correctness. |
| Lock library | Add **`filelock`** dependency (v1). |
| Validation depth (datasets) | **Format-version gate + structural/size integrity** on open. |
| Dataset mismatch policy | **Always raise an actionable error** — datasets cannot auto-rebuild (no retained source). Only the FASTA cache auto-rebuilds (it has its source). |
| Atomic-creation scope | **Both** the `.gvlfa` cache **and** `gvl.write` dataset creation. |
| Back-compat | A dataset with **no `format_version`** is treated as the current layout (`1.0.0`) and loaded best-effort (no warning, no forced regeneration). |
| Out of scope | genoray `.gvi` and pysam `.fai`/`.gzi` are created by those libraries; making them atomic needs upstream changes. Documented as a limitation. |

---

## Architecture

A new small module **`python/genvarloader/_atomic.py`** owns the single reusable primitive both artifact types need — safely publishing a directory. Everything else reuses it.

### `_atomic.py` — safe directory publish (single responsibility)

```python
@contextmanager
def atomic_dir(
    dest: Path,
    *,
    overwrite: bool = False,
    lock: bool = True,
    timeout: float = DEFAULT_LOCK_TIMEOUT,
) -> Iterator[Path]:
    """Yield a private temp dir to build into; atomically publish it to `dest` on
    clean exit, or remove it on error. Optional best-effort lock avoids redundant
    concurrent builds but is never required for correctness."""
```

- **Temp location:** `<dest>.tmp.<pid>-<uniq>/` as a *sibling* of `dest` (same filesystem, so `os.replace` is atomic; avoids `EXDEV`).
- **Publish:** on clean context exit, `os.replace(tmp, dest)`. On exception, remove `tmp` and re-raise; `dest` is never touched.
- **Existing dest:** checked up front — exists & not `overwrite` → `FileExistsError`; exists & `overwrite` → **move-aside then rename**: `os.replace(dest, <dest>.old.<uniq>)`, then `os.replace(tmp, dest)`, then `rmtree(<dest>.old.<uniq>)`. Two fast metadata renames keep the dest-absent window minimal; if the second rename fails the `.old` copy is still recoverable.
- **Lock layer** (`filelock`): a `<dest>.lock` sibling, **internal with a 60s default `DEFAULT_LOCK_TIMEOUT`** (not exposed on the public API — the lock is only a best-effort optimization; atomic rename is the correctness guarantee). Flow is **double-checked**: acquire lock (with `timeout`) → re-validate `dest` (another job may have just published it → reuse, skip build) → else build. On lock timeout or a silent network-FS no-op, fall back to building anyway; atomic rename keeps that correct.

### `_fasta_cache.py` (from PR #206)

`build()` and `migrate_legacy()` refactored to publish through `atomic_dir` rather than writing `sequence.bin`/`metadata.json` into the live directory. `ensure_cache` already has the format-version gate + blake2b fingerprint; it gains the lock + atomic publish and a double-check re-classification inside the lock. Stale/corrupt caches still auto-rebuild (the source FASTA is available).

### `_write.py`

`write()`'s body runs against a temp dir yielded by `atomic_dir(path, overwrite=overwrite, ...)`; the final `os.replace` publishes the complete `*.gvl/`. The existing `overwrite` / `FileExistsError` semantics move into `atomic_dir`. `Metadata` gains a real **`format_version: SemanticVersion`** field — bumped only when the on-disk layout changes, distinct from the existing package `version` field.

### `_open.py`

`_load_metadata()` gains a `_validate(metadata, path)` step:

1. **Format-version gate** — incompatible major (too-new *or* too-old) → actionable `ValueError` ("written by format N; regenerate with `gvl.write`"). Missing `format_version` → treat as `1.0.0`.
2. **Structural + size integrity** — required files exist, and each `.npy`/memmap `stat().st_size` equals the size implied by the shapes in metadata (`n_regions`, `ploidy`, `n_samples`, final offset values, track set). Mismatch → `ValueError` naming the file.

Datasets never auto-rebuild.

---

## Data flow

### A. Building the FASTA cache (`ensure_cache`)

1. Dispatch `.fa` vs `.gvlfa` and decide build / migrate / rebuild / reuse (unchanged).
2. If a build is needed:
   - `with atomic_dir(gvlfa_dir, lock=True, timeout=T) as tmp:`
     - lock acquired (or timed out → proceed)
     - **double-check:** re-classify `gvlfa_dir`; if another job just published a fresh cache, skip building and reuse it
     - else write `sequence.bin` + `metadata.json` into `tmp`
   - context exit → `os.replace(tmp, gvlfa_dir)`
3. return `(meta, data_path)`.

### B. Writing a dataset (`gvl.write`)

1. `with atomic_dir(path, overwrite=overwrite, lock=True, timeout=T) as tmp:`
   - existing-dest handled up front (`FileExistsError` unless `overwrite`)
   - every write (`input_regions.arrow`, `regions.npy`, `genotypes/*`, `intervals/**`, `metadata.json` with `format_version`) targets `tmp`
2. exit → atomic publish to `path`. An interrupted write leaves only an orphan `<path>.tmp.*`, never a half-written `path`.

### C. Opening a dataset (`Dataset.open`)

1. `_load_metadata()` reads `metadata.json`.
2. `_validate(metadata, path)`: format-version gate, then structural + size integrity.
3. construct readers as today.

### Orphan / lock-file policy

Temp dirs are removed on exception. A *crashed* process can still leave an orphan `<dest>.tmp.*`; these are harmless (distinctly named, ignored on open) and documented rather than swept (sweeping risks deleting a live job's temp). `<dest>.lock` files persist empty and are reused.

---

## Error handling

| Situation | Behavior |
|---|---|
| Lock timeout / silent network-FS no-op | `log.info`, build anyway — atomic rename keeps it correct (never an error) |
| Build raises mid-way | temp dir removed; exception propagates; live `dest` untouched |
| Dataset `dest` exists, `overwrite=False` | `FileExistsError` (preserved behavior) |
| FASTA format too-new | `ValueError` "newer than supported" (existing behavior) |
| Dataset format too-new / too-old-incompatible | actionable `ValueError` → regenerate with `gvl.write` |
| Dataset missing `format_version` | treat as `1.0.0`, load (no warning) |
| Dataset integrity fail (missing file / size mismatch) | `ValueError` naming the file → regenerate |
| FASTA cache integrity fail | auto-rebuild (source available), else `ValueError` |

**Correctness invariant:** a reader never observes a partially-written `dest`, and two builders never write the same file.

---

## Test plan

### `tests/unit/test_atomic.py` (the primitive)
- publishes on clean exit (dest appears, temp gone); removes temp + leaves no dest on exception
- `overwrite=False` + existing dest → `FileExistsError`; `overwrite=True` replaces
- lock double-check: `build_fn` runs once when dest becomes valid mid-wait; lock timeout → falls back to build, no error

### Concurrency regression (`multiprocessing`, `@pytest.mark.slow`) — the #21 fix
- N processes `ensure_cache` the same `.fa` concurrently → resulting `sequence.bin` is valid and **byte-identical** to a single-process build; no corruption
- N processes `gvl.write` the same path (overwrite) → exactly one valid, openable dataset; no partial dirs left as `path`

### Format / validation
- dataset `format_version` too-new → `ValueError`; missing → loads as `1.0.0`; truncated/missing `.npy` → `ValueError` naming it
- write→open round-trip validates clean; back-compat: a dataset with the `format_version` field stripped still opens

### Regression
- full fast suite green (write + open are now atomic); `gen` + a `Dataset` round-trip

Cross-platform: `filelock` covers posix/Windows; `os.replace` is atomic within a directory on both.

---

## Out of scope (documented limitations)

- **genoray `.gvi`** (VCF/PGEN variant index) and **pysam `.fai`/`.gzi`** (FASTA index) are created by their respective libraries; making their creation atomic/locked requires upstream changes. Concurrent jobs relying on these still depend on those libraries' own behavior.
- No per-region / per-sample integrity beyond total byte-size checks (full-content hashing of multi-GB datasets is intentionally avoided as too expensive — same tradeoff `_data_size_ok` documents for the FASTA cache).

---

## Resolved implementation decisions

1. **Overwrite publish (non-empty existing dir).** `os.replace` cannot replace a non-empty dir on POSIX, so `overwrite=True` uses **move-aside then rename**: `os.replace(dest, <dest>.old.<uniq>)` → `os.replace(tmp, dest)` → `rmtree(<dest>.old.<uniq>)`. Two fast metadata renames keep the dest-absent window minimal, and a failure of the second rename leaves the old data recoverable under `.old` (vs. an `rmtree`-first approach that spans the whole delete and loses old data on a mid-delete crash).
2. **Lock timeout / API surface.** `DEFAULT_LOCK_TIMEOUT = 60s`, kept **internal** — `lock`/`timeout` are not exposed on `gvl.write` / `Reference.from_path` / `Fasta`. The lock is a best-effort optimization only; atomic rename is the correctness guarantee, so tuning is rarely needed.
3. **`format_version`.** Current dataset layout is **`1.0.0`**. Bump **MAJOR** only when an existing dataset can no longer be read correctly by new code (incompatible layout change); minor/patch for backward-compatible additions. A missing `format_version` field is treated as `1.0.0`.
