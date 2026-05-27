# Test Coverage Initiative — Design

**Date**: 2026-05-25
**Status**: spec
**Follows**: 2026-05-24 test-suite overhaul (Phase 7 shipped CI coverage job)
**Baseline**: 63% overall (see `2026-05-24-test-audit-coverage-baseline.txt`)

## Goal

Cover risky/critical modules and specific untested user-facing paths. Not coverage-for-its-own-sake.

Two motivating observations from the baseline:

1. The lowest-reported coverage modules — `_dataset/_genotypes.py` (5%) and `_dataset/_intervals.py` (12%) — are almost entirely `@nb.njit` kernels. coverage.py cannot instrument numba-compiled code, so those numbers are a measurement artifact, not untested code. These kernels are nevertheless correctness-critical (haplotype reconstruction, AF filtering, exonic variant choice) and deserve explicit behavior tests.
2. Several pure-Python user-facing modules genuinely lack tests: `_torch.py` (31%), `_bigwig.py` (47%), `_fasta.py` (51%), parts of `_dataset/_impl.py` (64%), `data_registry.py` (22%).

## Non-goals

- Hitting a global coverage percentage target.
- New fixture machinery, property-testing frameworks, or speculative test builders.
- Refactoring tested code for testability.
- Network-dependent tests for `data_registry.py` download paths.

## Shape

Three waves in a single bundled PR (solo-maintainer preference). Waves are commit boundaries. After Wave 2 lands, decide whether Wave 3 ships in the same PR or defers.

### Where tests live

Follow existing tiers — no new directories.

- Kernel tests → `tests/unit/test_kernels_genotypes.py`, `tests/unit/test_kernels_intervals.py`. Direct calls into numba functions with hand-built numpy arrays. No fixtures, no dataset.
- Reader tests → extend existing `tests/unit/test_bigwig.py`, `tests/unit/test_fasta.py`, `tests/tracks/`. New `tests/unit/test_torch.py`.
- API-surface tests → extend `tests/dataset/test_dataset.py` / `test_indexing.py`; new `tests/dataset/test_with_methods.py` for `with_*` branches.

## Wave 1 — Kernel correctness

Each kernel gets a parametrized scenario matrix. Inputs are hand-built numpy arrays; expected outputs computed by inspection (no oracle code).

**`reconstruct_haplotype_from_sparse`** (`_dataset/_genotypes.py`)

- reference-only (no variants)
- single SNV
- atomized insertion
- atomized deletion
- deletion spanning region start
- deletion spanning region end
- overlapping variants — first ALT wins
- shift consumes ref-only
- shift consumes ref + variant
- shift exactly at variant boundary
- shift exceeds region
- annotation buffers populated (variant idx + ref pos)
- negative `ref_start` padding

**`get_diffs_sparse`**: abbreviated matrix of the same scenarios, exercising both code paths (with/without `q_starts`+`v_starts`, with/without `keep`).

**`filter_af`**: min only, max only, both, neither (no-op short circuit), `(2, n_slices)` offsets layout.

**`choose_exonic_variants`**: variant fully inside, spans start, spans end, entirely outside.

**`intervals_to_tracks`** (`_dataset/_intervals.py`): empty interval set, single interval, multiple non-overlapping. Insertion-fill interactions tested at the `_tracks` level, not here.

No property-based testing. Plain `pytest.parametrize`.

## Wave 2 — User-facing readers + DataLoader

**`_torch.py`** (`tests/unit/test_torch.py`)

- Wrap existing tiny test dataset in `DataLoader` with `num_workers=0` and `num_workers=2`.
- Assert: batch shapes, dtype, deterministic ordering with seeded `RandomSampler`.
- Collate handles `RaggedSeqs` correctly.
- `to_dataset` `return_indices` flag round-trips.
- Skip GPU-specific paths.

**`_bigwig.py`**: extend existing tests with — missing chrom contract (verify from code), `float32` value dtype, intervals reader with `out=` preallocated buffer, range fully off-contig, error message when path doesn't exist.

**`_fasta.py`**: contig-name normalization (`chr1` ↔ `1`), missing contig raises with helpful message, padded vs ragged read modes, range past contig end, `Reader.read()` protocol round-trip.

## Wave 3 — API surface

**`Dataset.with_*` chain** (`tests/dataset/test_with_methods.py`): one test per method confirming (a) returns a new lazy view (frozen dataclass), (b) original unchanged, (c) rejection of invalid args raises a clear message. Methods enumerated from `_impl.py`: `with_seqs`, `with_tracks`, `with_annotations`, `with_indels`, `with_jitter`, `with_seed`, `with_insertion_fill`, `with_return_indices`, and any others present at implementation time.

**`_open.py`**: missing dataset dir, missing reference when required, version mismatch in metadata, presence/absence of intervals dir.

**`_indexing.py`**: tuple vs single index, slice with step, negative indices, out-of-bounds, fancy boolean mask, empty selection.

**`_query.py`**: AF filter combined with exonic filter, empty filter result.

**`_reference.py`** splice path: SVAR vs FASTA reference, missing exon, splice across contig boundary (should error).

**`data_registry.py`**: registry resolves already-cached paths. Download path gated behind env var — `# pragma: no cover` and document.

## Test discipline

- No coverage-chasing. Unreachable or numba-shadowed branches get `# pragma: no cover`, not fake tests.
- No new fixture machinery. Use the existing tiny test dataset and `tests/_builders/`. Do not extend builders speculatively.
- No mocks for things that can run for real (bigtools, FASTA reader). Use the test data registry.
- `@pytest.mark.slow` stays opt-in; nothing here needs it.

## Coverage measurement

- Track overall coverage and per-module coverage.
- Update `.coveragerc` to **omit** numba-only modules (`_dataset/_intervals.py`) and **exclude** numba-jitted functions in mixed modules (`_dataset/_genotypes.py`) from the coverage gate. They're verified via Wave 1 behavior tests, not by coverage.py.
- Target for pure-Python modules: **≥80% line coverage** for `_torch`, `_bigwig`, `_fasta`, `_open`, `_indexing`, `_impl` (excluding `__repr__`/`__str__`). Other modules: best-effort, no gate.
- Do not raise the global gate in this PR. Adjust after Wave 3 lands and the numbers are real.

## Risks

- Numba kernel tests trigger first-run JIT on CI; `@nb.njit(cache=True)` should mitigate after the first run but the first run is slow. Acceptable — these are unit-tier and infrequent.
- Some `with_*` methods may have hidden interactions (e.g. `with_indels` + `with_seqs="annotated"`). Those interaction tests live in `tests/dataset/`, not `tests/unit/`.
- `data_registry.py` network path is untestable offline; documented as `# pragma: no cover`.

## Out of scope (explicitly deferred)

- Branch coverage in `_dataset/_impl.py` beyond `with_*` methods and the obvious user-facing branches.
- `_dataset/_reference.py` non-splice branches (already at 65%, mostly load-paths exercised by integration tests).
- `_dataset/_haps.py` (already 86%) — the missing lines are mostly unreachable error paths.
- `_dataset/_write.py` (already 91%).

## Success criteria

- Wave 1 tests pass; behavior matches existing integration tests (no kernel regressions surfaced).
- Wave 2 modules hit ≥80% line coverage on pure-Python lines.
- Wave 3 lands `with_*` matrix; remaining items either land or are explicitly deferred in a follow-up note.
- One bundled PR; commits separated by wave so reviewers can read it in order.
