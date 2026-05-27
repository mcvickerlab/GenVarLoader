# Test Coverage Deeper Initiative — Design

**Date**: 2026-05-25
**Status**: spec
**Follows**: 2026-05-25 test-coverage initiative (PR #195) which raised baseline to 74%

## Goal

Deepen tests of code already covered (cross-mode equivalence, kernel cross-checks, oracle parity for all output modes, determinism) and fill remaining gaps in important pure-Python modules. Pin down subtle invariants (NaN handling, contig-boundary edges, empty-shape datasets). Not coverage-for-its-own-sake.

## Non-goals

- Coverage-percentage chasing.
- New fixture machinery or test builders beyond what's already in `tests/_builders/`.
- Network-dependent tests for `data_registry` download paths.
- Refactoring code for testability (note real issues separately).
- CLI tests (no CLI exists).

## Existing infrastructure to lean on

- `tests/integration/dataset/test_ds_haps.py` already parametrizes over `vcf`/`pgen`/`svar` and validates `haplotypes` output against `tests/data/consensus/` bcftools-generated FASTAs.
- `tests/data/1kg_consensus/` has ground-truth FASTAs for the 1kg slow tier.
- `tests/conftest.py` exposes `phased_vcf_gvl`, `phased_pgen_gvl`, `phased_svar_gvl`, `consensus_dir`, `reference`.
- `pytest_cases` is the established style for scenario matrices in this repo.

## Shape

Single bundled PR, three groups of commits, each landable independently.

### Group A — Deepen tests of covered code

**A1. Oracle parity across output modes.** `test_ds_haps` covers `haplotypes`. Add equivalent parity tests against `consensus/` ground truth for:
- `with_seqs("annotated")` — verify `AnnotatedHaps.haps` matches the consensus FASTA, and that `var_idx` / `ref_pos` are internally consistent (reference positions monotonic in the absence of indels, variant idx values correspond to actual variant records, `-1` only at padded positions).
- `with_seqs("reference")` — verify the reference path returns the unaltered reference slice for each region. Trivial but currently untested at the integration tier.

Reuse the existing `vcf`/`pgen`/`svar` parametrization.

**A2. Cross-mode equivalence.** For the same `(region, sample)` query:
- `RaggedDataset` and `ArrayDataset` (via `with_len(int)`) must return identical content when the requested length matches the ragged length. Where ragged length differs from fixed length, the array path must produce the ragged content padded/truncated according to the documented rule.
- `VCF`/`PGEN`/`SVAR` opens of the same underlying genotypes must return identical haplotypes.
- Sample-name order vs. integer-index order must round-trip identically.

**A3. Kernel cross-checks (equivalence between code paths).**
- `get_diffs_sparse` fast path (no `q_starts`/`v_starts`) must agree with the slow path (with all spanning args) when no variant spans a region boundary.
- `filter_af` 1-D and 2-D `geno_offsets` layouts must produce equivalent `keep` arrays for the same logical input.

**A4. Determinism tests.**
- `with_seed(s)` applied twice → bitwise identical output across two opens of the same dataset.
- `with_jitter(j)` + `with_seed(s)` → same jitter offsets across repeated reads.
- DataLoader with seeded `Generator` → same batch order across two runs (`pixi -e py310`).

### Group B — Fill remaining gaps

**B5. `_query.py` filter combinations.** Currently 65%. Test AF filter + exonic-only + sample subset interactions:
- AF + exonic-only together (intersection of keeps).
- Empty filter result (no variants survive both filters).
- Sample subset interacting with AF filter (per-sample AF differs).

**B6. `_ragged.py` utilities (59%).** Test the missing public-surface utilities — likely conversion (`to_padded`, slicing, dtype coercion). Read the module's `__all__` first; only test public API.

**B7. `_variants/_sitesonly.py` (74%).** The sites-only VCF read path. Missing lines suggest filter handling and edge cases.
- Sites-only VCF with no INFO fields requested.
- Sites-only VCF with all INFO fields requested.
- Sites-only VCF on a region with no variants.

**B8. `_dataset/_utils.py` (61%).** Read the module; pick public helpers worth testing. Skip private one-offs.

**B9. `_torch.py` collate + `return_indices`.**
- `Dataset.to_dataloader(..., return_indices=True)` produces batches where the indices align with the data.
- Custom `collate_fn` is respected.
- `Dataset.to_dataset(transform=fn)` applies the transform.

### Group C — Subtle invariants

**C10. AF filter with NaN.** What does `filter_af` do when an allele frequency is NaN? Test current behavior; if it silently keeps NaN variants (likely — `nan >= 0.05` is False, `nan <= 0.5` is False, both bounds reject), document the contract with a test. If behavior is buggy (NaN slips through one bound but not the other), surface as `DONE_WITH_CONCERNS`.

**C11. Spanning deletion at contig boundary.** A deletion that runs off the END of a contig. The reconstruction kernel pads with `N` per the existing `case_spanning_del_end` pattern; verify this through the full Dataset API, not just the kernel.

**C12. Empty / single-sample / single-region shapes.**
- `Dataset` with one region and many samples — indexing, slicing, DataLoader collate.
- `Dataset` with many regions and one sample — same.
- Empty selection via boolean mask or empty integer array (already covered for indexing; ensure also covered for downstream consumers).

**C13. Write → open round-trip on edge inputs.**
- Empty BED → write should either succeed and produce an empty dataset, or raise a clear error.
- Single-variant VCF with a single-region BED.
- Overlapping BED regions (same chrom, ranges intersect).
- BED with a contig missing from the reference FASTA.

For each: write the dataset, open it, exercise the simplest valid query, and confirm either the documented happy path or the documented error.

## Test discipline

- No new fixture machinery. Use `tests/conftest.py` fixtures and `tests/_builders/` only.
- No mocks for things that can run for real.
- `@pytest.mark.slow` for anything that touches the 1kg fixtures or runs many parameterized iterations.
- `# pragma: no cover` for any truly unreachable branch encountered.
- If a test reveals a real bug, the implementer surfaces it as `DONE_WITH_CONCERNS` and the controller decides whether to fix or document.

## File layout

- A1 → extend `tests/integration/dataset/test_ds_haps.py` (or new file `test_ds_haps_modes.py` if the existing one grows past ~150 lines).
- A2 → new `tests/integration/dataset/test_cross_mode_equivalence.py`.
- A3 → extend the existing kernel test files in `tests/unit/dataset/genotypes/`.
- A4 → new `tests/integration/dataset/test_determinism.py`.
- B5 → new `tests/unit/dataset/test_query.py` (or extend existing if one exists — discover at impl time).
- B6 → new `tests/unit/ragged/test_ragged_utils.py`.
- B7 → extend `tests/unit/variants/test_variants_info_fields.py` (already exists) or new `test_sitesonly.py` if scope diverges.
- B8 → new `tests/unit/dataset/test_dataset_utils.py`.
- B9 → extend `tests/unit/test_torch.py`.
- C10 → extend `tests/unit/dataset/genotypes/test_filter_af.py`.
- C11 → extend `tests/integration/dataset/` with a small contig-boundary fixture.
- C12 → extend `tests/integration/dataset/test_subset.py` if it exists, or `test_dataset.py`.
- C13 → new `tests/integration/dataset/test_write_edge_cases.py`.

## Coverage measurement

Re-run `pixi run -e dev pytest tests --cov=python/genvarloader --cov-report=term` at the end. Save to `docs/superpowers/specs/2026-05-25-test-coverage-deeper-after.txt`. Expect overall to land around 78-82%.

## Risks

- Some equivalence tests may reveal real discrepancies (e.g. VCF vs PGEN handling subtly differs at some edge). When this happens: surface as DONE_WITH_CONCERNS with a specific repro; do not adjust the test to mask divergence.
- `AnnotatedHaps` parity tests need to know the exact shape contract for `var_idx`/`ref_pos` arrays; the implementer reads `_types.py` and `_genotypes.py` annotation kernel logic to ground assertions.
- C13 (write round-trip on edge inputs) may surface previously-unreported bugs in `_write.py`. Acceptable — flag and continue.

## Out of scope (defer)

- `data_registry.py` network-path coverage (download tests).
- Refactoring large files identified during this work.
- `_torch.py` GPU device tests.
- Performance/benchmark tests.

## Success criteria

- All A/B/C items either land as tests OR are explicitly deferred with a one-line note.
- No reduction in coverage on previously-covered modules.
- Real bugs discovered during the work are filed as issues or PRs separate from this initiative.
- One bundled PR; commits grouped by A/B/C boundaries.
