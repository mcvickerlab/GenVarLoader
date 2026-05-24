# Test Suite Overhaul Design

**Date:** 2026-05-24
**Status:** Draft

## Motivation

The recent internal refactor campaign exposed cleaner seams in the codebase:

- Smaller composable units (`Haps`, `Ref`, `SplicePlan`, `_Variants`, reconstruction kernels, `Ragged*`).
- Components can be constructed from in-memory inputs (numpy/pyarrow) rather than written-out `.gvl` directories.
- Reconstruction kernels are callable directly.

The existing test suite predates these seams: it leans heavily on `gvl.Dataset.open(...)` against on-disk fixtures in `tests/data/*.gvl`, with no `conftest.py` and module-scoped `DATASET = ...` constants. The suite is slow, redundant in places, and fails to cover several polymorphism axes — notably `with_settings`, `subset_to`, `__getitem__` return-type polymorphism, the output-mode matrix, spliced/GTF code paths, and VCF/PGEN/SVAR parity across all output modes.

This is a full overhaul: introduce a two-tier suite, build a shared in-memory builder library, audit existing tests, set up `pytest-cov` reporting, and adopt `pytest-cases` for the matrix-shaped work.

## Goals

1. Make unit-level testing of components possible without on-disk `Dataset` fixtures.
2. Trim slow end-to-end tests aggressively, retaining a small canonical integration tier.
3. Track coverage (report-only, no failing gate).
4. Audit existing tests to identify redundant/useless tests, untested code paths, and uncovered polymorphisms.
5. Standardize the parametrization story around `pytest-cases` for complex/matrix cases.

## Non-Goals

- Rewriting the Rust test (`tests/test_bigwig.rs`).
- Touching ground-truth regeneration scripts (`tests/data/generate_*_ground_truth.py`).
- Refactoring benchmarks (`tests/tracks/bench_cpu_gpu.py`).
- Hard coverage thresholds / per-module coverage gates.
- Hypothesis / property-based testing (may be revisited per-kernel later).

## Architecture

### Directory layout

```
tests/
├── conftest.py              # session/module fixtures for curated .gvl dirs + ref paths
├── _builders/               # in-memory constructors (shared infra; NOT a test package)
│   ├── __init__.py
│   ├── genotypes.py         # sparse genotype arrays, region/sample idx maps
│   ├── variants.py          # _Variants tables (pyarrow) without reading VCF
│   ├── haps.py              # assemble a Haps from in-memory pieces
│   ├── ref.py               # in-memory Ref / fake FASTA Reader
│   ├── tracks.py            # RaggedIntervals, fake BigWig Reader
│   ├── splice.py            # SplicePlan / GTF-shaped frame
│   └── readers.py           # fakes implementing the Reader protocol
├── unit/
│   ├── ragged/              # _ragged, RaggedVariants ops
│   ├── reconstruct/         # haplotype + track reconstruction kernels
│   ├── variants/            # _Variants.from_table, info_fields filter, dosage gating
│   ├── haps/                # Haps.from_path, var_fields plumbing (unit-level)
│   ├── tracks/              # insertion-fill strategies, realignment
│   ├── splice/              # SplicePlan.permutation, get_splice_bed
│   ├── ref/                 # Ref reading + jitter
│   └── dataset/             # __getitem__, subset_to, with_settings, with_seqs polymorphism
├── integration/
│   ├── test_write_read_roundtrip.py    # one canonical roundtrip per variant source
│   ├── test_variant_source_parity.py   # VCF == PGEN == SVAR across output modes
│   ├── test_output_mode_matrix.py      # output-mode polymorphism via Dataset.open
│   └── test_1kg_smoke.py               # one trimmed 1kg sanity check (slow-marked)
└── data/                               # curated golden inputs only; trim aggressively
```

### Tier rules

- **Unit tier never calls `Dataset.open`** and never reads `tests/data/*.gvl`. If a `Dataset`-shaped object is needed, it is constructed via builders or the test operates on the component directly.
- **Integration tier** is the only place on-disk gvl datasets are loaded; all paths come from `tests/conftest.py` fixtures.
- The Rust test stays in place.

## Builder strategy

Builders are the load-bearing piece of this overhaul. They take plain Python / numpy / pyarrow inputs and return the *real* internal type. Mocks/fakes are reserved for the `Reader` protocol boundary.

### Shapes

```python
# tests/_builders/genotypes.py
def make_sparse_genotypes(
    *, n_regions=2, n_samples=3, ploidy=2,
    variants_per_region: int | list[int] | None = None,
    seed: int = 0,
) -> SparseGenotypes: ...

# tests/_builders/variants.py
def make_variants_table(
    *, contig: str = "chr1",
    positions: Sequence[int],
    refs: Sequence[str],
    alts: Sequence[str],
    info: dict[str, list] | None = None,
) -> pa.Table: ...

def make_variants(table: pa.Table | None = None, **table_kwargs) -> _Variants: ...

# tests/_builders/haps.py
def make_haps(
    *, regions=None, samples=None,
    variants: _Variants | None = None,
    genotypes: SparseGenotypes | None = None,
    ref: Ref | None = None,
    var_fields: Sequence[str] = (),
) -> Haps: ...

# tests/_builders/ref.py
def make_ref(contig_seqs: dict[str, bytes]) -> Ref: ...

# tests/_builders/tracks.py
def make_ragged_intervals(per_region: list[list[tuple[int, int, float]]]) -> RaggedIntervals: ...

# tests/_builders/readers.py
class FakeBigWigReader:        # implements Reader protocol
    name: str
    dtype = np.float32
    contigs: list[str]
    coords = "intervals"
    chunked = False
    def read(self, contig, starts, ends) -> RaggedIntervals: ...

class FakeFastaReader: ...     # contigs from dict, coords="bytes"
```

### Rules

1. **Default to minimum viable input.** Every kwarg has a sensible default; tests pass only what they're asserting on. `make_haps()` with no args produces a valid 2-region, 3-sample, ploidy-2 toy.
2. **Builders compose, they don't branch on mode.** No `make_haps(mode="annotated")`. Use `make_haps(...).with_annotations(...)` (the real API).
3. **No hidden disk I/O.** A builder that writes a temp file is a code smell — promote the test to `integration/`.
4. **Builders live in `_builders/`, not `conftest.py`.** Fixtures wrap builders when scope/caching matters; otherwise tests call builders directly. Avoids fixture sprawl.
5. **One file per internal concept**, not per test file. Builders are shared infra.

### Migration trick

When porting an existing test, first ask: "what's the smallest set of builder kwargs that reproduces the input?" If the answer is "I need a real `.gvl` directory," it's an integration test — move it, don't force it into the unit tier.

## Polymorphism matrix

Centralize axes so every relevant test parametrizes from the same source of truth.

```python
# tests/unit/dataset/_axes.py  (mirror in integration/)
OUTPUT_MODES = ["haplotypes", "reference", "annotated", "variants"]
LEN_MODES    = ["ragged", "padded"]
JITTER       = [0, 5]
RC_NEG       = [False, True]
SPLICED      = [False, True]
VARIANT_SRC  = ["vcf", "pgen", "svar"]   # integration only
```

| Axis | Unit | Integration |
|---|---|---|
| Output mode (`with_seqs`) | yes — built on in-memory Haps/Ref | yes — `test_output_mode_matrix.py` via `Dataset.open` |
| `with_settings` lazy reload | yes — assert reload triggered by `var_fields`/`jitter` only | one smoke case |
| `subset_to` | yes — assert lazy view, index mapping | one smoke case |
| `__getitem__` polymorphism | yes — return type per output-mode combo | — |
| `len` modes (ragged vs padded) | yes | parametrized over canonical roundtrip |
| Jitter / rc_neg | yes (kernel-level) | parametrized over canonical roundtrip |
| Splicing | yes — `SplicePlan.permutation`, `get_splice_bed` edges | one spliced roundtrip |
| Insertion-fill strategies | yes — each strategy gets its own kernel test | — |
| VCF/PGEN/SVAR parity | — | yes — `test_variant_source_parity.py` |

Use `pytest.mark.parametrize` with `ids=` so failures read clearly (`test_getitem[haps-ragged-jitter5-rc]`). Where combinations would explode, prune to meaningful subsets — no blind cross-products.

## pytest-cases usage

Add `pytest-cases` to the `dev` pixi env. Use where it earns its keep; fall back to plain `parametrize` for trivial cases.

### Use pytest-cases for

- **Complex object parametrization** — `Haps`, `_Variants`, `SplicePlan` constructions where each case sets several related fields together. e.g. `cases_haps.py` with `case_minimal()`, `case_with_indels()`, `case_spliced_multi_exon()`, `case_with_annotations()` each returning a built `Haps`.
- **Cross-product matrices** — `@parametrize_with_cases` plus multiple `@fixture`-decorated case modules cleanly produces output-mode × len-mode × jitter combinations without manual `itertools.product`.
- **Fixture × parameter mixing** — polymorphism tests (`__getitem__`, `with_settings`, `subset_to`) need a fixture-built base object varied across cases. pytest-cases' `@fixture` + `parametrize_with_cases` handles this.
- **Variant source parity** — one set of "expected output" cases parametrized once, applied across `vcf`/`pgen`/`svar` source fixtures.

### Use plain `parametrize` for

- Single-axis enum lists (e.g. just `LEN_MODES`).
- File-local parametrization where a separate `cases_*.py` is overkill.

### Conventions

- Case modules named `cases_<topic>.py`, colocated with their tests.
- Case functions return *built objects* (via the builders), not raw kwargs.
- Shared axes (`OUTPUT_MODES`, etc.) become case generators when consumed via pytest-cases.

## Coverage tooling

- Add `pytest-cov` and `pytest-cases` to the `dev` pixi env.
- `pyproject.toml`:

  ```toml
  [tool.coverage.run]
  source = ["python/genvarloader"]
  branch = true
  omit = ["*/tests/*", "*/_builders/*"]

  [tool.coverage.report]
  show_missing = true
  skip_covered = false
  exclude_lines = ["pragma: no cover", "if TYPE_CHECKING:", "raise NotImplementedError"]
  ```

- `pixi run -e dev test-cov` → `pytest --cov --cov-report=term-missing --cov-report=html`
- HTML report → `htmlcov/` (gitignored).
- CI: print coverage summary; **no failing threshold**. Upload `htmlcov/` as an artifact.
- Branch coverage on — catches missed `if`/`else` polymorphism branches that the matrix is meant to flush out.

## Audit method

The audit is a structured pre-rewrite pass. Deliverable: `docs/superpowers/specs/2026-05-24-test-audit.md` with three lists.

1. **Delete candidates** — tests matching any of:
   - Tautological assertions (`assert ds is ds`, re-asserting fixture state).
   - Duplicate coverage of an integration path already covered elsewhere.
   - Tests for behavior that no longer exists post-refactor.
   - "Smoke tests" that only verify `Dataset.open` doesn't raise (replaced by one canonical roundtrip).
2. **Port candidates** — tests that are valuable but unnecessarily E2E; move to unit tier once a builder exists.
3. **Keep-as-integration** — true regression tests for write/read roundtrips, parity, golden 1kg checks.

### Mechanics

- Run `pytest --cov` baseline → record per-file coverage and per-test duration.
- For each test file, classify every test function into one bucket. Output as a Markdown table.
- Cross-reference uncovered lines against the polymorphism axes; file gap items inline in the audit doc.

## Migration phasing

Sequenced so the suite is never broken; each step is a separate PR.

1. **Bootstrap.** Add `pytest-cov` + `pytest-cases` config, create empty `_builders/`, `unit/`, `integration/`. Move existing tests under `integration/` wholesale (no rewrites). Suite stays green.
2. **Conftest centralization.** Extract on-disk `.gvl` paths into `tests/conftest.py` fixtures; replace top-level `DATASET = gvl.Dataset.open(...)` module constants with fixtures.
3. **Audit pass.** Produce the audit doc; sign-off before deletions.
4. **Delete pass.** Apply the agreed-upon redundancies in one PR.
5. **Builders + unit tier.** Implement builders module-by-module, port one component at a time in this order: ragged → reconstruct → variants → haps → tracks → splice → dataset polymorphism. Each component is a separate PR.
6. **Integration trim.** Once unit coverage lands for an area, prune duplicate integration tests in that area.
7. **CI report.** Wire coverage HTML report into CI as an artifact.

## Open questions

- None at design time. Decisions on hard coverage thresholds, Hypothesis adoption, and Rust test integration are deferred to follow-up work.

## Out of scope

- Rewriting `tests/test_bigwig.rs`.
- Ground-truth regeneration scripts.
- Benchmarks.
- Hard coverage thresholds.
- Property-based / Hypothesis testing.
