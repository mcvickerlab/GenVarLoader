# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

<!-- rtk-instructions v2 -->
## RTK (Rust Token Killer) - Token-Optimized Commands

### Golden Rule

**Always prefix commands with `rtk`**. If RTK has a dedicated filter, it uses it. If not, it passes through unchanged. This means RTK is always safe to use.

**Important**: Even in command chains with `&&`, use `rtk`:
```bash
# ❌ Wrong
git add . && git commit -m "msg" && git push

# ✅ Correct
rtk git add . && rtk git commit -m "msg" && rtk git push
```

### RTK Commands by Workflow

#### Build & Compile (80-90% savings)
```bash
rtk cargo build         # Cargo build output
rtk cargo check         # Cargo check output
rtk cargo clippy        # Clippy warnings grouped by file (80%)
rtk tsc                 # TypeScript errors grouped by file/code (83%)
rtk lint                # ESLint/Biome violations grouped (84%)
rtk prettier --check    # Files needing format only (70%)
rtk next build          # Next.js build with route metrics (87%)
```

#### Test (90-99% savings)
```bash
rtk cargo test          # Cargo test failures only (90%)
rtk vitest run          # Vitest failures only (99.5%)
rtk playwright test     # Playwright failures only (94%)
rtk test <cmd>          # Generic test wrapper - failures only
```

#### Git (59-80% savings)
```bash
rtk git status          # Compact status
rtk git log             # Compact log (works with all git flags)
rtk git diff            # Compact diff (80%)
rtk git show            # Compact show (80%)
rtk git add             # Ultra-compact confirmations (59%)
rtk git commit          # Ultra-compact confirmations (59%)
rtk git push            # Ultra-compact confirmations
rtk git pull            # Ultra-compact confirmations
rtk git branch          # Compact branch list
rtk git fetch           # Compact fetch
rtk git stash           # Compact stash
rtk git worktree        # Compact worktree
```

Note: Git passthrough works for ALL subcommands, even those not explicitly listed.

#### GitHub (26-87% savings)
```bash
rtk gh pr view <num>    # Compact PR view (87%)
rtk gh pr checks        # Compact PR checks (79%)
rtk gh run list         # Compact workflow runs (82%)
rtk gh issue list       # Compact issue list (80%)
rtk gh api              # Compact API responses (26%)
```

#### JavaScript/TypeScript Tooling (70-90% savings)
```bash
rtk pnpm list           # Compact dependency tree (70%)
rtk pnpm outdated       # Compact outdated packages (80%)
rtk pnpm install        # Compact install output (90%)
rtk npm run <script>    # Compact npm script output
rtk npx <cmd>           # Compact npx command output
rtk prisma              # Prisma without ASCII art (88%)
```

#### Files & Search (60-75% savings)
```bash
rtk ls <path>           # Tree format, compact (65%)
rtk read <file>         # Code reading with filtering (60%)
rtk grep <pattern>      # Search grouped by file (75%)
rtk find <pattern>      # Find grouped by directory (70%)
```

#### Analysis & Debug (70-90% savings)
```bash
rtk err <cmd>           # Filter errors only from any command
rtk log <file>          # Deduplicated logs with counts
rtk json <file>         # JSON structure without values
rtk deps                # Dependency overview
rtk env                 # Environment variables compact
rtk summary <cmd>       # Smart summary of command output
rtk diff                # Ultra-compact diffs
```

#### Infrastructure (85% savings)
```bash
rtk docker ps           # Compact container list
rtk docker images       # Compact image list
rtk docker logs <c>     # Deduplicated logs
rtk kubectl get         # Compact resource list
rtk kubectl logs        # Deduplicated pod logs
```

#### Network (65-70% savings)
```bash
rtk curl <url>          # Compact HTTP responses (70%)
rtk wget <url>          # Compact download output (65%)
```

#### Meta Commands
```bash
rtk gain                # View token savings statistics
rtk gain --history      # View command history with savings
rtk discover            # Analyze Claude Code sessions for missed RTK usage
rtk proxy <cmd>         # Run command without filtering (for debugging)
rtk init                # Add RTK instructions to CLAUDE.md
rtk init --global       # Add RTK to ~/.claude/CLAUDE.md
```

### Token Savings Overview

| Category | Commands | Typical Savings |
|----------|----------|-----------------|
| Tests | vitest, playwright, cargo test | 90-99% |
| Build | next, tsc, lint, prettier | 70-87% |
| Git | status, log, diff, add, commit | 59-80% |
| GitHub | gh pr, gh run, gh issue | 26-87% |
| Package Managers | pnpm, npm, npx | 70-90% |
| Files | ls, read, grep, find | 60-75% |
| Infrastructure | docker, kubectl | 85% |
| Network | curl, wget | 65-70% |

Overall average: **60-90% token reduction** on common development operations.
<!-- /rtk-instructions -->

---

## Overview

GenVarLoader is a Python/Rust hybrid library for efficiently loading genomic data with genetic variation to train sequence models. It reconstructs haplotypes and re-aligns functional genomic tracks on the fly without writing personalized genomes to disk.

## Commands

All commands require the `pixi` package manager. Use `pixi run -e dev <task>` for development tasks.

```bash
# Generate test data (required before first test run)
pixi run -e dev gen

# Run all tests (pytest + cargo) — runs the WHOLE tree (tests/dataset/, tests/unit/, etc.)
pixi run -e dev test

# Run a single pytest test
pixi run -e dev pytest tests/dataset/test_dataset.py::test_name -v

# Run a directory's tests (cover BOTH dataset and unit when changing shared code)
pixi run -e dev pytest tests/dataset tests/unit -q

# Lint (cover python/ AND tests/ — `python/` alone misses test-only issues like unused imports)
pixi run -e dev ruff check python/ tests/
pixi run -e dev ruff format python/ tests/
pixi run -e dev typecheck

# Build docs
pixi run -e docs doc
```

The build system uses Maturin (Rust + Python).

**IMPORTANT — rebuild Rust before testing Rust changes:** `pixi run -e dev pytest` (and `pixi run -e dev test`) do **not** rebuild the Rust extension. After editing anything in `src/`, run `pixi run -e dev maturin develop --release` first, or pytest silently imports the *stale* compiled extension — parity/integration tests then pass or fail against the old binary, not your change. (`cargo test`/`cargo-test` compile from source and are unaffected; this only bites the Python tests that import the extension.)

**Before pushing a change that renames/removes a public symbol or touches shared code, run the full tree** (`pixi run -e dev pytest tests -q`, or the full `pixi run -e dev test`). Scoped runs like `pytest tests/dataset` skip `tests/unit/` (e.g. `tests/unit/dataset/test_build_reconstructor.py`), so a stale reference there fails only in CI.

## Architecture

### Hybrid Python/Rust

- `python/genvarloader/` — main Python package
- `src/` — Rust extension (BigWig interval extraction via PyO3/bigtools)

### Core Data Flow

**Writing:** `write(bed, variants, bigwigs) → dataset_dir/`
1. Normalizes variants (left-align, bi-allelic, atomized)
2. Extracts BigWig intervals and re-aligns them to haplotype coordinates when indels are present
3. Stores metadata, sparse genotypes, and interval data

**Reading:** `Dataset.open(path, reference?) → RaggedDataset`
1. Loads metadata and region index map
2. Initializes lazy readers (`Haps` from genotypes, or `Ref` from reference)
3. Eager indexing `dataset[region_idx, sample_idx]` triggers data loading

### Key Modules

- `_dataset/_impl.py` — `Dataset`, `RaggedDataset`, `ArrayDataset` classes; main user API
- `_dataset/_write.py` — dataset writing pipeline
- `_dataset/_reconstruct.py` — haplotype and track reconstruction from stored data
- `_dataset/_genotypes.py` — genotype handling (VCF/PGEN sparse storage)
- `_dataset/_tracks.py` — track re-alignment to account for indels
- `_variants/` — variant record structures and VCF/PGEN reading
- `_bigwig.py` — `BigWigs` reader wrapping the Rust backend
- `_ragged.py` — ragged array utilities built on `seqpro.rag.Ragged`
- `_types.py` — `Reader` protocol, `AnnotatedHaps`, type aliases

### Key Abstractions

**`Reader` protocol** (`_types.py`): Abstract interface for all data sources (VCF, PGEN, BigWig, FASTA). Implementors must provide `read()`, `name`, `dtype`, `contigs`, `coords`, `chunked`.

**`AnnotatedHaps`**: Haplotype sequences with parallel arrays for variant indices and reference coordinates. Dtype is `S1` (single byte per nucleotide).

**Ragged arrays**: Variable-length data throughout. `RaggedIntervals`, `RaggedSeqs`, `RaggedTracks`, `RaggedAnnotatedHaps` all wrap `seqpro.rag.Ragged`. Use `.to_padded()` to materialize into dense arrays.

**`Dataset`** (frozen dataclass): Lazy view over stored data. Subsetting via `subset_to()` returns a new lazy view; eager access is `dataset[region, sample]`. The return type varies based on whether a reference genome and tracks are present.

### Dataset Directory Layout

```
dataset_dir/
├── metadata.json          # sample names, contigs, ploidy, max_jitter
├── input_regions.arrow    # BED regions + region index map
├── genotypes/             # sparse genotype storage (if variants provided)
└── intervals/             # track data (or annot_intervals/ with annotation)
```

## Maintaining the `genvarloader` skill

`skills/genvarloader/SKILL.md` is an AI-agent reference for gvl's public Python API. **Any PR that changes the public API must also update this skill.** Public API = anything exported in `python/genvarloader/__init__.py` `__all__`, plus the docstrings, signatures, and defaults of `gvl.write`, `Dataset.open`, and every `Dataset.with_*` method.

In scope:
- New, removed, or renamed public symbols
- Changed signatures, defaults, or accepted literal values (e.g. new `with_seqs` kind)
- New output modes, insertion-fill strategies, or splice/site-only behavior
- Changed bcftools/plink2 preprocessing requirements
- Changed on-disk format that affects how users open datasets

When a change ships, update the relevant section of the skill and re-check the "Common gotchas" and "Where to look next" pointer table. The skill is published to https://www.skills.sh/ as `mcvickerlab/GenVarLoader` (installable via `npx skills add mcvickerlab/GenVarLoader`); keep it accurate against `main`.

## Docs audit before feature/breaking-change PRs

Before opening any PR that adds a user-facing feature or makes a breaking change, audit and update the user-facing docs so they stay consistent with the code:

- `README.md` (features, installation, requirements)
- `docs/source/*.md` — especially `api.md`, `faq.md`, `write.md`, `dataset.md`, `format.md`, `index.md`
- `skills/genvarloader/SKILL.md` (see "Maintaining the `genvarloader` skill" above)

Check for: now-false claims (deleted backends, removed deps, changed defaults, renamed/removed symbols), new user-facing config or environment variables that need documenting, and changed installation/preprocessing (bcftools/plink2) requirements.

**`api.md` must stay in sync with `__all__`.** Every symbol exported in `python/genvarloader/__init__.py`'s `__all__` needs an autodoc entry in `docs/source/api.md`; adding a public symbol without one silently drops it from the rendered API reference. Quick check:

```bash
python -c "import re,genvarloader as g; api=open('docs/source/api.md').read(); print('MISSING:', [n for n in g.__all__ if n not in api] or 'none')"
```

The auto-generated `docs/source/changelog.md` (built from commit messages via `changelog.md.j2`) does **not** count as documentation — never treat a changelog entry as a substitute for prose docs. This gate complements the skill-maintenance rule above: public-API changes must update the skill, and any user-facing change must also keep the prose docs true.

## Rust migration roadmap

Any task that mentions "rust" (adding or porting Rust code, touching `src/`, or migrating numba/Python hot paths) **must** read `docs/roadmaps/rust-migration.md` before starting and update it as part of the work — tick completed tasks, record measurement results under the relevant checkpoint, and set the phase status marker (⬜/🚧/✅) + PR link. The roadmap is the source of truth for migration sequencing and the byte-identical parity contract.

## Development Notes

- **Pixi environments**: Use `-e dev` for development, `-e docs` for documentation, `-e py310`/`py311`/`py312`/`py313` for Python version testing. Platform is linux-64.
- **Ruff config**: E501 (line length) is ignored.
- **Pyrefly**: Configured permissively; type annotations follow patterns in `_types.py`.
- **Conventional commits**: Project uses commitizen for versioning.
- **Test markers**: `@pytest.mark.slow` for slow tests (excluded by default).
