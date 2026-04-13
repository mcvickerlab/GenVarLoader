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

# Run all tests (pytest + cargo)
pixi run -e dev test

# Run a single pytest test
pixi run -e dev pytest tests/dataset/test_dataset.py::test_name -v

# Lint
pixi run -e dev ruff check python/
pixi run -e dev basedpyright python/

# Build docs
pixi run -e docs doc
```

The build system uses Maturin (Rust + Python). Rust code is compiled automatically when running tests via pixi.

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

## Development Notes

- **Pixi environments**: Use `-e dev` for development, `-e docs` for documentation, `-e py310`/`py311`/`py312`/`py313` for Python version testing. Platform is linux-64.
- **Ruff config**: E501 (line length) is ignored.
- **BasedPyright**: Configured permissively; type annotations follow patterns in `_types.py`.
- **Conventional commits**: Project uses commitizen for versioning.
- **Test markers**: `@pytest.mark.slow` for slow tests (excluded by default).
