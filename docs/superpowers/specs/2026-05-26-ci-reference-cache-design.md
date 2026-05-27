# CI hg38 Reference Cache — Design

**Date:** 2026-05-26
**Status:** Approved (design)

## Problem

CI is slow because every test job re-downloads the full hg38 reference. The `test`
and `test-cov` pixi tasks both `depends-on = ["gen", "gen-1kg"]`, and `gen`
(`tests/data/generate_ground_truth.py`) downloads `hg38.fa.gz` (~1 GB) from UCSC via
`pooch`, then `gzip -dc | bgzip` and `samtools faidx` it. This runs in all five CI
jobs (`pytest` on py310/py311/py312/py313 plus the py312 `coverage` job), each on a
fresh `ubuntu-latest` runner, so the ~1 GB download and its post-processing repeat on
every job of every run.

The tests only exercise a handful of contigs (toy data on chr19/chr20 with a chr1
empty region; 1kg on chr21/chr22), but the whole-genome file is downloaded regardless.
`pixi`'s `cache: true` caches the conda environment, not the downloaded FASTA.

## Goal

Eliminate the repeated 1 GB download (and its bgzip/faidx) from CI by caching the
prepared reference across runs, with no changes to test data, test contigs, or the
generation logic.

## Non-goals

- Migrating test data onto chr21/chr22 or any other contigs.
- Committing a subset reference into the repo or hosting it as a release asset.
- Caching generated `consensus/`, `vcf/`, or `pgen/` outputs (these regenerate fast
  relative to the download and stay on the per-run critical path unchanged).
- Changing `pixi` tasks or the generation scripts' behavior.

## Approach

Add a GitHub Actions cache step for the reference directory in
`.github/workflows/test.yaml`, in both the `pytest` matrix job and the `coverage` job,
placed before the step that runs the pixi `test` / `test-cov` task.

```yaml
- name: Cache hg38 reference
  uses: actions/cache@v4
  with:
    path: tests/data/fasta
    key: hg38-ref-c1dd8706   # first 8 chars of the pooch known_hash; bump if the reference changes
```

### What is cached

The entire `tests/data/fasta/` directory, which after a successful `gen` contains:

- `hg38.fa.gz` — the raw pooch download.
- `hg38.fa.bgz` — the bgzipped reference used by the datasets.
- `hg38.fa.bgz.fai` / `hg38.fa.bgz.csi` — the samtools index.

Caching all of these means that on a cache hit `gen` short-circuits every expensive
step:

- `pooch.retrieve` finds `hg38.fa.gz` already present with a matching `known_hash`, so
  it does not re-download.
- The `if not reference.with_suffix(".bgz").exists()` guard skips the bgzip.
- The `if not reference.with_suffix(".csi").exists()` guard skips `samtools faidx`.

`tests/data/fasta` is already gitignored, so caching it does not interact with tracked
files.

### Cache key strategy (Decision 1)

The key is a **static string derived from the reference's `known_hash`**
(`hg38-ref-c1dd8706`), not `hashFiles(...)` over the generation script.

Rationale: the reference identity changes essentially never, while
`generate_ground_truth.py` changes frequently. Keying on the script would invalidate
the cache and force a fresh 1 GB download on unrelated edits. A static key tied to the
reference hash only invalidates when the reference itself changes.

Trade-off: if the reference is ever swapped (new `known_hash`), the key must be bumped
by hand. To make this discoverable, add a short comment at the `known_hash` line in
`tests/data/generate_ground_truth.py` pointing to the workflow cache key.

No `restore-keys` fallback is used: a partial/older reference cache is not desirable
(we want an exact, hash-matched reference or a clean miss).

### First-run parallelism (Decision 2)

On a cold key (first adoption, or after a deliberate key bump), all five jobs start in
parallel and each misses the cache, so up to five simultaneous downloads occur — no
worse than today. Once any one job finishes and saves the cache, every subsequent run
(and every job in it) hits.

We accept this rather than introducing a serialized `prepare-reference` warm-up job
with a `needs:` dependency. The warm-up adds workflow complexity to optimize a cost
paid only once per reference version. It can be added later if the cold-run cost
proves painful.

## Affected files

- `.github/workflows/test.yaml` — add the cache step to the `pytest` job and the
  `coverage` job.
- `tests/data/generate_ground_truth.py` — add a comment at the `known_hash` line
  noting that the CI cache key must be bumped if the reference changes. (No behavior
  change.)

## Success criteria

- After one warm run on a branch, subsequent CI runs restore `tests/data/fasta` from
  cache and perform no hg38 download, bgzip, or faidx (verifiable in job logs: the
  pooch download line and bgzip/faidx steps do not execute; the cache step reports a
  hit).
- Test results are unchanged (same pass/skip/xfail counts as before the change).
- Editing `generate_ground_truth.py` (without touching the reference) does **not**
  invalidate the reference cache.

## Risks

- **GitHub cache eviction / limits:** repo caches are capped at 10 GB with LRU
  eviction, and branch caches only fall back to the default branch. A bgzipped hg38 is
  well under the cap; eviction would simply cause a one-time re-download, which is
  safe.
- **Stale reference after a swap:** mitigated by the bump-the-key comment; a forgotten
  bump would serve the old reference. Because `gen` regenerates all downstream ground
  truth from whatever reference is present, a mismatched cached reference would surface
  as test failures rather than silent wrong results.
