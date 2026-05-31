# Handoff: codspeed bench suite + profiling (resume on another machine)

**Written:** 2026-05-30. The **entire cluster (carter AND cellar) is undergoing a
multi-day migration** — no cluster filesystem will be reachable. Dev resumes on a local
machine or cloud VM. **git is the only transport**, so both the code and the build-source
data travel in this branch (see "Data").
**Branch:** `feat/bench-codspeed-profiling` (10 commits + handoff/format/bundle commits ahead of `main`).
**Spec:** `docs/superpowers/specs/2026-05-29-codspeed-perf-tracking-design.md`
**Plan:** `docs/superpowers/plans/2026-05-29-codspeed-perf-tracking.md`

## TL;DR of where things stand

The benchmark + profiling **suite is built and works**. There is **one open problem**:
the committed benchmark dataset (`tests/benchmarks/data/chr22_geuv.gvl/`) is a **broken
build** — its regions are massively over-extended (one to 3.1 Mb) and its track intervals
are inflated 11 MB → 36 MB. Root cause is understood (below). The fix is a **pending
decision** between three options; all require regenerating the dataset from source, which
needs `/carter` data (a portable copy has been staged — see "Data").

Everything except the dataset correctness is done and reviewed.

## How to get the branch on the new machine

The branch MUST be pushed before the cluster goes down (no filesystem access afterward).
Caveats hit while pushing from here:
- `fork` (`git@github.com:d-laub/genome-loader.git`) is a **dead URL** — repo renamed. Set the
  correct fork: `git remote set-url fork git@github.com:d-laub/GenVarLoader.git` (verify the name).
- `origin` = `mcvickerlab/GenVarLoader` (push needs write access).
- The pre-push hook runs `ruff-format`; formatting is already committed, so it should pass now.

Push (run yourself so interactive auth works — type with a leading `!` in Claude, or in a shell):
```bash
git push -u fork feat/bench-codspeed-profiling     # after fixing the fork URL
# or, if you have write access:  git push -u origin feat/bench-codspeed-profiling
```
On the new machine: `git clone <remote> && git checkout feat/bench-codspeed-profiling`.
The clone contains EVERYTHING: code, the committed (broken) `.gvl`, and the build-source tar.

## What is DONE (suite — works, reviewed task-by-task)

Implemented per the plan, Tasks 1–8 + a final review polish:

- `pixi.toml`: `pytest-codspeed` dep; tasks `bench`, `bench-local`, `profile-{haps,tracks,variants}`,
  `memray-{haps,tracks,variants}`.
- `tests/benchmarks/_capture.py` (+ `test_capture.py`): capture-and-replay helper that records
  the first call's args of a hot numba fn by patching the **consumer** module namespace
  (gvl imports these by name). 2 unit tests pass.
- `tests/benchmarks/_indices.py`: shared `batch_indices(n_regions, n_samples, n)` used by
  conftest, e2e, and the profiler.
- `tests/benchmarks/conftest.py`: session fixtures — `bench_dataset` (opens the committed
  dataset, skips if absent) and `captured_{haplotypes,diffs,intervals_to_tracks,realign_tracks}`.
  NOTE: a `captured_germline_ccfs` fixture was intentionally dropped — `_infer_germline_ccfs`
  never fires on this data (no CCF field); the e2e `variants` bench covers that path.
  NOTE: `captured_realign_tracks` uses the `with_seqs("haplotypes").with_tracks(...)` path
  because `shift_and_realign_tracks_sparse` only fires there (the tracks-only path never
  re-aligns).
- `tests/benchmarks/test_micro.py`: 4 micro-benches (get_diffs_sparse, reconstruct_haplotypes_from_sparse,
  intervals_to_tracks, shift_and_realign_tracks_sparse). Warmup is OUTSIDE the timed region;
  buffer-writer kernels asserted via `np.asarray(out).size > 0`.
- `tests/benchmarks/test_e2e.py`: 5 e2e benches (haplotypes, annotated, variants, tracks,
  tracks_only) at `with_len(16384)`.
- `tests/benchmarks/profiling/profile.py`: `--mode {haplotypes,tracks,variants}`,
  `NUMBA_NUM_THREADS=1`; py-spy/memray outputs are gitignored.
- `docs/superpowers/REGRESSIONS.md`: appended "Profiling results (local, chr22 GEUVADIS
  slice)". Key local finding (0.24.x side only): the dominant per-batch cost in ALL three
  modes is **awkward-array ragged assembly** (`numpyarray._carry`, `_kernels.__call__`,
  `concat`) driven by `_query.py:_getitem_unspliced` + `reverse_complement_ragged`, ABOVE
  gvl's own numba kernels. Most promising optimization target = that ragged-assembly path.
  CAVEAT: this is 0.24.x only; no 0.6.1 comparison was run locally.

Verification (all green on the current tree): `pixi run -e dev pytest tests/benchmarks -p no:cov -q`
→ 11 passed (9 benches + 2 capture unit tests). `pixi run -e dev bench-local` → 9 benches pass.

## The OPEN PROBLEM (must fix before this is mergeable)

### Symptom
The committed `chr22_geuv.gvl` has `intervals/read-depth/intervals.npy` = 36 MB and region
spans up to **3,138,304 bp** (mean 205,800 bp) for nominal 16,384 bp windows.

### Root cause (confirmed, deterministic — NOT non-determinism)
In `python/genvarloader/_dataset/_write.py::write`:
```
gvl_bed = _write_from_pgen(..., extend_to_length)   # :246 reassigns bed; extends region ENDS via variants
_write_regions(path, gvl_bed, contigs)              # :259 writes regions.npy (extended ends)
_write_track(path, gvl_bed, tr, samples, max_mem)   # :264 extracts BigWig intervals over those ends
```
With `extend_to_length=True` (the default), region ends are pushed out using the variant set
so haplotypes still reach the requested length after deletions. Track intervals are then
extracted over those **variant-extended** regions. So stored track size is deterministically
coupled to the variant set.

### How the current committed dataset got broken
During the size-fix (commit history below) I restricted the source variants to the **nominal**
16,384 bp windows with `plink2 --extract bed0`. That removed the just-past-the-window variants
that `extend_to_length` needs to find where to stop, so the extension ran away. Measured from
`regions.npy` (same 165 starts):

| build | mean region span | max span | intervals.npy |
|---|---|---|---|
| full variants (FAITHFUL) | 17,184 bp (~window + 0.8 kb real extension) | 20,331 bp | 11 MB |
| restricted (CURRENT, BROKEN) | 205,800 bp | 3,138,304 bp | 36 MB |

Real `extend_to_length` extension is tiny (<4 kb worst case). The 3.1 Mb spans are pure
artifact of starving the extender.

### The fix decision (PENDING — pick one, then regenerate)
The original reason for restricting variants: `gvl.write` stores the **entire** source variant
table in `genotypes/variants.arrow` (~1M chr22 variants ≈ 94 MB), even though only ~77K overlap
the windows. Reducing region count does NOT shrink it — only restricting the source does. So
we must shrink the source WITHOUT starving `extend_to_length`:

1. **Window + safe flank extract (recommended — faithful + small).** `--extract bed0` over
   each window ± ~50 kb (real extension <4 kb, so this preserves correct extension and faithful
   ~17 kb regions / 11 MB intervals). Est. `variants.arrow` ~15–20 MB, total ~30 MB.
   Verify after rebuild: `regions.npy` spans ≈ 17 kb (NOT 200 kb).
2. **Disable `extend_to_length` (simplest + smallest).** `gvl.write(..., extend_to_length=False)`
   + restrict variants to exact windows. `variants.arrow` ~6 MB, intervals ~11 MB, total ~18 MB,
   internally consistent. Tradeoff: haplotypes are reference-padded after deletions instead of
   extended — a valid gvl mode but not the default user path.
3. **Full variants, accept size.** No restriction; faithful but `variants.arrow` ~94 MB
   (~105 MB committed). Defeats the size goal.

I recommended #1. The user paused to discuss before choosing — **get/confirm the decision,
then implement in `tests/benchmarks/data/build_realistic.py`.**

### IMPORTANT side effect to handle: the masked reference
`build_realistic.py::build_masked_reference` masks chr22 to `N` outside the **nominal** windows.
`extend_to_length` extends ~0.8–4 kb past the window, so reconstructed haplotype bases in the
extension zone are currently `N`. For a perf benchmark that's harmless (extension is computed
from variant positions, not reference content), but if you want faithful bases, widen the mask
by a matching flank when you regenerate.

## Data — what's needed and where it is

Running the existing benchmarks/profiling needs ONLY the committed `.gvl` + masked reference
(self-contained, already in git) — no cluster filesystem. **Regenerating** the dataset (the
fix) needs the build inputs, which are **committed into this branch** as a tarball (the whole
cluster is down, so this is the only reliable transport):

```
tests/benchmarks/data/_handoff_source.tar   # ~25 MB, committed (TEMPORARY — remove after the fix)
```
Extract on the new machine: `tar -C /tmp -xf tests/benchmarks/data/_handoff_source.tar`
→ `/tmp/source/` containing:
```
chr22_5s.pgen / .pvar.zst / .psam   # chr22, 5 samples, ALL chr22 variants (unrestricted → any fix option works)
bw_chr22/                            # the 5 GEUVADIS read-depth bigwigs (13 MB)
sample_id_to_bigwig.csv              # sample → bigwig basename map (our 5)
chr22_egenes.raw.bed                 # original recount3 egenes BED (zero-width TSS points)
chr22_egenes.bed                     # the WIDENED 165 windows actually used
chr22.masked.fa.gz(.fai/.gzi)        # masked reference
samples.txt
```
Then point `build_realistic.py` at `/tmp/source/` (replace the `/carter` `PLINK_PREFIX`,
`RNA_DIR`, `REF_FASTA` constants). The unrestricted PGEN supports any of the three fix options.

Provenance, if you ever need to rebuild the bundle from scratch (cluster back, or fresh fetch):
genotypes `/carter/users/dlaub/data/1kGP/plink2/hg38.norm.{pgen,psam,pvar.zst}`; RNA-seq
`/carter/users/dlaub/data/1kGP-rna-seq/{sample_id_to_bigwig.csv,bw_chr22/,chr22_egenes.bed}`
(bigwigs = recount3 study ERP001942, publicly re-downloadable); masking reference
`tests/data/fasta/hg38.fa.bgz` (run `pixi run -e dev gen`).

The 5 samples (deterministic, in `samples.txt`): HG00096, HG00097, HG00099, HG00100, HG00101.

## Resume checklist (new machine)

1. `git checkout feat/bench-codspeed-profiling`; `pixi install -e dev`.
2. Sanity: `pixi run -e dev pytest tests/benchmarks -p no:cov -q` → 11 passed (uses the
   committed BROKEN-but-functional dataset; benchmarks still run).
3. Decide fix option (1/2/3 above) with the user.
4. Edit `tests/benchmarks/data/build_realistic.py`:
   - Currently it does `plink2 ... --extract bed0 <widened windows>` in `slice_pgen()`.
   - Option 1: change the extract BED to windows ± ~50 kb (build a flanked range file).
   - Option 2: drop `--extract` AND pass `extend_to_length=False` to `gvl.write` in `build_dataset()`.
   - Point the source-path constants (`PLINK_PREFIX`, `RNA_DIR`, `REF_FASTA`) at the extracted
     `/tmp/source/` (from `_handoff_source.tar`).
5. Rebuild: `pixi run -e dev python tests/benchmarks/data/build_realistic.py`.
6. **VERIFY THE FIX:** regions must be ~17 kb, not 200 kb:
   ```bash
   pixi run -e dev python -c "import numpy as np; r=np.load('tests/benchmarks/data/chr22_geuv.gvl/regions.npy'); s=r[:,2]-r[:,1]; print('mean',s.mean(),'max',s.max())"
   ```
   Expect mean ~17 k, max ~20 k (option 1) or ~16384 (option 2). Check `intervals.npy` ≈ 11 MB.
7. `pixi run -e dev bench-local` → 9 pass. Re-run profiling if you want fresh numbers
   (`pixi run -e dev profile-tracks` etc.), and update the REGRESSIONS.md profiling section
   if the dataset changed materially.
8. Commit the corrected dataset (the `data/` dir is gitignored — use `git add -f`). Then see
   "git history note" below about the now-stale data in earlier commits.

## git history note (read before merging)

History was rewritten once (`git filter-branch`) to scrub an 89 MB `variants.arrow` blob that
an earlier build committed — `.git` went 171 MB → 4.6 MB. Side effect: in the intermediate
commits `6524a03`..`9ce69a9` the rewrite swapped in the final `intervals.npy` but not its
matching `offsets.npy`, so those **intermediate** commits have mismatched intervals/offsets
(cosmetic — only HEAD's tree is used). The user **does not use squash merges**, so when you
land this, land the real commits. After the dataset fix you'll be replacing the data blobs
again; consider whether a final `filter-branch`/`filter-repo` pass is worth it to keep the
branch's blob history tidy, or just commit the corrected data forward and move on.

## Commit log (branch)
```
0d7097d test(bench): consolidate batch-index logic into shared helper
6e18060 test(bench): restrict benchmark variants to regions (msg now stale post-rewrite)
9ce69a9 docs(REGRESSIONS): add local py-spy/memray profiling results for hot paths
37f1f15 test(bench): add py-spy/memray profiling driver for hot paths
e7f8247 test(bench): add end-to-end reconstructor benchmarks
bb7a07f test(bench): add micro-benchmarks for hot numba reconstruction kernels
ea3af63 test(bench): add session fixtures for dataset + captured numba args
6524a03 test(bench): add committed chr22 1kGP+GEUVADIS realistic slice + build script
3e33a74 test(bench): add capture-and-replay helper for micro-benchmark inputs
d84f43f build(bench): add pytest-codspeed dep and bench/profile pixi tasks
```
