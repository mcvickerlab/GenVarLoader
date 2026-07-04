# SVAR2 gvl MVP — Real-Data Validation (chr21)

**Date:** 2026-07-03
**Task:** Plan Task 3 (Task D) — prove the real-data plumbing works through both the
SVAR1 (gvl `Dataset` over `.svar`) and SVAR2 (`SparseVar2Source` over `.svar2`) backends
on real germline + somatic chr21 stores. Correctness is already proven by the test suite
(Task 2 e2e oracle); this exercises the real-data path end-to-end.

**Env:** `pixi run -e default`, genoray 2.15.0 wheel. Reference FASTA
`/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa`.
**Work dir (outside repo):** `/carter/users/dlaub/repos/for_loukik/svar2_mvp` (`$W`).

## Resolved contig names (Step 1)

All three inputs use `chr21` — **no naming mismatch**:

| File | Contig |
| --- | --- |
| Reference FASTA (`GRCh38.d1.vd1.fa`) | `chr21` |
| `chr21.bcf` (germline, 1000G) | `chr21` |
| `gdc.chr21.bcf` (somatic, GDC) | `chr21` |

So `<GERMLINE_CHROM>` = `<SOMATIC_CHROM>` = `chr21` throughout. The germline `.csi` was
already present (no re-index needed).

## Cohort sizes

| Source | Samples | Ploidy | Raw records (chr21) |
| --- | --- | --- | --- |
| germline (1000G) | 3202 | 2 | 1,002,753 |
| somatic (GDC) | 16007 | 2 | 4,525,689 |

## Build pipeline (Steps 2–3)

Per source: `bcftools norm -m -any --atomize` → **filter symbolic/breakend** → build both
stores (`build_stores.py`: `SparseVar.from_vcf` for `.svar`, `run_conversion_pipeline` for
`.svar2`). Orchestrated by `$W/build_source.sh` (kept outside the repo), submitted via
`sbatch -p carter-compute` (germline 32G, somatic 128G, 8 cpus each).

### Two environment/data issues found and fixed (API/infra drift)

1. **Compute-node `libstdc++` / GLIBCXX shadowing.** On the compute nodes the inherited
   `LD_LIBRARY_PATH` puts the gcc11 module's `libstdc++.so.6` (which lacks
   `GLIBCXX_3.4.30`) ahead of the pixi env's newer one, so importing genoray crashed at
   `llvmlite → numba` load:
   `OSError: .../gcc11/.../libstdc++.so.6: version 'GLIBCXX_3.4.30' not found (required by
   .../libLLVM-14.so)`. The login node is unaffected (its `miniforge3/lib` on
   `LD_LIBRARY_PATH` rescues it). **Fix:** prepend the pixi env lib dir to
   `LD_LIBRARY_PATH` in the job so the env's own `libstdc++` (which has `GLIBCXX_3.4.30`)
   wins the loader search. Baked into `build_source.sh`.

2. **SVAR2 rejects symbolic/breakend ALTs (short-read only).** genoray's
   `normalize::atomize_record` returns `SymbolicAllele` for symbolic (`<DEL>`,
   `<INS:ME:ALU>`, `<DUP>`, `<INV>`, …) and breakend ALTs, and the reader `.expect()`s on
   it — so a single SV record **panics the whole conversion**
   (`worker thread 'read-chr21' panicked`). `*`/`.` alleles are silently skipped, not
   errored. **Fix:** drop symbolic ("other") + breakend ("bnd") types with
   `bcftools view -V other,bnd` before building, and — for a **fair** benchmark — build
   *both* stores from the identical filtered input. Filtering impact:

   | Source | Records before | Records after filter | Symbolic/bnd dropped |
   | --- | --- | --- | --- |
   | germline | 1,002,753 | 1,001,385 | **1,368** (702 `<DEL>`, 260 `<INS:ME:ALU>`, 195 `<DUP>`, 174 `<INS>`, 17 `<INS:ME:LINE1>`, 12 `<INS:ME:SVA>`, 7 `<INV>`, 1 `<DEL:ME>`) |
   | somatic | 4,525,689 | 4,525,689 | **0** (GDC somatic calls are SNV/indel only) |

   This friction motivated a new roadmap milestone on the genoray side — **M13: opt-in
   skip for out-of-scope alleles during conversion** (drop instead of panic), in
   `genoray:docs/roadmap/svar-2.md`.

Build wall-clock: germline ~11 min; somatic ~2h1m (peak RSS ~10.9 GB).

## Backend validation (Step 4) — `validate.py`, 2 regions × chr21

Regions (0-based half-open): `(20_000_000, 20_001_000)` [1000 bp] and
`(30_000_000, 30_000_500)` [500 bp]. `validate.py` ran **as-written — no API drift** in the
`SparseVar2Source.reconstruct` / `sv.decode` / `gvl.write` / `gvl.Dataset.open` signatures.

### germline (`$W/germline`)

- **SVAR2** (`SparseVar2Source`): `n_samples=3202 ploidy=2`; hap ragged
  `rows=12808` (= 2 regions × 3202 samples × 2 ploidy ✓), `min_len=499 max_len=1000`
  (1000 bp window full-length; the 499 is a single-base net deletion) — sane. `decode`
  returned a non-empty `seqpro.rag.Ragged`.
- **SVAR1** (gvl `Dataset` over `.svar`): dataset written (3202 samples, 2 regions) and
  opened cleanly; `with_seqs("haplotypes")` returned a `seqpro.rag.Ragged`.

### somatic (`$W/somatic`)

- **SVAR2**: `n_samples=16007 ploidy=2`; hap ragged `rows=64028` (= 2 × 16007 × 2 ✓),
  `min_len=498 max_len=1001` (a −2 DEL and a +1 INS at the extremes) — sane. `decode`
  returned a non-empty `Ragged`.
- **SVAR1**: dataset written (16007 samples, 2 regions) and opened; haplotypes `Ragged`.

Both backends return non-empty, correctly-shaped haplotypes and variants on real data for
both cohorts.

## Store sizes (Step 5)

`du` on the four stores:

Apparent sizes (`du -sb`); decimal labels (MB = 10⁶ B, GB = 10⁹ B):

| Store | SVAR1 (`.svar`) | SVAR2 (`.svar2`) | SVAR2 / SVAR1 | SVAR2 advantage |
| --- | --- | --- | --- | --- |
| germline (3202 smp) | 1,149,533,941 B (1.15 GB) | 202,842,586 B (203 MB) | 0.176× | **5.67× smaller** |
| somatic (16007 smp) | 55,578,073 B (55.6 MB) | 38,184,053 B (38.2 MB) | 0.687× | **1.46× smaller** |

SVAR2 wins on size for **both** cohorts. The germline win is dramatic (5.7×): 1000G
carries many **common, high-allele-frequency** variants, which SVAR2's cost model routes
to the 1-bit dense matrix — far cheaper than SVAR1's `u32` pointers. Somatic mutations are
**rare / near-private** (low AF), so they stay sparse in both formats and the gap narrows
to 1.46×. This is exactly the empirical distribution SVAR2 was designed to exploit.

## Artifacts

- Store builder: `tmp/svar2_mvp/build_stores.py` (committed)
- Backend validator: `tmp/svar2_mvp/validate.py` (committed)
- Job orchestrator: `$W/build_source.sh` (outside repo — norm/filter/build wrapper with
  the `LD_LIBRARY_PATH` fix; not committed)
- Four stores + normalized/filtered BCFs live under `$W` (outside the repo).

## Feeds Task 4 (benchmark)

The four stores are the benchmark inputs. Early size signal above; latency (hap/variant,
all-samples-per-region) is measured next.
