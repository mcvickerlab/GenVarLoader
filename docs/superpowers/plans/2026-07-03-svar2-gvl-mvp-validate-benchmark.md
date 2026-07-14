# SVAR2 gvl MVP — Validate & Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the SVAR2 gvl value proposition on real chr21 data — an e2e track test, real-data validation, and a SVAR1-vs-SVAR2 benchmark — before wiring SVAR2 into the Dataset (Task B, deferred).

**Architecture:** SVAR2 reconstruction runs through the already-validated `SparseVar2Source` adapter (`python/genvarloader/_dataset/_svar2_source.py`), which queries genoray `SparseVar2.overlap_batch` live. Task C adds the missing track-path test; Tasks D/E exercise the adapter on real germline/somatic chr21 stores. No changes to the SVAR 1.0 path.

**Tech Stack:** Rust (PyO3/maturin) kernels, Python (numpy, polars, seqpro), genoray 2.15.0 wheel, bcftools/samtools, SLURM (`sbatch -p carter-compute`), pixi (`-e default`).

**Spec:** `docs/superpowers/specs/2026-07-03-svar2-gvl-mvp-validate-benchmark-design.md`

## Global Constraints

- **Worktree guard:** all work happens in `/carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel`. Every shell/subagent step must first `cd` there and verify `git rev-parse --show-toplevel` ends in `svar2-m6b-kernel`. The gvl worktree is a **different repo** from genoray.
- **Env:** run Python/pytest via `pixi run -e default ...`. Only the `default` env is installed.
- **Rust rebuild:** after any Rust edit, `pixi run -e default maturin develop` (editable install does not auto-rebuild). Not needed for this plan — no Rust edits are planned.
- **Rust tests** (if ever needed): `pixi run -e default cargo test --no-default-features [FILTER]`.
- **Commits:** prek hooks are intentionally not installed here — use `git commit --no-verify`.
- **Commit trailer:** end every commit message with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **genoray** is the pre-built **2.15.0 wheel** in gvl's `default` env (not editable).
- **Reference FASTA:** `/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa` (GDC GRCh38; `.fai` present).
- **Real data:** `/carter/users/dlaub/repos/for_loukik/chr21.bcf` (germline, 3202 samples, no `.csi`), `/carter/users/dlaub/repos/for_loukik/gdc.chr21.bcf` (somatic, 16007 samples, `.csi` present).
- **DEL-anchor convention:** genoray `decode`/`overlap_batch` return empty ALT for pure DELs; the anchor `ref[pos]` is injected downstream. Honor this in any oracle.
- **Out of scope:** Task B (Dataset dispatch) — deferred to its own brainstorm after Task E.

---

### Task 1: Clean up merged genoray worktrees (Task A)

**Files:** none in this repo. Operates on the **genoray** repo at `/carter/users/dlaub/projects/genoray`.

**Interfaces:**
- Consumes: nothing.
- Produces: nothing consumed by later tasks (independent housekeeping).

- [ ] **Step 1: Confirm each target worktree has no uncommitted work**

```bash
for wt in svar-2-m6b svar-2-m6c svar-2-m6-core; do
  echo "=== $wt ==="
  git -C /carter/users/dlaub/projects/genoray/.claude/worktrees/$wt status --porcelain 2>&1 || echo "(missing)"
done
```
Expected: empty output under each header (clean). If any prints file paths, **STOP** and report — do not remove a worktree with uncommitted work.

- [ ] **Step 2: Remove the three worktrees**

```bash
git -C /carter/users/dlaub/projects/genoray worktree remove .claude/worktrees/svar-2-m6b
git -C /carter/users/dlaub/projects/genoray worktree remove .claude/worktrees/svar-2-m6c
git -C /carter/users/dlaub/projects/genoray worktree remove .claude/worktrees/svar-2-m6-core
```
Expected: no output (success). If a worktree is already gone, `git worktree remove` errors with "is not a working tree" — that's fine, continue.

- [ ] **Step 3: Verify removal**

```bash
git -C /carter/users/dlaub/projects/genoray worktree list
```
Expected: none of `svar-2-m6b`, `svar-2-m6c`, `svar-2-m6-core` appear.

No commit (worktree removal is not tracked in this repo).

---

### Task 2: End-to-end test for `realign_tracks` (Task C)

**Files:**
- Create: `tests/test_svar2_realign_tracks.py`

**Interfaces:**
- Consumes: `genvarloader._dataset._svar2_source.SparseVar2Source.realign_tracks(contig, regions, tracks, track_offsets, params, strategy_id, base_seed, shifts=None, parallel=False) -> Ragged[np.float32]` (shape `(R, S, P, None)`); `genvarloader._dataset._tracks.shift_and_realign_track_sparse(offset_idx, geno_v_idxs, geno_offsets, v_starts, ilens, shift, track, query_start, out, params, keep=None, strategy_id=0, base_seed=0, query=0, hap=0)` (pure-Python oracle, fills `out` in place); genoray `SparseVar2` + `sv._readers[contig].decode_batch([(start, end)])`.
- Produces: nothing consumed by later tasks.

**Oracle rationale (from spec):** the pure-Python SVAR1 `shift_and_realign_track_sparse` is a distinct implementation from the SVAR2 **Rust** kernel under test and already carries the DEL-anchor branch (`_tracks.py:755`). Feed it genoray's decoded `(pos, ilen)` per hap via a trivial synthetic layout. DEL-only variants bypass insertion-fill, so the strategy is irrelevant.

- [ ] **Step 1: Write the failing test**

Create `tests/test_svar2_realign_tracks.py`:

```python
"""End-to-end validation of the SVAR2 track-realign adapter path.

Builds a DEL-only SVAR2 store, realigns a reference track through gvl's SVAR2
path (SparseVar2Source.realign_tracks, the Rust two-source kernel), and compares
per-(region, sample, ploid) against gvl's INDEPENDENT pure-Python SVAR1 track
realign (shift_and_realign_track_sparse) fed genoray's materialized decode
records. Agreement proves the SVAR2 Rust track kernel matches the trusted SVAR1
realign semantics — including the DEL anchor.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

# 40 bp reference (chr1). Two pure DELs chosen to match the reference exactly:
#   POS 4  GTA>G  -> 0-based pos 3, ilen -2  (ref[3:6] == "GTA")
#   POS 10 GGG>G  -> 0-based pos 9, ilen -2  (ref[9:12] == "GGG")
_REF = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"
_VCF = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=40>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1
chr1\t4\t.\tGTA\tG\t.\t.\t.\tGT\t1|0\t1|1
chr1\t10\t.\tGGG\tG\t.\t.\t.\tGT\t0|1\t1|0
"""


@pytest.fixture(scope="module")
def svar2_del_store(tmp_path_factory) -> Path:
    from genoray import _core

    d = tmp_path_factory.mktemp("svar2_del")
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    vcf = d / "in.vcf"
    vcf.write_text(_VCF)
    bcf = d / "in.bcf"
    subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", str(bcf)], check=True)

    out = d / "store"
    _core.run_conversion_pipeline(
        str(bcf), str(ref), ["chr1"], str(out), ["S0", "S1"],
        25_000, 2, 1, 8 * 1024 * 1024,
    )
    assert (out / "meta.json").exists(), "conversion did not finish"
    return out


def test_svar2_realign_tracks_matches_svar1_oracle(svar2_del_store):
    import genoray
    from genvarloader._dataset._svar2_source import SparseVar2Source
    from genvarloader._dataset._tracks import shift_and_realign_track_sparse

    contig = "chr1"
    q_start, q_end = 0, 40
    region_len = q_end - q_start
    regions = [(q_start, q_end)]

    sv = genoray.SparseVar2(str(svar2_del_store))
    S, P = sv.n_samples, sv.ploidy
    assert (S, P) == (2, 2)

    # A per-region reference track (f32). Random-but-fixed so a positional bug
    # can't hide behind a monotonic ramp.
    rng = np.random.default_rng(0)
    track = rng.random(region_len).astype(np.float32)

    strategy_id = 0            # irrelevant for DEL-only (insertion-fill unused)
    params = np.zeros(1, np.float64)
    base_seed = 0

    # --- SVAR2 path under test: one region, expanded internally to R*S*P haps ---
    src = SparseVar2Source(sv)
    out_rag = src.realign_tracks(
        contig,
        regions,
        track,                                  # flat per-region track buffer
        np.array([0, region_len], np.int64),    # (R+1) offsets
        params,
        strategy_id,
        base_seed,
        shifts=None,                            # no jitter
        parallel=False,
    )

    # --- oracle: genoray decode records -> pure-Python SVAR1 realign, per hap ---
    raw = sv._readers[contig].decode_batch([(q_start, q_end)])
    R, So, Po = int(raw["n_regions"]), int(raw["n_samples"]), int(raw["ploidy"])
    assert (R, So, Po) == (1, S, P)
    off = np.asarray(raw["off"])        # (H+1,) per-hap variant offsets
    d_pos = np.asarray(raw["pos"])
    d_ilen = np.asarray(raw["ilen"])

    # Non-triviality: haps carry a varying number of DELs.
    per_hap_counts = (off[1:] - off[:-1]).tolist()
    assert per_hap_counts == [1, 1, 2, 1], per_hap_counts

    for s in range(S):
        for p in range(P):
            h = (0 * S + s) * P + p                # region-major h=(r*S+s)*P+p
            gi0, gi1 = int(off[h]), int(off[h + 1])
            pos_h = np.ascontiguousarray(d_pos[gi0:gi1], np.int32)
            ilen_h = np.ascontiguousarray(d_ilen[gi0:gi1], np.int32)
            n_h = gi1 - gi0

            # Independently size the hap: region length + sum of (negative) ilens.
            exp_len = region_len + int(ilen_h.sum())

            got = np.asarray(out_rag[0, s, p])
            assert got.shape[0] == exp_len, (
                f"(s={s},p={p}) SVAR2 len {got.shape[0]} != expected {exp_len} "
                f"(ilen={ilen_h.tolist()})"
            )

            # Synthetic single-hap SVAR1 layout: v_idxs 0..n_h, one group.
            geno_v_idxs = np.arange(n_h, dtype=np.int32)
            geno_offsets = np.array([0, n_h], np.int64)
            expected = np.empty(exp_len, np.float32)
            shift_and_realign_track_sparse(
                offset_idx=0,
                geno_v_idxs=geno_v_idxs,
                geno_offsets=geno_offsets,
                v_starts=pos_h,
                ilens=ilen_h,
                shift=0,
                track=track,
                query_start=q_start,
                out=expected,
                params=params,
                strategy_id=strategy_id,
                base_seed=base_seed,
                query=0,
                hap=h,
            )
            np.testing.assert_allclose(
                got, expected, rtol=0, atol=0,
                err_msg=f"(s={s},p={p}) SVAR2 track != SVAR1 oracle "
                        f"(pos={pos_h.tolist()}, ilen={ilen_h.tolist()})",
            )
```

- [ ] **Step 2: Run the test**

```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel
pixi run -e default pytest tests/test_svar2_realign_tracks.py -v
```
Expected: **PASS**. This test validates already-working code (the SVAR2 track kernel), so it should pass on first run. If it FAILS:
- On the `per_hap_counts` assert → the decode layout differs from expectation; print `per_hap_counts` and adjust the expected list to the observed counts (the store is deterministic).
- On the length or `assert_allclose` → a **real** discrepancy between the SVAR2 Rust track kernel and the SVAR1 oracle. Do **not** loosen the tolerance. Use superpowers:systematic-debugging: inspect the failing `(s,p)`'s `got` vs `expected` arrays and the DEL anchor handling (`_tracks.py:755` vs the Rust `shift_and_realign_tracks_from_svar2`).

- [ ] **Step 3: Commit**

```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel
git add tests/test_svar2_realign_tracks.py
git commit --no-verify -m "test: e2e SVAR2 realign_tracks vs SVAR1 pure-Python oracle

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Real-data validation on chr21 (Task D)

**Nature:** exploratory validation, **not** pytest. Deliverable is a committed notes file recording what was built and observed. Involves real BCFs and SLURM — has human-checkpointable steps.

**Files:**
- Create: `tmp/svar2_mvp/build_stores.py` (store-builder script)
- Create: `tmp/svar2_mvp/validate.py` (backend-comparison script)
- Create: `docs/superpowers/notes/2026-07-03-svar2-mvp-validation.md` (results/notes)

**Interfaces:**
- Consumes: genoray `SparseVar.from_vcf(out, VCF(bcf), max_mem, samples=..., overwrite=True)`, `genoray._core.run_conversion_pipeline(bcf, ref, chroms, out, samples, chunk_size, ploidy, threads, long_allele_cap)`, `SparseVar2(store)`, `SparseVar2.decode(contig, regions)`, `SparseVar2Source.reconstruct(contig, regions, ref_, ref_offsets, pad_char, shifts=None, output_length=-1)`, `gvl.write(path, bed, variants=..., samples=..., overwrite=True)`, `gvl.Dataset.open(path, reference=...)`.
- Produces: two store pairs on disk (`<work>/germline.svar`, `<work>/germline.svar2`, `<work>/somatic.svar`, `<work>/somatic.svar2`) consumed by Task 4 (benchmark).

Let `WORK=/carter/users/dlaub/repos/for_loukik/svar2_mvp` (create it; big outputs live outside the repo). Let `REF=/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa`.

- [ ] **Step 1: Resolve contig naming and index the germline BCF**

```bash
mkdir -p /carter/users/dlaub/repos/for_loukik/svar2_mvp
REF=/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa
echo "=== FASTA contigs (chr21 region) ==="; grep -E "^>.*(21|chr21)\b" "$REF.fai" 2>/dev/null || cut -f1 "$REF.fai" | grep -E "21$|chr21"
for b in /carter/users/dlaub/repos/for_loukik/chr21.bcf /carter/users/dlaub/repos/for_loukik/gdc.chr21.bcf; do
  echo "=== $b contigs ==="; bcftools view -h "$b" | grep -E "^##contig" | grep -E "ID=(chr)?21," | head
done
# germline lacks a .csi:
[ -f /carter/users/dlaub/repos/for_loukik/chr21.bcf.csi ] || bcftools index --csi /carter/users/dlaub/repos/for_loukik/chr21.bcf
```
Expected: identify the contig name each file uses. **Checkpoint:** if the FASTA uses `chr21` but a BCF uses `21` (common for 1000G germline), record the mismatch — later steps must query each store with **its own** contig name, and `run_conversion_pipeline`/`from_vcf` must be given the BCF's chrom name (`21`), while `reconstruct`/`decode` use that same store's naming. Note the resolved names in the notes file (Step 6).

- [ ] **Step 2: Normalize+atomize both BCFs to biallelic**

```bash
cd /carter/users/dlaub/repos/for_loukik
for src in chr21 gdc.chr21; do
  bcftools norm -m -any --atomize -Ob -o svar2_mvp/${src}.norm.bcf ${src}.bcf
  bcftools index --csi svar2_mvp/${src}.norm.bcf
done
```
Expected: `svar2_mvp/chr21.norm.bcf` and `svar2_mvp/gdc.chr21.norm.bcf` created. genoray requires normalized+atomized biallelic input.
**Checkpoint (compute):** the somatic file (1.1 GB, 16007 samples) `norm` is heavy — if it does not finish in a few minutes interactively, wrap Steps 2–3 for the somatic source in an `sbatch -p carter-compute` job script and wait for completion before Step 4's somatic validation.

- [ ] **Step 3: Build both stores per source**

Create `tmp/svar2_mvp/build_stores.py`:

```python
"""Build .svar (SVAR1) and .svar2 (SVAR2) stores from a normalized biallelic BCF."""
import sys
from pathlib import Path

from genoray import VCF, SparseVar, _core

def build(bcf: str, chrom: str, samples: list[str], out_prefix: str, ploidy: int):
    bcf = str(bcf)
    # SVAR 1.0
    SparseVar.from_vcf(f"{out_prefix}.svar", VCF(bcf), "8g", overwrite=True)
    # SVAR 2.0
    _core.run_conversion_pipeline(
        bcf, "/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa",
        [chrom], f"{out_prefix}.svar2", samples,
        25_000, ploidy, 8, 8 * 1024 * 1024,
    )
    print(f"built {out_prefix}.svar and {out_prefix}.svar2")

if __name__ == "__main__":
    # argv: <norm.bcf> <chrom> <out_prefix>
    bcf, chrom, out_prefix = sys.argv[1], sys.argv[2], sys.argv[3]
    import subprocess
    samples = subprocess.run(
        ["bcftools", "query", "-l", bcf], capture_output=True, text=True, check=True
    ).stdout.split()
    build(bcf, chrom, samples, out_prefix, ploidy=2)
```

Run germline interactively; somatic via SLURM:

```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel
W=/carter/users/dlaub/repos/for_loukik/svar2_mvp
# germline (use the germline BCF's own chrom name from Step 1, e.g. chr21 or 21):
pixi run -e default python tmp/svar2_mvp/build_stores.py $W/chr21.norm.bcf <GERMLINE_CHROM> $W/germline
# somatic (heavy) — submit and wait:
sbatch -p carter-compute --wrap "cd $PWD && pixi run -e default python tmp/svar2_mvp/build_stores.py $W/gdc.chr21.norm.bcf <SOMATIC_CHROM> $W/somatic"
```
Replace `<GERMLINE_CHROM>`/`<SOMATIC_CHROM>` with the names resolved in Step 1.
Expected: four store dirs — `germline.svar`, `germline.svar2`, `somatic.svar`, `somatic.svar2`. **Checkpoint:** monitor the sbatch job (`squeue -u $USER`); proceed to Step 4's somatic checks only after it completes and `somatic.svar2/meta.json` exists.

- [ ] **Step 4: Validate both backends return sane output**

Create `tmp/svar2_mvp/validate.py`:

```python
"""Spot-check that gvl returns non-empty, sane haplotypes + variants through
both the SVAR1 (gvl Dataset over .svar) and SVAR2 (SparseVar2Source over .svar2)
backends, on a handful of regions x a few samples. Correctness is already proven
by the test suite; this proves the REAL-DATA plumbing works."""
import sys
from pathlib import Path

import numpy as np
import genvarloader as gvl
from genoray import SparseVar2
from genvarloader._dataset._svar2_source import SparseVar2Source

REF = "/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa"

def main(prefix: str, chrom: str):
    # A few small regions (0-based, half-open) in a variant-dense chr21 window.
    regions = [(20_000_000, 20_001_000), (30_000_000, 30_000_500)]

    # --- SVAR2 backend (adapter direct) ---
    sv2 = SparseVar2(f"{prefix}.svar2")
    print(f"[svar2] n_samples={sv2.n_samples} ploidy={sv2.ploidy}")
    ref_bytes = _contig_ref(REF, chrom)
    src = SparseVar2Source(sv2)
    hap = src.reconstruct(
        chrom, regions,
        np.frombuffer(ref_bytes, np.uint8),
        np.array([0, len(ref_bytes)], np.int64),
        pad_char=ord("N"), shifts=None, output_length=-1,
    )
    lens = np.asarray(hap.offsets)
    print(f"[svar2] hap ragged rows={len(lens) - 1} "
          f"min_len={int(np.diff(lens).min())} max_len={int(np.diff(lens).max())}")
    var = sv2.decode(chrom, regions)
    print(f"[svar2] decode variants: {var}")

    # --- SVAR1 backend (gvl Dataset over .svar) ---
    import polars as pl
    bed = pl.DataFrame({
        "chrom": [chrom] * len(regions),
        "chromStart": [s for s, _ in regions],
        "chromEnd": [e for _, e in regions],
    })
    ds_path = f"{prefix}.gvl"
    gvl.write(ds_path, bed, variants=f"{prefix}.svar", overwrite=True)
    ds = gvl.Dataset.open(ds_path, reference=REF).with_seqs("haplotypes")
    seqs = ds[:len(regions), :3]   # a few regions x first 3 samples
    print(f"[svar1] gvl haplotypes sample shape/type: {type(seqs)}")

def _contig_ref(fasta: str, chrom: str) -> bytes:
    import pysam
    return pysam.FastaFile(fasta).fetch(chrom).encode()

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])   # argv: <prefix> <chrom>
```

```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel
W=/carter/users/dlaub/repos/for_loukik/svar2_mvp
pixi run -e default python tmp/svar2_mvp/validate.py $W/germline <GERMLINE_CHROM>
pixi run -e default python tmp/svar2_mvp/validate.py $W/somatic  <SOMATIC_CHROM>
```
Expected: for each source, non-empty haplotype rows with sane lengths (≈ region length ± indels), non-empty decode variants, and the gvl SVAR1 Dataset opening without error. **Checkpoint:** if a chosen region is empty (no variants), pick a denser window (inspect with `bcftools view -H <norm.bcf> <chrom>:20000000-20010000 | head`) and rerun. Adapt the API calls to the actual installed signatures if they drift (record any drift in the notes).

- [ ] **Step 5: Record store sizes now (needed for Task 4)**

```bash
W=/carter/users/dlaub/repos/for_loukik/svar2_mvp
du -sh $W/germline.svar $W/germline.svar2 $W/somatic.svar $W/somatic.svar2
```
Expected: four sizes. Somatic `.svar2` is expected smaller than `.svar`.

- [ ] **Step 6: Write and commit the validation notes**

Create `docs/superpowers/notes/2026-07-03-svar2-mvp-validation.md` capturing: resolved contig names per file, exact commands run, sample/ploidy counts printed, hap-row counts + length ranges per backend, decode variant summaries, the four `du` sizes, and any API drift encountered. Then:

```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel
git add tmp/svar2_mvp/build_stores.py tmp/svar2_mvp/validate.py docs/superpowers/notes/2026-07-03-svar2-mvp-validation.md
git commit --no-verify -m "chore: SVAR2 MVP real-data validation scripts + notes

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
(The store dirs under `WORK` are outside the repo and are not committed.)

---

### Task 4: Benchmark SVAR1 vs SVAR2 (Task E)

**Nature:** measurement script producing a table. Depends on Task 3's four stores.

**Files:**
- Create: `tmp/svar2_mvp/benchmark.py`
- Create: `docs/superpowers/notes/2026-07-03-svar2-mvp-benchmark.md` (results table)

**Interfaces:**
- Consumes: the four stores from Task 3; the same backend calls listed in Task 3's Interfaces.
- Produces: a results table (documentation only).

**Fairness rule (from spec):** query the **same workload** on both backends — **all samples for a fixed region set** — matching genoray's per-contig/all-samples query granularity. Warm caches, N repeats, report **median**. Record the caveat that this is adapter-vs-Dataset (Task B not wired), so SVAR2 latency excludes gvl batching/collation.

- [ ] **Step 1: Write the benchmark script**

Create `tmp/svar2_mvp/benchmark.py`:

```python
"""Benchmark SVAR1 (gvl Dataset over .svar) vs SVAR2 (SparseVar2Source over
.svar2): hap latency, variant latency, store size, for one source prefix.
Fair workload: ALL samples for a fixed region set. Warm caches, median of N."""
import sys
import time
import subprocess
from statistics import median

import numpy as np
import genvarloader as gvl
from genoray import SparseVar2
from genvarloader._dataset._svar2_source import SparseVar2Source

REF = "/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa"
N = 5  # repeats

def _contig_ref(fasta, chrom):
    import pysam
    return pysam.FastaFile(fasta).fetch(chrom).encode()

def _timed(fn, warmup=1):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(N):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return median(ts)

def main(prefix, chrom):
    regions = [(20_000_000, 20_001_000), (30_000_000, 30_000_500),
               (40_000_000, 40_001_000)]
    ref_bytes = _contig_ref(REF, chrom)
    ref_u8 = np.frombuffer(ref_bytes, np.uint8)
    ref_off = np.array([0, len(ref_bytes)], np.int64)

    # SVAR2 backend
    sv2 = SparseVar2(f"{prefix}.svar2")
    src = SparseVar2Source(sv2)
    svar2_hap = _timed(lambda: src.reconstruct(
        chrom, regions, ref_u8, ref_off, pad_char=ord("N"),
        shifts=None, output_length=-1))
    svar2_var = _timed(lambda: sv2.decode(chrom, regions))

    # SVAR1 backend (all samples, same regions)
    import polars as pl
    bed = pl.DataFrame({"chrom": [chrom] * len(regions),
                        "chromStart": [s for s, _ in regions],
                        "chromEnd": [e for _, e in regions]})
    ds_path = f"{prefix}.gvl"
    ds = gvl.Dataset.open(ds_path, reference=REF)
    ds_hap = ds.with_seqs("haplotypes")
    ds_var = ds.with_seqs("variants")
    n_s = sv2.n_samples
    svar1_hap = _timed(lambda: ds_hap[:len(regions), :n_s])
    svar1_var = _timed(lambda: ds_var[:len(regions), :n_s])

    def du(path):
        return subprocess.run(["du", "-sb", path], capture_output=True,
                              text=True).stdout.split()[0]

    print(f"source={prefix.split('/')[-1]} chrom={chrom} n_samples={n_s} "
          f"regions={len(regions)} N={N}")
    print(f"  hap_latency_s   svar1={svar1_hap:.4f}  svar2={svar2_hap:.4f}")
    print(f"  var_latency_s   svar1={svar1_var:.4f}  svar2={svar2_var:.4f}")
    print(f"  store_bytes     svar1={du(prefix + '.svar')}  "
          f"svar2={du(prefix + '.svar2')}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])   # argv: <prefix> <chrom>
```

- [ ] **Step 2: Run the benchmark for both sources**

```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel
W=/carter/users/dlaub/repos/for_loukik/svar2_mvp
pixi run -e default python tmp/svar2_mvp/benchmark.py $W/germline <GERMLINE_CHROM>
pixi run -e default python tmp/svar2_mvp/benchmark.py $W/somatic  <SOMATIC_CHROM>
```
Expected: printed hap/variant latencies + store bytes per source. **Checkpoint:** the somatic all-samples decode (16007 samples) may be large — if a single call is very slow or OOMs, reduce the region set (fewer/smaller windows) uniformly for **both** backends to keep the comparison fair, and note the reduced workload. If `ds.with_seqs("variants")` indexing differs from the installed API, adapt and record the drift.

- [ ] **Step 3: Assemble the results table and commit**

Create `docs/superpowers/notes/2026-07-03-svar2-mvp-benchmark.md` with the 2×3 table (germline/somatic × {hap latency, variant latency, store size}) filled from Step 2's output, the exact workload (regions, n_samples, N), and the recorded caveat (adapter-vs-Dataset; somatic is where SVAR2 layout should win on size). Then:

```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel
git add tmp/svar2_mvp/benchmark.py docs/superpowers/notes/2026-07-03-svar2-mvp-benchmark.md
git commit --no-verify -m "chore: SVAR2 vs SVAR1 gvl benchmark script + results

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 4: Summarize the value proposition**

Report to the user: the filled table, whether SVAR2 wins on store size (esp. somatic) and how hap/variant latency compares, and a recommendation on whether Task B (Dataset wiring) is worth the invasive integration — plus the concrete all-samples-per-batch cost signal the benchmark revealed, to feed the Task B brainstorm.

---

## Self-Review

**Spec coverage:**
- Task A (cleanup) → Task 1. ✓
- Task C (realign_tracks e2e test) → Task 2, with the spec's pure-Python SVAR1 oracle. ✓
- Task D (real-data validation) → Task 3 (build both stores per source, validate both backends, small scope, germline interactive / somatic sbatch, contig-name checkpoint, reference FASTA). ✓
- Task E (benchmark: hap latency, variant latency, store size, fair all-samples workload, median of warm repeats, 2×3 table, caveat) → Task 4. ✓
- Task B (Dataset dispatch) → intentionally out of scope; noted in Global Constraints and fed by Task 4 Step 4. ✓

**Placeholder scan:** The only intentional fill-ins are `<GERMLINE_CHROM>`/`<SOMATIC_CHROM>`, which are *resolved values* from Task 3 Step 1 (a genuine runtime discovery, not a deferred design decision), and the notes-file contents (recorded observations, which cannot be pre-written). All code steps contain complete, runnable code. No "TBD"/"add error handling"/"similar to Task N".

**Type consistency:** `SparseVar2Source.reconstruct` / `.realign_tracks` signatures match `_svar2_source.py`; `shift_and_realign_track_sparse` params match `_tracks.py:708`; `run_conversion_pipeline` positional args match the reconstruct-test call and `gvl.write`/`SparseVar.from_vcf` match their source signatures. Hap indexing `h=(r*S+s)*P+p` is consistent between Task 2 and the decode layout.
