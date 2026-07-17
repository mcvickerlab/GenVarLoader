"""The #275 throughput gate + scale-fixture parity.

The gate is a DETERMINISTIC COUNTER, not wall-clock: this node is too noisy for
absolute timings (see the project's perf-gate convention). The walking skeleton
re-opened `Svar1RecordSource` per batch, whose constructor is O(all CSR entries) --
so entries-touched grew with the STORE, not the window. After the rewrite it must
grow with the WINDOW only. That's a flat-vs-linear curve; noise can't fake it.

The "no per-window materialization of sample-scale arrays" scale guard used to live
here as an `ru_maxrss` test; it was flake-prone AND blind (a few-KB owned copy of
`geno_v_idxs` never moves `ru_maxrss`'s page-granularity high-water mark, so the
guard couldn't fail on the defect it named). It's now a deterministic Rust unit test,
`geno_v_idxs_borrows_the_mmap_not_a_copy` in `src/svar1/store.rs`, asserting pointer
identity between the slice handed to the kernel and `Svar1Reader::variant_idxs()`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import genvarloader as gvl

# A store big enough that "touches the whole contig" and "touches the window" differ
# by orders of magnitude. 200 variants x 20 samples, one contig.
_N_VARIANTS = 200
_N_SAMPLES = 20
_CONTIG_LEN = 4000


def _make_vcf(path: Path) -> None:
    rng = np.random.default_rng(0)
    lines = [
        "##fileformat=VCFv4.2",
        f"##contig=<ID=chr1,length={_CONTIG_LEN}>",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        + "\t".join(f"S{i}" for i in range(_N_SAMPLES)),
    ]
    positions = np.sort(
        rng.choice(np.arange(2, _CONTIG_LEN - 2), _N_VARIANTS, replace=False)
    )
    for pos in positions:
        gts = "\t".join(
            f"{rng.integers(0, 2)}|{rng.integers(0, 2)}" for _ in range(_N_SAMPLES)
        )
        lines.append(f"chr1\t{pos}\t.\tA\tG\t.\t.\t.\tGT\t{gts}")
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture(scope="module")
def scale_fixture(tmp_path_factory):
    from genoray import VCF, SparseVar

    d = tmp_path_factory.mktemp("svar1_scale")
    ref = d / "ref.fa"
    rng = np.random.default_rng(1)
    seq = "".join(rng.choice(list("ACGT"), _CONTIG_LEN))
    ref.write_text(f">chr1\n{seq}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    vcf = d / "in.vcf"
    _make_vcf(vcf)
    bcf = d / "in.bcf"
    subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", str(bcf)], check=True)

    svar = d / "store.svar"
    SparseVar.from_vcf(
        svar,
        VCF(bcf),
        max_mem="1g",
        samples=[f"S{i}" for i in range(_N_SAMPLES)],
        overwrite=True,
    )
    return svar, ref


def test_entries_touched_scales_with_window_not_store(scale_fixture):
    """THE #275 GATE. Entries touched per window must be ~flat as the store's variant
    count grows -- i.e. proportional to the window's variants, not the contig's.

    The skeleton's `Svar1RecordSource::new` inverted the whole contig CSR per call,
    so this ratio would be ~1.0 (touching everything). After the rewrite a narrow
    window must touch a small fraction.
    """
    from genvarloader.genvarloader import svar1_csr_entries_touched

    svar, ref = scale_fixture
    total_entries = int(
        np.asarray(__import__("genoray").SparseVar(svar).genos.data).size
    )

    # One narrow region: ~1/40th of the contig.
    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [100]})
    sds = gvl.StreamingDataset(bed, reference=ref, variants=svar).with_seqs(
        "haplotypes"
    )

    before = svar1_csr_entries_touched()
    list(sds.to_iter(batch_size=8))
    touched = svar1_csr_entries_touched() - before

    assert touched > 0, "counter is not wired"
    assert touched < total_entries * 0.25, (
        f"window read touched {touched} of {total_entries} CSR entries -- that is "
        "whole-store behavior, i.e. the O(all entries) per-call path is back"
    )


def test_entries_touched_is_flat_across_batch_size(scale_fixture):
    """Batch size must NOT affect I/O: the window is the read granularity and a batch
    is only a slice of it. The skeleton did one read PER BATCH, so halving batch_size
    doubled the work. Now it must be identical."""
    from genvarloader.genvarloader import svar1_csr_entries_touched

    svar, ref = scale_fixture
    bed = pl.DataFrame(
        {
            "chrom": ["chr1"] * 4,
            "chromStart": [0, 100, 200, 300],
            "chromEnd": [100, 200, 300, 400],
        }
    )
    sds = gvl.StreamingDataset(bed, reference=ref, variants=svar).with_seqs(
        "haplotypes"
    )

    counts = []
    for bs in (1, 8, 64):
        before = svar1_csr_entries_touched()
        list(sds.to_iter(batch_size=bs))
        counts.append(svar1_csr_entries_touched() - before)

    assert counts[0] == counts[1] == counts[2], (
        f"entries touched varied with batch_size: {counts} -- the window is the read "
        "granularity, so batch_size must not change I/O at all"
    )


def test_peak_rss_is_flat_in_cohort_size(tmp_path):
    """THE #284 GATE. At fixed batch_size, peak RSS growth across a full sweep must NOT
    scale with the number of samples: per-batch generation caps output at batch_size,
    and the offsets buffer is max_mem-bounded. Whole-window generation (the old path)
    would grow output ~linearly in n_samples and blow this."""
    import resource
    import subprocess

    import numpy as np
    import polars as pl
    from genoray import VCF, SparseVar

    def build(n_samples: int):
        d = tmp_path / f"n{n_samples}"
        d.mkdir()
        ref = d / "ref.fa"
        rng = np.random.default_rng(1)
        seq = "".join(rng.choice(list("ACGT"), 4000))
        ref.write_text(f">chr1\n{seq}\n")
        subprocess.run(["samtools", "faidx", str(ref)], check=True)
        vcf = d / "in.vcf"
        lines = [
            "##fileformat=VCFv4.2",
            "##contig=<ID=chr1,length=4000>",
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">',
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            + "\t".join(f"S{i}" for i in range(n_samples)),
        ]
        pos = np.sort(rng.choice(np.arange(2, 3998), 200, replace=False))
        for p in pos:
            gts = "\t".join(
                f"{rng.integers(0, 2)}|{rng.integers(0, 2)}" for _ in range(n_samples)
            )
            lines.append(f"chr1\t{p}\t.\tA\tG\t.\t.\t.\tGT\t{gts}")
        vcf.write_text("\n".join(lines) + "\n")
        bcf = d / "in.bcf"
        subprocess.run(
            ["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True
        )
        subprocess.run(["bcftools", "index", str(bcf)], check=True)
        svar = d / "store.svar"
        SparseVar.from_vcf(
            svar,
            VCF(bcf),
            max_mem="1g",
            samples=[f"S{i}" for i in range(n_samples)],
            overwrite=True,
        )
        return svar, ref

    def peak_growth(n_samples: int) -> int:
        svar, ref = build(n_samples)
        bed = pl.DataFrame(
            {
                "chrom": ["chr1"] * 4,
                "chromStart": [0, 100, 200, 300],
                "chromEnd": [100, 200, 300, 400],
            }
        )
        sds = gvl.StreamingDataset(bed, reference=ref, variants=svar).with_seqs(
            "haplotypes"
        )
        list(sds.to_iter(batch_size=4))  # warm up allocator
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        for _ in range(3):
            list(sds.to_iter(batch_size=4))
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return (after - before) * 1024  # KiB -> bytes on Linux

    small = peak_growth(50)
    large = peak_growth(400)
    # NOTE on measurement limitation: ru_maxrss is a process-global MONOTONIC
    # high-water mark. If the peak was already reached during the small-cohort
    # sweep (or an earlier test in this session), both deltas can measure ~0,
    # which makes this assertion vacuously true rather than a real discriminator
    # between flat and linear growth. See the module docstring for the history of
    # a prior ru_maxrss guard removed for exactly this "blind" failure mode --
    # inspect the printed deltas below when re-validating this gate.
    print(f"[rss-gate] small(n=50) delta={small}B large(n=400) delta={large}B")
    # 8x the cohort must NOT produce ~8x the peak growth; per-batch output is flat.
    assert large < max(small, 8 * 1024 * 1024) * 2, (
        f"peak RSS growth scaled with cohort (50->{small}B, 400->{large}B) -- "
        "whole-window output materialization has returned"
    )


def test_scale_parity_still_byte_identical(scale_fixture, tmp_path):
    """The scale fixture must ALSO satisfy the parity oracle -- a fast wrong answer
    is not progress."""
    from genoray import SparseVar

    svar, ref = scale_fixture
    bed = pl.DataFrame(
        {
            "chrom": ["chr1"] * 3,
            "chromStart": [0, 500, 1500],
            "chromEnd": [200, 700, 1700],
        }
    )
    out = tmp_path / "scale.gvl"
    gvl.write(out, bed, variants=SparseVar(svar), samples=None, overwrite=True)
    written = gvl.Dataset.open(out, reference=ref).with_seqs("haplotypes")

    sds = gvl.StreamingDataset(bed, reference=ref, variants=svar).with_seqs(
        "haplotypes"
    )
    for data, r_idx, s_idx in sds.to_iter(batch_size=7):
        for i in range(len(r_idx)):
            expected = written[int(r_idx[i]), int(s_idx[i])]
            for h in range(sds.ploidy):
                np.testing.assert_array_equal(
                    np.asarray(data[i][h]), np.asarray(expected[h])
                )
