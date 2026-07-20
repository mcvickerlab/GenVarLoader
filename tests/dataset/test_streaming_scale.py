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
from genvarloader._dataset._streaming import _Svar1Backend

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


def test_generate_batch_output_is_flat_in_cohort_size(tmp_path):
    """THE #284 GATE, as a DETERMINISTIC counter -- not `ru_maxrss` (an earlier version
    of this test used a `ru_maxrss` before/after delta across `to_iter` sweeps; it
    measured 0B at both 50 and 400 samples, even up to 20000 samples / batch_size=4096
    in exploratory testing, because the tiny fixture's output never crosses a page
    boundary. That guard was BLIND -- it could not fail on the defect it named -- so it
    was replaced with this one. See the module docstring for the same lesson learned
    about the prior `ru_maxrss` scale guard.)

    `_Svar1Backend.generate_batch(r_idx, s_idx, o_starts, o_stops, lo, hi)` allocates
    its output for exactly the `hi - lo` rows sliced IN -- the batch rows are chosen
    before generation, not after. So the returned `Ragged`'s total byte count IS the
    backend's internal peak output allocation for that call. At a fixed `batch_size`,
    that byte count depends only on the batch's rows (region lengths x ploidy), never
    on how many samples the *window* covers.

    A whole-window regression (the #284 defect) would instead generate output for
    every sample in the window before slicing a batch out of it, so the returned
    array's size (or the backend's peak allocation) would scale with `len(s_idx)` --
    i.e. with the cohort size. This test proves the window covers the full cohort
    (`len(s_idx) == n_samples`) while the generated batch's byte count stays IDENTICAL
    between a 50-sample and a 400-sample cohort.
    """
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
            # SNP-only (REF=A/ALT=G, no indels): every haplotype's length equals the
            # region's length regardless of genotype, so batch output bytes are
            # deterministic across cohorts of different size.
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

    def batch_output_bytes(n_samples: int) -> int:
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
        backend = sds._backend
        assert isinstance(backend, _Svar1Backend), (
            "test requires the real SVAR1 backend, not _Svar2Backend or the "
            "whole-window test seam"
        )

        r_idx, s_idx = next(iter(sds._plan()))
        # Under default max_mem the first (and only, at this fixture size) window
        # covers the WHOLE cohort -- prove that, so a batch staying flat below is
        # actually evidence of per-batch generation, not just a tiny window.
        assert len(s_idx) == n_samples, (
            f"expected the window to cover the whole cohort ({n_samples} samples), "
            f"got {len(s_idx)} -- test no longer proves what it claims"
        )

        o_starts, o_stops = backend.read_window(r_idx, s_idx)
        batch_size = 4
        assert batch_size <= len(r_idx) * len(s_idx)
        data = backend.generate_batch(r_idx, s_idx, o_starts, o_stops, 0, batch_size)
        return int(data.data.nbytes)

    bytes_50 = batch_output_bytes(50)
    bytes_400 = batch_output_bytes(400)
    print(f"[batch-output-gate] n=50 bytes={bytes_50} n=400 bytes={bytes_400}")

    assert bytes_50 > 0, "counter is not wired (batch produced no output)"
    # A whole-window regression (issue #284) would make generated output scale with
    # len(s_idx) == n_samples; per-batch generation makes it depend only on
    # batch_size, so 50 and 400 samples must produce EXACTLY the same batch output
    # size (the fixture is SNP-only, so haplotype length == region length always).
    assert bytes_400 == bytes_50, (
        f"batch output bytes scaled with cohort size (50->{bytes_50}B, "
        f"400->{bytes_400}B) -- whole-window output materialization has returned"
    )


def test_svar2_generate_batch_output_is_flat_in_cohort_size(tmp_path):
    """SVAR2 #284 gate: a fixed-batch_size call's output bytes are identical between a
    50- and a 400-sample cohort (per-batch generation), while the window covers the
    whole cohort. Clone of the SVAR1 test above, SparseVar2 + .svar2.

    `SparseVar2.from_vcf` (unlike SVAR1's `SparseVar.from_vcf`) takes `source` as a
    plain path -- no `VCF(...)` wrapper -- and has no `samples=` subsetting kwarg; it
    always converts every sample in the VCF header. It requires either `reference=` or
    `no_reference=True`. This fixture is SNP-only with no out-of-scope records, so
    `no_reference=True, skip_out_of_scope=True` (the pattern
    `tests/benchmarks/data/build_svar2_stream_bulk.py` uses) avoids REF-vs-FASTA
    validation entirely.
    """
    import subprocess

    import numpy as np
    import polars as pl
    from genoray import SparseVar2

    from genvarloader._dataset._streaming import _Svar2Backend

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
            # SNP-only (REF=A/ALT=G, no indels): every haplotype's length equals the
            # region's length regardless of genotype, so batch output bytes are
            # deterministic across cohorts of different size.
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
        svar2 = d / "store.svar2"
        SparseVar2.from_vcf(
            svar2, bcf, no_reference=True, skip_out_of_scope=True, overwrite=True
        )
        return svar2, ref

    def batch_output_bytes(n_samples: int) -> int:
        svar2, ref = build(n_samples)
        bed = pl.DataFrame(
            {
                "chrom": ["chr1"] * 4,
                "chromStart": [0, 100, 200, 300],
                "chromEnd": [100, 200, 300, 400],
            }
        )
        sds = gvl.StreamingDataset(bed, reference=ref, variants=svar2).with_seqs(
            "haplotypes"
        )
        backend = sds._backend
        assert isinstance(backend, _Svar2Backend), (
            "test requires the real SVAR2 backend, not _Svar1Backend or the "
            "whole-window test seam"
        )

        r_idx, s_idx = next(iter(sds._plan()))
        # Under default max_mem the first (and only, at this fixture size) window
        # covers the WHOLE cohort -- prove that, so a batch staying flat below is
        # actually evidence of per-batch generation, not just a tiny window.
        assert len(s_idx) == n_samples, (
            f"expected the window to cover the whole cohort ({n_samples} samples), "
            f"got {len(s_idx)} -- test no longer proves what it claims"
        )

        window = backend.read_window(r_idx, s_idx)  # SVAR2: opaque bundle
        batch_size = 4
        assert batch_size <= len(r_idx) * len(s_idx)
        # Production (Phase 2) generates via a recycled `Svar2ReconBuf`: fill exactly
        # one batch's rows, then drain them -- mirrors the `_iter_batches` "sync"
        # drive's per-drained-batch call, not the old single-shot `generate_batch`.
        from genvarloader.genvarloader import Svar2ReconBuf

        buf = Svar2ReconBuf(backend.ploidy)
        backend._fill_super_batch(
            r_idx, s_idx, window, 0, batch_size, buf, parallel=False
        )
        data = backend._drain(buf, 0, batch_size)
        return int(data.data.nbytes)

    bytes_50 = batch_output_bytes(50)
    bytes_400 = batch_output_bytes(400)
    print(f"[svar2 batch-output-gate] n=50 bytes={bytes_50} n=400 bytes={bytes_400}")

    assert bytes_50 > 0, "counter is not wired (batch produced no output)"
    # A whole-window regression (issue #284) would make generated output scale with
    # len(s_idx) == n_samples; per-batch generation makes it depend only on
    # batch_size, so 50 and 400 samples must produce EXACTLY the same batch output
    # size (the fixture is SNP-only, so haplotype length == region length always).
    assert bytes_400 == bytes_50, (
        f"batch output bytes scaled with cohort size (50->{bytes_50}B, "
        f"400->{bytes_400}B) -- whole-window output materialization has returned"
    )


def test_svar2_super_batch_buffer_is_flat_in_cohort_size(tmp_path):
    """The super-batch reconstruct buffer's byte count is IDENTICAL between a 50- and a
    400-sample cohort at a fixed super-batch size (cohort-independent, #284), while the
    read window covers the whole cohort. Generalizes the per-batch flatness gate above:
    that test proves `_drain`'s output is flat in cohort size for a fixed `batch_size`;
    this one proves the recycled `Svar2ReconBuf` itself -- what `_fill_super_batch`
    actually fills -- never scales with the cohort, only with the fixed super-batch
    row count (`_super_batch_rows`). Clone of the builder in
    `test_svar2_generate_batch_output_is_flat_in_cohort_size` above.

    Correctness note: rows are C-order (region, sample), so row = region_i * n_samples
    + sample_j. With `_super_batch_rows` forced to 16 and n_samples >= 50, the first 16
    window rows are always region 0, samples 0..15 in BOTH cohorts -- the SAME region
    (same fixed width) regardless of how many samples follow them in the window.
    Because the fixture is SNP-only (REF/ALT both 1bp), every haplotype's length equals
    the region width no matter which allele each sample carries, so the buffer's byte
    count is genotype-independent -- 16 rows x ploidy 2 x region-0 width 100 = 3200B in
    either cohort. (The genotype *content* does diverge between the 50- and 400-sample
    builds -- reseeding rng(1) then drawing 2*n_samples values per position shifts the
    stream -- but that only changes which bases appear, not the byte count.) So a
    fixed-16-row super-batch fill must produce identical buffer bytes across cohorts; if
    it doesn't, the buffer is retaining/copying something sized by the whole window
    rather than just the filled super-batch rows.
    """
    import subprocess

    import numpy as np
    import polars as pl
    from genoray import SparseVar2

    from genvarloader._dataset._streaming import _Svar2Backend
    from genvarloader.genvarloader import Svar2ReconBuf

    def build(n_samples: int):
        d = tmp_path / f"sb_n{n_samples}"
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
            # SNP-only (REF=A/ALT=G, no indels): haplotype length == region length
            # regardless of genotype, so buffer bytes are deterministic across
            # cohorts of different size.
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
        svar2 = d / "store.svar2"
        SparseVar2.from_vcf(
            svar2, bcf, no_reference=True, skip_out_of_scope=True, overwrite=True
        )
        return svar2, ref

    def measure(n_samples: int) -> int:
        svar2, ref = build(n_samples)
        bed = pl.DataFrame(
            {
                "chrom": ["chr1"] * 4,
                "chromStart": [0, 100, 200, 300],
                "chromEnd": [100, 200, 300, 400],
            }
        )
        sds = gvl.StreamingDataset(bed, reference=ref, variants=svar2).with_seqs(
            "haplotypes"
        )
        backend = sds._backend
        assert isinstance(backend, _Svar2Backend), (
            "test requires the real SVAR2 backend, not _Svar1Backend or the "
            "whole-window test seam"
        )
        # Force a small FIXED super-batch so it cannot silently track the cohort.
        backend._super_batch_rows = 16

        r_idx, s_idx = next(iter(sds._plan()))
        assert len(s_idx) == n_samples, (
            f"expected the window to cover the whole cohort ({n_samples} samples), "
            f"got {len(s_idx)} -- test no longer proves what it claims"
        )

        window = backend.read_window(r_idx, s_idx)  # SVAR2: opaque bundle
        n_rows = len(r_idx) * len(s_idx)
        buf = Svar2ReconBuf(backend.ploidy)
        backend._fill_super_batch(
            r_idx, s_idx, window, 0, min(16, n_rows), buf, parallel=False
        )
        return int(buf.total_bytes)

    bytes_50 = measure(50)
    bytes_400 = measure(400)
    print(f"[svar2 super-batch flatness] n=50 {bytes_50}B n=400 {bytes_400}B")

    assert bytes_50 > 0, "counter is not wired (super-batch fill produced no bytes)"
    # A whole-window regression (issue #284) would size the buffer by the cohort
    # (len(s_idx) == n_samples); a truly fixed super-batch fills the same 16 rows --
    # same region, same samples 0..15 -- regardless of how many more samples follow
    # in the window, so the buffer's byte count must be EXACTLY identical.
    assert bytes_50 == bytes_400, (
        f"super-batch buffer scaled with cohort size (50->{bytes_50}B, "
        f"400->{bytes_400}B); the super-batch must be a fixed chunk of window rows, "
        "cohort-independent (#284)"
    )


def test_svar2_engine_output_is_flat_in_cohort_size(tmp_path):
    """PR-3 Task 4: the SVAR2 #284 cohort-independence gate driven end-to-end through
    the `"svar2_engine"` pipeline-engine strategy, not the "sync" super-batch path the
    two tests above exercise. Same invariant, same builder pattern (clone of
    `test_svar2_super_batch_buffer_is_flat_in_cohort_size`'s fixture-building above):
    at a fixed `batch_size`, the PEAK per-batch drained output byte count is IDENTICAL
    between a 50- and a 400-sample cohort, while the plan still covers every sample
    (`base.n_samples == n_samples`). A whole-window/whole-cohort regression under the
    engine (e.g. the producer materializing per-cohort-scale metadata, or draining
    more than `batch_size` rows per yielded batch) would scale a single batch's output
    with the cohort size, even though every batch still carries only `batch_size` rows.

    NOTE 1: the PR-3 task brief names a `svar2_scale_fixture_factory` fixture that does
    not exist in this test module -- the real SVAR2 scale tests above build their own
    stores inline via `tmp_path`. This test mirrors THEIR builder instead.

    NOTE 2: the brief's own example code SUMS `data.data.nbytes` across every batch
    `to_iter` yields for the whole sweep. That sum is NOT cohort-independent by
    construction -- a 400-sample cohort has 8x as many (region, sample) cells as a
    50-sample one (same 4-region bed), so it always yields 8x as many batches and the
    running total scales with the cohort regardless of whether the engine is correct.
    Measured once as written it failed with exactly that 8x ratio (50->40000B,
    400->320000B), confirming the test as specified could never pass and was not
    exercising #284 at all. #284 is about PEAK per-batch/per-super-batch allocation,
    not total bytes moved over a full epoch, so this test takes the MAX single-batch
    byte count across the sweep instead of the sum -- consistent with the two
    sync-path scale tests above, which also measure one representative call's output,
    never an accumulated total.
    """
    import subprocess

    import numpy as np
    import polars as pl
    from genoray import SparseVar2

    def build(n_samples: int):
        d = tmp_path / f"eng_n{n_samples}"
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
            # SNP-only: haplotype length == region length regardless of genotype, so
            # drained output bytes are deterministic across cohorts of different size.
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
        svar2 = d / "store.svar2"
        SparseVar2.from_vcf(
            svar2, bcf, no_reference=True, skip_out_of_scope=True, overwrite=True
        )
        return svar2, ref

    def _with_strategy(sds, strategy):
        """Test-only seam (PR-3 Task 3, duplicated here per the prior reviewer's
        note that a third copy in a third test module is acceptable): force a
        `_prefetch_strategy` on a clone of an already-constructed StreamingDataset."""
        import copy

        clone = copy.copy(sds)
        object.__setattr__(clone, "_prefetch_strategy", strategy)
        return clone

    def one(n_samples: int) -> tuple[int, int]:
        svar2, ref = build(n_samples)
        bed = pl.DataFrame(
            {
                "chrom": ["chr1"] * 4,
                "chromStart": [0, 100, 200, 300],
                "chromEnd": [100, 200, 300, 400],
            }
        )
        base = gvl.StreamingDataset(
            bed, reference=ref, variants=svar2, max_mem="64MB"
        ).with_seqs("haplotypes")
        sds = _with_strategy(base, "svar2_engine")
        max_bytes = 0
        for data, _r, _s in sds.to_iter(batch_size=4, return_indices=True):
            max_bytes = max(max_bytes, int(np.asarray(data.data).nbytes))
        return max_bytes, base.n_samples

    small, n_small = one(50)
    large, n_large = one(400)
    print(
        f"[svar2 engine output-gate] n=50 peak_bytes={small} n=400 peak_bytes={large}"
    )

    assert n_small == 50 and n_large == 400, (
        "plan did not cover the whole cohort at one or both sizes -- test no longer "
        "proves what it claims"
    )
    assert small > 0, "counter is not wired (engine drained no output)"
    assert small == large, (
        f"engine peak per-batch output scaled with cohort size (50->{small}B, "
        f"400->{large}B) -- cohort-scale residency has returned under the "
        "svar2_engine strategy (#284)"
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
