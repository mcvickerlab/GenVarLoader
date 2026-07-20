"""PR 2: Svar2ReconBuf super-batch fill + drain — byte-identical to the per-batch
read-bound FFI, and self-consistent across drain boundaries."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
import pytest
from numpy.typing import NDArray
from seqpro.rag import Ragged

# Core-utilization fixture: SNP-only, wide enough (rows * ploidy >> num_threads) that
# a single core-saturating super-batch under GVL_FORCE_PARALLEL actually engages
# multiple cores, not just clears the size gate. Sized so each window is ~2400 rows
# (8 regions x 300 samples) -> one ~4800-haplotype parallel super-batch, big enough
# that the accumulated cpu/wall signal clears the noise floor over repeated timed
# passes. See test_super_batch_engages_multiple_cores below.
_CU_N_VARIANTS = 200
_CU_N_SAMPLES = 300
_CU_CONTIG_LEN = 4000
_CU_N_REGIONS = 8
_CU_REGION_WIDTH = 100


@dataclass(frozen=True)
class _Svar2ScaleFixture:
    bed: pl.DataFrame
    reference_path: Path
    svar2_path: Path


@pytest.fixture(scope="module")
def svar2_scale_fixture(tmp_path_factory) -> _Svar2ScaleFixture:
    """Self-contained SVAR2 store: chr1 reference + a SNP-only VCF (200 variants x 300
    samples, random 0|1 genotypes), converted with `no_reference=True,
    skip_out_of_scope=True` (mirrors `test_streaming_scale.py`'s
    `test_svar2_generate_batch_output_is_flat_in_cohort_size` builder, replicated here
    so this module doesn't depend on a helper nested inside another task's test
    function)."""
    from genoray import SparseVar2

    d = tmp_path_factory.mktemp("svar2_core_util")
    ref = d / "ref.fa"
    rng = np.random.default_rng(1)
    seq = "".join(rng.choice(list("ACGT"), _CU_CONTIG_LEN))
    ref.write_text(f">chr1\n{seq}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    vcf = d / "in.vcf"
    lines = [
        "##fileformat=VCFv4.2",
        f"##contig=<ID=chr1,length={_CU_CONTIG_LEN}>",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        + "\t".join(f"S{i}" for i in range(_CU_N_SAMPLES)),
    ]
    positions = np.sort(
        rng.choice(np.arange(2, _CU_CONTIG_LEN - 2), _CU_N_VARIANTS, replace=False)
    )
    for pos in positions:
        # SNP-only (REF=A/ALT=G): no indels, so haplotype reconstruction stays cheap
        # per-row and the fixture's cost scales predictably with rows * ploidy.
        gts = "\t".join(
            f"{rng.integers(0, 2)}|{rng.integers(0, 2)}" for _ in range(_CU_N_SAMPLES)
        )
        lines.append(f"chr1\t{pos}\t.\tA\tG\t.\t.\t.\tGT\t{gts}")
    vcf.write_text("\n".join(lines) + "\n")

    bcf = d / "in.bcf"
    subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", str(bcf)], check=True)

    svar2 = d / "store.svar2"
    SparseVar2.from_vcf(
        svar2, bcf, no_reference=True, skip_out_of_scope=True, overwrite=True
    )

    starts = [i * _CU_REGION_WIDTH for i in range(_CU_N_REGIONS)]
    bed = pl.DataFrame(
        {
            "chrom": ["chr1"] * _CU_N_REGIONS,
            "chromStart": starts,
            "chromEnd": [s + _CU_REGION_WIDTH for s in starts],
        }
    )
    return _Svar2ScaleFixture(bed=bed, reference_path=ref, svar2_path=svar2)


def _plan_windows(sds):
    """(r_idx, s_idx) windows exactly as the sync drive sees them."""
    return list(sds._plan())


def _per_batch_reference(
    backend,
    r_idx: NDArray[np.intp],
    s_idx: NDArray[np.intp],
    window: dict[str, object],
    lo: int,
    hi: int,
) -> Ragged:
    """Parity reference (Phase-2 PR 2): the *old* `_Svar2Backend.generate_batch`/
    `_reconstruct_batch_reference` body, moved here once production switched over to
    the super-batch `_fill_super_batch`/`_drain` path (Task 3) -- so production has
    ONE obvious reconstruct path, and this stays as the independent oracle the
    super-batch fill+drain path is checked byte-identical against.
    """
    from genvarloader.genvarloader import reconstruct_haplotypes_from_svar2_readbound

    P = backend.ploidy
    (
        region_starts,
        orig_samples,
        vk_snp,
        vk_indel,
        dense_snp,
        dense_indel,
        region_bounds,
        shifts,
        ref_,
        ref_offsets,
    ) = backend._gather_rows(r_idx, s_idx, window, lo, hi)
    contig_idx = cast(int, window["contig_idx"])
    contig = backend._contigs[contig_idx]
    m = hi - lo

    data, offsets = reconstruct_haplotypes_from_svar2_readbound(
        backend._store,
        contig,
        region_starts,
        orig_samples,
        vk_snp,
        vk_indel,
        dense_snp,
        dense_indel,
        region_bounds,
        shifts,
        ref_,
        ref_offsets,
        np.uint8(backend._ref.pad_char),
        np.int64(-1),  # ragged output (no fixed output_length)
        False,  # parallel: per-batch reconstruct is tiny (~batch_size*ploidy
        #        haplotypes); the 96-thread rayon fork/join costs more than it saves
        #        here (measured 1.2-1.8x faster serial).
        False,  # filter_exonic (splicing out of scope)
    )
    return Ragged.from_offsets(
        np.asarray(data).view("S1"), (m, P, None), np.asarray(offsets, np.int64)
    )


def test_super_batch_fill_drain_matches_per_batch_ffi(
    svar2_multicontig_fixture,
) -> None:
    """Reconstructing a whole super-batch then draining batch_size slices is
    byte-identical to reconstructing each batch_size slice on its own via the
    Phase-1/PR-1 per-batch FFI (parallel=False)."""
    import genvarloader as gvl
    from genvarloader.genvarloader import Svar2ReconBuf, svar2_reconstruct_super_batch

    assert svar2_reconstruct_super_batch is not None

    fx = svar2_multicontig_fixture
    sds = gvl.StreamingDataset(
        fx.bed, reference=fx.reference_path, variants=fx.svar2_path
    ).with_seqs("haplotypes")
    backend = sds._backend
    assert backend is not None
    P = backend.ploidy

    for r_idx, s_idx in _plan_windows(sds):
        window = backend.read_window(r_idx, s_idx)
        n_rows = len(r_idx) * len(s_idx)

        # (a) super-batch: one fill over ALL window rows, drain per batch_size.
        buf = Svar2ReconBuf(P)
        backend._fill_super_batch(r_idx, s_idx, window, 0, n_rows, buf, parallel=False)
        assert buf.n_rows == n_rows
        drained = []
        for lo in range(0, n_rows, 4):
            hi = min(lo + 4, n_rows)
            rag = backend._drain(buf, lo, hi)
            drained.append(rag.data.view("S1").copy())
        got = np.concatenate(drained) if drained else np.empty(0, "S1")

        # (b) reference: per-batch FFI, parallel=False (the PR-1 path).
        ref_parts = []
        for lo in range(0, n_rows, 4):
            hi = min(lo + 4, n_rows)
            rag = _per_batch_reference(backend, r_idx, s_idx, window, lo, hi)
            ref_parts.append(np.asarray(rag.data).view("S1").copy())
        ref = np.concatenate(ref_parts) if ref_parts else np.empty(0, "S1")

        np.testing.assert_array_equal(got, ref)


def test_super_batch_engages_multiple_cores(svar2_scale_fixture, monkeypatch) -> None:
    """With a core-saturating super-batch, cpu_secs/wall rises materially above the
    single-core (~1x) PR-1 baseline. GVL_FORCE_PARALLEL removes the size gate so the
    fixture still dispatches rayon; the signal is threads engaged, not speed.

    The ratio is accumulated over many repeated passes so the totals reach tens of ms
    -- a single pass is only a few ms of CPU, where timer quantization and scheduler
    jitter can flip a tight ratio across the threshold (a real, reproduced flake). The
    bound stays loose (1.3) for shared-node noise; the point is *threads engaged*, not
    a precise speedup. Skipped rather than asserted on single-core nodes, where the
    ratio cannot exceed ~1 no matter how well the work parallelizes.
    """
    import os
    import time

    import pytest

    import genvarloader as gvl

    if len(os.sched_getaffinity(0)) < 2:
        pytest.skip("core-utilization gate is meaningless on a single-core node")

    # Coverage instrumentation traces the Python driver loop serially, adding
    # CPU+wall overhead around the GIL-free Rust rayon section and pinning cpu/wall
    # near ~1.1 no matter how well the reconstruct parallelizes -- the same reason
    # this gate is meaningless on a single-core node. Skip when a coverage session
    # is active so the threshold is only asserted where the measurement is valid.
    try:
        import coverage
    except ImportError:
        coverage = None
    if coverage is not None and coverage.Coverage.current() is not None:
        pytest.skip(
            "cpu/wall parallelism signal is invalid under coverage instrumentation"
        )

    monkeypatch.setenv("GVL_FORCE_PARALLEL", "1")
    fx = svar2_scale_fixture
    sds = gvl.StreamingDataset(
        fx.bed, reference=fx.reference_path, variants=fx.svar2_path
    ).with_seqs("haplotypes")

    for _ in sds.to_iter(batch_size=8):  # warm import/JIT/FASTA-cache off the clock
        pass

    # Accumulate cpu/wall over repeated passes: totals in the tens-of-ms range make the
    # ratio robust to the sub-ms timer jitter that flakes a single ~3ms pass.
    repeats = 30
    cpu, wall = 0.0, 0.0
    n = 0
    for _ in range(repeats):
        cpu0, wall0 = time.process_time(), time.perf_counter()
        for _data, _r, _s in sds.to_iter(batch_size=8):
            n += 1
        cpu += time.process_time() - cpu0
        wall += time.perf_counter() - wall0

    ratio = cpu / wall if wall > 0 else 0.0
    print(
        f"[svar2 core-util] cpu={cpu:.3f}s wall={wall:.3f}s ratio={ratio:.2f} "
        f"batches={n} repeats={repeats}"
    )
    assert ratio > 1.3, (
        f"super-batch reconstruct did not engage multiple cores (cpu/wall={ratio:.2f}); "
        "expected >1.3 with GVL_FORCE_PARALLEL"
    )
