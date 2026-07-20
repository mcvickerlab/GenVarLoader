"""Task 12 (issue #276): the PGEN analog of Tasks 7+8's VCF parity gates.

`gvl.write` decodes PGEN genotypes via genoray's Python pgenlib +
`dense2sparse`. `StreamingDataset`'s PGEN backend decodes the SAME `.pgen`
via genoray's Rust `ChunkAssembler` (`RecordStreamEngine` ->
`PgenWindowFiller`, `src/record_stream/pgen.rs`). These are two INDEPENDENT
decoders; byte-identical end-to-end haplotype parity (mirroring Task 8) is
only possible if they agree on the variant TABLE first -- the cheapest
layer to check, and far cheaper to debug than an opaque haplotype byte-diff
(mirroring Task 7).

`pgen_snp_ins_del_multi` (see conftest.py) is the PGEN conversion of the
SAME `vcf_snp_ins_del_multi.vcf.gz` used by `test_streaming_vcf_parity.py`
(via `plink2 --make-pgen`), so this module's variant table/tie-break caveat
is identical to that module's -- see its docstring for detail.

Not in scope here: decode-vs-decode comparison between the VCF and PGEN
backends themselves (Task 10 already pinned `PgenWindowFiller`'s
`DecodedWindow` byte-equal to `VcfWindowFiller`'s on shared variants). This
module only compares each backend's streamed table/haplotypes against ITS
OWN written-dataset oracle.
"""

from __future__ import annotations

import numpy as np

import genvarloader as gvl
from genvarloader.genvarloader import RecordStreamEngine


def test_pgen_streamed_variant_table_matches_written(pgen_snp_ins_del_multi, tmp_path):
    f = pgen_snp_ins_del_multi
    contig_len = int(f.regions["chromEnd"][0])

    # Reference bytes for the engine's contig table -- must match exactly what
    # gvl.write's dataset was opened against (`f.fasta`, single-record FASTA).
    ref_seq = "".join(f.fasta.read_text().splitlines()[1:])
    assert len(ref_seq) == contig_len

    # 1. Write the oracle dataset (Python pgenlib + dense2sparse decode path).
    out = tmp_path / "ds"
    gvl.write(out, f.regions, variants=str(f.pgen), overwrite=True)
    written = gvl.Dataset.open(out, reference=f.fasta)
    oracle = written._seqs.ffi_static  # type: ignore[attr-defined]

    # 2. Stream the same window's table (Rust ChunkAssembler decode path). No
    # jobs are registered on the engine -- `debug_decode_window` bypasses the
    # producer/consumer plan entirely and decodes one ad hoc window.
    eng = RecordStreamEngine(
        "pgen",
        str(f.pgen),
        f.sample_names,
        f.ploidy,
        [f.contig],
        [ref_seq.encode()],
        [],
        [],
        [],
        [],
        [],
        None,  # fasta_path=None -- matches gvl.write's PGEN parity (no read-time left-align)
        ord("N"),
        False,
        32,
    )
    v_starts, ilens, alt_alleles, alt_offsets = eng.debug_decode_window(
        0, [0], [contig_len], 0, f.n_samples
    )

    # 3. The fixture's single region spans the whole contig, so the streamed
    # window's table and the dataset's whole-PGEN stored table cover the exact
    # same variant set -- direct array comparison, no restriction needed.
    streamed_v_starts = np.asarray(v_starts, dtype=np.int32)
    streamed_ilens = np.asarray(ilens, dtype=np.int32)
    streamed_alt = np.frombuffer(bytes(alt_alleles), dtype=np.uint8)
    streamed_alt_offsets = np.asarray(alt_offsets, dtype=np.int64)

    # Sanity: the fixture actually exercises all four variant classes (else
    # this test proves nothing) -- 5 variants (multiallelic split into 2).
    assert len(streamed_v_starts) == 5
    assert (streamed_ilens == 0).sum() == 3  # SNP + 2 split-multiallelic atoms
    assert (streamed_ilens > 0).any()  # insertion
    assert (streamed_ilens < 0).any()  # deletion

    np.testing.assert_array_equal(
        streamed_v_starts, oracle.v_starts, err_msg="v_starts (POS) mismatch"
    )
    np.testing.assert_array_equal(
        streamed_ilens, oracle.ilens, err_msg="ilens mismatch"
    )
    np.testing.assert_array_equal(
        streamed_alt_offsets, oracle.alt_offsets, err_msg="alt_offsets mismatch"
    )
    np.testing.assert_array_equal(
        streamed_alt, oracle.alt_alleles, err_msg="ALT bytes mismatch"
    )


# ---------------------------------------------------------------------------
# End-to-end oracle (mirrors Task 8). The table gate above proves the
# streamed variant TABLE byte-equals the written table for the same PGEN --
# so if any test below fails, the divergence is NOT in decode; it is in the
# transpose or the reconstruction arg-assembly (`RecordBackend::generate`,
# `src/record_stream/engine.rs`), neither of which the table gate covers.
# ---------------------------------------------------------------------------


def _assert_cell_matches(streamed, expected, ploidy: int) -> None:
    """Per-haplotype comparison (mirrors `test_streaming_vcf_parity.py`'s
    pattern): haplotype lengths are ragged across cells (indels shift
    length), so compare haplotype-by-haplotype rather than assuming a single
    dense array shape."""
    for h in range(ploidy):
        np.testing.assert_array_equal(np.asarray(streamed[h]), np.asarray(expected[h]))


def test_pgen_streaming_matches_written_all_cells(pgen_snp_ins_del_multi, tmp_path):
    """Base oracle: single region spanning the whole 250bp contig, 3 samples
    -> 3 cells. `batch_size=2` still exercises a within-window batch
    boundary (2, then 1) even with only one region."""
    f = pgen_snp_ins_del_multi
    gvl.write(tmp_path / "ds", f.regions, variants=str(f.pgen))
    written = gvl.Dataset.open(tmp_path / "ds", reference=f.fasta).with_seqs(
        "haplotypes"
    )
    sds = gvl.StreamingDataset(
        f.regions, reference=str(f.fasta), variants=str(f.pgen)
    ).with_seqs("haplotypes")

    seen = set()
    for data, r_idx, s_idx in sds.to_iter(batch_size=2):
        for k in range(len(r_idx)):
            r, s = int(r_idx[k]), int(s_idx[k])
            _assert_cell_matches(data[k], written[r, s], sds.ploidy)
            seen.add((r, s))

    assert seen == {
        (r, s) for r in range(written.shape[0]) for s in range(written.shape[1])
    }


def test_pgen_streaming_matches_written_all_cells_multi_region(
    pgen_snp_ins_del_multi, vcf_snp_ins_del_multi_regions, tmp_path
):
    """Multi-region coverage: `vcf_snp_ins_del_multi_regions` splits the SAME
    contig `pgen_snp_ins_del_multi` uses into 3 disjoint sub-regions (see its
    docstring in conftest.py), so a single window's genotype CSR must be
    expanded per-(region, sample) -- exactly the axis `RecordBackend::generate`
    had a Critical bug on for VCF (Task 3b) that a single-region test cannot
    exercise, and which the PGEN backend shares the same code path for.
    3 regions x 3 samples = 9 cells; `batch_size=4` does NOT evenly divide 9
    (4, 4, 1), covering the window/batch-boundary requirement simultaneously.
    """
    f = pgen_snp_ins_del_multi
    regions = vcf_snp_ins_del_multi_regions
    gvl.write(tmp_path / "ds", regions, variants=str(f.pgen))
    written = gvl.Dataset.open(tmp_path / "ds", reference=f.fasta).with_seqs(
        "haplotypes"
    )
    sds = gvl.StreamingDataset(
        regions, reference=str(f.fasta), variants=str(f.pgen)
    ).with_seqs("haplotypes")

    seen = set()
    for data, r_idx, s_idx in sds.to_iter(batch_size=4):
        for k in range(len(r_idx)):
            r, s = int(r_idx[k]), int(s_idx[k])
            _assert_cell_matches(data[k], written[r, s], sds.ploidy)
            seen.add((r, s))

    assert seen == {
        (r, s) for r in range(written.shape[0]) for s in range(written.shape[1])
    }


def test_pgen_streaming_matches_written_unsorted_samples(
    pgen_unsorted_samples, tmp_path
):
    """REGRESSION (issue #276 final-review blocker): the ultimate proof for the
    PGEN sample-ordering fix. `pgen_unsorted_samples`'s `.psam` physical order
    is `S10, S2, S1`; the public `sample_idx` order `gvl.write`/`gvl.Dataset`
    use is the lexicographically-sorted `S1, S10, S2` (`"S10" < "S2"`). Each
    sample carries a distinct SNP, so a physical-vs-sorted column mix-up in
    `PgenWindowFiller` yields the WRONG sample's haplotypes. This test FAILS
    before the fix (streamed cell (r, s) != written cell (r, s)) and PASSES
    after. Every prior PGEN fixture used pre-sorted `s0/s1/s2` names and so
    could not catch the bug.
    """
    f = pgen_unsorted_samples
    gvl.write(tmp_path / "ds", f.regions, variants=str(f.pgen))
    written = gvl.Dataset.open(tmp_path / "ds", reference=f.fasta).with_seqs(
        "haplotypes"
    )
    # Sanity: the written dataset's samples ARE in sorted-name order, and the
    # .psam physical order is genuinely unsorted (else the test proves nothing).
    assert list(written.samples) == ["S1", "S10", "S2"]

    sds = gvl.StreamingDataset(
        f.regions, reference=str(f.fasta), variants=str(f.pgen)
    ).with_seqs("haplotypes")
    assert list(sds.samples) == ["S1", "S10", "S2"]

    seen = set()
    for data, r_idx, s_idx in sds.to_iter(batch_size=2):
        for k in range(len(r_idx)):
            r, s = int(r_idx[k]), int(s_idx[k])
            _assert_cell_matches(data[k], written[r, s], sds.ploidy)
            seen.add((r, s))

    assert seen == {
        (r, s) for r in range(written.shape[0]) for s in range(written.shape[1])
    }


def test_pgen_streaming_matches_written_all_cells_multi_contig(
    pgen_multi_contig, tmp_path
):
    """Multi-contig coverage: `pgen_multi_contig` has 2 regions on `chr1` and
    2 on `chr2` (distinct SNP/INS/DEL sets per contig); `_plan`'s window
    traversal is per-contig-run (`_window_regions=64` default groups each
    contig's 2 regions into one window), so this also exercises the
    per-contig window boundary. 4 regions x 3 samples = 12 cells (6/contig);
    `batch_size=5` does NOT evenly divide 6, giving a (5, 1) split within
    each contig's window -- boundary coverage on both axes at once.
    """
    f = pgen_multi_contig
    gvl.write(tmp_path / "ds", f.regions, variants=str(f.pgen))
    written = gvl.Dataset.open(tmp_path / "ds", reference=f.fasta).with_seqs(
        "haplotypes"
    )
    sds = gvl.StreamingDataset(
        f.regions, reference=str(f.fasta), variants=str(f.pgen)
    ).with_seqs("haplotypes")

    seen = set()
    for data, r_idx, s_idx in sds.to_iter(batch_size=5):
        for k in range(len(r_idx)):
            r, s = int(r_idx[k]), int(s_idx[k])
            _assert_cell_matches(data[k], written[r, s], sds.ploidy)
            seen.add((r, s))

    assert seen == {
        (r, s) for r in range(written.shape[0]) for s in range(written.shape[1])
    }
