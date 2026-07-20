"""Task 7 (issue #276): the #1-risk parity gate.

`gvl.write` decodes VCF genotypes via genoray's Python cyvcf2 + `dense2sparse`.
`StreamingDataset`'s VCF backend decodes the SAME VCF via genoray's Rust
`ChunkAssembler` (`RecordStreamEngine` -> `VcfWindowFiller`,
`src/record_stream/vcf.rs`). These are two INDEPENDENT decoders; byte-identical
end-to-end haplotype parity (Task 8) is only possible if they agree on the
variant TABLE first -- the cheapest layer to check, and far cheaper to debug
than an opaque haplotype byte-diff.

This module pins that: the streamed window's local variant table
(`v_starts`/`ilens`/`alt` via `RecordStreamEngine.debug_decode_window`, a
test-only accessor added in `src/record_stream/engine.rs`) must exactly equal
the written dataset's stored variant table (`Dataset._seqs.ffi_static`, the
same FFI-ready arrays the reconstruction kernels consume) for the SAME VCF.

Not in scope here (Task 8): end-to-end haplotype byte parity. This test never
reconstructs a haplotype -- it only compares the two decoders' POS/ILEN/ALT
tables.

Same-POS tie-break (issue #300, corrected): the streamed decoder (Rust
`ChunkAssembler`) and the written oracle (`gvl.write` / `.gvi`) both
tie-break same-POS atoms by FILE ORDER -- the streamed side via a
`BinaryHeap` keyed on `(pos, seq)` where `seq = record_seq<<32 | atom_ix` is
a monotonic file-row counter (no lexicographic ALT comparison anywhere in
that `Ord` impl), the written side via genoray's `.gvi` file-row index (no
ALT sort). Since `gvl.write` only accepts pre-split biallelic input (one
atom per record), `(pos, record-order)` agreement between the two decoders
holds BY CONSTRUCTION. `vcf_snp_ins_del_multi`'s pos=149 pair happens to
also be in lexicographic order, so it alone wouldn't distinguish a
file-order tie-break from a (wrong) lexicographic one;
`test_same_pos_var_idxs_file_order` below uses `vcf_same_pos_nonlex` /
`vcf_same_pos_triallelic`, where file order and lexicographic order
genuinely diverge, to prove the invariant is real and not coincidental.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import genvarloader as gvl
from genvarloader.genvarloader import RecordStreamEngine


def test_vcf_streamed_variant_table_matches_written(vcf_snp_ins_del_multi, tmp_path):
    f = vcf_snp_ins_del_multi
    contig_len = int(f.regions["chromEnd"][0])

    # Reference bytes for the engine's contig table -- must match exactly what
    # gvl.write's dataset was opened against (`f.fasta`, single-record FASTA).
    ref_seq = "".join(f.fasta.read_text().splitlines()[1:])
    assert len(ref_seq) == contig_len

    # 1. Write the oracle dataset (Python cyvcf2 + dense2sparse decode path).
    out = tmp_path / "ds"
    gvl.write(out, f.regions, variants=str(f.vcf), overwrite=True)
    written = gvl.Dataset.open(out, reference=f.fasta)
    oracle = written._seqs.ffi_static  # type: ignore[attr-defined]

    # 2. Stream the same window's table (Rust ChunkAssembler decode path). No
    # jobs are registered on the engine -- `debug_decode_window` bypasses the
    # producer/consumer plan entirely and decodes one ad hoc window.
    eng = RecordStreamEngine(
        "vcf",
        str(f.vcf),
        f.sample_names,
        f.ploidy,
        [f.contig],
        [ref_seq.encode()],
        [],
        [],
        [],
        [],
        [],
        None,  # fasta_path=None -- matches gvl.write's VCF parity (no read-time left-align)
        ord("N"),
        False,
        32,
        -1,  # output_length: ragged (unused by debug_decode_window, no generation here)
    )
    v_starts, ilens, alt_alleles, alt_offsets = eng.debug_decode_window(
        0, [0], [contig_len], 0, f.n_samples
    )

    # 3. The fixture's single region spans the whole contig, so the streamed
    # window's table and the dataset's whole-VCF stored table cover the exact
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


def test_vcf_streamed_variant_table_covers_all_variant_classes(vcf_snp_ins_del_multi):
    """Non-tautological check on the fixture itself: confirms the committed
    VCF's positions/ILENs are what the docstring/module doc claim, independent
    of the streamed-vs-written comparison above."""
    import subprocess

    f = vcf_snp_ins_del_multi
    out = subprocess.run(
        ["bcftools", "view", "-H", str(f.vcf)],
        check=True,
        capture_output=True,
    ).stdout.decode()
    rows = [line.split("\t") for line in out.strip().splitlines()]
    assert len(rows) == 5

    records = pl.DataFrame(
        {
            "pos": [int(r[1]) for r in rows],
            "ref": [r[3] for r in rows],
            "alt": [r[4] for r in rows],
        }
    ).with_columns(
        ilen=pl.col("alt").str.len_bytes().cast(pl.Int32)
        - pl.col("ref").str.len_bytes().cast(pl.Int32)
    )

    assert records.filter(pl.col("ilen") == 0).height == 3  # SNP + 2 multiallelic atoms
    assert records.filter(pl.col("ilen") > 0).height == 1  # insertion
    assert records.filter(pl.col("ilen") < 0).height == 1  # deletion
    # The multiallelic split: both atoms anchor at the same POS.
    assert records["pos"].value_counts().filter(pl.col("count") > 1).height == 1


# ---------------------------------------------------------------------------
# Task 8 (issue #276): the end-to-end oracle. Task 7 (above) proves the
# streamed variant TABLE byte-equals the written table for the same VCF -- so
# if any test below fails, the divergence is NOT in decode; it is in the
# transpose or the reconstruction arg-assembly (`RecordBackend::generate`,
# `src/record_stream/engine.rs`), neither of which the table gate covers.
# ---------------------------------------------------------------------------


def _assert_cell_matches(streamed, expected, ploidy: int) -> None:
    """Per-haplotype comparison (mirrors `test_streaming_parity.py`'s
    pattern): haplotype lengths are ragged across cells (indels shift
    length), so compare haplotype-by-haplotype rather than assuming a single
    dense array shape."""
    for h in range(ploidy):
        np.testing.assert_array_equal(np.asarray(streamed[h]), np.asarray(expected[h]))


def test_vcf_streaming_matches_written_all_cells(vcf_snp_ins_del_multi, tmp_path):
    """Base oracle, mirroring `test_streaming_parity.py`'s
    `test_streaming_matches_written_all_cells` shape for the VCF backend:
    single region spanning the whole 250bp contig, 3 samples -> 3 cells.
    `batch_size=2` still exercises a within-window batch boundary (2, then
    1) even with only one region."""
    f = vcf_snp_ins_del_multi
    gvl.write(tmp_path / "ds", f.regions, variants=str(f.vcf))
    written = gvl.Dataset.open(tmp_path / "ds", reference=f.fasta).with_seqs(
        "haplotypes"
    )
    sds = gvl.StreamingDataset(
        f.regions, reference=str(f.fasta), variants=str(f.vcf)
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


def test_vcf_streaming_matches_written_all_cells_multi_region(
    vcf_snp_ins_del_multi, vcf_snp_ins_del_multi_regions, tmp_path
):
    """Multi-region coverage: `vcf_snp_ins_del_multi_regions` splits the same
    VCF/FASTA's contig into 3 disjoint sub-regions (see its docstring in
    conftest.py), so a single window's genotype CSR must be expanded
    per-(region, sample) -- exactly the axis `RecordBackend::generate` had a
    Critical bug on (Task 3b) that a single-region test cannot exercise.
    3 regions x 3 samples = 9 cells; `batch_size=4` does NOT evenly divide 9
    (4, 4, 1), covering the window/batch-boundary requirement simultaneously.
    """
    f = vcf_snp_ins_del_multi
    regions = vcf_snp_ins_del_multi_regions
    gvl.write(tmp_path / "ds", regions, variants=str(f.vcf))
    written = gvl.Dataset.open(tmp_path / "ds", reference=f.fasta).with_seqs(
        "haplotypes"
    )
    sds = gvl.StreamingDataset(
        regions, reference=str(f.fasta), variants=str(f.vcf)
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


# ---------------------------------------------------------------------------
# Task 5 (issue #300): lock the file-order same-POS tie-break invariant with
# GENUINE (non-lexicographic) fixtures, correcting the previously-wrong
# "lexicographic ALT order" docstrings (see module docstring above and the
# `vcf_snp_ins_del_multi` / `pgen_snp_ins_del_multi` fixture docstrings in
# conftest.py).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture_name",
    [
        "vcf_same_pos_nonlex",
        "pgen_same_pos_nonlex",
        "vcf_same_pos_triallelic",
        "pgen_same_pos_triallelic",
    ],
)
def test_same_pos_var_idxs_file_order(request, fixture_name, tmp_path):
    """Annotated `var_idxs` parity vs. the written oracle for same-POS
    fixtures where FILE ORDER != LEXICOGRAPHIC ALT ORDER by construction
    (see `vcf_same_pos_nonlex`/`vcf_same_pos_triallelic` docstrings in
    conftest.py for the exact variant tables and genotypes).

    This is expected to PASS: both the streamed `ChunkAssembler` and the
    written oracle tie-break same-POS atoms by file order (issue #300), so
    their `var_idxs` must agree even though a lexicographic tie-break would
    have produced a DIFFERENT (and here, wrong) answer -- a pass on these
    fixtures is not a coincidence the way it would be on
    `vcf_snp_ins_del_multi`'s pos=149 pair (see module docstring).
    """
    f = request.getfixturevalue(fixture_name)
    variants = str(f.vcf) if hasattr(f, "vcf") else str(f.pgen)

    out = tmp_path / "ds"
    gvl.write(out, f.regions, variants=variants, overwrite=True)
    written = gvl.Dataset.open(out, reference=f.fasta)
    ds = written.with_seqs("annotated")
    sds = gvl.StreamingDataset(
        f.regions, reference=str(f.fasta), variants=variants
    ).with_seqs("annotated")

    n_cells = 0
    saw_same_pos_variant = False
    for data, r_idx, s_idx in sds.to_iter(batch_size=4):
        for row in range(len(r_idx)):
            exp = ds[int(r_idx[row]), int(s_idx[row])]  # RaggedAnnotatedHaps
            for h in range(sds.ploidy):
                got_vidx = np.asarray(data.var_idxs[row][h])
                exp_vidx = np.asarray(exp.var_idxs[h])
                np.testing.assert_array_equal(
                    got_vidx,
                    exp_vidx,
                    err_msg=(
                        f"{fixture_name} row={row} hap={h}: streamed "
                        "var_idxs must match the written (file-order) "
                        "oracle for same-POS atoms even when file order != "
                        "lexicographic ALT order (#300)"
                    ),
                )
                if np.any(exp_vidx >= 0):
                    saw_same_pos_variant = True
            n_cells += 1
    assert n_cells > 0, f"{fixture_name}: fixture yielded no cells"
    assert saw_same_pos_variant, (
        f"{fixture_name}: no variant-carrying haplotype seen; the "
        "same-POS var_idxs comparison proves nothing without one"
    )


def test_vcf_streaming_matches_written_all_cells_multi_contig(
    vcf_multi_contig, tmp_path
):
    """Multi-contig coverage: `vcf_multi_contig` has 2 regions on `chr1` and 2
    on `chr2` (distinct SNP/INS/DEL sets per contig); `_plan`'s window
    traversal is per-contig-run (`_window_regions=64` default groups each
    contig's 2 regions into one window), so this also exercises the
    per-contig window boundary. 4 regions x 3 samples = 12 cells (6/contig);
    `batch_size=5` does NOT evenly divide 6, giving a (5, 1) split within
    each contig's window -- boundary coverage on both axes at once.
    """
    f = vcf_multi_contig
    gvl.write(tmp_path / "ds", f.regions, variants=str(f.vcf))
    written = gvl.Dataset.open(tmp_path / "ds", reference=f.fasta).with_seqs(
        "haplotypes"
    )
    sds = gvl.StreamingDataset(
        f.regions, reference=str(f.fasta), variants=str(f.vcf)
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
