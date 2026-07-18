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
"""

from __future__ import annotations

import numpy as np
import polars as pl

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
