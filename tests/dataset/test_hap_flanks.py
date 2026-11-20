"""Haplotype ("personalized") variant flanks: ``flank_source="haplotype"`` (issue #430).

A dense synthetic VCF puts SNPs, insertions and deletions within a few bp of
each other, so most flanks overlap neighbouring ALTs. The oracle is gvl's own
reconstructed haplotype over the whole contig: for every variant the haplotype
carries, ``flank5 . ALT . flank3`` must be exactly the haplotype around it.
Checked for SVAR1 (``.svar``) and SVAR2 (``.svar2``, bare-ALT deletions), for
both the ride-along ``flank_tokens`` and ``variant-windows`` outputs.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import genvarloader as gvl
from genvarloader import VarWindowOpt

#       1234567890123456789012345678901234567890
_REF = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"
_VCF = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=40>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1
chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|1
chr1\t5\t.\tT\tC\t.\t.\t.\tGT\t1|1\t0|1
chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1
chr1\t10\t.\tG\tC\t.\t.\t.\tGT\t1|1\t1|0
chr1\t12\t.\tGTA\tG\t.\t.\t.\tGT\t1|1\t0|1
chr1\t16\t.\tT\tA\t.\t.\t.\tGT\t1|0\t1|1
chr1\t18\t.\tG\tGCC\t.\t.\t.\tGT\t0|1\t1|0
chr1\t22\t.\tGGC\tG\t.\t.\t.\tGT\t1|1\t1|0
chr1\t25\t.\tT\tG\t.\t.\t.\tGT\t0|1\t1|1
chr1\t38\t.\tG\tA\t.\t.\t.\tGT\t1|0\t0|1
"""
L = 4
_ALPHABET = b"ACGT"
_UNK = 4


@pytest.fixture(scope="module")
def _src(tmp_path_factory) -> tuple[Path, Path]:
    d = tmp_path_factory.mktemp("hapflank_src")
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)
    vcf = d / "in.vcf"
    vcf.write_text(_VCF)
    bcf = d / "in.bcf"
    subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", str(bcf)], check=True)
    return bcf, ref


@pytest.fixture(scope="module")
def _datasets(_src, tmp_path_factory) -> dict[str, gvl.Dataset]:
    from genoray import VCF, SparseVar, SparseVar2, _core
    from genoray._pipeline_args import FieldSpec, PlanSettings, RegionSpec

    bcf, ref = _src
    d = tmp_path_factory.mktemp("hapflank_ds")
    svar = d / "store.svar"
    SparseVar.from_vcf(svar, VCF(bcf), max_mem="1g", samples=["S0", "S1"])
    svar2 = d / "store.svar2"
    _core.run_conversion_pipeline(
        vcf_path=str(bcf),
        reference_path=str(ref),
        output_dir=str(svar2),
        regions=RegionSpec(chroms=["chr1"], samples=["S0", "S1"]),
        fields=FieldSpec(),
        plan=PlanSettings(
            chunk_size=25_000, max_threads=1, long_allele_capacity=8 * 1024 * 1024
        ),
        ploidy=2,
    )
    # One region over the whole contig so the reconstructed haplotype is the
    # complete oracle (flank bases past the contig are padding on both sides).
    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [40]})
    out = {}
    for name, variants in (("svar", SparseVar(svar)), ("svar2", SparseVar2(svar2))):
        path = d / f"{name}.gvl"
        gvl.write(path, bed, variants=variants, samples=None, overwrite=True)
        out[name] = gvl.Dataset.open(path, reference=ref).with_settings(
            rc_neg=False, var_fields=["alt", "ilen", "start"]
        )
    return out


def _lut() -> np.ndarray:
    lut = np.full(256, _UNK, np.uint8)
    for i, c in enumerate(_ALPHABET):
        lut[c] = i
    return lut


def _expected(ds: gvl.Dataset):
    """Per-variant (f5, allele, f3) byte strings from gvl's reconstructed haplotypes.

    Variant rows are walked in (sample, ploid) C-order. Each row's haplotype is
    padded with ``N`` (the reference pad char) on both sides; a variant's offset
    into it is its start shifted by the ilen of every earlier variant on the row.
    """
    haps = ds.with_seqs("haplotypes").with_len("ragged")[0, :]
    var = ds.with_seqs("variants")[0, :]
    hap_data = np.asarray(haps.data).view(np.uint8).tobytes()
    hap_off = np.concatenate([[0], np.cumsum(np.asarray(haps.lengths).ravel())])
    v_off = np.concatenate([[0], np.cumsum(np.asarray(var.start.lengths).ravel())])
    all_starts = np.asarray(var.start.data).tolist()
    all_ilens = np.asarray(var.ilen.data).tolist()
    out = []
    for row in range(len(hap_off) - 1):
        hap = hap_data[hap_off[row] : hap_off[row + 1]]
        padded = b"N" * L + hap + b"N" * L
        shift = 0
        for k in range(v_off[row], v_off[row + 1]):
            st, il = all_starts[k], all_ilens[k]
            alen = 1 + max(il, 0)  # atomized: anchor + inserted bases
            pos = L + st + shift
            out.append(
                (
                    padded[pos - L : pos],
                    padded[pos : pos + alen],
                    padded[pos + alen : pos + alen + L],
                )
            )
            shift += il
    return out


@pytest.mark.parametrize("store", ["svar", "svar2"])
def test_flank_tokens_match_haplotype(_datasets, store):
    ds = _datasets[store]
    lut = _lut()
    flat = (
        ds.with_seqs("variants")
        .with_output_format("flat")
        .with_settings(
            flank_length=L,
            token_alphabet=_ALPHABET,
            unknown_token=_UNK,
            flank_source="haplotype",
        )[0, :]
    )
    got = np.asarray(flat.flank_tokens.data).reshape(-1, 2 * L)
    exp = _expected(ds)
    assert len(exp) == len(got) > 0
    n_personal = 0
    ref_flat = (
        ds.with_seqs("variants")
        .with_output_format("flat")
        .with_settings(flank_length=L, token_alphabet=_ALPHABET, unknown_token=_UNK)[
            0, :
        ]
    )
    ref_tok = np.asarray(ref_flat.flank_tokens.data).reshape(-1, 2 * L)
    for i, (f5, _a, f3) in enumerate(exp):
        e = lut[np.frombuffer(f5 + f3, np.uint8)]
        np.testing.assert_array_equal(got[i], e, err_msg=f"{store} variant {i}")
        n_personal += not np.array_equal(got[i], ref_tok[i])
    # The fixture is dense: most flanks must differ from the reference ones.
    assert n_personal >= len(exp) // 2, (n_personal, len(exp))


@pytest.mark.parametrize("store", ["svar", "svar2"])
def test_variant_windows_match_haplotype(_datasets, store):
    ds = _datasets[store]
    lut = _lut()
    opt = VarWindowOpt(
        flank_length=L,
        token_alphabet=_ALPHABET,
        unknown_token=_UNK,
        flank_source="haplotype",
    )
    w = ds.with_output_format("flat").with_seqs("variant-windows", opt)[0, :]
    exp = _expected(ds)
    a_off = np.asarray(w.alt_window.seq_offsets)
    r_off = np.asarray(w.ref_window.seq_offsets)
    a_dat = np.asarray(w.alt_window.data)
    r_dat = np.asarray(w.ref_window.data)
    starts = np.asarray(w.fields["start"].data)
    ilens = np.asarray(w.fields["ilen"].data)
    ref = np.frombuffer(_REF.encode(), np.uint8)
    assert len(exp) == len(a_off) - 1 > 0
    for i, (f5, allele, f3) in enumerate(exp):
        # alt_window: exactly the haplotype around the variant.
        np.testing.assert_array_equal(
            a_dat[a_off[i] : a_off[i + 1]],
            lut[np.frombuffer(f5 + allele + f3, np.uint8)],
            err_msg=f"{store} alt_window {i}",
        )
        # ref_window: same haplotype context, focal variant reverted to REF.
        s, e = starts[i], starts[i] - min(ilens[i], 0) + 1
        np.testing.assert_array_equal(
            r_dat[r_off[i] : r_off[i + 1]],
            lut[
                np.concatenate(
                    [np.frombuffer(f5, np.uint8), ref[s:e], np.frombuffer(f3, np.uint8)]
                )
            ],
            err_msg=f"{store} ref_window {i}",
        )


def test_svar_svar2_identical(_datasets):
    toks = []
    for store in ("svar", "svar2"):
        flat = (
            _datasets[store]
            .with_seqs("variants")
            .with_output_format("flat")
            .with_settings(
                flank_length=L,
                token_alphabet=_ALPHABET,
                unknown_token=_UNK,
                flank_source="haplotype",
            )[0, :]
        )
        toks.append(np.asarray(flat.flank_tokens.data))
    np.testing.assert_array_equal(toks[0], toks[1])


def test_default_is_reference(_datasets):
    ds = _datasets["svar"]
    assert ds._seqs.flank_source == "reference"
    assert (
        VarWindowOpt(
            flank_length=1, token_alphabet=b"ACGT", unknown_token=4
        ).flank_source
        == "reference"
    )


def test_bad_flank_source_rejected(_datasets):
    with pytest.raises(ValueError, match="flank_source"):
        _datasets["svar"].with_settings(flank_source="personal")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="flank_source"):
        VarWindowOpt(
            flank_length=1,
            token_alphabet=b"ACGT",
            unknown_token=4,
            flank_source="x",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("kind", ["variants", "variant-windows"])
def test_unphased_union_rejected(_datasets, kind):
    ds = _datasets["svar"].with_output_format("flat")
    if kind == "variants":
        ds = ds.with_seqs("variants").with_settings(
            flank_length=L,
            token_alphabet=_ALPHABET,
            unknown_token=_UNK,
            flank_source="haplotype",
        )
        with pytest.raises(ValueError, match="unphased_union"):
            ds.with_settings(unphased_union=True)
    else:
        opt = VarWindowOpt(
            flank_length=L,
            token_alphabet=_ALPHABET,
            unknown_token=_UNK,
            flank_source="haplotype",
        )
        with pytest.raises(ValueError, match="unphased_union"):
            ds.with_settings(unphased_union=True).with_seqs("variant-windows", opt)
