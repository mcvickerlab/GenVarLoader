"""Integration oracle test for SVAR2 INFO/FORMAT field routing (Task 3.1).

Gates the remaining wiring (Tasks 2.3/2.4) that routes scalar-numeric INFO/FORMAT
field values -- discovered and accepted already (Tasks 2.1/2.2) -- into gvl's
``RaggedVariants`` / variant-windows outputs. Written BEFORE that wiring lands,
so it is EXPECTED TO FAIL (RED) right now; it must fail because the field
VALUES are missing from the output, not because the fixture/store/API is broken.

Oracle: the source VCF, parsed independently with ``cyvcf2`` (never hardcoded
from this repo's own decode output). Coordinate convention: VCF ``POS`` is
1-based; genoray/gvl positions are 0-based, so every oracle key uses
``POS - 1``.

Fixture routing (self-asserted below via ``SparseVar2._find_ranges``, not
assumed):
    - chr1:3 (0-based 2), A>G -- carried by exactly ONE haplotype (S0/hap0)
      out of 6 in the cohort -> cost model routes this to the VAR_KEY channel.
    - chr1:10 (0-based 9), G>C -- carried by ALL 6 haplotypes (hom in every
      sample) -> cost model routes this to the DENSE channel.
    - chr2:5 (0-based 4), A>T -- carried by exactly ONE haplotype (S1/hap0)
      -> VAR_KEY channel, on the second contig.

INFO ``AF`` (Float) is deliberately omitted from the chr1:10 record's INFO to
pin the missing-value fill (NaN, per genoray's ``StoredField`` semantics for a
field declared with no explicit ``default``). ``NS`` (Integer) and FORMAT
``DP`` (Integer) are always present. AF/NS are distinct across every variant;
DP is distinct across every sample within a variant -- both are deliberate
(repeated values would make a broken variant<->value or sample<->value
association silently pass).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# --- fixture: 2 contigs, 3 samples, ploidy 2 (6 haplotypes/contig) ----------

_REF1 = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"  # chr1, 40bp; idx2='A', idx9='G'
_REF2 = "ACGT" * 7 + "AC"  # chr2, 30bp; idx4='A'
assert len(_REF1) == 40 and _REF1[2] == "A" and _REF1[9] == "G"
assert len(_REF2) == 30 and _REF2[4] == "A"

_VCF = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=40>
##contig=<ID=chr2,length=30>
##INFO=<ID=AF,Number=1,Type=Float,Description="Allele frequency">
##INFO=<ID=NS,Number=1,Type=Integer,Description="Number of samples with data">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\tS2
chr1\t3\t.\tA\tG\t.\t.\tAF=0.1;NS=5\tGT:DP\t1|0:10\t0|0:20\t0|0:30
chr1\t10\t.\tG\tC\t.\t.\tNS=6\tGT:DP\t1|1:11\t1|1:21\t1|1:31
chr2\t5\t.\tA\tT\t.\t.\tAF=0.42;NS=2\tGT:DP\t0|0:12\t1|0:22\t0|0:32
"""


@pytest.fixture(scope="module")
def _src(tmp_path_factory) -> tuple[Path, Path]:
    d = tmp_path_factory.mktemp("svar2_fields_src")
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_REF1}\n>chr2\n{_REF2}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    vcf = d / "in.vcf"
    vcf.write_text(_VCF)
    bcf = d / "in.bcf"
    subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", str(bcf)], check=True)
    return bcf, ref


@pytest.fixture(scope="module")
def svar2_fields_store(_src, tmp_path_factory) -> Path:
    from genoray import SparseVar2
    from genoray._svar2_fields import FormatField, InfoField

    bcf, ref = _src
    out = tmp_path_factory.mktemp("svar2_fields") / "store.svar2"
    SparseVar2.from_vcf(
        out=out,
        source=bcf,
        reference=ref,
        info_fields=[InfoField("AF"), InfoField("NS")],
        format_fields=[FormatField("DP")],
        overwrite=True,
    )
    assert (out / "meta.json").exists(), "svar2 conversion did not finish"
    return out


def _build_oracle(bcf_path: Path) -> tuple[dict[tuple[str, int], dict], list[str]]:
    """Parse the source VCF with cyvcf2 (the oracle) into
    ``{(contig, pos0): {"AF": float | None, "NS": int | None,
    "carriers": {(sample_idx, hap_idx): dp_int}}}``.

    ``carriers`` is derived from cyvcf2's own decoded genotypes (allele == 1),
    not hardcoded -- it is the ground truth for which (sample, haplotype)
    pairs a read-bound decode kernel must emit a variant record for.
    """
    from cyvcf2 import VCF as _CyVCF

    vcf = _CyVCF(str(bcf_path))
    try:
        samples = list(vcf.samples)
        oracle: dict[tuple[str, int], dict] = {}
        for rec in vcf:
            contig = rec.CHROM
            pos0 = rec.POS - 1  # VCF POS is 1-based; genoray/gvl positions are 0-based.
            af = rec.INFO.get("AF")
            ns = rec.INFO.get("NS")
            dp = np.asarray(rec.format("DP")).reshape(-1)
            carriers: dict[tuple[int, int], int] = {}
            for s_i, gt in enumerate(rec.genotypes):
                for p_i, allele in enumerate(gt[:2]):
                    if allele == 1:
                        carriers[(s_i, p_i)] = int(dp[s_i])
            oracle[(contig, pos0)] = {
                "AF": None if af is None else float(af),
                "NS": None if ns is None else int(ns),
                "carriers": carriers,
            }
        return oracle, samples
    finally:
        vcf.close()


@pytest.fixture(scope="module")
def oracle_and_samples(_src) -> tuple[dict[tuple[str, int], dict], list[str]]:
    bcf, _ref = _src
    return _build_oracle(bcf)


def _build_dataset(tmp_path: Path, name: str, bed: pl.DataFrame, store: Path, ref: Path):
    import genvarloader as gvl
    from genoray import SparseVar2

    d = tmp_path / name
    gvl.write(d, bed, variants=SparseVar2(store), samples=None, overwrite=True)
    return gvl.Dataset.open(d, reference=ref)


_VAR_FIELDS = ["alt", "start", "ilen", "AF", "NS", "DP"]


# --- self-assert: fixture actually exercises BOTH channels ------------------


def test_svar2_fields_store_has_fields_and_routes_both_channels(svar2_fields_store):
    """Sanity gate for the fixture itself (not the wiring under test).

    ``available_fields`` must list AF/NS/DP (Task 2.1/2.2 discovery), and a
    query spanning chr1 must show BOTH a non-empty var_key window (chr1:3,
    carried by 1/6 haplotypes) AND a non-empty dense window (chr1:10, carried
    by 6/6 haplotypes) -- else half the provenance logic (Task 2.3/2.4) would
    be silently untested by the tests below.
    """
    import genoray

    sv = genoray.SparseVar2(str(svar2_fields_store))
    assert set(sv.available_fields) == {"AF", "NS", "DP"}, sv.available_fields

    d = sv._find_ranges("chr1", [0], [40], samples=None)
    vk_snp_range = np.asarray(d["vk_snp_range"], np.int64)  # (R*S*P, 2)
    dense_snp_range = np.asarray(d["dense_snp_range"], np.int64)  # (R, 2)

    vk_width = int((vk_snp_range[:, 1] - vk_snp_range[:, 0]).sum())
    dense_width = int(dense_snp_range[0, 1] - dense_snp_range[0, 0])
    assert vk_width >= 1, (
        f"expected chr1:3 (1/6 haplotypes) to route to var_key, but vk_snp_range "
        f"is empty ({vk_snp_range.tolist()})"
    )
    assert dense_width >= 1, (
        f"expected chr1:10 (6/6 haplotypes) to route to dense, but "
        f"dense_snp_range is empty ({dense_snp_range.tolist()})"
    )


# --- shared oracle-comparison helper (diploid RaggedVariants) ---------------


def _assert_diploid_fields(
    rv,
    region_contigs: list[str],
    samples: list[str],
    oracle: dict[tuple[str, int], dict],
    sv,
) -> None:
    """Compare a diploid (ploidy-2) ``RaggedVariants`` against the oracle.

    Checks, per decoded (region, sample, ploid, variant): AF/NS by position,
    DP by (position, sample) -- AND dtype (no widening) -- AND completeness
    (every oracle carrier for a queried contig was actually decoded, so a
    silently-dropped call is caught, not just a wrong value).
    """
    af_dtype = sv.available_fields["AF"].dtype
    ns_dtype = sv.available_fields["NS"].dtype
    dp_dtype = sv.available_fields["DP"].dtype
    assert np.asarray(rv["AF"].data).dtype == af_dtype
    assert np.asarray(rv["NS"].data).dtype == ns_dtype
    assert np.asarray(rv["DP"].data).dtype == dp_dtype

    start_ak = rv.start.to_ak().to_list()
    af_ak = rv["AF"].to_ak().to_list()
    ns_ak = rv["NS"].to_ak().to_list()
    dp_ak = rv["DP"].to_ak().to_list()

    seen: dict[tuple[str, int], set[tuple[int, int]]] = {}
    for r, contig in enumerate(region_contigs):
        for s_i in range(len(start_ak[r])):
            for p_i in range(len(start_ak[r][s_i])):
                for v_i, pos0 in enumerate(start_ak[r][s_i][p_i]):
                    key = (contig, int(pos0))
                    assert key in oracle, f"decoded variant not in oracle: {key}"
                    exp = oracle[key]

                    got_af = af_ak[r][s_i][p_i][v_i]
                    if exp["AF"] is None:
                        assert got_af != got_af, (
                            f"expected NaN AF (missing in VCF) at {key}, got {got_af}"
                        )
                    else:
                        expected_af = float(np.asarray(exp["AF"], dtype=af_dtype))
                        assert got_af == expected_af, (
                            f"AF mismatch at {key}: {got_af} != {expected_af}"
                        )

                    expected_ns = int(np.asarray(exp["NS"], dtype=ns_dtype))
                    assert ns_ak[r][s_i][p_i][v_i] == expected_ns, (
                        f"NS mismatch at {key}: {ns_ak[r][s_i][p_i][v_i]} != {expected_ns}"
                    )

                    assert (s_i, p_i) in exp["carriers"], (
                        f"decoded a call not marked as a carrier in the oracle: "
                        f"{key} sample={samples[s_i]} hap={p_i}"
                    )
                    expected_dp = int(
                        np.asarray(exp["carriers"][(s_i, p_i)], dtype=dp_dtype)
                    )
                    got_dp = dp_ak[r][s_i][p_i][v_i]
                    assert got_dp == expected_dp, (
                        f"DP mismatch at {key} sample={samples[s_i]}: "
                        f"{got_dp} != {expected_dp}"
                    )

                    seen.setdefault(key, set()).add((s_i, p_i))

    for key, exp in oracle.items():
        contig, _pos0 = key
        if contig not in region_contigs or not exp["carriers"]:
            continue
        assert seen.get(key) == set(exp["carriers"]), (
            f"missing/extra decoded carriers at {key}: "
            f"expected {set(exp['carriers'])}, got {seen.get(key)}"
        )


# --- Test 1: single-contig ---------------------------------------------------


def test_svar2_ragged_variants_fields(
    tmp_path, svar2_fields_store, oracle_and_samples, _src
):
    import genoray

    oracle, samples = oracle_and_samples
    _bcf, ref = _src
    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [40]})
    ds = _build_dataset(tmp_path, "d1.gvl", bed, svar2_fields_store, ref)
    ds = ds.with_seqs("variants").with_settings(var_fields=_VAR_FIELDS)
    assert ds.samples == samples

    rv = ds[:, :]
    sv = genoray.SparseVar2(str(svar2_fields_store))
    _assert_diploid_fields(rv, ["chr1"], samples, oracle, sv)


# --- Test 2: multi-contig, interleaved (exercises the row-reorder path) ----


def test_svar2_ragged_variants_fields_multicontig(
    tmp_path, svar2_fields_store, oracle_and_samples, _src
):
    import genoray

    oracle, samples = oracle_and_samples
    _bcf, ref = _src
    # Interleaved chr2/chr1/chr2/chr1, out of natural order -> >1 contig group.
    bed = pl.DataFrame(
        {
            "chrom": ["chr2", "chr1", "chr2", "chr1"],
            "chromStart": [0, 0, 15, 20],
            "chromEnd": [15, 20, 30, 40],
        }
    )
    ds = _build_dataset(tmp_path, "d2.gvl", bed, svar2_fields_store, ref)
    ds = ds.with_seqs("variants").with_settings(var_fields=_VAR_FIELDS)

    rv = ds[:, :]
    sv = genoray.SparseVar2(str(svar2_fields_store))
    _assert_diploid_fields(rv, ["chr2", "chr1", "chr2", "chr1"], samples, oracle, sv)


# --- Test 3: FORMAT per-sample, explicitly NOT sample 0 ---------------------


def test_svar2_ragged_variants_format_not_sample0(
    tmp_path, svar2_fields_store, oracle_and_samples, _src
):
    oracle, samples = oracle_and_samples
    _bcf, ref = _src
    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [40]})
    ds = _build_dataset(tmp_path, "d3.gvl", bed, svar2_fields_store, ref)
    ds = ds.with_seqs("variants").with_settings(var_fields=_VAR_FIELDS)

    rv = ds[:, :]
    start_ak = rv.start.to_ak().to_list()
    dp_ak = rv["DP"].to_ak().to_list()

    dense_pos0 = 9  # chr1:10 (1-based) -> 0-based 9; carried by ALL samples/haps.
    exp = oracle[("chr1", dense_pos0)]

    query_sample = "S1"
    s_i = samples.index(query_sample)
    assert s_i != 0, "must query a sample other than sample 0 (index 0)"
    assert exp["carriers"][(s_i, 0)] != exp["carriers"][(0, 0)], (
        "fixture bug: the queried sample's DP must differ from sample 0's DP "
        "at this variant, else a broken FORMAT sample-stride would go undetected"
    )

    found = False
    for r in range(len(start_ak)):
        for p_i in range(len(start_ak[r][s_i])):
            for v_i, pos0 in enumerate(start_ak[r][s_i][p_i]):
                if int(pos0) != dense_pos0:
                    continue
                found = True
                got_dp = dp_ak[r][s_i][p_i][v_i]
                assert got_dp == exp["carriers"][(s_i, p_i)], (
                    f"DP mismatch for sample {query_sample} at chr1:{dense_pos0}: "
                    f"{got_dp} != {exp['carriers'][(s_i, p_i)]}"
                )
    assert found, (
        f"expected sample {query_sample} to carry the dense variant at "
        f"chr1:{dense_pos0}"
    )


# --- Test 4: unphased_union ---------------------------------------------------


def test_svar2_ragged_variants_fields_unphased_union(
    tmp_path, svar2_fields_store, oracle_and_samples, _src
):
    """``with_settings(unphased_union=True)`` folds ploidy 2->1: ALT occurrences
    from both haplotypes are concatenated per sample (no dedup, no sort). AF/NS
    are per-variant (hap-independent) and DP is per-sample (identical for every
    haplotype of that sample in this fixture), so each decoded occurrence must
    still match the oracle by (contig, start) / (contig, start, sample), and the
    per-sample occurrence COUNT must equal the number of carrying haplotypes.
    """
    oracle, samples = oracle_and_samples
    _bcf, ref = _src
    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [40]})
    ds = _build_dataset(tmp_path, "d4.gvl", bed, svar2_fields_store, ref)
    ds = ds.with_seqs("variants").with_settings(
        var_fields=_VAR_FIELDS, unphased_union=True
    )

    rv = ds[:, :]
    assert rv.start.shape[-2] == 1, "unphased_union must fold ploidy to 1"

    start_ak = rv.start.to_ak().to_list()
    af_ak = rv["AF"].to_ak().to_list()
    ns_ak = rv["NS"].to_ak().to_list()
    dp_ak = rv["DP"].to_ak().to_list()

    counts: dict[tuple[str, int, int], int] = {}
    for s_i in range(len(start_ak[0])):
        for pos0, af, ns, dp in zip(
            start_ak[0][s_i][0], af_ak[0][s_i][0], ns_ak[0][s_i][0], dp_ak[0][s_i][0]
        ):
            key = ("chr1", int(pos0))
            assert key in oracle, f"decoded variant not in oracle: {key}"
            exp = oracle[key]
            if exp["AF"] is None:
                assert af != af, f"expected NaN AF at {key}, got {af}"
            else:
                assert af == pytest.approx(exp["AF"], rel=0, abs=1e-6), (
                    f"AF mismatch at {key}: {af} != {exp['AF']}"
                )
            assert ns == exp["NS"], f"NS mismatch at {key}: {ns} != {exp['NS']}"
            sample_dp = {v for (si, _p), v in exp["carriers"].items() if si == s_i}
            assert sample_dp, f"sample {samples[s_i]} unexpectedly carries {key}"
            assert len(sample_dp) == 1, "fixture bug: DP must be uniform per sample"
            assert dp == next(iter(sample_dp)), (
                f"DP mismatch at {key} sample={samples[s_i]}: {dp} != {sample_dp}"
            )
            counts[(*key, s_i)] = counts.get((*key, s_i), 0) + 1

    for key, exp in oracle.items():
        contig, _pos0 = key
        if contig != "chr1":
            continue
        per_sample_hap_count: dict[int, int] = {}
        for si, _p in exp["carriers"]:
            per_sample_hap_count[si] = per_sample_hap_count.get(si, 0) + 1
        for s_i, expected_count in per_sample_hap_count.items():
            got_count = counts.get((*key, s_i), 0)
            assert got_count == expected_count, (
                f"union occurrence count mismatch at {key} sample={samples[s_i]}: "
                f"{got_count} != {expected_count}"
            )


# --- Test 5: variant-windows, including an EMPTY-group fill-value case -----


def test_svar2_variant_windows_fields(
    tmp_path, svar2_fields_store, oracle_and_samples, _src
):
    import genvarloader as gvl

    oracle, samples = oracle_and_samples
    _bcf, ref = _src
    # Region 0 covers both chr1 variants; region 1 (chr1:20-40) has NONE.
    bed = pl.DataFrame(
        {"chrom": ["chr1", "chr1"], "chromStart": [0, 20], "chromEnd": [20, 40]}
    )
    ds = _build_dataset(tmp_path, "d5.gvl", bed, svar2_fields_store, ref)
    opt = gvl.VarWindowOpt(flank_length=3, token_alphabet=b"ACGT", unknown_token=4)
    ds = (
        ds.with_output_format("flat")
        .with_seqs("variant-windows", opt)
        .with_settings(
            var_fields=_VAR_FIELDS,
            dummy_variant=gvl.DummyVariant(alt=b"N", ref=b"N"),
        )
    )
    win = ds[:, :]
    assert "AF" in win.fields and "NS" in win.fields and "DP" in win.fields

    P = 2
    S = len(samples)
    start_off = np.asarray(win.fields["start"].offsets)
    start_data = np.asarray(win.fields["start"].data)
    af_data = np.asarray(win.fields["AF"].data)
    ns_data = np.asarray(win.fields["NS"].data)
    dp_data = np.asarray(win.fields["DP"].data)

    def _group(r: int, s_i: int, p_i: int) -> tuple[int, int]:
        g = (r * S + s_i) * P + p_i
        return int(start_off[g]), int(start_off[g + 1])

    # Region 0: real variants, checked against the oracle exactly as in the
    # diploid RaggedVariants test.
    for s_i in range(S):
        for p_i in range(P):
            lo, hi = _group(0, s_i, p_i)
            for i in range(lo, hi):
                pos0 = int(start_data[i])
                key = ("chr1", pos0)
                assert key in oracle, f"decoded variant not in oracle: {key}"
                exp = oracle[key]
                if exp["AF"] is None:
                    assert af_data[i] != af_data[i], (
                        f"expected NaN AF at {key}, got {af_data[i]}"
                    )
                else:
                    assert af_data[i] == pytest.approx(exp["AF"], abs=1e-6)
                assert ns_data[i] == exp["NS"]
                assert (s_i, p_i) in exp["carriers"], (
                    f"decoded a call not marked as carrier: {key} sample={samples[s_i]}"
                )
                assert dp_data[i] == exp["carriers"][(s_i, p_i)]

    # Region 1: variant-free -> the dummy fill must appear (exactly 1 entry per
    # group), with the documented fill values: NaN for the float AF column, 0
    # for the integer NS/DP columns (DummyVariant.info was left empty).
    for s_i in range(S):
        for p_i in range(P):
            lo, hi = _group(1, s_i, p_i)
            assert hi - lo == 1, (
                f"expected exactly 1 dummy variant in the empty group "
                f"(region 1, sample {samples[s_i]}, hap {p_i}), got {hi - lo}"
            )
            assert af_data[lo] != af_data[lo], (
                f"expected NaN AF fill for the empty group, got {af_data[lo]}"
            )
            assert ns_data[lo] == 0, f"expected 0 NS fill, got {ns_data[lo]}"
            assert dp_data[lo] == 0, f"expected 0 DP fill, got {dp_data[lo]}"
