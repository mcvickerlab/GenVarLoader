"""Shared builder: turn a (ReferenceSpec, VcfDocument) into on-disk gvl inputs
plus a `bcftools consensus` haplotype oracle.

Used by BOTH the property-test module (random reference-consistent draws) and
the session conftest fixture / `gen` task (one fixed standardized document), so
the generation logic lives in exactly one place.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from vcfixture import Number, ReferenceBuilder, Type, VcfBuilder

SEQ_LEN = 20


def _run(cmd: list[str], input: bytes | None = None) -> bytes:
    try:
        prc = subprocess.run(cmd, check=True, capture_output=True, input=input)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"command failed: {' '.join(e.cmd)}\n"
            f"stdout: {e.stdout.decode(errors='replace')}\n"
            f"stderr: {e.stderr.decode(errors='replace')}"
        ) from e
    return prc.stdout


@dataclass
class Case:
    """On-disk artifacts + oracle for one built case."""

    ref_path: Path
    gvl_path: dict[str, Path]
    consensus_dir: Path
    bed_path: Path
    vcf_path: Path  # normalized, bgzipped+indexed .vcf.gz
    pgen_path: Path | None
    svar_path: Path | None
    truth: object  # vcfixture.GroundTruth
    samples: list[str]
    regions: pl.DataFrame  # columns: index, chrom, start, end, strand


def _normalize(raw_vcf: bytes, ref: Path) -> bytes:
    """Left-align, then atomize + split multiallelics (canonicalize)."""
    vcf = _run(["bcftools", "norm", "-f", str(ref)], input=raw_vcf)
    vcf = _run(
        ["bcftools", "norm", "-a", "--atom-overlaps", ".", "-f", str(ref), "-m", "-"],
        input=vcf,
    )
    return vcf


def _bgzip_index(vcf_text: bytes, out_gz: Path) -> Path:
    out_vcf = out_gz.with_suffix("")  # strip .gz -> .vcf
    out_vcf.write_bytes(vcf_text)
    _run(["bcftools", "view", "-O", "z", "-o", str(out_gz), str(out_vcf)])
    _run(["bcftools", "index", str(out_gz)])
    return out_gz


def _derive_bed(
    vcf_gz: Path, extra_regions: pl.DataFrame | None
) -> pl.DataFrame:
    """Group variant positions into SEQ_LEN-wide regions (Phase-1 logic, keyed
    off the *normalized* VCF positions). Optionally append manual regions."""
    df = pl.read_csv(
        vcf_gz,
        separator="\t",
        comment_prefix="#",
        has_header=False,
        truncate_ragged_lines=True,
    ).select(
        pl.nth(0).cast(pl.Utf8).alias("chrom"),
        pl.nth(1).cast(pl.Int64).alias("pos"),
    )
    bed = (
        df.group_by("chrom", maintain_order=True)
        .agg(
            "pos",
            (pl.col("pos").diff().fill_null(0) > SEQ_LEN).cum_sum().alias("group"),
        )
        .explode("pos", "group")
        .group_by("chrom", "group", maintain_order=True)
        .agg(
            start=pl.col("pos").min() - SEQ_LEN // 2,
            end=pl.col("pos").min() + SEQ_LEN // 2,
        )
        .drop("group")
    )
    if extra_regions is not None:
        bed = bed.vstack(extra_regions.select("chrom", "start", "end"))
    rng = np.random.default_rng(0)
    bed = bed.with_row_index()
    strand = pl.Series("strand", rng.choice(["+", "-"], size=bed.height))
    return bed.hstack([strand])


def _write_consensus(
    ref: Path, vcf_gz: Path, bed: pl.DataFrame, samples: list[str], out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for row_nr, chrom, start, end in bed.select(
        "index", "chrom", "start", "end"
    ).iter_rows():
        seq = _run(["samtools", "faidx", str(ref), f"{chrom}:{start + 1}-{end}"])
        for sample in samples:
            for hap in range(2):
                out_fa = out_dir / f"source_{sample}_nr{row_nr}_h{hap}.fa"
                _run(
                    [
                        "bcftools", "consensus", "-H", str(hap + 1),
                        "-s", sample, "-o", str(out_fa), str(vcf_gz),
                    ],
                    input=seq,
                )
                _run(["samtools", "faidx", str(out_fa)])


def build_case(
    spec,
    doc,
    workdir,
    *,
    sources: tuple[str, ...] = ("vcf", "pgen", "svar"),
    normalize: bool = True,
    extra_regions: pl.DataFrame | None = None,
) -> Case:
    """Build a complete case from a (ReferenceSpec, VcfDocument).

    Steps: write reference -> render+(optionally normalize) VCF -> derive BED ->
    bcftools consensus oracle -> PGEN -> SVAR -> gvl.write per requested source.
    """
    import genvarloader as gvl
    from genoray import PGEN, VCF, SparseVar

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # 1. Reference (bgzip + faidx).
    ref_path = spec.write(workdir / "ref.fa.bgz")

    # 2. Render + canonicalize VCF.
    raw = doc.render().encode()
    vcf_text = _normalize(raw, ref_path) if normalize else raw
    vcf_gz = _bgzip_index(vcf_text, workdir / "filtered.vcf.gz")

    samples = list(doc.samples)

    # 3. BED from normalized positions (+ optional manual regions).
    bed = _derive_bed(vcf_gz, extra_regions)
    bed_path = workdir / "source.bed"
    bed.select(
        "chrom",
        "start",
        "end",
        pl.lit(".").alias("name"),
        pl.lit(".").alias("score"),
        "strand",
    ).write_csv(bed_path, include_header=False, separator="\t")

    # 4. Haplotype oracle.
    consensus_dir = workdir / "consensus"
    _write_consensus(ref_path, vcf_gz, bed, samples, consensus_dir)

    # 5. PGEN.
    pgen_path: Path | None = None
    if "pgen" in sources:
        pgen_path = workdir / "filtered.pgen"
        _run(
            [
                "plink2", "--vcf", str(vcf_gz), "--make-pgen",
                "--vcf-half-call", "r", "--out", str(pgen_path.with_suffix("")),
            ]
        )

    # 6. SVAR (with cached AFs for Track 1b).
    svar_path: Path | None = None
    if "svar" in sources:
        svar_path = workdir / "filtered.svar"
        SparseVar.from_vcf(svar_path, VCF(vcf_gz), "50mb")
        SparseVar(svar_path).cache_afs()

    # 7. gvl.write per source.
    gvl_path: dict[str, Path] = {}
    if "vcf" in sources:
        reader = VCF(vcf_gz)
        if not reader._valid_index():
            reader._write_gvi_index()
        reader._load_index()
        out = workdir / "ds.vcf.gvl"
        gvl.write(path=out, bed=bed_path, variants=reader, max_jitter=2)
        gvl_path["vcf"] = out
    if "pgen" in sources:
        out = workdir / "ds.pgen.gvl"
        gvl.write(path=out, bed=bed_path, variants=PGEN(pgen_path), max_jitter=2)
        gvl_path["pgen"] = out
    if "svar" in sources:
        out = workdir / "ds.svar.gvl"
        gvl.write(path=out, bed=bed_path, variants=SparseVar(svar_path), max_jitter=2)
        gvl_path["svar"] = out

    return Case(
        ref_path=ref_path,
        gvl_path=gvl_path,
        consensus_dir=consensus_dir,
        bed_path=bed_path,
        vcf_path=vcf_gz,
        pgen_path=pgen_path,
        svar_path=svar_path,
        truth=doc.truth(),
        samples=samples,
        regions=bed,
    )


# (contig, 1-based pos, REF) — engineered into the reference so bcftools norm
# leaves these records in place. Relabeled from the Phase-1 chr19/chr20 set.
_REF_OVERWRITES: list[tuple[str, int, str]] = [
    ("chr1", 111, "N"),
    ("chr1", 1010696, "GAGACGGGGCC"),
    ("chr1", 1110696, "A"),
    ("chr1", 1210696, "C"),
    ("chr1", 1210697, "T"),
    ("chr2", 14370, "N"),
    ("chr2", 17330, "N"),
    ("chr2", 1110696, "G"),
    ("chr2", 1234567, "A"),
]

# (contig, 1-based anchor pos, guard base) — break leftward repeats 5' of indels.
_FLANK_GUARDS: list[tuple[str, int, str]] = [
    ("chr1", 1010696, "T"),
    ("chr1", 1110696, "G"),
    ("chr2", 1234567, "T"),
]

_SESSION_CONTIGS: list[tuple[str, int]] = [
    ("chr1", 1_300_000),
    ("chr2", 1_300_000),
]


def session_reference():
    """Fixed standardized reference (chr1/chr2) for the session fixture + `gen`.

    Replaces Phase-1 ``_synthetic.write_synthetic_reference``: engineers REF
    bases at every variant locus, plants 5' flank guards before indel anchors,
    and N-masks the chr1 telomere (chr1[0:150]) for parity with hg38 (consumed
    by tests/unit/dataset/test_ref_ds.py). The rest of each contig is filled by
    ReferenceBuilder's default base.
    """
    b = ReferenceBuilder()
    for contig, length in _SESSION_CONTIGS:
        b.add_contig(contig, length)
    for contig, pos, ref in _REF_OVERWRITES:
        b.set_seq(contig, pos - 1, ref)  # 0-based
    for contig, pos, guard in _FLANK_GUARDS:
        b.set_base(contig, pos - 2, guard)  # base immediately 5' of anchor
    b.set_seq("chr1", 0, "N" * 150)  # telomere mask LAST so it wins over 111
    return b.build()


def session_document(spec):
    """Fixed standardized source document (relabel of the Phase-1 source VCF).

    Replaces Phase-1 ``_synthetic.build_source_vcf``. VCFv4.0 is used so the
    Number=. INFO fields parse under the noodles VCF reader SparseVar relies on.
    """
    contigs = [(c, l) for c, l in _SESSION_CONTIGS]
    b = VcfBuilder(samples=["s0", "s1", "s2"], contigs=contigs, fileformat="VCFv4.0")
    b.info("NS", Number.ONE, Type.INTEGER)
    b.info("AN", Number.ONE, Type.INTEGER)
    b.info("AC", Number.DOT, Type.INTEGER)
    b.info("DP", Number.ONE, Type.INTEGER)
    b.info("AF", Number.DOT, Type.FLOAT)
    b.info("AA", Number.ONE, Type.STRING)
    b.info("DB", Number.FLAG, Type.FLAG)
    b.info("H2", Number.FLAG, Type.FLAG)
    b.fmt("GT")
    b.fmt("VAF", Number.A, Type.FLOAT)
    b.fmt("GQ", Number.ONE, Type.INTEGER)
    b.fmt("DP", Number.ONE, Type.INTEGER)
    b.fmt("HQ", Number.fixed(2), Type.INTEGER)
    b.filter("q10", "Quality below 10")
    b.filter("s50", "Less than 50% of samples have data")

    # chr1 block (relabeled from chr19).
    b.record("chr1", 111, ref="N", alt=["C"], gt=["0|0", "0|0", "0/1"])
    b.record("chr1", 1010696, ref="GAGA", alt=["G"], gt=["1|0", "0|0", "0/0"])
    b.record("chr1", 1010696, ref="GAGACGG", alt=["G"], gt=["0|0", "0|0", "0/1"])
    b.record("chr1", 1010696, ref="GAGACGGGGCC", alt=["G"], gt=["0|1", "1|1", "0/0"])
    b.record("chr1", 1110696, ref="A", alt=["TTT"], gt=["0|1", "1|1", "0/0"])
    b.record("chr1", 1110696, ref="A", alt=["G"], gt=["0|0", "0|0", "0/1"])
    b.record("chr1", 1210696, ref="C", alt=["G"], gt=["1|.", "0/1", "1|1"])
    b.record("chr1", 1210696, ref="C", alt=["G"], gt=[".|1", "0|0", "0/0"])
    b.record("chr1", 1210697, ref="T", alt=["G"], gt=["0/0", "1|0", "0/1"])
    b.record("chr1", 1210697, ref="T", alt=["A"], gt=["0/0", "1|0", "0/1"])

    # chr2 block (relabeled from chr20) — carries INFO/IDs/FILTERs.
    b.record(
        "chr2", 14370, ref="N", alt=["A"], ids=["rs6054257"], qual=29.0, filter=(),
        gt=["0|0", "1|0", "1/1"],
        info={"NS": 3, "DP": 14, "AF": [0.5], "DB": True, "H2": True},
    )
    b.record(
        "chr2", 17330, ref="N", alt=["A"], qual=3.0, filter=["q10"],
        gt=["0|0", "0|1", "0/0"], info={"NS": 3, "DP": 11, "AF": [0.017]},
    )
    b.record(
        "chr2", 1110696, ref="G", alt=["A", "T"], ids=["rs6040355"], qual=67.0,
        filter=(), gt=["1|2", "2|1", "2/2"],
        info={"NS": 2, "DP": 10, "AF": [0.333, 0.667], "AA": "T", "DB": True},
    )
    b.record(
        "chr2", 1234567, ref="A", alt=["GA", "AC"], ids=["microsat1"], qual=50.0,
        filter=(), gt=["0/1", "0/2", "./."],
        info={"NS": 3, "DP": 9, "AA": "G", "AN": 6, "AC": [3, 1]},
    )
    return b.build()
