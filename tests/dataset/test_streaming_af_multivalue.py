"""Issue #324: multi-value ``INFO/AF`` must be declined SYMMETRICALLY.

``gvl.write``'s ``_attach_af_column`` (``_write.py``) inspects the **data**, not
the header: a record carrying more than one AF value has an ambiguous ALT->AF
mapping (e.g. a ``Number=.`` field left un-subset after a ``bcftools norm -m``
split, so a bi-allelic ``G>A`` still lists ``AF=0.333,0.667``), so it warns and
declines to cache AF. ``Dataset.with_settings(min_af=...)`` then raises the usual
AF-missing guard.

Streaming's ``_VcfBackend.has_cached_af`` was **header-only**
(``_declared_info_fields(("AF",))``), so it reported AF available and read it
live, with genoray's ``resolve_scalar`` silently taking the **first** value --
streaming filtered where written raised, breaking the streaming<->written
byte-identity contract.

Header ``Number=`` is deliberately NOT the criterion here. The written path
trusts the data, so streaming must too; keying off ``Number=.`` would diverge
all over again for a ``Number=A``-declared-but-multi-valued file, and would
wrongly decline the single-valued ``Number=.`` case that written accepts.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import polars as pl
import pytest

import genvarloader as gvl

_REF = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"  # chr1, 40bp
assert len(_REF) == 40

# Bi-allelic records whose `Number=.` INFO/AF still carries the full un-subset
# multiallelic list (POS 3) -- the ALT->AF mapping is ambiguous.
_VCF_MULTIVALUE_AF = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=40>
##INFO=<ID=AF,Number=.,Type=Float,Description="Allele frequency">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1
chr1\t3\t.\tA\tG\t.\t.\tAF=0.333,0.667\tGT\t1|0\t0|1
chr1\t16\t.\tT\tC\t.\t.\tAF=0.5\tGT\t1|1\t0|1
"""

# Same `Number=.` declaration, but every record carries exactly ONE value, so
# the mapping is unambiguous and AF filtering must still WORK. This is the
# anti-over-correction control: a header-only fix would wrongly decline it.
_VCF_SINGLE_VALUE_DOT_AF = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=40>
##INFO=<ID=AF,Number=.,Type=Float,Description="Allele frequency">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1
chr1\t3\t.\tA\tG\t.\t.\tAF=0.333\tGT\t1|0\t0|1
chr1\t16\t.\tT\tC\t.\t.\tAF=0.5\tGT\t1|1\t0|1
"""


def _af_vcf_case(vcf_text: str, d: Path) -> tuple[pl.DataFrame, Path, Path]:
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    vcf = d / "in.vcf"
    vcf.write_text(vcf_text)
    vcf_gz = d / "in.vcf.gz"
    subprocess.run(["bcftools", "view", "-Oz", "-o", str(vcf_gz), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", "-t", str(vcf_gz)], check=True)

    regions = pl.DataFrame(
        {"chrom": ["chr1"], "chromStart": [0], "chromEnd": [len(_REF)]}
    )
    return regions, ref, vcf_gz


def _n_variants(sds) -> int:
    """Total variants across the stream, via each batch's ragged `alt` offsets."""
    return sum(int(b[0].alt.offsets[-1]) for b in sds.to_iter(batch_size=2))


def test_af_missing_guard_raises_vcf_multivalue_af(tmp_path):
    """Streaming twin of ``test_write_multivalue_af_writes_without_af_column``.

    The written path declines to cache AF here, so its ``min_af`` raises.
    Streaming must raise the SAME guard rather than silently filtering on
    genoray's first value.
    """
    regions, ref, vcf_gz = _af_vcf_case(_VCF_MULTIVALUE_AF, tmp_path)
    sds = (
        gvl.StreamingDataset(regions, reference=str(ref), variants=str(vcf_gz))
        .with_seqs("variants")
        .with_settings(min_af=0.1)
    )
    with pytest.raises(RuntimeError, match="AFs cached"):
        next(iter(sds.to_iter(batch_size=2)))


def test_multivalue_af_does_not_disturb_unfiltered_streaming(tmp_path):
    """Declining AF must not break a stream that never asked to filter by it.

    The guard is reached only when ``min_af``/``max_af`` is set, so an ordinary
    variants stream over the same ambiguous-AF VCF keeps working.
    """
    regions, ref, vcf_gz = _af_vcf_case(_VCF_MULTIVALUE_AF, tmp_path)
    sds = gvl.StreamingDataset(
        regions, reference=str(ref), variants=str(vcf_gz)
    ).with_seqs("variants")
    assert _n_variants(sds) > 0


def test_single_value_number_dot_af_still_filters(tmp_path):
    """Anti-over-correction: ``Number=.`` alone must NOT disable AF filtering.

    Only an actual multi-value record is ambiguous. Keying the decline off the
    header would wrongly decline here, making streaming *stricter* than written.
    """
    regions, ref, vcf_gz = _af_vcf_case(_VCF_SINGLE_VALUE_DOT_AF, tmp_path)
    filtered = (
        gvl.StreamingDataset(regions, reference=str(ref), variants=str(vcf_gz))
        .with_seqs("variants")
        .with_settings(min_af=0.4)
    )
    unfiltered = gvl.StreamingDataset(
        regions, reference=str(ref), variants=str(vcf_gz)
    ).with_seqs("variants")

    # Must not raise, and must actually filter: AF=0.333 (POS 3) drops, AF=0.5
    # (POS 16) survives -- strictly fewer variants, but not zero.
    n_filtered = _n_variants(filtered)
    assert 0 < n_filtered < _n_variants(unfiltered)
