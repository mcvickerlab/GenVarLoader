"""Regenerate the committed toy *inputs* (reference, normalized VCF, PGEN,
SVAR, BED) under tests/data/ via the shared `build_case` pipeline.

Phase 2: the consensus FASTAs and phased_dataset.*.gvl/ datasets are NO LONGER
committed — they are built per-session in conftest (and per-example in the
property tests) from the same `session_document`. This task only persists the
inputs that the self-contained write/track/edge tests and the FASTA-only unit
tests consume.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

WDIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WDIR.parent / "_builders"))


def main() -> None:
    import polars as pl
    from case import SEQ_LEN, build_case, session_document, session_reference

    spec = session_reference()
    doc = session_document(spec)

    # Manual coverage regions: the chr1:1010696 spanning deletion (coupled to
    # test_write_edge_cases) and a no-variant region on chr1.
    extra = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [1010696, 500_000],
            "end": [1010696 + SEQ_LEN, 500_000 + SEQ_LEN],
        }
    )

    with tempfile.TemporaryDirectory() as tmp:
        case = build_case(spec, doc, Path(tmp), extra_regions=extra)

        fasta_dir = WDIR / "fasta"
        fasta_dir.mkdir(parents=True, exist_ok=True)
        for suffix in ("", ".fai", ".gzi"):
            src = case.ref_path.parent / (case.ref_path.name + suffix)
            if src.exists():
                shutil.copy(src, fasta_dir / ("synthetic.fa.bgz" + suffix))

        vcf_dir = WDIR / "vcf"
        if vcf_dir.exists():
            shutil.rmtree(vcf_dir)
        vcf_dir.mkdir(parents=True)
        for suffix in ("", ".csi", ".tbi"):
            src = case.vcf_path.parent / (case.vcf_path.name + suffix)
            if src.exists():
                shutil.copy(src, vcf_dir / ("filtered_source.vcf.gz" + suffix))

        pgen_dir = WDIR / "pgen"
        if pgen_dir.exists():
            shutil.rmtree(pgen_dir)
        pgen_dir.mkdir(parents=True)
        for ext in (".pgen", ".pvar", ".psam"):
            src = case.pgen_path.with_suffix(ext)
            if src.exists():
                shutil.copy(src, pgen_dir / ("filtered_source" + ext))

        svar_dst = WDIR / "filtered.svar"
        if svar_dst.exists():
            shutil.rmtree(svar_dst)
        shutil.copytree(case.svar_path, svar_dst)

        shutil.copy(case.bed_path, WDIR / "source.bed")


if __name__ == "__main__":
    main()
