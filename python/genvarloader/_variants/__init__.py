from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import polars as pl
from attrs import define
from numpy.typing import ArrayLike, NDArray

from .._utils import _normalize_contig_name
from ._genotypes import (
    Genotypes,
    PgenGenos,
    VCFGenos,
)
from ._records import Records, VLenAlleles

__all__ = []


@define
class DenseGenotypes:
    """Dense array(s) of genotypes."""

    positions: NDArray[np.int32]
    """Shape: (variants)"""
    size_diffs: NDArray[np.int32]
    """Shape : (variants). Difference in length between the REF and the ALT alleles."""
    ref: VLenAlleles
    """Shape: (variants). REF alleles."""
    alt: VLenAlleles
    """Shape: (variants). ALT alleles."""
    genotypes: NDArray[np.int8]
    """Shape: (samples, ploid, variants)"""
    offsets: NDArray[np.int32]
    """Shape: (regions + 1). Offsets for the index boundaries of each region such
        that variants for region `i` are `positions[offsets[i] : offsets[i+1]]`,
        `size_diffs[offsets[i] : offsets[i+1]]`, ..., etc."""


@define
class Variants:
    records: Records
    genotypes: Genotypes
    _sample_idxs: Optional[NDArray[np.intp]] = None

    @classmethod
    def from_file(cls, path: Union[str, Path, Dict[str, Path]]):
        if isinstance(path, (str, Path)):
            path = Path(path)
            first_path = path
        elif isinstance(path, dict):
            first_path = next(iter(path.values()))

        vcf_suffix = re.compile(r"\.[vb]cf(\.gz)?$")

        if vcf_suffix.search(first_path.suffix):
            return cls.from_vcf(path)
        elif first_path.suffix == ".pgen":
            return cls.from_pgen(path)
        else:
            raise ValueError("Unsupported file type.")

    @property
    def chunked(self) -> bool:
        return self.genotypes.chunked

    @property
    def samples(self) -> NDArray[np.str_]:
        if self._sample_idxs is None:
            return self.genotypes.samples
        return self.genotypes.samples[self._sample_idxs]

    @property
    def n_samples(self) -> int:
        if self._sample_idxs is None:
            return self.genotypes.n_samples
        return len(self._sample_idxs)

    @property
    def ploidy(self) -> int:
        return self.genotypes.ploidy

    @classmethod
    def from_vcf(cls, vcf: Union[str, Path, Dict[str, Path]]):
        """Currently does not support multi-allelic sites, but does support *split*
        multi-allelic sites. Note that SVs and "other" variants are also not supported.
        VCFs can be prepared by running:

        .. code-block:: bash

            bcftools view -i 'TYPE="snp" || TYPE="indel"' <file.bcf> \\
            | bcftools norm \\
            -a \\
            --atom-overlaps . \\
            -m - \\
            -f <ref.fa> \\
            -O b \\
            -o <norm.bcf>
        
        """
        records = Records.from_vcf(vcf)

        genotypes = VCFGenos(vcf, records.contig_offsets)
        return cls(records, genotypes)

    @classmethod
    def from_pgen(cls, pgen: Union[str, Path, Dict[str, Path]]):
        """Currently does not support multi-allelic sites, but does support *split*
        multi-allelic sites. Note that SVs and "other" variants are also not supported.
        A PGEN can be prepared from a VCF by running:

        .. code-block:: bash

            bcftools view -i 'TYPE="snp" || TYPE="indel"' <file.bcf> \\
            | bcftools norm \\
            -a \\
            --atom-overlaps . \\
            -m - \\
            -f <ref.fa> \\
            -O b \\
            -o <norm.bcf>

            plink2 --make-pgen \\
            --bcf <norm.bcf> \\
            --vcf-half-call r \\
            --out <prefix>
        
        """
        if isinstance(pgen, str):
            pgen = Path(pgen)

        if isinstance(pgen, Path):
            pgen = {"_all": pgen}

        psams: Dict[str, pl.DataFrame] = {}
        samples = None
        for contig, path in pgen.items():
            with open(path.with_suffix(".psam")) as f:
                cols = [c.strip("#") for c in f.readline().strip().split()]

            psam = pl.read_csv(
                path.with_suffix(".psam"),
                separator="\t",
                has_header=False,
                skip_rows=1,
                new_columns=cols,
                dtypes={
                    "FID": pl.Utf8,
                    "IID": pl.Utf8,
                    "SID": pl.Utf8,
                    "PAT": pl.Utf8,
                    "MAT": pl.Utf8,
                    "SEX": pl.Utf8,
                },
            )

            if samples is None:
                samples = psam["IID"].to_numpy()
            else:
                intersection = np.intersect1d(samples, psam["IID"].to_numpy())
                if np.any(samples != intersection):
                    raise ValueError("Sample names do not match across contigs.")

            psams[contig] = psam
        assert samples is not None

        records = Records.from_pvar(
            {c: p.with_suffix(".pvar") for c, p in pgen.items()}
        )

        genotypes = PgenGenos(pgen, samples, records.contigs)
        return cls(records, genotypes)

    def subset_samples(self, samples: ArrayLike):
        samples = np.atleast_1d(np.asarray(samples, dtype=str))
        geno_sample_idxs, sample_idxs = np.intersect1d(
            self.genotypes.samples, samples, return_indices=True
        )[1:]
        if len(sample_idxs) != len(samples):
            raise ValueError("Some samples were not found")
        self._sample_idxs = geno_sample_idxs[sample_idxs]

    def read(
        self,
        contig: str,
        starts: ArrayLike,
        ends: ArrayLike,
        sample: Optional[ArrayLike] = None,
        ploid: Optional[ArrayLike] = None,
    ):
        contig = _normalize_contig_name(contig, self.records.contigs)  # pyright: ignore[reportAssignmentType]
        if contig is None:
            return None

        starts = np.atleast_1d(np.asarray(starts, dtype=int))
        ends = np.atleast_1d(np.asarray(ends, dtype=int))

        recs = self.records.vars_in_range(contig, starts, ends)
        if recs.start_idxs.size == 0:
            return None

        if sample is not None:
            sample = np.atleast_1d(np.asarray(sample, dtype=str))
            geno_sample_idxs, sample_idxs = np.intersect1d(
                self.genotypes.samples, sample, return_indices=True
            )[1:]
            sample_idxs = geno_sample_idxs[sample_idxs]
            if len(sample_idxs) != len(sample):
                raise ValueError("Some samples were not found")
        else:
            sample_idxs = None

        if ploid is not None:
            ploid = np.atleast_1d(np.asarray(ploid, dtype=int))

        # (s p v)
        genos = self.genotypes.read(
            contig, recs.start_idxs, recs.end_idxs, sample_idxs, ploid
        )

        return DenseGenotypes(
            recs.positions,
            recs.size_diffs,
            recs.refs,
            recs.alts,
            genos,
            recs.offsets,
        )

    def read_for_haplotype_construction(
        self,
        contig: str,
        starts: ArrayLike,
        ends: ArrayLike,
        sample: Optional[ArrayLike] = None,
        ploid: Optional[ArrayLike] = None,
    ) -> Tuple[Optional[DenseGenotypes], NDArray[np.int32]]:
        starts = np.atleast_1d(np.asarray(starts, dtype=int))
        ends = np.atleast_1d(np.asarray(ends, dtype=int))

        _contig = _normalize_contig_name(contig, self.records.contigs)  # pyright: ignore[reportAssignmentType]
        if _contig is None:
            return None, ends.astype(np.int32)
        else:
            contig = _contig

        recs, max_ends = self.records.vars_in_range_for_haplotype_construction(
            contig, starts, ends
        )
        if recs is None:
            return None, ends

        if sample is not None:
            sample = np.atleast_1d(np.asarray(sample, dtype=str))
            geno_sample_idxs, sample_idxs = np.intersect1d(
                self.genotypes.samples, sample, return_indices=True
            )[1:]
            sample_idxs = geno_sample_idxs[sample_idxs]
            if len(sample_idxs) != len(sample):
                raise ValueError("Some samples were not found")
        else:
            sample_idxs = None

        if ploid is not None:
            ploid = np.atleast_1d(np.asarray(ploid, dtype=int))

        # (s p v)
        genos = self.genotypes.read(
            contig, recs.start_idxs, recs.end_idxs, sample_idxs, ploid
        )

        return (
            DenseGenotypes(
                recs.positions,
                recs.size_diffs,
                recs.refs,
                recs.alts,
                genos,
                recs.offsets,
            ),
            max_ends,
        )
