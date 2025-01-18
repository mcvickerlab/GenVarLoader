from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cyvcf2
import numpy as np
import polars as pl
from attrs import define, field
from numpy.typing import ArrayLike, NDArray

from .._utils import _normalize_contig_name
from ._genotypes import (
    DosageFieldError,
    Genotypes,
    PgenGenos,
    VCFGenos,
)
from ._records import Records, VLenAlleles
from ._utils import path_is_pgen, path_is_vcf

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
    offsets: NDArray[np.int64]
    """Shape: (regions + 1). Offsets for the index boundaries of each region such
        that variants for region `i` are `positions[offsets[i] : offsets[i+1]]`,
        `size_diffs[offsets[i] : offsets[i+1]]`, ..., etc."""


@define
class Variants:
    """Variant records and genotypes. Should not be directly instantiated, use
    :meth:`from_file` instead."""

    records: Records
    genotypes: Genotypes
    phased: bool
    dosage_field: Optional[str] = None
    _sample_idxs: Optional[NDArray[np.intp]] = field(default=None, alias="_sample_idxs")

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path, Dict[str, Path]],
        phased=True,
        dosage_field: Optional[str] = None,
    ) -> "Variants":
        """Create a Variants instances from a VCF or PGEN file(s). If a dictionary is provided, the keys should be
        contig names and the values should be paths to the corresponding VCF or PGEN.

        Parameters
        ----------
        path
            Path to a VCF or PGEN file or a mapping from contig names to paths.
        phased
            Whether the genotypes should be treated as phased.
        dosage_field
            The name of the dosage field in the VCF file. This is currently only applicable and required for
            unphased genotypes.
        """
        if isinstance(path, (str, Path)):
            path = Path(path)
            first_path = path
        elif isinstance(path, dict):
            first_path = next(iter(path.values()))

        if path_is_vcf(first_path):
            return cls._from_vcf(path, phased, dosage_field)
        elif path_is_pgen(first_path):
            return cls._from_pgen(path, phased)
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
        if self.phased:
            return self.genotypes.ploidy
        else:
            return 1

    @classmethod
    def _from_vcf(
        cls,
        vcf: Union[str, Path, Dict[str, Path]],
        phased: bool,
        dosage_field: Optional[str],
    ) -> "Variants":
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
        if not phased and dosage_field is None:
            raise ValueError("Dosage field is required for unphased genotypes.")

        elif not phased and dosage_field is not None:
            _vcf = cyvcf2.VCF(vcf)
            try:
                dosage_field_info = _vcf.get_header_type(dosage_field)
            except KeyError:
                raise DosageFieldError(
                    f"Dosage field '{dosage_field}' not found in VCF header."
                )

            if dosage_field_info["Number"] not in {"1", "A"}:
                raise DosageFieldError(
                    f"Dosage field '{dosage_field}' must have Number equal to '1' or 'A'."
                )

        records = Records.from_vcf(vcf)

        genotypes = VCFGenos(vcf, records.contig_offsets)
        return cls(records, genotypes, phased, dosage_field)

    @classmethod
    def _from_pgen(
        cls, pgen: Union[str, Path, Dict[str, Path]], phased: bool
    ) -> "Variants":
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
                schema_overrides={
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
        return cls(records, genotypes, phased)

    def subset_samples(self, samples: ArrayLike):
        """Subset the samples to be read from the genotypes.

        Parameters
        ----------
        samples
            List of sample names."""
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
    ) -> Optional[DenseGenotypes]:
        """Read genotypes for a given contig and range of positions.

        Parameters
        ----------
        contig
            Name of the contig/chromosome.
        starts
            Start coordinates, 0-based.
        ends
            End coordinates, 0-based exclusive.
        sample
            List of sample names.
        ploid
            List of ploid numbers.
        """
        contig = _normalize_contig_name(contig, self.records.contigs)  # pyright: ignore[reportAssignmentType]
        if contig is None:
            return None

        starts = np.atleast_1d(np.asarray(starts, dtype=int))
        ends = np.atleast_1d(np.asarray(ends, dtype=int))

        recs = self.records.vars_in_range(contig, starts, ends)
        if recs is None:
            return

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
        """Read genotypes for a given contig and range of positions. This method is intended for haplotype construction
        and returns the maximum end position for each region.

        Parameters
        ----------
        contig
            Name of the contig/chromosome.
        starts
            Start coordinates, 0-based.
        ends
            End coordinates, 0-based exclusive.
        sample
            List of sample names.
        ploid
            List of ploid numbers.
        """
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
