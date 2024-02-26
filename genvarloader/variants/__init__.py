from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
from attrs import define
from numpy.typing import ArrayLike

from ..types import DenseGenotypes
from .genotypes import Genotypes, MemmapGenos, PgenGenos, VCFGenos, ZarrGenos
from .records import Records

__all__ = [
    "PgenGenos",
    "ZarrGenos",
    "MemmapGenos",
    "VCFGenos",
    "Variants",
    "Records",
]


@define
class Variants:
    records: Records
    genotypes: Genotypes

    @property
    def chunked(self):
        return self.genotypes.chunked

    @property
    def samples(self):
        return self.genotypes.samples

    @property
    def n_samples(self):
        return self.genotypes.n_samples

    @property
    def ploidy(self):
        return self.genotypes.ploidy

    @classmethod
    def from_vcf(cls, vcf: Union[Path, Dict[str, Path]]):
        records = Records.from_vcf(vcf)

        try:
            genotypes = ZarrGenos(vcf)
        except FileNotFoundError:
            genotypes = VCFGenos(vcf)

        return cls(records, genotypes)

    @classmethod
    def from_pgen(cls, pgen: Union[Path, Dict[str, Path]], sample_names: ArrayLike):
        if isinstance(pgen, Path):
            pgen = {"_all": pgen}
        records = Records.from_pvar(
            {c: p.with_suffix(".pvar") for c, p in pgen.items()}
        )
        sample_names = np.atleast_1d(np.asarray(sample_names))
        genotypes = PgenGenos(pgen, sample_names)
        return cls(records, genotypes)

    def read(
        self,
        contig: str,
        starts: ArrayLike,
        ends: ArrayLike,
        samples: Optional[ArrayLike] = None,
        ploidies: Optional[ArrayLike] = None,
    ):
        starts = np.atleast_1d(np.asarray(starts, dtype=int))
        ends = np.atleast_1d(np.asarray(ends, dtype=int))

        recs = self.records.vars_in_range(contig, starts, ends)
        if recs is None:
            return None

        if samples is not None:
            samples = np.atleast_1d(np.asarray(samples, dtype=str))
            geno_sample_idxs, sample_idxs = np.intersect1d(
                self.genotypes.samples, samples, return_indices=True
            )[1:]
            if len(sample_idxs) != len(samples):
                raise ValueError("Some samples were not found")
        else:
            geno_sample_idxs = None

        if ploidies is not None:
            ploidies = np.atleast_1d(np.asarray(ploidies, dtype=int))

        # (s p v)
        genos = self.genotypes.read(
            contig, recs.start_idxs, recs.end_idxs, geno_sample_idxs, ploidies
        )

        if samples is not None:
            genos = genos[sample_idxs]

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
        samples: Optional[ArrayLike] = None,
        ploidies: Optional[ArrayLike] = None,
    ):
        starts = np.atleast_1d(np.asarray(starts, dtype=int))
        ends = np.atleast_1d(np.asarray(ends, dtype=int))

        recs, max_ends = self.records.vars_in_range_for_haplotype_construction(
            contig, starts, ends
        )
        if recs is None:
            return None

        if samples is not None:
            samples = np.atleast_1d(np.asarray(samples, dtype=str))
            geno_sample_idxs, sample_idxs = np.intersect1d(
                self.genotypes.samples, samples, return_indices=True
            )[1:]
            if len(sample_idxs) != len(samples):
                raise ValueError("Some samples were not found")
        else:
            geno_sample_idxs = None

        if ploidies is not None:
            ploidies = np.atleast_1d(np.asarray(ploidies, dtype=int))

        # (s p v)
        genos = self.genotypes.read(
            contig, recs.start_idxs, recs.end_idxs, geno_sample_idxs, ploidies
        )

        if samples is not None:
            genos = genos[sample_idxs]

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
