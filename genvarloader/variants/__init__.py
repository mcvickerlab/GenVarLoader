from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from attrs import define
from numpy.typing import ArrayLike, NDArray

from ..util import normalize_contig_name
from .genotypes import Genotypes, MemmapGenos, PgenGenos, VCFGenos, ZarrGenos
from .records import Records, VLenAlleles

__all__ = [
    "PgenGenos",
    "ZarrGenos",
    "MemmapGenos",
    "VCFGenos",
    "Variants",
    "Records",
]


@define
class DenseGenotypes:
    """Dense array(s) of genotypes.

    Attributes
    ----------
    positions : NDArray[np.int32]
        Shape: (variants)
    size_diffs : NDArray[np.int32]
        Shape : (variants). Difference in length between the REF and the ALT alleles.
    ref : VLenAlleles
        Shape: (variants). REF alleles.
    alt : VLenAlleles
        Shape: (variants). ALT alleles.
    genotypes : NDArray[np.int8]
        Shape: (samples, ploid, variants)
    offsets : NDArray[np.uint32], optional
        Shape: (regions + 1). Offsets for the index boundaries of each region such
        that variants for region `i` are `positions[offsets[i] : offsets[i+1]]`,
        `size_diffs[offsets[i] : offsets[i+1]]`, ..., etc.
    """

    positions: NDArray[np.int32]
    size_diffs: NDArray[np.int32]
    ref: VLenAlleles
    alt: VLenAlleles
    genotypes: NDArray[np.int8]
    offsets: NDArray[np.uint32]


@define
class Variants:
    records: Records
    genotypes: Genotypes
    _sample_idxs: Optional[NDArray[np.intp]] = None

    @property
    def chunked(self):
        return self.genotypes.chunked

    @property
    def samples(self):
        if self._sample_idxs is None:
            return self.genotypes.samples
        return self.genotypes.samples[self._sample_idxs]

    @property
    def n_samples(self):
        if self._sample_idxs is None:
            return self.genotypes.n_samples
        return len(self._sample_idxs)

    @property
    def ploidy(self):
        return self.genotypes.ploidy

    @classmethod
    def from_vcf(cls, vcf: Union[str, Path, Dict[str, Path]], use_cache: bool = True):
        if isinstance(vcf, str):
            vcf = Path(vcf)

        records = Records.from_vcf(vcf)

        if use_cache:
            try:
                genotypes = ZarrGenos(vcf)
            except FileNotFoundError:
                genotypes = VCFGenos(vcf, records.contig_offsets)
        else:
            genotypes = VCFGenos(vcf, records.contig_offsets)

        return cls(records, genotypes)

    # TODO: read sample names from .psam file by using #IID or IID column. Implement a .psam reader.
    @classmethod
    def from_pgen(
        cls, pgen: Union[str, Path, Dict[str, Path]], sample_names: ArrayLike
    ):
        if isinstance(pgen, str):
            pgen = Path(pgen)

        if isinstance(pgen, Path):
            pgen = {"_all": pgen}
        records = Records.from_pvar(
            {c: p.with_suffix(".pvar") for c, p in pgen.items()}
        )
        sample_names = np.atleast_1d(np.asarray(sample_names))
        genotypes = PgenGenos(pgen, sample_names)
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
        contig = normalize_contig_name(contig, self.records.contigs)
        if contig is None:
            return None

        starts = np.atleast_1d(np.asarray(starts, dtype=int))
        ends = np.atleast_1d(np.asarray(ends, dtype=int))

        recs = self.records.vars_in_range(contig, starts, ends)
        if recs is None:
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
            geno_sample_idxs = self._sample_idxs

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

        contig = normalize_contig_name(contig, self.records.contigs)
        if contig is None:
            return None, ends.astype(np.int32)

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
            if len(sample_idxs) != len(sample):
                raise ValueError("Some samples were not found")
        else:
            geno_sample_idxs = None

        if ploid is not None:
            ploid = np.atleast_1d(np.asarray(ploid, dtype=int))

        # (s p v)
        genos = self.genotypes.read(
            contig, recs.start_idxs, recs.end_idxs, geno_sample_idxs, ploid
        )

        if sample is not None:
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
