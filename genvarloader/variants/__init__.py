from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from attrs import define
from numpy.typing import ArrayLike, NDArray

from ..util import normalize_contig_name
from .genotypes import (
    Genotypes,
    MemmapGenos,
    NumpyGenos,
    PgenGenos,
    VCFGenos,
    VIdxGenos,
    ZarrGenos,
)
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
    def from_vcf(
        cls,
        vcf: Union[str, Path, Dict[str, Path]],
        use_cache: bool = True,
        chunk_shape: Optional[Tuple[int, int, int]] = None,
    ):
        """Currently does not support multi-allelic sites, but does support *split*
        multi-allelic sites. Note that SVs and "other" variants are also not supported.
        VCFs can be prepared by running:
        ```bash
        bcftools view -i 'TYPE="snp" || TYPE="indel"' <file.bcf> \\
        | bcftools norm \\
            -a \\
            --atom-overlaps . \\
            -m - \\
            -f <ref.fa> \\
            -O b \\
            -o <norm.bcf>
        ```
        """
        records = Records.from_vcf(vcf)

        if use_cache:
            try:
                genotypes = ZarrGenos(vcf)
            except FileNotFoundError:
                genotypes = VCFGenos(vcf, records.contig_offsets)
                genotypes = ZarrGenos.from_recs_genos(
                    records, genotypes, chunk_shape=chunk_shape
                )
        else:
            genotypes = VCFGenos(vcf, records.contig_offsets)
        return cls(records, genotypes)

    @classmethod
    def from_gvl(cls, path: Union[str, Path, Dict[str, Path]]):
        """Construct a Variants object from GVL files. The path(s) must end with `.gvl`.

        Parameters
        ----------
        path : Union[str, Path, Dict[str, Path]]
            Path to the GVL file(s).

        Returns
        -------
        Variants
        """
        records = Records.from_gvl_arrow(path)
        genotypes = ZarrGenos(path)
        return cls(records, genotypes)

    # TODO: read sample names from .psam file by using #IID or IID column. Implement a .psam reader.
    @classmethod
    def from_pgen(
        cls, pgen: Union[str, Path, Dict[str, Path]], sample_names: ArrayLike
    ):
        """Currently does not support multi-allelic sites, but does support *split*
        multi-allelic sites. Note that SVs and "other" variants are also not supported.
        A PGEN can be prepared from a VCF by running:
        ```bash
        bcftools view -i 'TYPE="snp" || TYPE="indel"' <file.bcf> \\
        | bcftools norm \\
            -a \\
            --atom-overlaps . \\
            -m - \\
            -f <ref.fa> \\
            -O b \\
            -o <norm.bcf> \\
        plink2 --make-pgen \\
            --bcf <norm.bcf> \\
            --vcf-half-call r \\
            --out <prefix>
        ```
        """
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

    def in_memory(self):
        if not isinstance(self.genotypes, NumpyGenos):
            genotypes = NumpyGenos.from_recs_genos(self.records, self.genotypes)
            return self.__class__(self.records, genotypes)
        return self

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
        contig = normalize_contig_name(
            contig, self.records.contigs
        )  # pyright: ignore[reportAssignmentType]
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

        _contig = normalize_contig_name(
            contig, self.records.contigs
        )  # pyright: ignore[reportAssignmentType]
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

    def vidx(
        self,
        contigs: ArrayLike,
        starts: ArrayLike,
        length: int,
        samples: ArrayLike,
        ploidies: ArrayLike,
    ):
        if not isinstance(self.genotypes, VIdxGenos):
            raise ValueError(
                f"Genotypes {self.genotypes} does not support vectorized indexing."
            )

        contigs = np.atleast_1d(np.asarray(contigs, dtype=str))
        starts = np.atleast_1d(np.asarray(starts, dtype=int))
        samples = np.atleast_1d(np.asarray(samples, dtype=str))
        ploidies = np.atleast_1d(np.asarray(ploidies, dtype=int))

        recs = self.records.vidx_vars_in_range(contigs, starts, length)
        if recs is None:
            return None

        genos = self.genotypes.vidx(
            contigs, recs.start_idxs, recs.end_idxs, samples, ploidies
        )

        return DenseGenotypes(
            recs.positions,
            recs.size_diffs,
            recs.refs,
            recs.alts,
            genos,
            recs.offsets,
        )

    def vidx_for_haplotype_construction(
        self,
        contigs: ArrayLike,
        starts: ArrayLike,
        length: int,
        samples: ArrayLike,
        ploidies: ArrayLike,
    ):
        if not isinstance(self.genotypes, VIdxGenos):
            raise ValueError(
                f"Genotypes {self.genotypes} does not support vectorized indexing."
            )

        contigs = np.atleast_1d(np.asarray(contigs, dtype=np.str_))
        starts = np.atleast_1d(np.asarray(starts, dtype=np.int32))
        samples = np.atleast_1d(np.asarray(samples, dtype=np.str_))
        ploidies = np.atleast_1d(np.asarray(ploidies, dtype=np.intp))

        recs, max_ends = self.records.vidx_vars_in_range_for_haplotype_construction(
            contigs, starts, length
        )
        if recs is None:
            return None, max_ends

        unique_samples, inverse = np.unique(samples, return_inverse=True)
        if missing := set(unique_samples).difference(self.samples):
            raise ValueError(f"Samples {missing} were not found")
        key_idx, query_idx = np.intersect1d(
            self.samples, unique_samples, return_indices=True, assume_unique=True
        )[1:]
        sample_idx = key_idx[query_idx[inverse]]

        if (ploidies >= self.ploidy).any():
            raise ValueError("Ploidies requested exceed maximum ploidy")

        genos = self.genotypes.vidx(
            contigs, recs.start_idxs, recs.end_idxs, sample_idx, ploidies
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
