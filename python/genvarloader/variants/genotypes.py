from pathlib import Path
from typing import Dict, List, Optional, Protocol, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from genvarloader.utils import get_rel_starts

try:
    import pgenlib

    PGENLIB_INSTALLED = True
except ImportError:
    PGENLIB_INSTALLED = False

try:
    import cyvcf2

    CYVCF2_INSTALLED = True
except ImportError:
    CYVCF2_INSTALLED = False


class Genotypes(Protocol):
    chunked: bool
    samples: NDArray[np.str_]
    ploidy: int

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    def read(
        self,
        contig: str,
        start_idxs: ArrayLike,
        end_idxs: ArrayLike,
        sample_idx: Optional[ArrayLike] = None,
        haplotype_idx: Optional[ArrayLike] = None,
    ) -> NDArray[np.int8]:
        """Read genotypes from a contig from index i to j, 0-based exclusive.

        Parameters
        ----------
        contig : str
            Name of the contig/chromosome.
        start_idxs : NDArray[np.intp]
            Start indices, 0-based.
        end_idxs : NDArray[np.intp]
            End indices, 0-based exclusive.
        sample_idx : NDArray[np.intp], optional
            Indices of the samples to include. Must be unique.
        haplotype_idx : NDArray[np.intp], optional
            Indices of the haplotypes to include. Must be unique.

        Returns
        -------
        genotypes : NDArray[np.int8]
            Shape: (samples ploidy variants). Genotypes for each query region.
        """
        ...


class PgenGenos(Genotypes):
    chunked = False
    ploidy = 2

    def __init__(
        self,
        paths: Union[str, Path, Dict[str, Path]],
        sample_names: NDArray[np.str_],
        contigs: List[str],
    ) -> None:
        if not PGENLIB_INSTALLED:
            raise ImportError(
                "pgenlib must be installed to use PGEN files for genotypes."
            )
        if isinstance(paths, (str, Path)):
            paths = {"_all": Path(paths)}

        if len(paths) == 0:
            raise ValueError("No paths provided.")
        n_samples = None
        n_variants = 0
        for p in paths.values():
            if not p.exists():
                raise FileNotFoundError(f"PGEN file {p} not found.")
            if n_samples is None:
                n_samples = pgenlib.PgenReader(bytes(p)).get_raw_sample_ct()  # type: ignore
            n_variants += pgenlib.PgenReader(bytes(p)).get_variant_ct()  # type: ignore
        self.paths = paths
        self.n_variants = n_variants
        self.samples = sample_names
        self.contigs = contigs
        self.handles: Optional[Dict[str, "pgenlib.PgenReader"]] = None

    def _pgen(self, contig: str, sample_idx: Optional[NDArray[np.uint32]]):
        return pgenlib.PgenReader(bytes(self.paths[contig]), sample_subset=sample_idx)  # type: ignore

    def read(
        self,
        contig: str,
        start_idxs: ArrayLike,
        end_idxs: ArrayLike,
        sample_idx: Optional[ArrayLike] = None,
        haplotype_idx: Optional[ArrayLike] = None,
    ) -> NDArray[np.int8]:
        if self.handles is None and "_all" in self.paths:
            handle = self._pgen("_all", None)
            self.handles = {c: handle for c in self.contigs}
        elif self.handles is None:
            self.handles = {c: self._pgen(c, None) for c in self.contigs}

        start_idxs = np.atleast_1d(np.asarray(start_idxs, dtype=np.int32))
        end_idxs = np.atleast_1d(np.asarray(end_idxs, dtype=np.int32))

        n_vars = (end_idxs - start_idxs).sum()
        # (v s*2)
        genotypes = np.empty((n_vars, self.n_samples * self.ploidy), dtype=np.int32)

        for i, (s, e) in enumerate(zip(start_idxs, end_idxs)):
            if s == e:
                continue
            rel_s = s - start_idxs[i]
            rel_e = e - start_idxs[i]
            self.handles[contig].read_alleles_range(
                s, e, allele_int32_out=genotypes[rel_s:rel_e]
            )

        # (v s*2)
        genotypes = genotypes.astype(np.int8)
        # (s*2 v)
        genotypes = genotypes.swapaxes(0, 1)
        # (s 2 v)
        genotypes = np.stack([genotypes[::2], genotypes[1::2]], axis=1)

        if sample_idx is not None:
            _sample_idx = np.atleast_1d(np.asarray(sample_idx, dtype=np.intp))
        else:
            _sample_idx = slice(None)

        if haplotype_idx is not None:
            _haplotype_idx = np.atleast_1d(np.asarray(haplotype_idx, dtype=np.intp))
        else:
            _haplotype_idx = slice(None)

        genotypes = genotypes[_sample_idx, _haplotype_idx]
        # re-order samples to be in query order
        # if sample_sorter is not None and (np.arange(n_samples) != sample_sorter).any():
        #     genotypes = genotypes[sample_sorter]

        return genotypes


class VCFGenos(Genotypes):
    chunked = False
    ploidy = 2

    def __init__(
        self, vcfs: Union[str, Path, Dict[str, Path]], contig_offsets: Dict[str, int]
    ) -> None:
        if not CYVCF2_INSTALLED:
            raise ImportError(
                "cyvcf2 must be installed to use VCF files for genotypes."
            )

        if isinstance(vcfs, (str, Path)):
            vcfs = {"_all": Path(vcfs)}

        self.paths = vcfs
        samples = None
        for p in self.paths.values():
            if not p.exists():
                raise FileNotFoundError(f"VCF file {p} not found.")
            if samples is None:
                f = cyvcf2.VCF(str(p))  # type: ignore
                samples = np.array(f.samples)
                f.close()
                del f
        self.samples = samples  # type: ignore
        self.handles: Optional[Dict[str, "cyvcf2.VCF"]] = None
        self.contig_offsets = contig_offsets

    def close(self):
        if self.handles is not None:
            for handle in self.handles.values():
                handle.close()

    def read(
        self,
        contig: str,
        start_idxs: ArrayLike,
        end_idxs: ArrayLike,
        sample_idx: Optional[ArrayLike] = None,
        haplotype_idx: Optional[ArrayLike] = None,
    ) -> NDArray[np.int8]:
        start_idxs = np.atleast_1d(np.asarray(start_idxs, dtype=np.int32))
        end_idxs = np.atleast_1d(np.asarray(end_idxs, dtype=np.int32))

        if self.handles is None and "_all" not in self.paths:
            self.handles = {
                c: cyvcf2.VCF(str(p), lazy=True)  # type: ignore
                for c, p in self.paths.items()
            }
        elif self.handles is None:
            first_path = next(iter(self.paths.values()))
            handle = cyvcf2.VCF(str(first_path), lazy=True)  # type: ignore
            self.handles = {c: handle for c in handle.seqnames}

        if sample_idx is None:
            _sample_idx = slice(None)
        else:
            _sample_idx = np.atleast_1d(np.asarray(sample_idx, dtype=np.int32))

        if haplotype_idx is None:
            _haplotype_idx = slice(None)
        else:
            _haplotype_idx = np.atleast_1d(np.asarray(haplotype_idx, dtype=np.int32))

        n_variants = (end_idxs - start_idxs).sum()
        # (s p v)
        genos = np.empty(
            (len(self.samples), self.ploidy, n_variants),
            dtype=np.int8,
        )

        if genos.size == 0:
            return genos

        # (n_queries)
        geno_idxs = get_rel_starts(start_idxs, end_idxs)
        finish_idxs = np.empty_like(geno_idxs)
        finish_idxs[:-1] = geno_idxs[1:]
        finish_idxs[-1] = n_variants
        offset = self.contig_offsets[contig]
        for i, v in enumerate(self.handles[contig](contig), start=offset):
            # (n_queries)
            overlapping_query_intervals = (i >= start_idxs) & (i < end_idxs)
            if overlapping_query_intervals.any():
                if v.is_sv or v.var_type == "unknown":
                    continue
                # (n_valid)
                place_idx = geno_idxs[overlapping_query_intervals]
                genos[..., place_idx] = v.genotype.array()[:, : self.ploidy, None]
                # increment idxs for next iteration
                geno_idxs[overlapping_query_intervals] += 1
            if (geno_idxs == finish_idxs).all():
                break

        genos = genos[_sample_idx, _haplotype_idx]
        # cyvcf2 encoding: 0, 1, -1 => gvl/pgen encoding: 0, 1, -9
        genos[genos == -1] = -9

        return genos

    def init_handles(self) -> Dict[str, "cyvcf2.VCF"]:
        if self.handles is None and "_all" not in self.paths:
            handles = {
                c: cyvcf2.VCF(str(p), lazy=True)  # pyright: ignore
                for c, p in self.paths.items()
            }
        elif self.handles is None:
            handle = cyvcf2.VCF(str(next(iter(self.paths.values()))), lazy=True)  # pyright: ignore
            handles = {c: handle for c in handle.seqnames}
        else:
            handles = self.handles
        return handles

    def __del__(self) -> None:
        if self.handles is not None:
            for handle in self.handles.values():
                handle.close()
