from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

import cyvcf2
import joblib
import numpy as np
import pgenlib
from numpy.typing import ArrayLike, NDArray

from .._utils import _get_rel_starts


class DosageFieldError(Exception):
    """Exception raised when the dosage field is not found in the VCF record."""


__all__ = []


class Genotypes(Protocol):
    chunked: bool
    samples: NDArray[np.str_]
    ploidy: int
    handles: Optional[Dict[str, Any]]

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    def close(self):
        """Close any open file handles and set handles to None. Should be a no-op if
        the handles are already closed or the interface does not use handles."""
        ...

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

    def read_genos_and_dosages(
        self,
        contig: str,
        start_idxs: ArrayLike,
        end_idxs: ArrayLike,
        dosage_field: str,
        sample_idx: Optional[ArrayLike] = None,
        haplotype_idx: Optional[ArrayLike] = None,
    ) -> Tuple[NDArray[np.int8], NDArray[np.float32]]: ...

    def multiprocess_read(
        self,
        contig: str,
        start_idxs: ArrayLike,
        end_idxs: ArrayLike,
        sample_idx: Optional[ArrayLike] = None,
        haplotype_idx: Optional[ArrayLike] = None,
        n_jobs: int = 1,
    ) -> NDArray[np.int8]:
        """Read genotypes in parallel from a contig from index i to j, 0-based exclusive.

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
        n_jobs : int, optional
            Number of jobs to run in parallel.

        Returns
        -------
        genotypes : NDArray[np.int8]
            Shape: (samples ploidy variants). Genotypes for each query region.
        """
        start_idxs = np.atleast_1d(np.asarray(start_idxs, dtype=np.int32))
        end_idxs = np.atleast_1d(np.asarray(end_idxs, dtype=np.int32))
        n_jobs = min(n_jobs, len(start_idxs))
        starts = np.array_split(start_idxs, n_jobs)
        ends = np.array_split(end_idxs, n_jobs)

        self.close()  # close any open handles so they aren't pickled and cause an error
        tasks = [
            joblib.delayed(self.read)(contig, s, e, sample_idx, haplotype_idx)
            for s, e in zip(starts, ends)
        ]
        with joblib.Parallel(n_jobs=n_jobs) as parallel:
            split_genos = parallel(tasks)
        genos = np.concatenate(split_genos, axis=-1)  # type: ignore
        return genos

    def multiprocess_read_genos_and_dosages(
        self,
        contig: str,
        start_idxs: ArrayLike,
        end_idxs: ArrayLike,
        dosage_field: str,
        sample_idx: Optional[ArrayLike] = None,
        haplotype_idx: Optional[ArrayLike] = None,
        n_jobs: int = 1,
    ) -> Tuple[NDArray[np.int8], NDArray[np.float32]]:
        """Read genotypes in parallel from a contig from index i to j, 0-based exclusive.

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
        n_jobs : int, optional
            Number of jobs to run in parallel.

        Returns
        -------
        genotypes : NDArray[np.int8]
            Shape: (samples ploidy variants). Genotypes for each query region.
        """
        start_idxs = np.atleast_1d(np.asarray(start_idxs, dtype=np.int32))
        end_idxs = np.atleast_1d(np.asarray(end_idxs, dtype=np.int32))
        n_jobs = min(n_jobs, len(start_idxs))
        starts = np.array_split(start_idxs, n_jobs)
        ends = np.array_split(end_idxs, n_jobs)

        self.close()  # close any open handles so they aren't pickled and cause an error
        tasks = [
            joblib.delayed(self.read_genos_and_dosages)(
                contig, s, e, dosage_field, sample_idx, haplotype_idx
            )
            for s, e in zip(starts, ends)
        ]
        with joblib.Parallel(n_jobs=n_jobs) as parallel:
            # (s p v), (s v)
            results = parallel(tasks)
        split_genos, split_dosages = zip(*results)
        genos = np.concatenate(split_genos, axis=-1)  # type: ignore
        dosages = np.concatenate(split_dosages, axis=-1)  # type: ignore
        return genos, dosages


class PgenGenos(Genotypes):
    chunked = False
    ploidy = 2

    def __init__(
        self,
        paths: Union[str, Path, Dict[str, Path]],
        sample_names: NDArray[np.str_],
        contigs: List[str],
    ) -> None:
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
        self.current_sample_idx: Optional[NDArray[np.uint32]] = None

    def _pgen(self, contig: str, sample_idx: Optional[NDArray[np.uint32]]):
        return pgenlib.PgenReader(bytes(self.paths[contig]), sample_subset=sample_idx)  # type: ignore

    def close(self):
        if self.handles is not None:
            for handle in self.handles.values():
                handle.close()
        self.handles = None
        self.current_sample_idx = None

    def read(
        self,
        contig: str,
        start_idxs: ArrayLike,
        end_idxs: ArrayLike,
        sample_idx: Optional[ArrayLike] = None,
        haplotype_idx: Optional[ArrayLike] = None,
    ) -> NDArray[np.int8]:
        changed_sample_idx = False
        if sample_idx is not None:
            _sample_idx = np.atleast_1d(np.asarray(sample_idx, dtype=np.uint32))
            sample_sorter = np.argsort(_sample_idx)
            if self.current_sample_idx is None:
                self.current_sample_idx = _sample_idx[sample_sorter]
                changed_sample_idx = True
            elif np.array_equal(_sample_idx[sample_sorter], self.current_sample_idx):
                sample_sorter = slice(None)
        else:
            self.current_sample_idx = np.arange(self.n_samples, dtype=np.uint32)
            sample_sorter = slice(None)

        if haplotype_idx is not None:
            _haplotype_idx = np.atleast_1d(np.asarray(haplotype_idx, dtype=np.intp))
            if np.array_equal(haplotype_idx, np.arange(self.ploidy)):
                _haplotype_idx = slice(None)
        else:
            _haplotype_idx = slice(None)

        if self.handles is None or changed_sample_idx and "_all" in self.paths:
            handle = self._pgen("_all", self.current_sample_idx)
            self.handles = {c: handle for c in self.contigs}
        elif self.handles is None or changed_sample_idx:
            self.handles = {
                c: self._pgen(c, self.current_sample_idx) for c in self.contigs
            }

        start_idxs = np.atleast_1d(np.asarray(start_idxs, dtype=np.int32))
        end_idxs = np.atleast_1d(np.asarray(end_idxs, dtype=np.int32))

        vars_per_region = end_idxs - start_idxs
        n_vars = vars_per_region.sum()
        rel_start_idxs = _get_rel_starts(start_idxs, end_idxs)
        rel_end_idxs = rel_start_idxs + vars_per_region
        # (v s*2)
        genotypes = np.empty(
            (n_vars, len(self.current_sample_idx) * self.ploidy), dtype=np.int32
        )

        for s, e, rel_s, rel_e in zip(
            start_idxs, end_idxs, rel_start_idxs, rel_end_idxs
        ):
            if s == e:
                continue
            self.handles[contig].read_alleles_range(
                s, e, allele_int32_out=genotypes[rel_s:rel_e]
            )

        # (v s*2)
        genotypes = genotypes.astype(np.int8)
        # (s*2 v)
        genotypes = genotypes.swapaxes(0, 1)
        # (s 2 v)
        genotypes = np.stack([genotypes[::2], genotypes[1::2]], axis=1)

        genotypes = genotypes[sample_sorter, _haplotype_idx]

        return genotypes

    def read_genos_and_dosages(
        self,
        contig: str,
        start_idxs: ArrayLike,
        end_idxs: ArrayLike,
        dosage_field: str,
        sample_idx: Optional[ArrayLike] = None,
        haplotype_idx: Optional[ArrayLike] = None,
    ) -> Tuple[NDArray[np.int8], NDArray[np.float32]]:
        changed_sample_idx = False
        if sample_idx is not None:
            _sample_idx = np.atleast_1d(np.asarray(sample_idx, dtype=np.uint32))
            sample_sorter = np.argsort(_sample_idx)
            if self.current_sample_idx is None:
                self.current_sample_idx = _sample_idx[sample_sorter]
                changed_sample_idx = True
            elif np.array_equal(_sample_idx[sample_sorter], self.current_sample_idx):
                sample_sorter = slice(None)
        else:
            self.current_sample_idx = np.arange(self.n_samples, dtype=np.uint32)
            sample_sorter = slice(None)

        if haplotype_idx is not None:
            _haplotype_idx = np.atleast_1d(np.asarray(haplotype_idx, dtype=np.intp))
            if np.array_equal(haplotype_idx, np.arange(self.ploidy)):
                _haplotype_idx = slice(None)
        else:
            _haplotype_idx = slice(None)

        if self.handles is None or changed_sample_idx and "_all" in self.paths:
            handle = self._pgen("_all", self.current_sample_idx)
            self.handles = {c: handle for c in self.contigs}
        elif self.handles is None or changed_sample_idx:
            self.handles = {
                c: self._pgen(c, self.current_sample_idx) for c in self.contigs
            }

        start_idxs = np.atleast_1d(np.asarray(start_idxs, dtype=np.int32))
        end_idxs = np.atleast_1d(np.asarray(end_idxs, dtype=np.int32))

        vars_per_region = end_idxs - start_idxs
        n_vars = vars_per_region.sum()
        rel_start_idxs = _get_rel_starts(start_idxs, end_idxs)
        rel_end_idxs = rel_start_idxs + vars_per_region
        # (v s*2)
        genotypes = np.empty(
            (n_vars, len(self.current_sample_idx) * self.ploidy), dtype=np.int32
        )
        # (v s)
        dosages = np.empty((n_vars, len(self.current_sample_idx)), dtype=np.float32)

        for s, e, rel_s, rel_e in zip(
            start_idxs, end_idxs, rel_start_idxs, rel_end_idxs
        ):
            if s == e:
                continue
            self.handles[contig].read_alleles_range(
                s, e, allele_int32_out=genotypes[rel_s:rel_e]
            )
            self.handles[contig].read_dosages_range(s, e, dosages[rel_s:rel_e])

        # (v s*2)
        genotypes = genotypes.astype(np.int8)
        # (s*2 v)
        genotypes = genotypes.swapaxes(0, 1)
        # (s 2 v)
        genotypes = np.stack([genotypes[::2], genotypes[1::2]], axis=1)

        genotypes = genotypes[sample_sorter, _haplotype_idx]

        return genotypes, dosages.T


class VCFGenos(Genotypes):
    chunked = False
    ploidy = 2

    def __init__(
        self, vcfs: Union[str, Path, Dict[str, Path]], contig_offsets: Dict[str, int]
    ) -> None:
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
        self.handles = None

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
        geno_idxs = _get_rel_starts(start_idxs, end_idxs)
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

    def read_genos_and_dosages(
        self,
        contig: str,
        start_idxs: ArrayLike,
        end_idxs: ArrayLike,
        dosage_field: str,
        sample_idx: Optional[ArrayLike] = None,
        haplotype_idx: Optional[ArrayLike] = None,
    ) -> Tuple[NDArray[np.int8], NDArray[np.float32]]:
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
        genos = np.empty((len(self.samples), self.ploidy, n_variants), dtype=np.int8)
        dosages = np.empty((len(self.samples), n_variants), dtype=np.float32)

        if n_variants == 0:
            return genos, dosages

        # (n_queries)
        geno_idxs = _get_rel_starts(start_idxs, end_idxs)
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
                # (s, 1, 1) or (s, 1)?
                d = v.format(dosage_field)
                if d is None:
                    raise DosageFieldError(
                        f"Dosage field '{dosage_field}' not found for record {repr(v)}"
                    )
                # (s, 1, 1) or (s, 1)? -> (s, 1)
                dosages[:, place_idx] = d.squeeze()[:, None]
                # increment idxs for next iteration
                geno_idxs[overlapping_query_intervals] += 1
            if (geno_idxs == finish_idxs).all():
                break

        genos = genos[_sample_idx, _haplotype_idx]
        # cyvcf2 encoding: 0, 1, -1 => gvl/pgen encoding: 0, 1, -9
        genos[genos == -1] = -9

        return genos, dosages

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
