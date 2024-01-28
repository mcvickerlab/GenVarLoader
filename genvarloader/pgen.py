import gc
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Protocol, Tuple, Union, cast

import numba as nb
import numpy as np
import polars as pl
from loguru import logger
from numpy.typing import NDArray
from tqdm.auto import tqdm

from .types import DenseGenotypes, Variants, VLenAlleles

try:
    import pgenlib

    PGENLIB_INSTALLED = True
except ImportError:
    PGENLIB_INSTALLED = False

try:
    import tensorstore as ts

    TENSORSTORE_INSTALLED = True
except ImportError:
    TENSORSTORE_INSTALLED = False

try:
    import zarr
    from numcodecs.blosc import Blosc

    ZARR_INSTALLED = True
except ImportError:
    ZARR_INSTALLED = False


class Pgen(Variants):
    # pgenlib is exclusively diploid
    ploidy = 2
    # unknown genotypes are set to -9
    UNKNOWN = -9

    def __init__(
        self,
        paths: Union[str, Path, Mapping[str, Union[str, Path]]],
        samples: Optional[List[str]] = None,
        caches: Optional[Union[str, Path, Mapping[str, Union[str, Path]]]] = None,
    ) -> None:
        """Reads genotypes from PGEN files. Currently does not support multi-allelic
        sites, but does support *split* multi-allelic sites. This can be done by
        running `bcftools norm -a --atom-overlaps . -m - <file.vcf> -f <ref.fa>` and 
        creating the PGEN with the `--vcf-half-call r` option. All together:
        ```bash
        bcftools norm \\
            -a \\
            --atom-overlaps . \\
            -m - \\
            -f '<ref.fa>' \\
            -O b \\
            -o '<norm.bcf>' \\
            '<file.bcf>'
        plink2 --make-pgen \\
            --bcf '<norm.bcf>' \\
            --vcf-half-call r \\
            --out '<prefix>'
        ```

        Parameters
        ----------
        path : str | Path | Dict[str, str | Path]
            Path to any of the PGEN files (.pgen, .pvar, .psam) or their prefix. Or, a 
            dictionary mapping contig names to the PGEN file for that contig.
        samples : List[str], optional
            Which samples to include, by default all samples.
        caches : str | Path | Mapping[str, str | Path], optional
            Where to cache genotypes, writing them if they do not exist. By default
            uses the PGEN file which is slow for training. Recommended cache type is N5,
            by using the .n5 file extension on cache paths.

        Notes
        -----
        Writes a copy of the .pvar file as an Arrow file with extensions `.gvl.arrow` to
        speed up initialization (by 25x or more for larger files) as well as a
        `.ends.gvl.arrow` file containing the end positions of each variant.
        """
        if not PGENLIB_INSTALLED:
            raise ImportError("Pgenlib must be installed to read PGEN files.")

        if isinstance(paths, (str, Path)):
            _paths = Path(paths)
            pgen_paths = {"_all": _paths.with_suffix(".pgen")}
            self.split_by_contig = False
        elif isinstance(paths, Mapping):
            _paths = {
                contig: Path(path).with_suffix(".pgen")
                for contig, path in paths.items()
            }
            pgen_paths = _paths
            self.split_by_contig = True
        _first_path = pgen_paths[next(iter(pgen_paths))]

        try:
            psam_samples = pl.read_csv(
                _first_path.with_suffix(".psam"), separator="\t", columns=["IID"]
            )["IID"].to_numpy()
        except pl.ColumnNotFoundError:
            psam_samples = pl.read_csv(
                _first_path.with_suffix(".psam"), separator="\t", columns=["#IID"]
            )["#IID"].to_numpy()

        if samples is not None:
            _samples, sample_idx, _ = np.intersect1d(
                psam_samples, samples, return_indices=True
            )
            if len(_samples) != len(samples):
                raise ValueError("Got samples that are not in the pgen file.")
            self.samples = _samples
            self.sample_idx = sample_idx.astype(np.uint32)
        else:
            self.samples = psam_samples
            self.sample_idx = np.arange(len(psam_samples), dtype=np.uint32)

        self.n_samples = len(self.samples)

        ## sorted by starts ##
        self.v_starts: Dict[str, NDArray[np.int32]] = {}
        # difference in length between ref and alt, sorted by start
        self.v_diffs: Dict[str, NDArray[np.int32]] = {}
        # no multi-allelics
        self.ref: Dict[str, VLenAlleles] = {}
        self.alt: Dict[str, VLenAlleles] = {}
        ######################

        ## sorted by ends ##
        self.v_ends: Dict[str, NDArray[np.int32]] = {}
        self.v_diffs_sorted_by_ends: Dict[str, NDArray[np.int32]] = {}
        self.e2s_idx: Dict[str, NDArray[np.int32]] = {}
        self.max_del_q: Dict[str, NDArray[np.int32]] = {}
        ####################

        self.contig_offsets: Dict[str, int] = {}

        for contig, path in pgen_paths.items():
            pvar_path = path.with_suffix(".pvar")
            pvar_arrow_path = path.with_suffix(".gvl.arrow")
            ends_arrow_path = path.with_suffix(".ends.gvl.arrow")

            # arrow files exists and, if .pvar exists, were modified more recently
            if (
                pvar_arrow_path.exists()
                and ends_arrow_path.exists()
                and (
                    not pvar_path.exists()
                    or (
                        pvar_path.exists()
                        and pvar_arrow_path.stat().st_mtime > pvar_path.stat().st_mtime
                        and ends_arrow_path.stat().st_mtime > pvar_path.stat().st_mtime
                    )
                )
            ):
                pvar = pl.read_ipc(pvar_arrow_path)
                ends = pl.read_ipc(ends_arrow_path)
            else:
                logger.info(
                    "Did not find existing .gvl.arrow files, creating them. This may take several minutes."  # noqa: E501
                )

                with open(pvar_path, "r") as f:
                    skip_rows = 0
                    while f.readline().startswith("##"):
                        skip_rows += 1

                logger.info("Reading .pvar file...")
                pvar = pl.read_csv(
                    pvar_path,
                    separator="\t",
                    skip_rows=skip_rows,
                    columns=["#CHROM", "POS", "REF", "ALT"],
                    dtypes={"#CHROM": pl.Utf8, "POS": pl.Int32},
                )
                logger.info("Finished reading .pvar file.")

                if (pvar["ALT"].str.contains(",")).any():
                    raise RuntimeError(
                        f"""PGEN file {path} contains multi-allelic variants which are 
                        not yet supported by GenVarLoader. Split your multi-allelic 
                        variants with `bcftools norm -f <reference.fa> -a
                        --atom-overlaps . -m - <file.vcf>` then remake the PGEN file
                        with the `--vcf-half-call r` option."""
                    )

                logger.info("Writing .gvl.arrow files...")

                pvar = pvar.with_columns(
                    POS=pl.col("POS") - 1,  #! change to 0-indexed
                    ILEN=(
                        pl.col("ALT").str.len_bytes().cast(pl.Int32)
                        - pl.col("REF").str.len_bytes().cast(pl.Int32)
                    ),
                )

                # ends in reference coordiantes, 0-based **inclusive**
                # end_to_var_idx is a mapping from variants sorted by end to the
                # earliest positioned variant that has an end >= the end of the variant
                # at that index e.g. if the v0 has end 200, and v1 has end 100, then \
                # ends would be sorted as [v1, v0] and end_to_var_idx[0] = 1 and
                # end_to_var_idx[1] = 1
                ends = (
                    pvar.select(
                        "#CHROM",
                        "ILEN",
                        END=pl.col("POS")
                        - pl.col("ILEN").clip(upper_bound=0),  #! end-inclusive
                    )
                    .with_row_count("VAR_IDX")
                    .group_by("#CHROM", maintain_order=True)
                    .agg(
                        pl.all().sort_by("END"),
                        # make E2S_IDX relative to each contig
                        pl.int_range(0, pl.count(), dtype=pl.UInt32)
                        .sort_by("END")
                        .reverse()
                        .rolling_min(pvar.height, min_periods=1)
                        .reverse()
                        .alias("E2S_IDX"),
                    )
                    .explode(pl.exclude("#CHROM"))
                    .select("#CHROM", "END", "ILEN", "VAR_IDX", "E2S_IDX")
                )

                for_max_dels = ends.join(
                    pvar.select("POS").with_row_count("VAR_IDX"), on="VAR_IDX"
                )
                max_del_q = np.empty(for_max_dels.height, dtype=np.int32)
                last_idx = 0
                for _, part in for_max_dels.group_by("#CHROM", maintain_order=True):
                    part = part.sort("END")
                    _starts = part["POS"].to_numpy()
                    _ends = part["END"].to_numpy()
                    was_ends = np.empty(len(_ends) + 1, dtype=_ends.dtype)
                    was_ends[0] = 0
                    #! convert to end-exclusive, + 1
                    was_ends[1:] = _ends + 1
                    q = np.searchsorted(was_ends, _starts, side="right") - 1
                    max_del_q[last_idx : last_idx + part.height] = q
                    last_idx += part.height
                ends = ends.with_columns(MD_Q=max_del_q)

                pvar.write_ipc(pvar_arrow_path)
                ends.select("#CHROM", "END", "MD_Q", "ILEN", "E2S_IDX").write_ipc(
                    ends_arrow_path
                )

                logger.info("Finished writing .gvl.arrow files.")

            if contig == "_all":
                last_end = 0
                for _contig, partition in pvar.partition_by(
                    "#CHROM", maintain_order=True, as_dict=True
                ).items():
                    self.contig_offsets[_contig] = last_end
                    last_end = self.contig_offsets[_contig] + partition.height
                    self.v_starts[_contig] = partition["POS"].to_numpy()
                    # no multi-allelics
                    self.ref[_contig] = VLenAlleles.from_polars(partition["REF"])
                    self.alt[_contig] = VLenAlleles.from_polars(partition["ALT"])
                    self.v_diffs[_contig] = partition["ILEN"].to_numpy()

                for _contig, partition in ends.partition_by(
                    "#CHROM", as_dict=True
                ).items():
                    self.v_ends[_contig] = partition["END"].to_numpy()
                    self.v_diffs_sorted_by_ends[_contig] = partition["ILEN"].to_numpy()
                    self.e2s_idx[_contig] = np.empty(
                        partition.height + 1, dtype=np.uint32
                    )
                    self.e2s_idx[_contig][:-1] = partition["E2S_IDX"].to_numpy()
                    self.e2s_idx[_contig][-1] = partition.height
                    self.max_del_q[_contig] = partition["MD_Q"].to_numpy()

                # make all contigs map to the same pgen file
                pgen_path = pgen_paths["_all"]
                pgen_paths = {contig: pgen_path for contig in self.contig_offsets}
            else:
                self.contig_offsets[contig] = 0
                self.v_starts[contig] = pvar["POS"].to_numpy()
                self.v_diffs[contig] = pvar["ILEN"].to_numpy()
                self.v_ends[contig] = ends["END"].to_numpy()
                self.v_diffs_sorted_by_ends[contig] = ends["ILEN"].to_numpy()
                self.e2s_idx[contig] = np.empty(ends.height + 1, dtype=np.uint32)
                self.e2s_idx[contig][:-1] = ends["E2S_IDX"].to_numpy()
                self.e2s_idx[contig][-1] = ends.height
                self.max_del_q[contig] = ends["MD_Q"].to_numpy()

                # no multi-allelics
                self.ref[contig] = VLenAlleles.from_polars(pvar["REF"])
                self.alt[contig] = VLenAlleles.from_polars(pvar["ALT"])

        self.contigs = list(self.contig_offsets.keys())
        self.contig_starts_with_chr = self.infer_contig_prefix(self.contigs)
        self.n_variants = sum(len(p) for p in self.v_starts.values())

        if caches is not None:
            if isinstance(caches, (str, Path)):
                caches = {c: Path(caches) for c in self.contigs}
            else:
                caches = {c: Path(p) for c, p in caches.items()}

            cache_type = next(iter(caches.values())).suffix[1:]
            write_caches = any(not p.exists() for p in caches.values())

            if cache_type in ("n5", "zarr"):
                if write_caches:
                    pgen_genos = _PgenGenos(pgen_paths, self.n_samples, self.n_variants)
                    self.genotypes = _TStoreGenos.from_genos(
                        pgen_genos,
                        self.n_samples,
                        self.n_variants,
                        self.ploidy,
                        self.contig_offsets,
                        {c: len(p) for c, p in self.v_starts.items()},
                        caches,
                    )
                else:
                    self.genotypes = _TStoreGenos(caches)
            elif cache_type == "memmap":
                self.genotypes = _MemmapGenos()
            else:
                raise ValueError(
                    "Couldn't infer file type from cache's file extension."
                )
        else:
            self.genotypes = _PgenGenos(pgen_paths, self.n_samples, self.n_variants)

        self.chunked = self.genotypes.chunked
        self.nbytes = (
            sum(a.nbytes for a in self.v_diffs.values())
            + sum(a.nbytes for a in self.v_starts.values())
            + sum(a.nbytes for a in self.v_ends.values())
            + sum(a.nbytes for a in self.e2s_idx.values())
            + sum(a.nbytes for a in self.max_del_q.values())
            + sum(a.nbytes for a in self.ref.values())
            + sum(a.nbytes for a in self.alt.values())
        )

    def read(
        self, contig: str, starts: NDArray[np.int64], ends: NDArray[np.int64], **kwargs
    ) -> Optional[DenseGenotypes]:
        """Read genotypes that overlap the query regions.

        Parameters
        ----------
        contig : str
            Name of the contig/chromosome.
        starts : NDArray[int32]
            Start coordinates, 0-based.
        ends : int, NDArray[int32]
            End coordinates, 0-based exclusive.

        Returns
        -------
        Optional[DenseGenotypes]
            Genotypes for each query region.
        """
        samples = kwargs.get("sample", None)
        if samples is None:
            pgen_idx, query_idx = self.sample_idx, None
        else:
            len(samples)
            pgen_idx, query_idx = self.get_sample_idx(samples)

        ploid = kwargs.get("ploid", None)
        if ploid is None:
            ploid = np.arange(self.ploidy)
        else:
            ploid = np.asarray(ploid)

        starts, ends = (
            np.asarray(starts, dtype=np.int64),
            np.asarray(ends, dtype=np.int64),
        )

        contig = self.normalize_contig_name(contig)

        # contig is not present in PGEN, has no variants
        if contig not in self.contigs:
            return None

        _s_idxs = np.searchsorted(self.v_ends[contig], starts)
        # make idxs absolute
        s_idxs = self.e2s_idx[contig][_s_idxs] + self.contig_offsets[contig]
        e_idxs = (
            np.searchsorted(self.v_starts[contig], ends) + self.contig_offsets[contig]
        )

        if s_idxs.min() == e_idxs.max():
            return None

        n_var_per_region = e_idxs - s_idxs
        offsets = np.empty(len(n_var_per_region) + 1, dtype=np.uint32)
        offsets[0] = 0
        offsets[1:] = np.cumsum(n_var_per_region)

        rel_s_idxs = s_idxs - self.contig_offsets[contig]
        rel_e_idxs = e_idxs - self.contig_offsets[contig]

        positions = np.concatenate(
            [self.v_starts[contig][s:e] for s, e, in zip(rel_s_idxs, rel_e_idxs)]
        )
        size_diffs = np.concatenate(
            [self.v_diffs[contig][s:e] for s, e, in zip(rel_s_idxs, rel_e_idxs)]
        )
        ref = VLenAlleles.concat(
            *(self.ref[contig][s:e] for s, e, in zip(rel_s_idxs, rel_e_idxs))
        )
        alt = VLenAlleles.concat(
            *(self.alt[contig][s:e] for s, e, in zip(rel_s_idxs, rel_e_idxs))
        )

        genotypes = self.genotypes.read(contig, s_idxs, e_idxs, pgen_idx, ploid)

        out = DenseGenotypes(
            positions=positions,
            size_diffs=size_diffs,
            ref=ref,
            alt=alt,
            genotypes=genotypes,
            offsets=offsets,
        )

        return out

    def read_for_haplotype_construction(
        self,
        contig: str,
        starts: NDArray[np.int64],
        ends: NDArray[np.int64],
        **kwargs,
    ) -> Tuple[Optional[DenseGenotypes], NDArray[np.int64]]:
        """Read genotypes for haplotype construction. This is a special case of `read`
        where variants past (i.e. downstream of) the query regions can be included to
        ensure that haplotypes of desired length can be constructed. This is necessary
        because deletions can shorten the haplotype, so variants downstream of `end` may
        be needed to add more sequence to the haplotype.

        Parameters
        ----------
        contig : str
            Name of the contig/chromosome.
        starts : NDArray[int32]
            Start coordinates, 0-based.
        ends : int, NDArray[int32]
            End coordinates, 0-based exclusive.

        Returns
        -------
        Optional[DenseGenotypes]
            Genotypes for each query region.
        NDArray[np.int64]
            New ends for querying the reference genome such that enough sequence is
            available to get haplotypes of `target_length`.
        """
        samples = kwargs.get("sample", None)
        if samples is None:
            pgen_idx, query_idx = self.sample_idx, None
        else:
            len(samples)
            pgen_idx, query_idx = self.get_sample_idx(samples)

        ploid = kwargs.get("ploid", None)
        if ploid is None:
            ploid = np.arange(self.ploidy)
        else:
            ploid = np.asarray(ploid)

        starts, ends = np.atleast_1d(starts), np.atleast_1d(ends)

        contig = self.normalize_contig_name(contig)

        # contig is not present in PGEN, has no variants
        if contig not in self.contigs:
            return None, ends

        _s_idxs = np.searchsorted(self.v_ends[contig], starts)

        max_ends, _e_idxs = get_max_ends_and_idxs(
            self.v_ends[contig],
            self.v_diffs_sorted_by_ends[contig],
            self.max_del_q[contig],
            _s_idxs,
            ends,
        )

        s_idxs = self.e2s_idx[contig][_s_idxs]
        e_idxs = self.e2s_idx[contig][_e_idxs]

        # make idxs absolute
        s_idxs += self.contig_offsets[contig]
        e_idxs += self.contig_offsets[contig]

        if s_idxs.min() == e_idxs.max():
            return None, ends

        np.concatenate(
            [np.arange(s, e, dtype=np.uint32) for s, e in zip(s_idxs, e_idxs)]
        )
        n_var_per_region = e_idxs - s_idxs
        offsets = np.empty(len(n_var_per_region) + 1, dtype=np.uint32)
        offsets[0] = 0
        np.cumsum(n_var_per_region, out=offsets[1:])

        rel_s_idxs = s_idxs - self.contig_offsets[contig]
        rel_e_idxs = e_idxs - self.contig_offsets[contig]

        positions = np.concatenate(
            [self.v_starts[contig][s:e] for s, e, in zip(rel_s_idxs, rel_e_idxs)]
        )
        size_diffs = np.concatenate(
            [self.v_diffs[contig][s:e] for s, e, in zip(rel_s_idxs, rel_e_idxs)]
        )
        ref = VLenAlleles.concat(
            *(self.ref[contig][s:e] for s, e, in zip(rel_s_idxs, rel_e_idxs))
        )
        alt = VLenAlleles.concat(
            *(self.alt[contig][s:e] for s, e, in zip(rel_s_idxs, rel_e_idxs))
        )

        genotypes = self.genotypes.read(contig, s_idxs, e_idxs, pgen_idx, ploid)

        out = DenseGenotypes(
            positions=positions,
            size_diffs=size_diffs,
            ref=ref,
            alt=alt,
            genotypes=genotypes,
            offsets=offsets,
        )

        return (out, max_ends.astype(np.int64))

    def get_sample_idx(self, samples):
        _samples, _pgen_idx, query_idx = np.intersect1d(
            self.samples, samples, return_indices=True
        )
        if len(_samples) != len(samples):
            unrecognized_samples = set(samples) - set(_samples)
            raise ValueError(
                f"""Got samples that are not in the pgen file: 
                {unrecognized_samples}."""
            )
        pgen_idx: NDArray[np.uint32] = self.sample_idx[_pgen_idx]
        return pgen_idx, query_idx


class _Genotypes(Protocol):
    chunked: bool

    def read(
        self,
        contig: str,
        start_idxs: NDArray[np.integer],
        end_idxs: NDArray[np.integer],
        sample_idx: Optional[NDArray[np.integer]],
        haplotype_idx: Optional[NDArray[np.integer]],
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


class _PgenGenos(_Genotypes):
    chunked = False
    paths: Dict[str, Path]
    PLOIDY = 2

    def __init__(self, paths: Dict[str, Path], n_samples: int, n_variants: int) -> None:
        self.paths = paths
        self.n_samples = n_samples
        self.n_variants = n_variants

    def _pgen(self, contig: str, sample_idx: Optional[NDArray[np.uint32]]):
        return pgenlib.PgenReader(bytes(self.paths[contig]), sample_subset=sample_idx)

    def read(
        self,
        contig: str,
        start_idxs: NDArray[np.integer],
        end_idxs: NDArray[np.integer],
        sample_idx: Optional[NDArray[np.integer]],
        haplotype_idx: Optional[NDArray[np.integer]],
    ) -> NDArray[np.int8]:
        if sample_idx is None:
            n_samples = self.n_samples
            pgen_idx = None
            sample_sorter = None
        else:
            n_samples = len(sample_idx)
            sample_sorter = np.argsort(sample_idx)
            pgen_idx = sample_idx[sample_sorter].astype(np.uint32)

        n_vars = (end_idxs - start_idxs).sum()
        genotypes: NDArray[np.int32] = np.empty(
            (n_samples * self.PLOIDY, n_vars), dtype=np.int32
        )

        with self._pgen(contig, pgen_idx) as f:
            for i, (s, e) in enumerate(zip(start_idxs, end_idxs)):
                if s == e:
                    continue
                rel_s = s - start_idxs[i]
                rel_e = e - start_idxs[i]
                f.read_alleles_range(
                    s, e, allele_int32_out=genotypes[..., rel_s:rel_e], hap_maj=1
                )

        # (s, 2, v)
        genotypes = genotypes.reshape(n_samples, self.PLOIDY, -1)

        if haplotype_idx is not None:
            genotypes = genotypes[:, haplotype_idx]

        genotypes = genotypes.astype(np.int8)

        # re-order samples to be in query order
        if sample_sorter is not None and (np.arange(n_samples) != sample_sorter).any():
            genotypes = genotypes[sample_sorter]

        return genotypes


class _TStoreGenos(_Genotypes):
    chunked = True

    def __init__(self, paths: Dict[str, Path]) -> None:
        if not TENSORSTORE_INSTALLED:
            raise ImportError(
                "Tensorstore must be installed to use chunked array caches like Zarr and N5."  # noqa: E501
            )
        self.paths = paths
        first_path = next(iter(paths.values()))
        if all(first_path == p for p in paths.values()):
            tstore = self._open_tstore(next(iter(paths)))
            self.tstores = {contig: tstore for contig in paths}
        else:
            self.tstores = {contig: self._open_tstore(contig) for contig in paths}

    @classmethod
    def from_genos(
        cls,
        genotypes: _Genotypes,
        n_samples: int,
        n_variants: int,
        ploidy: int,
        contig_offsets: Dict[str, int],
        n_var_per_contig: Dict[str, int],
        cache_paths: Dict[str, Path],
        mem=int(1e9),
        chunk_shape=None,
    ) -> "_TStoreGenos":
        if chunk_shape is None:
            chunk_shape = (100, ploidy, int(1e5))

        if not TENSORSTORE_INSTALLED:
            raise ImportError(
                "Tensorstore must be installed to use chunked array caches like Zarr and N5."  # noqa: E501
            )

        first_path = next(iter(cache_paths.values()))
        one_tstore = all(first_path == p for p in cache_paths.values())
        if one_tstore:
            driver = first_path.suffix[1:]
            ts_open_kwargs = {
                "spec": {
                    "driver": driver,
                    "kvstore": {"driver": "file", "path": str(first_path)},
                },
                "dtype": np.int8,
                "shape": (n_samples, ploidy, n_variants),
                "create": True,
                "delete_existing": True,
                "chunk_layout": ts.ChunkLayout(  # pyright: ignore[reportGeneralTypeIssues]
                    chunk_shape=chunk_shape
                ),
            }
            ts_handle = ts.open(  # pyright: ignore[reportGeneralTypeIssues]
                **ts_open_kwargs
            ).result()

        c_n_variants = np.array(list(n_var_per_contig.values()))
        var_per_chunk = mem // (n_samples * ploidy)
        for (contig, path), c_n_vars in zip(cache_paths.items(), c_n_variants):
            c_offset = contig_offsets[contig]
            driver = path.suffix[1:]

            if not one_tstore:
                ts_open_kwargs = {
                    "spec": {
                        "driver": driver,
                        "kvstore": {"driver": "file", "path": str(path)},
                    },
                    "dtype": np.int8,
                    "shape": (n_samples, ploidy, c_n_vars),
                    "create": True,
                    "delete_existing": True,
                    "chunk_layout": ts.ChunkLayout(  # pyright: ignore[reportGeneralTypeIssues]
                        chunk_shape=chunk_shape
                    ),
                }
                ts_handle = ts.open(  # pyright: ignore[reportGeneralTypeIssues]
                    **ts_open_kwargs
                ).result()

            # write all genotypes to cache in chunks of up to 1 GB of memory, using up
            # to (8 + 1) * chunk size = 9 GB working space
            # [c_offset, c_offset + c_n_vars)
            # for contig files, c_offset always = 0
            # for one file, c_offset = cumsum of previous offsets
            # c_n_vars is the number of variants in the contig
            idxs = np.arange(
                c_offset, c_offset + c_n_vars + var_per_chunk, var_per_chunk
            )
            idxs[-1] = c_offset + c_n_vars
            for s_idx, e_idx in tqdm(
                zip(idxs[:-1], idxs[1:]),
                total=len(idxs) - 1,
                desc=f"Writing contig {contig} to cache.",
            ):
                genos = genotypes.read(
                    contig, np.array([s_idx]), np.array([e_idx]), None, None
                )
                ts_handle[  # pyright: ignore[reportUnboundVariable]
                    :, :, s_idx:e_idx
                ] = genos
                gc.collect()

        return cls(cache_paths)

    def _open_tstore(self, contig: str):
        ts_open_kwargs = {
            "spec": {
                "driver": "n5",
                "kvstore": {"driver": "file", "path": str(self.paths[contig])},
            },
            "open": True,
            "read": True,
        }

        return ts.open(  # pyright: ignore[reportGeneralTypeIssues]
            **ts_open_kwargs
        ).result()

    def read(
        self,
        contig: str,
        start_idxs: NDArray[np.integer],
        end_idxs: NDArray[np.integer],
        sample_idx: Optional[NDArray[np.integer]],
        haplotype_idx: Optional[NDArray[np.integer]],
    ) -> NDArray[np.int8]:
        # let tensorstore have the full query available to hopefully parallelize
        # reading variable length slices of genotypes
        sub_genos = [None] * len(start_idxs)
        _tstore = self.tstores[contig]

        if sample_idx is not None:
            _tstore = _tstore[sample_idx]
        if haplotype_idx is not None:
            _tstore = _tstore[:, haplotype_idx]

        for i, (s, e) in enumerate(zip(start_idxs, end_idxs)):
            # no variants in query regions
            if s == e:
                continue
            sub_genos.append(_tstore[..., s:e])

        genotypes = ts.concat(  # pyright: ignore[reportGeneralTypeIssues]
            sub_genos, axis=-1
        )[
            ts.d[0].translate_to[0]  # pyright: ignore[reportGeneralTypeIssues]
        ]

        genotypes = cast(NDArray[np.int8], genotypes.result())

        return genotypes


class _ZarrGenos(_Genotypes):
    chunked = True

    def __init__(self, paths: Dict[str, Path]) -> None:
        if not ZARR_INSTALLED:
            raise ImportError(
                "Zarr must be installed to use zarr-python to I/O Zarr stores."
            )

        self.paths = paths
        first_path = next(iter(paths.values()))
        if all(first_path == p for p in paths.values()):
            z_handle = self._open_store(next(iter(paths)))
            self.stores = {contig: z_handle for contig in paths}
        else:
            self.stores = {contig: self._open_store(contig) for contig in paths}

    def _open_store(self, contig: str):
        z_handle = zarr.open_array(
            store=str(self.paths[contig]),
            mode="r",
        )
        return z_handle

    def read(
        self,
        contig: str,
        start_idxs: NDArray[np.integer],
        end_idxs: NDArray[np.integer],
        sample_idx: Optional[NDArray[np.integer]],
        haplotype_idx: Optional[NDArray[np.integer]],
    ) -> NDArray[np.int8]:
        raise NotImplementedError

    @classmethod
    def from_genos(
        cls,
        genotypes: _Genotypes,
        n_samples: int,
        n_variants: int,
        ploidy: int,
        contig_offsets: Dict[str, int],
        n_var_per_contig: Dict[str, int],
        cache_paths: Dict[str, Path],
        mem=int(1e9),
        chunk_shape=None,
    ) -> "_ZarrGenos":
        if not ZARR_INSTALLED:
            raise ImportError(
                "Zarr must be installed to use zarr-python to write Zarr stores."
            )

        if chunk_shape is None:
            chunk_shape = (100, ploidy, int(1e5))

        first_path = next(iter(cache_paths.values()))
        one_store = all(first_path == p for p in cache_paths.values())
        if one_store:
            z_handle = zarr.open_array(
                store=str(first_path),
                dtype=np.int8,
                shape=(n_samples, ploidy, n_variants),
                mode="w",
                chunks=chunk_shape,  # pyright: ignore[reportGeneralTypeIssues]
                compressor=Blosc(shuffle=2),
            )

        c_n_variants = np.array(list(n_var_per_contig.values()))
        var_per_chunk = mem // (n_samples * ploidy)
        for (contig, path), c_n_vars in zip(cache_paths.items(), c_n_variants):
            c_offset = contig_offsets[contig]
            if not one_store:
                z_handle = zarr.open_array(
                    store=str(path),
                    dtype=np.int8,
                    shape=(n_samples, ploidy, c_n_vars),
                    mode="w",
                    chunks=chunk_shape,  # pyright: ignore[reportGeneralTypeIssues]
                    compressor=Blosc(shuffle=2),
                )

            # [c_offset, c_offset + c_n_vars)
            # for contig files, c_offset always = 0
            # for one file, c_offset = cumsum of previous offsets
            # c_n_vars is the number of variants in the contig
            idxs = np.arange(
                c_offset, c_offset + c_n_vars + var_per_chunk, var_per_chunk
            )
            idxs[-1] = c_offset + c_n_vars
            for s_idx, e_idx in tqdm(
                zip(idxs[:-1], idxs[1:]),
                total=len(idxs) - 1,
                desc=f"Writing contig {contig} to cache.",
            ):
                genos = genotypes.read(
                    contig, np.array([s_idx]), np.array([e_idx]), None, None
                )
                z_handle[  # pyright: ignore[reportUnboundVariable]
                    :, :, s_idx:e_idx
                ] = genos

        return cls(cache_paths)


class _MemmapGenos(_Genotypes):
    chunked = False

    def __init__(self) -> None:
        raise NotImplementedError

    def read(
        self,
        contig: str,
        start_idxs: NDArray[np.integer],
        end_idxs: NDArray[np.integer],
        sample_idx: Optional[NDArray[np.integer]],
        haplotype_idx: Optional[NDArray[np.integer]],
    ) -> NDArray[np.int8]:
        raise NotImplementedError


# @nb.njit(nogil=True, cache=True)
def get_max_ends_and_idxs(
    v_ends: NDArray[np.int32],
    v_diffs: NDArray[np.int32],
    nearest_nonoverlapping: NDArray[np.intp],
    start_idxs: NDArray[np.intp],
    query_ends: NDArray[np.int64],
) -> Tuple[NDArray[np.int32], NDArray[np.intp]]:
    max_ends: NDArray[np.int32] = np.empty(len(start_idxs), dtype=np.int32)
    end_idxs: NDArray[np.intp] = np.empty(len(start_idxs), dtype=np.intp)
    for r in nb.prange(len(start_idxs)):
        s = start_idxs[r]
        if s == len(v_ends):  # no variants in this region
            max_ends[r] = query_ends[r]
            end_idxs[r] = s
            continue

        w = -v_diffs[s:]  # flip sign so deletions have positive weight

        # to adjust q from [0, j) to [i, j)
        # (q[i:] - i).clip(0)
        q = (nearest_nonoverlapping[s:] - s).clip(0)
        max_end, end_idx = weighted_activity_selection(v_ends[s:], w, q, query_ends[r])
        max_ends[r] = max_end
        end_idxs[r] = s + end_idx
    return max_ends, end_idxs


# @nb.njit(nogil=True, cache=True)
def weighted_activity_selection(
    v_ends: NDArray[np.int32],
    w: NDArray[np.int32],
    q: NDArray[np.intp],
    query_end: int,
) -> Tuple[int, int]:
    """Implementation of the [weighted activity selection problem](https://en.wikipedia.org/wiki/Activity_selection_problem)
    to compute the maximum length of deletions that can occur for each region. This is
    used to adjust the end coordinates for reference sequence queries and include all
    variants for that are needed for haplotype construction.

    Parameters
    ----------
    v_ends : NDArray[np.int64]
        Shape: (variants). End coordinates of variants, 0-based inclusive.
    w : NDArray[np.int64]
        Shape: (variants). Weights of activities (i.e. deletion lengths).
    q : NDArray[np.intp]
        Shape: (variants). Nearest variant i such that i < j and variants are non-overlapping, q[j] = i.
    query_end : int
        Shape: (regions). End of query region.

    Returns
    -------
    max_ends : NDArray[np.int32]
        Shape: (regions). Maximum end coordinate for each query region.
    end_idxs : NDArray[np.intp]
        Shape: (regions). Index of the variant with the maximum end coordinate for each
        query region.

    Notes
    -----
    For the weighted activity selection problem, each deletion corresponds
    to an activity with weight equal to the length of the deletion. The goal is to
    compute the maximum total weight of deletions for each query region.

    Psuedocode from (Princeton slides)[https://www.cs.princeton.edu/~wayne/cs423/lectures/dynamic-programming-4up.pdf]:
    Given starts :math: `s_1, ..., s_n`, ends :math: `e_1, ..., e_n`, and weights
    :math: `w_1, ..., w_n`.
    Note that ends are sorted, :math: `e_1 <= ... <= e_n`.
    Let :math: `q_j` = largest index :math: `i < j` such that activity :math: `i` is
    compatible with :math: `j`.
    Let opt(j) = value of solution to the problem consisting of activities 1 to j
    Then,
        opt(0) = 0
    and
        opt(j) = max(w_j + opt(q_j), opt(j - 1))
    """
    n_vars = len(w)
    max_del: NDArray[np.int32] = np.empty(n_vars + 1, dtype=np.int32)
    max_del[0] = 0
    for j in range(1, n_vars + 1):
        max_del[j] = max(max_del[q[j - 1]] + w[j - 1], max_del[j - 1])
        v_del_end = v_ends[j - 1] - max_del[j] + 1  # + 1, v_ends is end-inclusive
        # if:
        # this variant more than satisfies query length
        # last variant doesn't span v_del_end
        if v_del_end > query_end and j > 1 and v_ends[j - 2] <= v_del_end:
            # then add the max deletion length up to but not including this variant
            # to the query end, and return the index of this variant for slicing
            return query_end + max_del[j - 1], j - 1
    return query_end + max_del[-1], n_vars
