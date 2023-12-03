import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple, Union, cast

import numba as nb
import numpy as np
import polars as pl
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


class Pgen(Variants):
    def __init__(
        self,
        paths: Union[str, Path, Mapping[str, Union[str, Path]]],
        samples: Optional[List[str]] = None,
        n5_store: Union[str, Path, bool] = False,
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
        n5_store : Union[str, Path, bool], optional
            Whether to cache a Zarr store of the genotypes, writing one if it does not
            exist. By default False. Can also provide a file path, otherwise defaults to
            the same path as the PGEN file(s) with the extension `.geno.zarr`.

        Notes
        -----
        Writes a copy of the .pvar file as an Arrow file with extensions `.gvl.arrow` to
        speed up initialization (by 25x or more for larger files) as well as a
        `.ends.gvl.arrow` file containing the end positions of each variant.
        """
        if not PGENLIB_INSTALLED:
            raise ImportError("Pgenlib must be installed to read PGEN files.")

        # pgenlib is exclusively diploid
        self.ploidy = 2
        # unknown genotypes are set to -9
        self.UNKNOWN = -9

        if isinstance(paths, (str, Path)):
            _paths = Path(paths)
            self.pgen_paths = {"_all": _paths.with_suffix(".pgen")}
            self.split_by_contig = False
        elif isinstance(paths, dict):
            _paths = {
                contig: Path(path).with_suffix(".pgen")
                for contig, path in paths.items()
            }
            self.pgen_paths = _paths
            self.split_by_contig = True
        _first_path = self.pgen_paths[next(iter(self.pgen_paths))]

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

        # sorted starts
        self.v_starts: Dict[str, NDArray[np.int32]] = {}
        # difference in length between ref and alt, sorted by start
        self.v_diffs: Dict[str, NDArray[np.int32]] = {}
        # sorted ends
        self.v_ends: Dict[str, NDArray[np.int32]] = {}
        # end_idx_to_start_idx maps from relative end idxs to relative variant idxs
        # such that self.v_ends[]
        self.end_idx_to_start_idx: Dict[str, NDArray[np.int32]] = {}
        # no multi-allelics
        self.ref: Dict[str, VLenAlleles] = {}
        self.alt: Dict[str, VLenAlleles] = {}

        self.contig_offsets: Dict[str, int] = {}

        for contig, path in self.pgen_paths.items():
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
                with open(pvar_path, "r") as f:
                    skip_rows = 0
                    while f.readline().startswith("##"):
                        skip_rows += 1

                pvar = pl.read_csv(
                    pvar_path,
                    separator="\t",
                    skip_rows=skip_rows,
                    columns=["#CHROM", "POS", "REF", "ALT"],
                    dtypes={"#CHROM": pl.Utf8, "POS": pl.Int32},
                )

                if (pvar["ALT"].str.contains(",")).any():
                    raise RuntimeError(
                        f"""PGEN file {path} contains multi-allelic variants which are 
                        not yet supported by GenVarLoader. Split your multi-allelic 
                        variants with `bcftools norm -a --atom-overlaps . -m - 
                        <file.vcf>` then remake the PGEN file with the `--vcf-half-call 
                        r` option."""
                    )

                pvar = pvar.with_columns(
                    POS=pl.col("POS") - 1,
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
                        END=pl.col("POS")
                        - pl.col("ILEN").clip_max(0),  #! end-inclusive
                    )
                    .group_by("#CHROM", maintain_order=True)
                    .agg(pl.all().sort_by("END"), VAR_IDX=pl.col("END").arg_sort())
                    .explode(pl.exclude("#CHROM"))
                )
                pvar.write_ipc(pvar_arrow_path)
                ends.write_ipc(ends_arrow_path)

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
                    self.end_idx_to_start_idx[_contig] = partition["VAR_IDX"].to_numpy()

                # make all contigs map to the same pgen file
                pgen_path = self.pgen_paths["_all"]
                self.pgen_paths = {contig: pgen_path for contig in self.contig_offsets}
            else:
                self.contig_offsets[contig] = 0
                self.v_starts[contig] = pvar["POS"].to_numpy()
                self.v_diffs[contig] = pvar["ILEN"].to_numpy()
                self.v_ends[contig] = ends["END"].to_numpy()
                self.end_idx_to_start_idx[contig] = ends["VAR_IDX"].to_numpy()
                # no multi-allelics
                self.ref[contig] = VLenAlleles.from_polars(pvar["REF"])
                self.alt[contig] = VLenAlleles.from_polars(pvar["ALT"])

        self.contigs = list(self.contig_offsets.keys())
        self.contig_starts_with_chr = self.infer_contig_prefix(self.contigs)

        # q_j = largest index i < j such that variant i is non-overlapping with j
        self.nearest_non_overlapping = {
            c: (
                np.searchsorted(self.v_ends[c], self.v_starts[c], side="right") - 1
            ).astype(np.int32)
            for c in self.contigs
        }

        self.n_variants = sum(len(p) for p in self.v_starts.values())

        if n5_store is not False:
            if not TENSORSTORE_INSTALLED:
                raise ImportError("Tensorstore must be installed to cache genotypes.")

            if n5_store is True:
                a_pgen_path = next(iter(self.pgen_paths.values()))
                cache_path = a_pgen_path.with_suffix(".geno.n5")
            else:
                cache_path = Path(n5_store)

            if not cache_path.exists():
                self.tstore = self._write_cache(cache_path)
            else:
                self.tstore = self._open_cache(cache_path)
            self.chunked = True
        else:
            self.tstore = None
            self.chunked = False

    def _pgen(self, contig: str, sample_idx: Optional[NDArray[np.uint32]]):
        if sample_idx is not None:
            sample_idx = np.sort(sample_idx)
        return pgenlib.PgenReader(
            bytes(self.pgen_paths[contig]), sample_subset=sample_idx
        )

    def _write_cache(self, cache_path: Path):
        ts_open_kwargs = {
            "spec": {
                "driver": "n5",
                "kvstore": {"driver": "file", "path": str(cache_path)},
            },
            "dtype": np.int8,
            "shape": (self.n_samples, self.ploidy, self.n_variants),
            "write": True,
            "create": True,
            "chunk_layout": ts.ChunkLayout(  # pyright: ignore[reportGeneralTypeIssues]
                chunk_shape=(100, self.ploidy, int(1e5))
            ),
        }

        ts_handle = ts.open(  # pyright: ignore[reportGeneralTypeIssues]
            **ts_open_kwargs
        ).result()

        # write all genotypes to cache in chunks of up to 1 GB of memory, using up to
        # (8 + 1) * chunk size = 9 GB working space
        chunksize = int(1e9)
        var_per_chunk = chunksize // (self.n_samples * self.ploidy)
        for c, c_offset in self.contig_offsets.items():
            with self._pgen(c, None) as f:
                n_vars = len(self.v_starts[c])
                idxs = np.arange(0, n_vars + var_per_chunk, var_per_chunk)
                idxs[-1] = min(idxs[-1], n_vars)
                pbar = tqdm(zip(idxs[:-1], idxs[1:]), total=len(idxs) - 1)
                for s_idx, e_idx in pbar:
                    pbar.set_description(f"Reading {c}:{s_idx}-{e_idx}")
                    genotypes = np.empty(
                        (self.n_samples * self.ploidy, var_per_chunk), dtype=np.int32
                    )
                    f.read_alleles_range(s_idx, e_idx, genotypes, hap_maj=1)
                    pbar.set_description(f"Writing {c}:{s_idx}-{e_idx}")
                    ts_handle[..., s_idx + c_offset : e_idx + c_offset].write(
                        genotypes.astype(np.int8).reshape(
                            self.n_samples, self.ploidy, -1
                        )
                    ).result()
                    pbar.set_description(f"Sleeping {c}:{s_idx}-{e_idx}")
                    time.sleep(10)

        return ts_handle

    def _open_cache(self, cache_path: Path):
        ts_open_kwargs = {
            "spec": {
                "driver": "n5",
                "kvstore": {"driver": "file", "path": str(cache_path)},
            },
            "open": True,
            "read": True,
            "dtype": np.int8,
            "shape": (self.n_samples, self.ploidy, self.n_variants),
        }

        return ts.open(  # pyright: ignore[reportGeneralTypeIssues]
            **ts_open_kwargs
        ).result()

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
            n_samples = self.n_samples
            pgen_idx, query_idx = self.sample_idx, None
        else:
            n_samples = len(samples)
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
        no_variant_mask = _s_idxs == len(self.v_ends[contig])
        _s_idxs = np.where(no_variant_mask, -1, _s_idxs)
        # make idxs absolute
        s_idxs = (
            self.end_idx_to_start_idx[contig][_s_idxs] + self.contig_offsets[contig]
        )
        e_idxs = (
            np.searchsorted(self.v_starts[contig], ends) + self.contig_offsets[contig]
        )
        e_idxs[no_variant_mask] = s_idxs[no_variant_mask]

        if s_idxs.min() == e_idxs.max():
            return None

        v_idxs = np.concatenate(
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

        # get alleles
        # (s*2, v)
        genotypes = np.empty((n_samples * self.ploidy, len(v_idxs)), dtype=np.int32)
        with self._pgen(contig, pgen_idx) as f:
            f.read_alleles_list(
                variant_idxs=v_idxs, allele_int32_out=genotypes, hap_maj=1
            )

        # (s, 2, v)
        genotypes = genotypes.astype(np.int8).reshape(n_samples, self.ploidy, -1)

        # re-order samples to be in query order
        if query_idx is not None:
            genotypes = genotypes[query_idx]

        genotypes = genotypes[:, ploid]

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
            n_samples = self.n_samples
            pgen_idx, query_idx = self.sample_idx, None
        else:
            n_samples = len(samples)
            pgen_idx, query_idx = self.get_sample_idx(samples)

        ploid = kwargs.get("ploid", None)
        if ploid is None:
            ploid = np.arange(self.ploidy)
        else:
            ploid = np.asarray(ploid)

        starts, ends = np.asarray(starts, dtype=np.int64), np.asarray(
            ends, dtype=np.int64
        )

        contig = self.normalize_contig_name(contig)

        # contig is not present in PGEN, has no variants
        if contig not in self.contigs:
            return None, ends

        _s_idxs = np.searchsorted(self.v_ends[contig], starts)
        no_variant_mask = _s_idxs == len(self.v_ends[contig])
        _s_idxs = np.where(no_variant_mask, -1, _s_idxs)
        s_idxs = self.end_idx_to_start_idx[contig][_s_idxs]
        e_idxs = np.searchsorted(self.v_starts[contig], ends)
        e_idxs[no_variant_mask] = s_idxs[no_variant_mask]

        max_deletion_lengths = weighted_activity_selection(
            r_start_idxs=s_idxs,
            r_end_idxs=e_idxs,
            e_idx_to_s_idx=self.end_idx_to_start_idx[contig],
            size_diffs=self.v_diffs[contig],
            nearest_non_overlapping=self.nearest_non_overlapping[contig],
        )
        max_ends = ends + max_deletion_lengths

        e_idxs = np.searchsorted(self.v_starts[contig], max_ends)

        # make idxs absolute
        s_idxs += self.contig_offsets[contig]
        e_idxs += self.contig_offsets[contig]
        e_idxs[no_variant_mask] = s_idxs[no_variant_mask]

        if s_idxs.min() == e_idxs.max():
            return None, max_ends

        v_idxs = np.concatenate(
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

        if self.tstore is None:
            # get alleles
            # (s*2, v)
            genotypes = np.empty((n_samples * self.ploidy, len(v_idxs)), dtype=np.int32)
            with self._pgen(contig, pgen_idx) as f:
                f.read_alleles_list(
                    variant_idxs=v_idxs, allele_int32_out=genotypes, hap_maj=1
                )

            # (s, 2, v)
            genotypes = genotypes.astype(np.int8).reshape(n_samples, self.ploidy, -1)

            # re-order samples to be in query order
            if query_idx is not None:
                genotypes = genotypes[query_idx]
        else:
            # let tensorstore have the full query available to hopefully parallelize
            # reading variable length slices of genotypes
            sub_genos = [None] * len(s_idxs)
            for i, (s_idx, e_idx) in enumerate(zip(s_idxs, e_idxs)):
                # no variants in query regions
                if s_idx == e_idx:
                    continue
                sub_genos.append(self.tstore[..., s_idx:e_idx])

            genotypes = ts.concat(  # pyright: ignore[reportGeneralTypeIssues]
                sub_genos, axis=-1
            )[
                ts.d[0].translate_to[0]  # pyright: ignore[reportGeneralTypeIssues]
            ]

            if query_idx is not None:
                genotypes = genotypes[query_idx]

            genotypes = cast(NDArray[np.int8], genotypes.result())

        out = DenseGenotypes(
            positions=positions,
            size_diffs=size_diffs,
            ref=ref,
            alt=alt,
            genotypes=genotypes[:, ploid],
            offsets=offsets,
        )

        return (out, max_ends)

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


@nb.njit(nogil=True, cache=True)
def weighted_activity_selection(
    r_start_idxs: NDArray[np.int32],
    r_end_idxs: NDArray[np.int32],
    e_idx_to_s_idx: NDArray[np.int32],
    size_diffs: NDArray[np.int32],
    nearest_non_overlapping: NDArray[np.int32],
) -> NDArray[np.int64]:
    """Implementation of the [weighted activity selection problem](https://en.wikipedia.org/wiki/Activity_selection_problem)
    to compute the maximum length of deletions that can occur for each region. This is
    used to adjust the end coordinates for reference sequence queries and include all
    variants for that are needed for haplotype construction.

    Parameters
    ----------
    r_start_idxs : NDArray[np.int32]
        Shape: (n_regions). The index of sorted_ends that is the first variant spanning
        the region of interest.
    r_end_idxs : NDArray[np.int32]
        Shape: (n_regions). The index of sorted_ends that is the last variant spanning
        the region of interest.
    e_idx_to_s_idx : NDArray[np.int32]
        Shape: (n_variants). The index of the variant in `v_starts` such that the start
        positions of `v_ends[i]` is `v_starts[e_idx_to_s_idx[i]]`.
    size_diffs : NDArray[np.int32]
        Shape: (n_variants). The difference in length between the ref and alt alleles.
        Sorted by start position.
    nearest_non_overlapping : NDArray[np.int32]
        Shape: (n_variants). The largest index `i < j` such that
        `v_ends[i] < v_starts[e_idx_to_s_idx][j]`.

    Returns
    -------
    max_deletion_lengths : NDArray[np.int64]
        Shape: (n_regions). The maximum length of deletions that can occur in each
        region.

    Notes
    -----
    For the weighted activity selection problem, each deletion corresponds
    to an activity with weight equal to the length of the deletion. The goal is to
    compute the maximum total weight of deletions for each query region.

    For right padding, we also need the final variant index (non-inclusive) that needs
    to be included to span the target length of each region and right pad sequences
    shorter than the query. Unfortunately, right padding can be arbitrarily long, so we
    make the choice to only include variants from expanding the query region by the max
    deletion length and then can right pad the sequence to the target length downstream
    when constructing haplotypes. This is a heuristic that should work well for most
    cases, but may not work well for very long deletions or queries near the end of a
    contig. However, genetic variation in telomeric regions is not well captured by most
    sequencing technologies anyway, so this should not be a problem in practice.

    Psuedocode from (Princeton slides)[https://www.cs.princeton.edu/~wayne/cs423/lectures/dynamic-programming-4up.pdf]:
    Given starts s_1, ..., s_n, ends e_1, ..., e_n, and weights w_1, ..., w_n.
    Note that ends are sorted, e_1 <= ... <= e_n.
    Let q_j = largest index i < j such that activity i is compatible with j
    i.e. q_j = argmax(e < s_j)
    Let opt(j) = value of solution to the problem consisting of activities 1 to j
    Then,
    opt(0) = 0
    else
    opt(j) = max(w_j + opt(q_j), opt(j - 1))
    """
    n_regions = len(r_start_idxs)
    max_deletion_lengths = np.empty(n_regions, dtype=np.int64)
    for i in nb.prange(n_regions):
        n_vars = r_end_idxs[i] - r_start_idxs[i]
        _e_idx_to_s_idx = e_idx_to_s_idx[r_start_idxs[i] : r_end_idxs[i]]
        _size_diffs = size_diffs[_e_idx_to_s_idx]
        _q = nearest_non_overlapping[_e_idx_to_s_idx]

        opt = np.empty(n_vars, dtype=np.int32)
        for j in range(n_vars):
            v = opt[_q[j]] + _size_diffs[j]
            if j == 0:
                opt[j] = min(v, 0)
            else:
                opt[j] = min(v, opt[j - 1])
        max_deletion_lengths[i] = -opt[-1]

    return max_deletion_lengths
