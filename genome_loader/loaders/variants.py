import gc
from abc import ABC, abstractmethod
from enum import Enum
from functools import reduce
from pathlib import Path
from typing import Optional, cast

import cyvcf2
import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from numpy.typing import NDArray

from genome_loader.loaders import Queries
from genome_loader.utils import PathType, validate_sample_sheet


class Variants(ABC):
    class MissingValue(Enum):
        REFERENCE = "reference"
        N = "N"

    @abstractmethod
    def sel(
        self,
        queries: Queries,
        length: int,
        **kwargs,
    ) -> Optional[
        tuple[NDArray[np.bytes_], NDArray[np.integer], NDArray[np.unsignedinteger]]
    ]:
        """Get the variants for specified regions.

        Parameters
        ----------
        queries : Queries
            Must have the following columns: contig, start, sample, ploid_idx
        length: int
        **kwargs : dict, optional
            sorted : bool
                Whether the queries are sorted by sample and ploid_idx.
            missing_value : 'reference' or 'N'
                What the replace missing values with (reference allele or 'N').

        Returns
        -------
        variants : ndarray[str]
            Flat array of all variants from the queries.
        positions : ndarray[int]
            Flat array of each variant's position.
        offsets : ndarray[int]
            Where each query's variants are in the result.
            For example, the first query's variants can be obtained as:
            >>> first_query_vars = variants[offsets[0] : region_offsets[1]]
        """
        raise NotImplementedError


# TODO: deprecate this class? Otherwise, handle missing genotypes.
class VCFVariants(Variants):
    def __init__(self, sample_sheet: PathType, threads: Optional[int] = None) -> None:
        raise NotImplementedError
        """Read batches of variants from VCFs using specific regions and samples.

        Note that the variants should be exclusively SNPs (i.e. no MNPs, indels, etc.).

        Parameters
        ----------
        sample_sheet : PathType
            A sample sheet (.csv) with at least two columns, 'sample' and 'vcf', that form a mapping from sample
            names to VCF files.
        threads : int, optional
            Threads for decompression, recommended to use =< 4 threads, by default None
        """
        _sample_sheet = pl.read_csv(sample_sheet)

        required_columns = ["sample", "vcf"]
        validate_sample_sheet(_sample_sheet, required_columns)

        self.sample_to_vcf: dict[str, str] = dict(
            zip(_sample_sheet["sample"], _sample_sheet["vcf"])
        )
        self.threads = threads if threads is not None else 1

    def sel(self, contigs, starts, length, samples, ploid_idx):
        raise NotImplementedError
        queries = pl.DataFrame(
            [
                pl.Series("contig", contigs, pl.Utf8),
                pl.Series("start", starts, pl.Int32),
                pl.Series("sample", samples, pl.Utf8),
                pl.Series("ploid_idx", ploid_idx, pl.UInt16),
            ]
        )
        res = self.get_variants(queries, length)
        if res is None:
            return None
        else:
            variants, variant_locs, variant_counts = res
        del queries
        gc.collect()
        return variants, variant_locs, variant_counts

    def get_variants(
        self, queries: pl.DataFrame, length: int
    ) -> Optional[tuple[NDArray[np.str_], NDArray[np.integer], NDArray[np.integer]]]:
        raise NotImplementedError

        def add_sam_string(queries: pl.DataFrame, length):
            return queries.with_column(
                # chrom:start-end
                pl.concat_str(
                    [
                        "chrom",
                        pl.lit(":"),
                        pl.col("start") + 1,  # samtools coordinates are 1-indexed
                        pl.lit("-"),
                        pl.col("start") + 1 + length,
                    ]
                ).alias("sam_str")
            )

        def get_vcf(sample: str):
            return cyvcf2.VCF(
                samples=[sample],
                fname=self.sample_to_vcf[sample],
                gts012=True,
                lazy=False,
                threads=self.threads,
            )

        ploid_idx = queries["ploid_idx"].to_numpy()

        queries = (
            queries.select(pl.exclude("ploid_idx"))
            .with_row_count()
            .sort(["sample", "contig", "start"])
            .pipe(add_sam_string, length=length)
        )

        # iterate through variants of query regions
        # Note: opportunity for concurrency as each sample is a different file?
        # May also be possible to use concurrency over regions.
        variant_ls = []
        variant_locs_ls = []
        variant_cnt_ls = []
        group: pl.DataFrame
        for group in queries.groupby("sample", maintain_order=True):  # type: ignore
            sample = group[0, "sample"]
            reader = get_vcf(sample)
            idx = 0
            for sam_str in group["sam_str"]:
                variant_cnt_ls.append(idx)
                bases_starts = [(v.gt_bases[0], v.start) for v in reader(sam_str)]
                variant_ls.append(np.array([x[0] for x in bases_starts]))
                variant_locs_ls.append(np.array([x[1] for x in bases_starts]))
                idx += len(bases_starts)

        # cast to arrays
        variants = np.array(variant_ls, dtype=object)
        variant_locs = np.array(variant_locs_ls, dtype=object)
        variant_cnts = np.array(variant_cnt_ls)

        # return if no variants
        if len(variants) == 0:
            return None

        # sort into original order
        sorter = queries["row_nr"].arg_sort().to_numpy()
        variants = variants[sorter]
        variant_locs = variant_locs[sorter]
        variant_cnts = variant_cnts[sorter]

        # IMPORTANT: assumes all variants are SNVs
        # concat array of arrays
        variants = np.concatenate(variants).astype("U3")
        variant_locs = np.concatenate(variant_locs).astype("i4")

        # split "X/X" into ['X', '/', 'X'] and select ['X', 'X']
        variants = variants.reshape(-1, 1).view("U1")[:, [0, 2]]

        # get the haplotype for each query
        ploid_idx = np.repeat(ploid_idx, repeats=variant_cnts)
        variants = variants[np.arange(len(variants)), ploid_idx].astype("S1")

        return variants, variant_locs, variant_cnts


class ZarrVariants(Variants):

    path: Path
    zarr: "SgkitDataset"

    class SgkitDataset(xr.Dataset):
        call_genotype: xr.DataArray
        call_genotype_mask: xr.DataArray
        sample_id: xr.DataArray
        variant_allele: xr.DataArray
        variant_contig: xr.DataArray
        variant_position: xr.DataArray
        attrs: dict[str, list[str]]

    def __init__(self, sgkit_zarr: PathType) -> None:
        """Read batches of variants from Zarrs using specific regions and samples.

        Note that the variants should be exclusively SNPs (i.e. no MNPs, indels, etc.).

        Parameters
        ----------
        sample_sheet : PathType
            A sample sheet (.csv) with at least two columns, 'sample' and 'zarr_variants',
            that form a mapping from sample names to Zarr files generated from VCFs using `sgkit`.
        """
        self.path = Path(sgkit_zarr)
        self.zarr = cast(
            ZarrVariants.SgkitDataset,
            xr.open_dataset(
                sgkit_zarr,
                engine="zarr",
                chunks={},
                concat_characters=False,
                drop_variables=[
                    "variant_filter",
                    "variant_id",
                    "variant_id_mask",
                    "variant_quality",
                    "call_genotype_phased",
                ],
            ),
        )

        self.zarr.variant_allele = self.zarr.variant_allele.astype("S1")
        self.zarr.variant_contig.load()
        # VCFs are 1-indexed and sgkit Datasets inherit the indexing of whatever
        # format they are generated from. Since queries are 0-indexed, adjust this.
        self.zarr_variant_position = self.zarr.variant_position.load() - 1
        self.zarr.sample_id.load()
        self.contig_offsets = np.searchsorted(
            self.zarr.variant_contig.values, self.zarr.attrs["contigs"]
        )

    def samples_to_sample_idxs(self, samples: pd.Series) -> NDArray[np.intp]:
        """Map sample names to sample indices as represented in the sgkit dataset.

        Example
        -------
        >>> variants = ZarrVariants('my/variants.zarr')
        >>> variants.sample_id.values
        array(['NCI-H660', 'OCI-AML5'], dtype=object)
        >>> variants.samples_to_sample_idxs(['OCI-AML5', 'OCI-AML5', 'OCI-AML5'])
        array([1, 1, 1])
        """
        uniq_samples = samples.cat.categories.values
        idxs = samples.cat.codes.values
        common_samples, _, idx_map = np.intersect1d(
            uniq_samples,
            self.zarr.sample_id,
            assume_unique=True,
            return_indices=True,
        )
        if len(common_samples) != len(uniq_samples):
            raise ValueError(
                f"Query samples not found: {np.setdiff1d(uniq_samples, common_samples, assume_unique=True)}"
            )
        return idx_map[idxs]

    def contigs_to_contig_idxs(
        self, contigs: pd.Series
    ) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """Get the unique contig indices and array of mapped contig indices i.e. those used in the sgkit VCF Dataset.

        Example
        -------
        >>> variants = ZarrVariants('my/variants.zarr')
        >>> variants.attrs['contigs']
        array(['1', 'X'], dtype=object)
        >>> variants.contigs_to_contig_idxs(['X', 'X', 'X'])
        (array([1]), array([1, 1, 1]))
        """
        uniq_contigs = contigs.cat.categories.values
        idxs = contigs.cat.codes.values
        common_contigs, _, idx_map = np.intersect1d(
            uniq_contigs,
            self.zarr.attrs["contigs"],
            assume_unique=True,
            return_indices=True,
        )
        if len(common_contigs) != len(uniq_contigs):
            raise ValueError(
                f"Query contigs not found: {np.setdiff1d(uniq_contigs, common_contigs, assume_unique=True)}"
            )
        return idx_map, idx_map[idxs]

    def sel(
        self, queries: Queries, length: int, **kwargs
    ) -> Optional[
        tuple[NDArray[np.bytes_], NDArray[np.integer], NDArray[np.unsignedinteger]]
    ]:
        sorted = kwargs.get("sorted", False)
        try:
            missing_value = self.MissingValue(kwargs.get("missing_value", "reference"))
        except ValueError:
            raise ValueError(
                "Keyword argument 'missing_value' must be 'reference' or 'N'."
            )

        if not isinstance(sorted, bool):
            raise TypeError("Keyword argument 'sorted' must be a bool.")

        sample_idx = self.samples_to_sample_idxs(queries.sample)
        _, contig_idx = self.contigs_to_contig_idxs(queries.contig)
        _queries = queries.assign(sample_idx=sample_idx, contig_idx=contig_idx)

        # get variants, their positions, and how many are in each query
        variant_ls: list[xr.DataArray] = []
        position_ls: list[xr.DataArray] = []
        count_ls: list[NDArray[np.unsignedinteger]] = []
        if not sorted:
            idx_ls = []
        for name, group in _queries.groupby(["sample", "ploid_idx"]):
            s_idx = group[0, "sample_idx"]
            p_idx = name[1]
            c_idx = group.contig_idx.to_numpy()
            s = group.starts.to_numpy()
            e = s + length

            indices, counts = self._intervals_to_idxs_and_cnts(c_idx, s, e)

            subset = self.zarr.isel(variants=indices, samples=s_idx, ploidy=p_idx)
            if missing_value is Variants.MissingValue.REFERENCE:
                subset.call_genotype = subset.call_genotype.where(
                    ~subset.call_genotype_mask, 0
                )
            s_p_vars = subset.variant_allele.isel(
                alleles=subset.call_genotype.as_numpy()
            )
            if missing_value is self.MissingValue.N:
                s_p_vars = s_p_vars.where(~subset.call_genotype_mask, "N")
            s_p_var_pos = subset.variant_position

            variant_ls.append(s_p_vars)
            position_ls.append(s_p_var_pos)
            count_ls.append(counts)
            if not sorted:
                idx_ls.append(group.index.values)

        # get counts
        variant_cnts: NDArray[np.unsignedinteger] = np.concatenate(count_ls)
        if sum(variant_cnts) == 0:
            return None

        # unsort
        if not sorted:
            idx = np.concatenate(idx_ls)
            unsorter = np.argsort(idx)
            variant_cnts = variant_cnts[unsorter]
            variant_ls = [variant_ls[i] for i in unsorter]
            position_ls = [position_ls[i] for i in unsorter]

        # get variants, positions, and offsets
        variants: NDArray[np.bytes_] = xr.concat(variant_ls, dim="variants").values
        positions: NDArray[np.int32] = xr.concat(position_ls, dim="variants").values
        offsets = np.zeros(len(variant_cnts) + 1, dtype=variant_cnts.dtype)
        variant_cnts.cumsum(out=offsets[1:])

        return variants, positions, offsets

    def _intervals_to_idxs_and_cnts(
        self,
        contig_idxs: NDArray[np.intp],
        starts: NDArray[np.integer],
        ends: NDArray[np.integer],
    ) -> tuple[NDArray[np.integer], NDArray[np.unsignedinteger]]:

        size = self.zarr.dims["variants"]
        slices = [
            self._pslice_to_slice(c, s, e)
            for (c, s, e) in zip(contig_idxs, starts, ends)
        ]
        indices = [np.arange(*sl.indices(size)) for sl in slices]
        counts: NDArray[np.unsignedinteger] = np.array([len(idx) for idx in indices])

        return np.concatenate(indices), counts  # type: ignore[no-untyped-call]

    def _pslice_to_slice(
        self,
        contig_idx: int,
        start: int,
        end: int,
    ) -> slice:

        c_start, c_end = self.contig_offsets[[contig_idx, contig_idx + 1]]
        c_pos = self.zarr.variant_position[c_start:c_end]
        start_index, end_index = c_start + np.searchsorted(c_pos.values, [start, end])

        return slice(start_index, end_index)
