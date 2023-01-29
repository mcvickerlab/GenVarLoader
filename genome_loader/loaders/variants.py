import gc
from abc import ABC, abstractmethod
from enum import Enum
from functools import reduce
from pathlib import Path
from typing import Optional, cast

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from genome_loader.loaders import Queries
from genome_loader.utils import PathType


class Variants:
    path: Path
    zarr: "Variants.SgkitDataset"

    class MissingValue(Enum):
        REFERENCE = "reference"
        N = "N"

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
            Variants.SgkitDataset,
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
        idxs = cast(NDArray[np.integer], samples.cat.codes.values)
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
        idxs = cast(NDArray[np.integer], contigs.cat.codes.values)
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
        sorted = kwargs.get("sorted", False)
        try:
            missing_value = self.MissingValue(kwargs.get("missing_value", "reference"))
        except ValueError:
            raise ValueError(
                "Keyword argument 'missing_value' must be 'reference' or 'N'."
            )

        if not isinstance(sorted, bool):
            raise TypeError("Keyword argument 'sorted' must be a bool.")

        sample_idx = self.samples_to_sample_idxs(queries["sample"])
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
                idx_ls.append(group.index.values)  # type: ignore

        # get counts
        variant_cnts: NDArray[np.unsignedinteger] = np.concatenate(count_ls)
        if sum(variant_cnts) == 0:
            return None

        # unsort
        if not sorted:
            idx = np.concatenate(idx_ls)  # type: ignore
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
