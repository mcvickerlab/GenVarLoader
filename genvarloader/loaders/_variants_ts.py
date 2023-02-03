import asyncio
import gc
from asyncio import Future
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union, cast

import numpy as np
import zarr
from numpy.typing import NDArray

from genvarloader.loaders import Queries
from genvarloader.loaders.types import _VCFTSDataset
from genvarloader.loaders.utils import ts_readonly_zarr
from genvarloader.types import PathType


class Variants:
    path: Path
    datasets: Dict[str, _VCFTSDataset]

    def __init__(self, zarrs: Union[Iterable[PathType], Dict[str, PathType]]) -> None:
        """Read batches of variants from Zarrs using specific regions and samples.

        Note that the variants should be exclusively SNPs (i.e. no MNPs, indels, etc.).

        Parameters
        ----------
        zarrs : iterable[path] or dict[str, path]
            A list of (or dictionary from sample IDs to) Zarr file paths that store VCFs for each sample.
            Whether a list or dict is used changes how queries can be made, since the sample IDs in the query
            will be matched against sample names in the loader. If using a list, sample IDs will be discovered
            from the Zarr files. Otherwise, they will be whatever sample ID is provided in the dictionary.
        """
        if isinstance(zarrs, dict):
            sample_ids = list(map(str, zarrs.keys()))
            paths = zarrs.values()
        else:
            sample_ids = None
            paths = zarrs
        self.datasets = {}
        for i, path in enumerate(map(Path, paths)):
            z = zarr.open_group(path, "r")
            call_genotype = ts_readonly_zarr(path / "call_genotype")
            variant_allele = ts_readonly_zarr(path / "variant_allele")
            variant_contig = ts_readonly_zarr(path / "variant_contig")
            variant_position = ts_readonly_zarr(path / "variant_position")
            contig_offsets = ts_readonly_zarr(path / "contig_offsets")
            if sample_ids is None:
                sample_id = cast(str, z["sample_id"][0])
            else:
                sample_id = sample_ids[i]
            self.datasets[sample_id] = _VCFTSDataset(
                call_genotype,
                variant_allele,
                variant_contig,
                variant_position,
                np.array(z.attrs["contigs"]),
                contig_offsets,
            )
            gc.collect()

    def contig_to_contig_idx(self, contig: str, sample: str) -> int:
        """Get the unique contig indices and array of mapped contig indices i.e. those used in the sgkit VCF Dataset.

        Example
        -------
        >>> variants = Variants('my/variants.zarr')
        >>> variants.attrs['contigs']
        array(['1', 'X'], dtype=object)
        >>> variants.contigs_to_contig_idxs(['X', 'X', 'X'])
        (array([1]), array([1, 1, 1]))
        """
        idx = np.flatnonzero(self.datasets[sample].contigs == contig)[0]
        return idx

    def sel(
        self,
        queries: Queries,
        length: int,
        **kwargs,
    ) -> Optional[
        tuple[NDArray[np.bytes_], NDArray[np.integer], NDArray[np.unsignedinteger]]
    ]:
        out = asyncio.run(self.async_sel(queries, length, **kwargs))
        return out

    async def async_sel(
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
        # get variants, their positions, and how many are in each query
        _queries = queries.reset_index(drop=True)
        groups = _queries.groupby(["contig", "sample"], sort=False)
        idx = np.concatenate([i for i in groups.indices.values()])

        variant_ls: List[Future[NDArray[np.uint8]]] = []
        position_ls: List[Future[NDArray[np.int32]]] = []
        count_ls: List[NDArray[np.unsignedinteger]] = []
        # NOTE: groupby preserves within-group order
        for (contig, sample), group in groups:
            c_idx = self.contig_to_contig_idx(contig, sample)
            s = group.start.to_numpy()
            e = s + length

            v_idx, p_idx, c_s_cnt = await self._intervals_to_idx(
                c_idx, s, e, sample, group.ploid_idx.to_numpy()  # type: ignore
            )

            allele_idx = self.datasets[sample].call_genotype[v_idx, p_idx]
            c_s_vars = self.datasets[sample].variant_allele[v_idx, allele_idx].read()
            c_s_var_pos = self.datasets[sample].variant_position[v_idx].read()

            variant_ls.append(c_s_vars)
            position_ls.append(c_s_var_pos)
            count_ls.append(c_s_cnt)

        # get counts
        variant_cnts: NDArray[np.unsignedinteger] = np.concatenate(count_ls).astype(
            "u4"
        )
        if sum(variant_cnts) == 0:
            return None

        # get variants, positions
        _variants = await asyncio.gather(*variant_ls)
        _positions = await asyncio.gather(*position_ls)
        variants = cast(NDArray[np.bytes_], np.concatenate(_variants).view("|S1"))
        # NOTE: VCF is 1-indexed, switch to 0-index
        positions = cast(NDArray[np.int32], np.concatenate(_positions) - 1)

        # resort
        resorter = np.argsort(idx)
        variant_cnts = variant_cnts[resorter]

        # get offsets
        offsets = np.zeros(len(variant_cnts) + 1, dtype=variant_cnts.dtype)
        variant_cnts.cumsum(out=offsets[1:])

        # resort
        split_variants = np.split(variants, offsets[1:])
        variants = np.concatenate([split_variants[i] for i in resorter])
        split_positions = np.split(positions, offsets[1:])
        positions = np.concatenate([split_positions[i] for i in resorter])

        return variants, positions, offsets

    async def _intervals_to_idx(
        self,
        contig_idx: int,
        starts: NDArray[np.integer],
        ends: NDArray[np.integer],
        sample: str,
        ploid_idx: NDArray[np.integer],
    ) -> Tuple[NDArray[np.integer], NDArray[np.integer], NDArray[np.unsignedinteger]]:

        c_start, c_end = (
            await self.datasets[sample]
            .contig_offsets[[contig_idx, contig_idx + 1]]
            .read()
        )
        c_pos = await self.datasets[sample].variant_position[c_start:c_end].read()
        # NOTE: VCF is 1-indexed and queries are 0-indexed, adjust
        s_e_idx = c_start + np.searchsorted(c_pos - 1, np.concatenate([starts, ends]))
        start_idxs, end_idxs = s_e_idx[: len(starts)], s_e_idx[len(starts) :]
        cnts: NDArray[np.unsignedinteger] = end_idxs - start_idxs
        v_idx = np.concatenate([np.r_[s:e] for s, e in zip(start_idxs, end_idxs)])
        p_idx = np.repeat(ploid_idx, cnts)
        return v_idx, p_idx, cnts
