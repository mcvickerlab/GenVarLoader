import asyncio
from asyncio import Future
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, cast

import numpy as np
import zarr
from numpy.typing import NDArray
from typing_extensions import Self

from genvarloader.loaders.types import Queries, _TStore, _VCFTSDataset
from genvarloader.loaders.utils import ts_readonly_zarr
from genvarloader.types import PathType


class Variants:
    """Loader for getting variants from Zarrs by querying specific regions and samples.

    NOTE:
    - Only supports SNPs (i.e. no MNPs, indels, etc.)
    - Doesn't use standard initialization i.e. `Variants(...)`
        Instead, use `Variants.create(...)` or the async version, `Variants.async_create(...)`
        to get an initialized instance. This is to support async (faster) initialization. If
        using this in a Jupyter notebook, you'll have to use `Variants.async_create(...)` since
        Jupyter runs an async loop.
    """

    paths: Dict[str, Path]
    datasets: Dict[str, _VCFTSDataset]

    def __init__(self) -> None:
        """Variants uses async initialization. Use Variants.create or Variants.async_create to
        create an instance."""
        pass

    @classmethod
    def create(
        cls,
        zarrs: Union[Iterable[PathType], Dict[str, PathType]],
        ts_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Self:
        """Create a Variants instance.

        Parameters
        ----------
        zarrs : iterable[path] or dict[str, path]
            A list of (or dictionary from sample IDs to) Zarr file paths that store VCFs for each sample.
            Whether a list or dict is used changes how queries can be made, since the sample IDs in the query
            will be matched against sample names in the loader. If using a list, sample IDs will be discovered
            from the Zarr files. Otherwise, they will be whatever sample ID is provided in the dictionary.

        Returns
        -------
        self : Variants
        """
        self = asyncio.run(cls.async_create(zarrs, ts_kwargs))
        return self

    @classmethod
    async def async_create(
        cls,
        zarrs: Union[Iterable[PathType], Dict[str, PathType]],
        ts_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Self:
        """Create a Variants instance asynchronously.

        Parameters
        ----------
        zarrs : iterable[path] or dict[str, path]
            A list of (or dictionary from sample IDs to) Zarr file paths that store VCFs for each sample.
            Whether a list or dict is used changes how queries can be made, since the sample IDs in the query
            will be matched against sample names in the loader. If using a list, sample IDs will be discovered
            from the Zarr files. Otherwise, they will be whatever sample ID is provided in the dictionary.
        ts_kwargs : dict[str, any]
            Keyword arguments to pass to tensorstore.open(). Useful e.g. to specify a shared cache pool across
            loaders.

        Returns
        -------
        self : Variants
        """
        self = cls()
        if isinstance(zarrs, dict):
            set_sample_ids = False
            sample_ids = list(map(str, zarrs.keys()))
            paths = zarrs.values()
        else:
            set_sample_ids = True
            sample_ids = []
            paths = zarrs

        if ts_kwargs is None:
            ts_kwargs = {}

        gvl_array_names = {
            "call_genotype",
            "contig_offsets",
            "variant_allele",
            "variant_contig",
        }

        gvl_array_futures: List[Future[List[_TStore]]] = []
        variant_positions: List[zarr.Group] = []
        contigs_ls: List[NDArray[np.object_]] = []

        for path in map(Path, paths):
            z = zarr.open_group(path, "r")

            arrays = [
                ts_readonly_zarr(path.resolve() / n, **ts_kwargs)
                for n in gvl_array_names
            ]
            variant_positions.append(z["variant_position"])  # type: ignore

            if set_sample_ids:
                sample_ids.append(cast(str, z["sample_id"][0]))

            gvl_array_futures.append(asyncio.gather(*arrays))

            contigs_ls.append(np.array(z.attrs["contigs"], dtype=object))

        gvl_arrays: List[List[_TStore]] = await asyncio.gather(*gvl_array_futures)

        self.datasets = {}
        for sample_id, gvl_arr, var_pos, contigs in zip(
            sample_ids, gvl_arrays, variant_positions, contigs_ls
        ):
            ds_kwargs = dict(zip(gvl_array_names, gvl_arr))
            ds_kwargs["variant_position"] = var_pos
            ds_kwargs["contigs"] = contigs
            ds = _VCFTSDataset(**ds_kwargs)
            self.datasets[sample_id] = ds

        self.paths = {s: p for s, p in zip(sample_ids, map(Path, paths))}
        return self

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
        _idx = np.flatnonzero(self.datasets[sample].contigs == contig)
        if len(_idx) == 0:
            raise RuntimeError(
                f"Query contig {contig} not in variant file for sample {sample}."
            )
        idx = _idx[0]
        return idx

    def sel(
        self,
        queries: Queries,
        length: int,
        **kwargs,
    ) -> Dict[str, NDArray]:
        """Get the variants for specified regions, if any, otherwise return None.

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
        out = asyncio.run(self.async_sel(queries, length, **kwargs))
        return out

    async def async_sel(
        self,
        queries: Queries,
        length: int,
        **kwargs,
    ) -> Dict[str, NDArray]:
        """Get the variants for specified regions, if any, otherwise return None.

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
        if not hasattr(self, "datasets"):
            raise RuntimeError(
                "Variants appears to be uninitialized. Did you remember to use Variants.create(...)?"
            )
        # get variants, their positions, and how many are in each query
        queries = cast(Queries, queries.reset_index(drop=True))
        groups = queries.groupby(["contig", "sample"], sort=False)

        allele_ls = []
        position_ls = []
        idx_ls = []
        count_ls = []
        count_idx_ls = []
        # # NOTE: groupby preserves within-group order
        for (contig, sample), group in groups:
            c_idx = self.contig_to_contig_idx(contig, sample)
            s = group.start.to_numpy()
            e = s + length

            a_idx, p_idx, c_s_counts, c_s_positions = self._intervals_to_idx_and_pos(
                c_idx, s, e, sample, group.ploid_idx.values  # type: ignore
            )

            allele_idx = self.datasets[sample].call_genotype[a_idx, p_idx]
            c_s_alleles = self.datasets[sample].variant_allele[a_idx, allele_idx].read()

            allele_ls.append(c_s_alleles)
            position_ls.append(c_s_positions)
            idx_ls.append(group.index.values.repeat(c_s_counts))
            count_ls.append(c_s_counts)
            count_idx_ls.append(group.index.values)

        # get counts
        counts = np.concatenate(count_ls)
        if counts.sum() == 0:
            return {
                "alleles": np.array([]),
                "positions": np.array([]),
                "offsets": np.zeros(len(queries) + 1, "i4"),
            }

        # get variants, positions and resort
        count_idx = np.concatenate(count_idx_ls)
        count_resorter = np.argsort(count_idx)
        counts = cast(NDArray[np.integer], counts[count_resorter])

        idx = np.concatenate(idx_ls)
        resorter = np.argsort(idx, kind="stable")
        _alleles = await asyncio.gather(*allele_ls)
        alleles = cast(
            NDArray[np.bytes_], np.concatenate(_alleles).view("|S1")[resorter]
        )
        positions = cast(NDArray[np.int32], np.concatenate(position_ls)[resorter])

        # get offsets
        offsets = np.zeros(len(counts) + 1, dtype=counts.dtype)
        counts.cumsum(out=offsets[1:])

        return {"alleles": alleles, "positions": positions, "offsets": offsets}

    def _intervals_to_idx_and_pos(
        self,
        contig_idx: int,
        starts: NDArray[np.integer],
        ends: NDArray[np.integer],
        sample: str,
        ploid_idx: NDArray[np.integer],
    ) -> Tuple[
        NDArray[np.integer], NDArray[np.integer], NDArray[np.integer], NDArray[np.int32]
    ]:

        c_pos = self.datasets[sample].variant_position.get(contig_idx, None)
        if c_pos is None:
            return (
                np.array([], np.int32),
                np.array([], ploid_idx.dtype),
                np.full_like(starts, 0, np.int32),
                np.array([], np.int32),
            )
        c_pos = cast(NDArray[np.int32], c_pos[:])
        c_start = self.datasets[sample].contig_offsets[contig_idx].read().result()
        # NOTE: VCF is 1-indexed and queries are 0-indexed, adjust
        c_pos -= 1
        s_e_idx = c_start + np.searchsorted(c_pos, np.concatenate([starts, ends]))
        start_idxs, end_idxs = np.split(s_e_idx, (len(starts),))
        cnts = end_idxs - start_idxs
        v_idx = np.concatenate([np.arange(s, e) for s, e in zip(start_idxs, end_idxs)])
        p_idx = np.repeat(ploid_idx, cnts)
        v_pos = c_pos[v_idx - c_start]
        return v_idx, p_idx, cnts, v_pos
