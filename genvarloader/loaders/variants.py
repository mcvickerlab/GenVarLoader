import asyncio
from asyncio import Future
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import polars as pl
import tiledbvcf
import zarr
from numpy.typing import NDArray
from typing_extensions import Self

from genvarloader.loaders.types import AsyncLoader, Loader, LoaderOutput, _TStore
from genvarloader.loaders.utils import ts_readonly_zarr
from genvarloader.types import PathType


class _VCFTSDataset:
    """A Zarr dataset for a single sample using TensorStore for I/O."""

    sample_id: str
    call_genotype: _TStore[np.int8]  # (v s p)
    variant_allele: _TStore[np.uint8]  # (v a)
    variant_contig: _TStore[np.int16]  # (v)
    variant_position: zarr.Group  # (v)
    contig_offsets: _TStore[np.integer]  # (c)
    contig_idx: Dict[str, int]
    contig_offset_idx: Dict[str, int]

    def __init__(self) -> None:
        self._initalized = False

    @classmethod
    async def create(cls, path: Path, ts_kwargs: Dict):
        self = cls()

        z = cast(zarr.Group, zarr.open_consolidated(str(path), mode="r"))

        self.sample_id = cast(str, z.attrs["sample_id"])

        # We have to eagerly read all the positions for a contig downstream
        # so there's no need to make tensorstores here.
        self.variant_position = cast(zarr.Group, z["variant_position"])

        # open tensorstores
        gvl_array_names = {
            "call_genotype",
            "contig_offsets",
            "variant_allele",
            "variant_contig",
        }
        arrays = [
            ts_readonly_zarr(path.resolve() / n, **ts_kwargs) for n in gvl_array_names
        ]
        gvl_arrays = await asyncio.gather(*arrays)
        self.call_genotype = gvl_arrays[0]
        self.contig_offsets = gvl_arrays[1]
        self.variant_allele = gvl_arrays[2]
        self.variant_contig = gvl_arrays[3]

        self.contig_idx = z.attrs["contig_idx"]
        self.contig_offset_idx = z.attrs["contig_offset_idx"]

        self._initalized = True

        return self


class Variants(AsyncLoader):
    """Loader for getting variants from Zarrs by querying specific regions and samples.

    NOTE:
    - Only supports SNPs (i.e. no MNPs, indels, etc.)
    - Doesn't use standard initialization i.e. `Variants(...)`
        Instead, use `Variants.create(...)` or the async version, `Variants.async_create(...)`
        to get an initialized instance. This is to support async (faster) initialization. If
        using this in a Jupyter notebook, you'll need to call `nest_asyncio.apply()` since
        Jupyter runs an async loop.
    """

    paths: Dict[str, Path]
    datasets: Dict[str, _VCFTSDataset]
    samples: List[str]

    def __init__(self) -> None:
        """Variants uses async initialization. Use Variants.create or Variants.async_create to
        create an instance."""
        self._initialized = False

    @classmethod
    def create(
        cls,
        zarrs: Iterable[PathType],
        sample_ids: Optional[List[str]] = None,
        ts_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Self:
        """Create a Variants instance asynchronously.

        Parameters
        ----------
        zarrs : iterable[path]
            A list of file paths to Variant Zarrs.
        sample_ids : list[str], optional
            A list of sample IDs corresponding to each Variant Zarr. If none given,
            sample IDs will be taken from the Variant Zarrs.
        ts_kwargs : dict[str, any]
            Keyword arguments to pass to tensorstore.open(). Useful to specify a shared cache pool across
            loaders, for example.

        Returns
        -------
        self : Variants
        """

        self = asyncio.run(cls.async_create(zarrs, sample_ids, ts_kwargs))
        return self

    @classmethod
    async def async_create(
        cls,
        zarrs: Iterable[PathType],
        sample_ids: Optional[List[str]] = None,
        ts_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Self:
        """Create a Variants instance asynchronously.

        Parameters
        ----------
        zarrs : iterable[path]
            A list of file paths to Variant Zarrs.
        sample_ids : list[str], optional
            A list of sample IDs corresponding to each Variant Zarr. If none given,
            sample IDs will be taken from the Variant Zarrs.
        ts_kwargs : dict[str, any]
            Keyword arguments to pass to tensorstore.open(). Useful to specify a shared cache pool across
            loaders, for example.

        Returns
        -------
        self : Variants
        """

        self = cls()
        paths = list(map(Path, zarrs))

        if ts_kwargs is None:
            ts_kwargs = {}

        dataset_coros = [_VCFTSDataset.create(path, ts_kwargs) for path in paths]
        datasets: List[_VCFTSDataset] = await asyncio.gather(*dataset_coros)

        if sample_ids is None:
            sample_ids = [d.sample_id for d in datasets]

        self.datasets = dict(zip(sample_ids, datasets))
        self.paths = {s: p for s, p in zip(sample_ids, paths)}
        self.samples = list(self.datasets.keys())
        self._initialized = True

        return self

    def sel(
        self,
        queries: pd.DataFrame,
        length: int,
        **kwargs,
    ) -> Dict[str, NDArray]:
        """Get the variants for specified regions, if any, otherwise return None.

        Parameters
        ----------
        queries : pd.DataFrame
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
        queries: pd.DataFrame,
        length: int,
        **kwargs,
    ) -> Dict[str, NDArray]:
        """Get the variants for specified regions, if any, otherwise return None.

        Parameters
        ----------
        queries : pd.DataFrame
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
        if not self._initialized:
            raise RuntimeError(
                "This Variants instance is uninitialized. Did you remember to use Variants.create(...)?"
            )
        # get variants, their positions, and how many are in each query
        queries = cast(pd.DataFrame, queries.reset_index(drop=True))
        groups = queries.groupby(["sample", "contig"], sort=False)

        allele_ls: List[Future[NDArray]] = []
        position_ls: List[NDArray[np.int32]] = []
        idx_ls: List[NDArray[np.integer]] = []
        count_ls: List[NDArray[np.integer]] = []
        count_idx_ls: List[NDArray[np.integer]] = []
        # NOTE: groupby preserves within-group order
        sample: str
        contig: str
        for (sample, contig), group in groups:
            s = group.start.to_numpy()
            e = s + length

            a_idx, p_idx, c_s_counts, c_s_positions = self._intervals_to_idx_and_pos(
                contig, s, e, sample, group.ploid_idx.to_numpy()
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
        contig: str,
        starts: NDArray[np.integer],
        ends: NDArray[np.integer],
        sample: str,
        ploid_idx: NDArray[np.integer],
    ) -> Tuple[
        NDArray[np.integer], NDArray[np.integer], NDArray[np.integer], NDArray[np.int32]
    ]:
        c_idx = self.datasets[sample].contig_idx[contig]
        c_pos = self.datasets[sample].variant_position.get(c_idx, None)
        if c_pos is None:
            return (
                np.array([], np.int32),
                np.array([], ploid_idx.dtype),
                np.full_like(starts, 0, np.int32),
                np.array([], np.int32),
            )
        c_pos = cast(NDArray[np.int32], c_pos[:])
        # NOTE: VCF is 1-indexed and queries are 0-indexed, adjust
        c_pos -= 1
        c_offset_idx = self.datasets[sample].contig_offset_idx[contig]
        c_start = self.datasets[sample].contig_offsets[c_offset_idx].read().result()
        s_e_idx = c_start + np.searchsorted(c_pos, np.concatenate([starts, ends]))
        start_idxs, end_idxs = np.split(s_e_idx, (len(starts),))
        cnts = end_idxs - start_idxs
        v_idx = np.concatenate([np.arange(s, e) for s, e in zip(start_idxs, end_idxs)])
        p_idx = np.repeat(ploid_idx, cnts)
        v_pos = c_pos[v_idx - c_start]
        return v_idx, p_idx, cnts, v_pos


class TileDBVariants(Loader):
    path: Path
    dataset: tiledbvcf.Dataset

    def __init__(self, tdb_path: PathType) -> None:
        self.path = Path(tdb_path)
        self.dataset = tiledbvcf.Dataset(tdb_path)

    def sel(self, queries: pd.DataFrame, length: int, **kwargs) -> LoaderOutput:
        pl_queries = (
            pl.from_pandas(queries)
            .with_columns(
                pl.col(pl.Int64).cast(pl.Int32), pl.col(pl.Categorical).cast(pl.Utf8)
            )
            .with_row_count()
        )

        # get the unique region strings from the queries
        regions = (
            pl_queries.lazy()
            .select("contig", "start")
            .unique()
            .with_columns(
                pl.col("start").clip_min(0) + 1,
                (pl.col("start").clip_min(0) + length).alias("end"),
            )
            .select(
                pl.concat_str(
                    ["contig", pl.lit(":"), "start", pl.lit("-"), "end"]
                ).alias("region")
            )
            .collect()["region"]
            .to_numpy()
        )

        # get the unique samples from the queries
        samples = pl_queries["sample"].unique()

        var_data = cast(
            pl.DataFrame,
            pl.from_arrow(
                self.dataset.read_arrow(
                    attrs=[
                        "contig",
                        "pos_start",
                        "alleles",
                        "sample_name",
                        "query_bed_start",
                    ],
                    samples=samples,
                    regions=regions,
                )
            ),
        )

        joined = (
            var_data
            # join against queries to get alleles & postiions for each query
            .join(
                pl_queries,
                left_on=["contig", "query_bed_start", "sample_name"],
                right_on=["contig", "start", "sample"],
            ).with_columns(
                # pick out the allele of interest for each query
                pl.col("alleles").arr.get(pl.col("ploid_idx"))
            )
        )

        # get alleles, positions, and offsets
        alleles = joined["alleles"].to_numpy().astype("|S1")
        positions = joined["pos_start"].to_numpy()
        counts = joined.groupby(["row_nr"]).count()["count"].to_numpy()
        offsets = np.zeros(len(counts) + 1, dtype=counts.dtype)
        counts.cumsum(out=offsets[1:])

        return {"alleles": alleles, "positions": positions, "offsets": offsets}
