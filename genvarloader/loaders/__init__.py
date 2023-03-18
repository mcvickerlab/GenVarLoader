import asyncio
from typing import Any, Callable, Dict, List, Optional, Union, cast

import numpy as np
import pandas as pd
import zarr
from numpy.typing import NDArray
from typing_extensions import assert_never

from genvarloader.loaders.coverage import Coverage, bin_coverage
from genvarloader.loaders.sequence import Sequence
from genvarloader.loaders.types import AsyncLoader, Loader, Queries
from genvarloader.loaders.utils import read_narrowpeak_as_queries, read_queries
from genvarloader.loaders.variants import Variants
from genvarloader.loaders.varseq import VarSequence
from genvarloader.types import PathType

try:
    from genvarloader.torch import SortedQuerySampler, TorchCollator
except ImportError as e:
    _TORCH_AVAILABLE = False
else:
    _TORCH_AVAILABLE = True


__all__ = [
    "Coverage",
    "Sequence",
    "Variants",
    "VarSequence",
    "bin_coverage",
    "read_queries",
    "read_narrowpeak_as_queries",
]


class GenVarLoader:
    def __init__(
        self,
        loaders: Dict[str, Union[Loader, AsyncLoader]],
        transforms: Optional[Dict[str, Callable[[NDArray], NDArray]]] = None,
    ) -> None:
        """Wrap multiple loaders to call them all at once with queries and bundle their outputs into a dictionary.

        Parameters
        ----------
        loaders : dict[str, Loader | AsyncLoader]
            A dictionary naming each loader that will be queried by the GenVarLoader.
        transforms : dict[str, (LoaderOutput) -> LoaderOutput], optional
            A dictionary of transformations (functions) to be applied to loader outputs, mapped by dictionary keys.

        Usage
        -----
        Suppose we have three loaders we're interested in and a set of queries:

        ```
        from glob import glob
        import genvarloaders.loaders as gvl

        sequence = gvl.Sequence('my/sequence.zarr')
        variants = gvl.Variants(glob('directory/of/variants/*.zarr'))
        coverage = gvl.Coverage('my/coverage.zarr')

        queries = gvl.read_queries('my/queries.csv')
        length = 600 # each query covers 600 bp
        encoding = 'bytes' # we want the nucleotides strings rather than e.g. one-hot encodings
        ```

        Instead of querying each of them individually, we can wrap them all with a `GenVarLoader` object to query
        them more conveniently:

        ```
        gvloader = gvl.GenVarLoader({
            'sequence': sequence,
            'variants': variants,
            'coverage': coverage
        })

        output = gvloader.sel(queries, length=length, encoding=encoding)
        ```

        This will give us an output that looks like:

        ```
        {
            'sequence': # NDArray with shape (n_queries length),
            'variants_alleles': # NDArray with shape (total_n_variants),
            'variants_positions': # NDArray with shape (total_n_variants),
            'variants_offsets': # NDArray with shape (n_queries),
            'coverage': # NDArray with shape (n_queries length)
        }
        ```
        """
        self.loaders = loaders
        self.async_loaders = {
            k: cast(AsyncLoader, v)
            for k, v in loaders.items()
            if hasattr(v, "async_sel")
        }
        self.sync_loaders = {
            k: cast(Loader, v)
            for k, v in loaders.items()
            if k not in self.async_loaders
        }
        self.transforms = {} if transforms is None else transforms

    def sel(self, queries: Queries, length: int, **kwargs) -> Dict[str, NDArray]:
        out = asyncio.run(self.async_sel(queries, length, **kwargs))
        return out

    async def async_sel(
        self, queries: Queries, length: int, **kwargs
    ) -> Dict[str, NDArray]:

        # get sync output
        out: Dict[str, NDArray] = {}
        for name, loader in self.sync_loaders.items():
            val = loader.sel(queries, length, **kwargs)
            if isinstance(val, np.ndarray):
                out[name] = val
            elif isinstance(val, dict):
                out.update({f"{name}_{k}": v for k, v in val.items()})
            else:
                assert_never(val)

        # get async output
        out_ls: List[Union[NDArray, Dict[str, NDArray]]] = await asyncio.gather(
            *[
                l.async_sel(queries, length, **kwargs)
                for l in self.async_loaders.values()
            ]
        )

        # add async output to output dictionary
        for name, val in zip(self.async_loaders.keys(), out_ls):
            if isinstance(val, np.ndarray):
                out[name] = val
            elif isinstance(val, dict):
                out.update({f"{name}_{k}": v for k, v in val.items()})
            else:
                assert_never(val)

        # apply transforms to output
        for name, val in out.items():
            if name in self.transforms:
                out[name] = self.transforms[name](val)

        return out

    def get_torch_collator(
        self,
        length: int,
        sel_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "TorchCollator":
        if not _TORCH_AVAILABLE:
            raise ImportError("Creating a torch collator requires PyTorch.")
        return TorchCollator(self, length, sel_kwargs)

    def get_sorted_query_sampler(
        self, queries_path: PathType, batch_size: int, seed: Optional[int] = None
    ) -> "SortedQuerySampler":
        if not _TORCH_AVAILABLE:
            raise ImportError("Creating a torch collator requires PyTorch.")

        queries = read_queries(queries_path)

        # reading variants is more of a bottleneck than reading sequences or coverage
        has_variants = any(
            isinstance(l, (Variants, VarSequence)) for l in self.loaders.values()
        )
        has_coverage = any(isinstance(l, Coverage) for l in self.loaders.values())

        # sort by sample *then* contig to optimize read performance for variants or variant sequences
        if has_variants or has_coverage:
            if "sample" not in queries.columns:
                raise ValueError(
                    "This GenVarLoader needs a sample column in the queries since it has one or more Variant or Coverage loaders."
                )
            sort_order = ["sample", "contig"]
        # sort by contig *then* sample to optimize read performance for sequence or coverage
        elif "sample" in queries.columns:
            sort_order = ["contig", "sample"]
        else:
            sort_order = ["contig"]

        return SortedQuerySampler(queries, batch_size, sort_order, seed)

    def write_queries(
        self,
        out_path: PathType,
        queries: Queries,
        length: int,
        sel_kwargs: Optional[Dict[str, Any]] = None,
        batch_size=1028,
    ):
        """Write queries to an AnnData compatible Zarr file. The file will contain datasets named corresponding to the GenVarLoader.

        Parameters
        ----------
        out_path : str, Path
        queries_path : str, Path
        length : int
        sel_kwargs : dict[str, Any], optional
            kwargs to pass to sel(), by default None
        batch_size : int, optional
            Number of queries to write at a time to limit memory requirements, by default 1028. Set this to be as large as possible.
        """
        if sel_kwargs is None:
            sel_kwargs = {}

        # reading variants is more of a bottleneck than reading sequences or coverage
        has_variants = any(
            isinstance(l, (Variants, VarSequence)) for l in self.loaders.values()
        )
        has_coverage = any(isinstance(l, Coverage) for l in self.loaders.values())

        # sort by sample *then* contig to optimize read performance for variants or variant sequences
        if has_variants or has_coverage:
            if "sample" not in queries.columns:
                raise ValueError(
                    "This GenVarLoader needs a sample column in the queries since it has one or more Variant or Coverage loaders."
                )
            sort_cols = ["sample", "contig"]
        # sort by contig *then* sample to optimize read performance for sequence or coverage
        elif "sample" in queries.columns:
            sort_cols = ["contig", "sample"]
        else:
            sort_cols = ["contig"]
        queries = queries.sort_values(sort_cols)  # type: ignore

        # split into batches
        n_equal_sized_batches, mod = divmod(len(queries), batch_size)
        end_equal_sized_batches = len(queries) - mod
        batches = np.split(
            queries.iloc[:end_equal_sized_batches], n_equal_sized_batches
        )
        if end_equal_sized_batches != len(queries):
            batches.append(queries.iloc[end_equal_sized_batches:])  # type: ignore
        batches = cast(List[Queries], batches)

        # write batches
        out_zarr = zarr.open_group(str(out_path))
        for batch in batches:
            out = self.sel(batch, length, **sel_kwargs)
            for k, v in out.items():
                if k not in out_zarr.array_keys():
                    out_zarr.create_dataset(k, shape=(len(queries), *v.shape[1:]))
                out_zarr[k][batch.index.to_numpy()] = v

    def write_queries_to_anndata(
        self,
        out_path: PathType,
        queries: Queries,
        length: int,
        X_feature: str,
        ohe: bool = True,
        sel_kwargs: Optional[Dict[str, Any]] = None,
        batch_size=1028,
    ):
        """Write queries to an AnnData compliant Zarr file.

        This GenVarLoader must have a gvl.Sequence or gvl.VarSequence available for X. This feature should
        then be specified as the argument to X. Other features will be put in obsm with corresponding names.

        Parameters
        ----------
        out_path : str, Path
        queries_path : str, Path
        length : int
        X : str
            Which feature to use as X. Must a gvl.Sequence or gvl.VarSequence.
        ohe : bool
            Whether to also include an OHE sequence in `obsm`.
        sel_kwargs : dict[str, Any], optional
            kwargs to pass to sel(), by default None
        batch_size : int, optional
            Number of queries to write at a time to limit memory requirements, by default 1028. Set this to be as large as possible.
        """
        X_loader = self.loaders[X_feature]
        assert isinstance(X_loader, (Sequence, VarSequence))

        if sel_kwargs is None:
            sel_kwargs = {"encoding": "bytes"}

        # setup AnnData Zarr
        array_attrs = {"encoding-type": "array", "encoding-version": "0.2.0"}
        compressor = zarr.Blosc("zstd", clevel=7, shuffle=-1)

        # init root
        root = zarr.open_group(str(out_path), mode="w-")
        root.attrs["encoding-type"] = "anndata"
        root.attrs["encoding-version"] = "0.1.0"

        # init X
        arr = root.require_dataset(
            "X",
            shape=(len(queries), length),
            chunks=(batch_size, None),
            dtype="|S1",
            compressor=compressor,
        )
        arr.attrs.update(array_attrs)

        # init obs, var, and obsm
        # obsm will hold all features other than the X_feature
        obs, var, obsm = root.require_groups(["obs", "var", "obsm"])
        if ohe:
            alphabet_len = len(X_loader.alphabet.array)
            arr = obsm.require_dataset(
                "X_ohe",
                shape=(len(queries), length, alphabet_len),
                chunks=(batch_size, None, None),
                dtype="u1",
                compressor=compressor,
            )
            arr.attrs.update(array_attrs)

        # set obs to be queries
        obs.attrs.update(
            {
                "_index": "index",
                "column-order": queries.columns.to_numpy(object),
                "encoding-type": "dataframe",
                "encoding-version": "0.2.0",
            }
        )
        arr = obs.create_dataset(
            "index", data=queries.index.to_numpy(), compressor=compressor
        )
        arr.attrs.update(array_attrs)
        val: pd.Series
        for col, val in queries.to_dict("series").items():
            arr = obs.create_dataset(col, data=val.to_numpy(), compressor=compressor)
            arr.attrs.update(array_attrs)

        # set var to be an index of size equal to length
        var.attrs.update(
            {
                "_index": "index",
                "column-order": np.array([], dtype=object),
                "encoding-type": "dataframe",
                "encoding-version": "0.2.0",
            }
        )
        arr = var.create_dataset(
            "index", data=np.arange(length, np.uint64), compressor=compressor
        )
        arr.attrs.update(array_attrs)

        # reading variants is more of a bottleneck than reading sequences or coverage
        has_variants = any(
            isinstance(l, (Variants, VarSequence)) for l in self.loaders.values()
        )
        has_coverage = any(isinstance(l, Coverage) for l in self.loaders.values())

        # sort by sample *then* contig to optimize read performance for variants or variant sequences
        if has_variants or has_coverage:
            if "sample" not in queries.columns:
                raise ValueError(
                    "This GenVarLoader needs a sample column in the queries since it has one or more Variant or Coverage loaders."
                )
            sort_cols = ["sample", "contig"]
        # sort by contig *then* sample to optimize read performance for sequence or coverage
        elif "sample" in queries.columns:
            sort_cols = ["contig", "sample"]
        else:
            sort_cols = ["contig"]
        queries = queries.sort_values(sort_cols)  # type: ignore

        # split into batches
        n_equal_sized_batches, mod = divmod(len(queries), batch_size)
        end_equal_sized_batches = len(queries) - mod
        batches = np.split(
            queries.iloc[:end_equal_sized_batches], n_equal_sized_batches
        )
        if end_equal_sized_batches != len(queries):
            batches.append(queries.iloc[end_equal_sized_batches:])  # type: ignore
        batches = cast(List[Queries], batches)

        # write batches
        for batch in batches:
            batch_idx = batch.index.to_numpy()

            out = self.sel(batch, length, **sel_kwargs)
            root["X"][batch_idx] = out.pop(X_feature)

            if ohe:
                X_ohe = X_loader.sel(batch, length, encoding="onehot")
                obsm["X_ohe"][batch_idx] = X_ohe

            for k, v in out.items():
                if k not in obsm.array_keys():
                    # create the dataset for the feature if it doesn't exist
                    arr = obsm.create_dataset(
                        k,
                        shape=(len(queries), *v.shape[1:]),
                        chunks=(batch_size, *[None] * (v.ndim - 1)),
                        dtype=v.dtype,
                        compressor=compressor,
                    )
                    arr.attrs.update(array_attrs)
                obsm[k][batch_idx] = v
