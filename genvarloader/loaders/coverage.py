"""Load & work with coverage data.

On-disk format for coverage data:

/
├── <contig 1> (samples length) uint16
├── <contig 2> (samples length) uint16
├── <contig ...> (samples length) uint16
├── read_count (samples) uint64
└── attrs (dictionary)
   ├── feature (str) what feature is this? depth? tn5? some other per-base feature?
   ├── samples (list[str])
   ├── contig_lengths (dict[str, int])
   └── ... extra items that may be relevant for a given feature
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Union, cast

import einops as ein
import numpy as np
import zarr
from numpy.typing import NDArray

from genvarloader.loaders.types import Queries, _TStore
from genvarloader.loaders.utils import ts_readonly_zarr
from genvarloader.types import PathType

_NORMALIZATION_METHODS = ["cpm"]


def cpm_normalization(
    counts: NDArray[np.uint16], total_counts: NDArray[np.uint64]
) -> NDArray[np.float64]:
    # (samples length) / (samples 1)
    norm_counts = counts * 1e6 / total_counts[:, None]
    return norm_counts


class Coverage:
    path: Path
    feature: str
    samples: NDArray[np.str_]

    def __init__(self, zarr_path: PathType) -> None:
        self.path = Path(zarr_path)
        root = cast(zarr.Group, zarr.open_consolidated(str(self.path), mode="r"))

        self.contig_lengths: Dict[str, int] = root.attrs["contig_lengths"]

        # Iterate over arrays under the group and grab them as TensorStores
        self.tstores: Dict[str, _TStore] = {}

        def add_array_to_tstores(p: str, val: Union[zarr.Group, zarr.Array]):
            if isinstance(val, zarr.Array) and p != "read_count":
                self.tstores[p] = ts_readonly_zarr(self.path / p).result()

        root.visititems(add_array_to_tstores)

        self._root = root
        self.samples = np.array(root.attrs["samples"])
        self._argsort_samples = self.samples.argsort()
        self._sorted_samples = self.samples[self._argsort_samples]
        self.feature = root.attrs["feature"]
        self.attrs = {
            k: v
            for k, v in root.attrs.items()
            if k not in ["samples", "features", "contig_lengths"]
        }

    def samples_to_sample_idx(self, samples: NDArray) -> NDArray[np.intp]:
        return self._argsort_samples[np.searchsorted(self._sorted_samples, samples)]

    def sel(
        self, queries: Queries, length: int, **kwargs
    ) -> NDArray[Union[np.uint16, np.float64]]:
        """Select coverage from a coverage Zarr i.e. read depth per base pair.

        Parameters
        ----------
        queries : Queries
        length : int
        **kwargs
            normalization : str
                Whether to apply normalization to the coverage. Can be None or "cpm".
                CPM normalization = counts * 10^9 / total_reads

        Returns
        -------
        ndarray[uint8 | float64]
            Coverage for the queries, perhaps normalized.
        """
        out = asyncio.run(self.async_sel(queries, length, **kwargs))
        return out

    async def async_sel(
        self, queries: Queries, length: int, **kwargs
    ) -> NDArray[Union[np.uint16, np.float64]]:
        """Select coverage from a coverage Zarr i.e. read depth per base pair.

        Parameters
        ----------
        queries : Queries
        length : int
        **kwargs
            normalization : str
                Whether to apply normalization to the coverage. Can be None or "cpm".
                CPM normalization = counts * 10^9 / total_reads

        Returns
        -------
        ndarray[uint16 | float64]
            Coverage for the queries, perhaps normalized.
        """

        out_shape = [len(queries), length]

        queries = cast(Queries, queries.reset_index(drop=True))

        queries["end"] = queries.start + length
        # map negative starts to 0
        queries["in_start"] = queries.start.clip(lower=0)
        # map ends > contig length to contig length
        queries["contig_length"] = queries.contig.replace(self.contig_lengths).to_numpy(
            np.int64
        )
        queries["in_end"] = np.minimum(queries.end, queries.contig_length)
        # get start, end index in output array
        queries["out_start"] = queries.in_start - queries.start
        queries["out_end"] = queries.in_end - queries.in_start
        queries["sample_idx"] = self.samples_to_sample_idx(queries["sample"].values)  # type: ignore

        def get_read(query):
            contig = query.contig
            sample_idx = query.sample_idx
            return self.tstores[contig][
                sample_idx, query.in_start : query.in_end
            ].read()

        # (q l)
        reads: List[NDArray] = await asyncio.gather(
            *[get_read(query) for query in queries.itertuples()]
        )

        # init array that will pad out-of-bound sequences
        out = np.zeros(out_shape, np.uint16)
        for i, (read, query) in enumerate(zip(reads, queries.itertuples())):
            # (1 l) = (l)
            out[i, query.out_start : query.out_end] = read

        # normalize counts
        normalization = kwargs.get("normalization", None)
        if normalization is not None and normalization not in _NORMALIZATION_METHODS:
            raise ValueError(
                "Got unrecognized normalization method. Should be one of:",
                _NORMALIZATION_METHODS,
            )
        elif normalization == "cpm":
            # total counts for each sample, respecting duplicate samples in query
            total_counts = cast(
                NDArray, self._root["read_count"][queries["sample_idx"].to_numpy()]
            )
            out = cpm_normalization(counts=out, total_counts=total_counts)

        # reverse negative stranded queries
        to_rev_comp = cast(NDArray[np.bool_], (queries["strand"] == "-").values)
        out[to_rev_comp] = np.flip(out[to_rev_comp], axis=-1)

        return out


def bin_coverage(coverage_array: NDArray, bin_width: int, normalize=False) -> NDArray:
    """Bin coverage by summing over non-overlapping windows.

    Parameters
    ----------
    coverage_array : ndarray
    bin_width : int
        Width of the windows to sum over. Must be an even divisor of the length
        of the coverage array. If not, raises an error. The length dimension is
        assumed to be the second dimension.
    normalize : bool, default False
        Whether to normalize by the length of the bin.

    Returns
    -------
    binned_coverage : ndarray
    """
    # coverage array (sample length [alphabet])
    length = coverage_array.shape[1]
    if length % bin_width != 0:
        raise ValueError("Bin width must evenly divide length.")
    binned_coverage = np.add.reduceat(
        coverage_array, np.arange(0, length, bin_width), axis=1
    )
    if normalize:
        binned_coverage /= bin_width
    return binned_coverage
