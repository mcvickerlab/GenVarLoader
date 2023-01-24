import logging
from pathlib import Path
from subprocess import CalledProcessError, run
from textwrap import dedent
from typing import Optional, Union

import numba
import numpy as np
import polars as pl
import zarr
from numcodecs import Blosc, Categorize
from numcodecs.abc import Codec
from numpy.typing import ArrayLike, NDArray

PathType = Union[str, Path]
IndexType = Union[int, slice, list[int], NDArray[np.int_], NDArray[np.uint]]

_DNA = [nuc.encode() for nuc in "ACacgtGTN"]
_DNA_COMP = [nuc.encode() for nuc in "TGtgcaCAN"]
DNA_COMPLEMENT = dict(zip(_DNA, _DNA_COMP))
_RNA = [nuc.encode() for nuc in "ACacguGUN"]
_RNA_COMP = [nuc.encode() for nuc in "UGugcaCAN"]
RNA_COMPLEMENT = dict(zip(_RNA, _RNA_COMP))

ALPHABETS: dict[str, NDArray[np.bytes_]] = {
    "DNA": np.array(_DNA),
    "RNA": np.array(_RNA),
}


logger = logging.getLogger(__name__)


def bytes_to_ohe(
    arr: NDArray[np.bytes_], alphabet: NDArray[np.bytes_]
) -> NDArray[np.uint8]:
    alphabet_size = len(alphabet)
    idx = np.empty_like(arr, dtype="u8")
    for i, char in enumerate(alphabet):
        idx[arr == char] = np.uint64(i)
    # out shape: (length alphabet)
    return np.eye(alphabet_size, dtype="u1")[idx]


def ohe_to_bytes(
    ohe_arr: NDArray[np.uint8], alphabet: NDArray[np.bytes_], ohe_axis=-1
) -> NDArray[np.bytes_]:
    # ohe_arr shape: (... alphabet)
    idx = ohe_arr.nonzero()[-1]
    if ohe_axis < 0:
        ohe_axis_idx = len(ohe_arr.shape) + ohe_axis
    else:
        ohe_axis_idx = ohe_axis_idx
    shape = tuple(d for i, d in enumerate(ohe_arr.shape) if i != ohe_axis_idx)
    # (regs length samples ploidy)
    return alphabet[idx].reshape(shape)


def read_bed(
    bed_file: PathType, region_idx: Optional[IndexType] = None
) -> pl.LazyFrame:
    """Read a BED-like file as a polars.LazyFrame. Must have a header and
    at minimum have columns named "chrom" and "start". If column "strand" not provided,
    it will be added assuming all regions are on positive strand."""
    bed = pl.scan_csv(
        bed_file,
        sep="\t",
        dtypes={"chrom": pl.Utf8, "start": pl.Int32, "strand": pl.Utf8},
    )
    if region_idx is not None:
        bed = bed.with_row_count("index")
        if isinstance(region_idx, slice):
            region_idx = np.arange(
                region_idx.start, region_idx.stop, region_idx.step, dtype="u4"
            )
        region_srs = pl.DataFrame(region_idx, columns=["index"])  # type: ignore
        bed = bed.join(region_srs.lazy(), on="index").drop("index")  # type: ignore
    if "strand" not in bed.columns:
        bed = bed.with_columns(pl.lit("+").alias("strand"))
    return bed


def df_to_zarr(
    df: pl.DataFrame, z: zarr.Group, compressor: Codec = Blosc("lz4", shuffle=-1)
):
    """Write all columns of a DataFrame to a Zarr group."""
    z.attrs["columns"] = df.columns  # to maintain order
    for col in df.get_columns():
        filters = []
        if col.dtype in {pl.List, pl.Struct, pl.Date, pl.Datetime, pl.Time, pl.Object}:
            msg = f"""
                DataFrame has dtypes incompatible with Zarr.
                Column name: {col.name}
                Column dtype: {col.dtype}
                """
            raise TypeError(dedent(msg).strip())
        elif col.dtype == pl.Utf8:
            data = col.to_numpy().astype("U")
            uniq = np.unique(data)
            # heuristic for choosing whether to categorize
            if len(uniq) < np.sqrt(np.prod(data.shape)):
                nbytes = int(-(-len(uniq) // 8))  # ceil
                filters.append(Categorize(uniq, data.dtype, astype=f"u{nbytes}"))
        else:
            data = col.to_numpy()
        z.create_dataset(col.name, data=data, filters=filters, compressor=compressor)


def zarr_to_df(z: zarr.Group) -> pl.DataFrame:
    series_ls: list[pl.Series] = []
    name: str
    col: zarr.Array
    for name, col in z.arrays():
        series_ls.append(pl.Series(name, col[:]))
    return pl.DataFrame(series_ls).select(z.attrs["columns"])


def order_as(a1: ArrayLike, a2: ArrayLike) -> NDArray[np.uint32]:
    """Get indices that would order ar1 as ar2, assuming all elements of a1 are in a2."""
    _, idx1, idx2 = np.intersect1d(a1, a2, assume_unique=True, return_indices=True)
    return idx1[idx2].astype("u4")


def get_complement_idx(
    comp_dict: dict[bytes, bytes], alphabet: NDArray[np.bytes_]
) -> NDArray[np.uint32]:
    """Get index to reorder alphabet that would give the complement."""
    idx = order_as([comp_dict[nuc] for nuc in alphabet], alphabet)
    return idx


def rev_comp_byte(
    byte_arr: NDArray[np.bytes_], complement_map: dict[bytes, bytes]
) -> NDArray[np.bytes_]:
    """Get reverse complement of byte (string) array.

    Parameters
    ----------
    byte_arr : ndarray[bytes]
        Array of shape (regions [samples] [ploidy] length) to complement.
    complement_map : dict[bytes, bytes]
        Dictionary mapping nucleotides to their complements.
    """
    out = np.empty_like(byte_arr)
    for nuc, comp in complement_map.items():
        if nuc == b"N":
            continue
        out[byte_arr == nuc] = comp
    return out[..., ::-1]


def rev_comp_ohe(ohe_arr: NDArray[np.uint8], has_N: bool) -> NDArray[np.uint8]:
    if has_N:
        np.concatenate(
            [np.flip(ohe_arr[..., :-1], -1), ohe_arr[..., -1][..., None]],
            axis=-1,
            out=ohe_arr,
        )
    else:
        ohe_arr = np.flip(ohe_arr, -1)
    return np.flip(ohe_arr, -2)


def run_shell(args, **kwargs):
    try:
        status = run(dedent(args).strip(), check=True, shell=True, **kwargs)
    except CalledProcessError as e:
        logging.error(e.stdout)
        logging.error(e.stderr)
        raise e
    return status


def validate_sample_sheet(sample_sheet: pl.DataFrame, required_columns: list[str]):
    missing_columns = [
        col for col in required_columns if col not in sample_sheet.columns
    ]
    if len(missing_columns) > 0:
        raise ValueError("Sample sheet is missing required columns:", missing_columns)
    if sample_sheet.select(pl.col(required_columns).is_null()).to_numpy().any():
        raise ValueError("Sample sheet contains missing values.")
