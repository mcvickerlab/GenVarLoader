from pathlib import Path
from textwrap import dedent
from typing import Union

import numpy as np
import polars as pl
import zarr
from numcodecs import Blosc, Categorize
from numcodecs.abc import Codec
from numpy.typing import ArrayLike, NDArray

PathType = Union[str, Path]
IndexType = Union[int, slice, NDArray[np.int_], NDArray[np.uint]]

_DNA = [nuc.encode() for nuc in "ACGTacgtN"]
_DNA_COMP = [nuc.encode() for nuc in "CATGcatgN"]
DNA_COMPLEMENT = dict(zip(_DNA, _DNA_COMP))
_RNA = [nuc.encode() for nuc in "ACGUacguN"]
_RNA_COMP = [nuc.encode() for nuc in "CAUGcaugN"]
RNA_COMPLEMENT = dict(zip(_RNA, _RNA_COMP))


def bytes_to_ohe(
    arr: NDArray[np.byte], alphabet: NDArray[np.byte]
) -> NDArray[np.uint8]:
    alphabet_size = len(alphabet)
    idx = np.empty_like(arr, dtype="u8")
    for i, char in enumerate(alphabet):
        idx[arr == char] = np.uint64(i)
    return np.eye(alphabet_size, dtype="u1")[idx]


def ohe_to_bytes(
    ohe_arr: NDArray[np.uint8], alphabet: NDArray[np.byte], ohe_axis=-1
) -> NDArray[np.byte]:
    idx = ohe_arr.nonzero()[-1]
    if ohe_axis < 0:
        ohe_axis_idx = len(ohe_arr.shape) + ohe_axis
    else:
        ohe_axis_idx = ohe_axis_idx
    shape = tuple(d for i, d in enumerate(ohe_arr.shape) if i != ohe_axis_idx)
    # (regs length samples ploidy)
    return alphabet[idx].reshape(shape)


def read_bed(bed_file: PathType):
    """Read a BED-like file as a polars.DataFrame. Must have a header and
    at minimum have columns named "chrom" and "start". If column "strand" not provided,
    it will be added assuming all regions are on positive strand."""
    bed = pl.read_csv(
        bed_file,
        sep="\t",
        dtype={"chrom": pl.Utf8, "start": pl.Int32, "strand": pl.Utf8},
    )
    if "strand" not in bed.columns:
        bed = bed.with_columns(pl.lit(np.repeat("+", len(bed))).alias("strand"))
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
    idx1, idx2 = np.intersect1d(a1, a2, assume_unique=True, return_indices=True)
    return idx1[idx2].astype("u4")


def get_complement_idx(
    comp_dict: dict[bytes, bytes], alphabet: NDArray[np.bytes_]
) -> NDArray[np.uint32]:
    """Get index to reorder alphabet that would give the complement."""
    idx = order_as([comp_dict[nuc] for nuc in alphabet], alphabet)
    return idx.astype("u4")


def rev_comp_byte(byte_arr: NDArray[np.bytes_], complement_map: dict[bytes, bytes]):
    """Get reverse complement of byte (string) array.

    Parameters
    ----------
    byte_arr : ndarray[bytes]
        Array of shape (regions length [samples] [ploidy] alphabet) to complement.
    complement_map : dict[bytes, bytes]
        Dictionary mapping nucleotides to their complements.
    """
    out = np.empty_like(byte_arr)
    for nuc, comp in complement_map.items():
        out[byte_arr == nuc] = comp
    return out


def rev_comp_ohe(
    ohe_arr: NDArray[np.uint8], complement_idx: NDArray[np.uint32]
) -> NDArray[np.uint8]:
    """Get reverse complement of onehot or probabilistic encoded array.

    Parameters
    ----------
    ohe_arr : ndarray[uint8]
        Array of shape (regions length [samples] [ploidy] alphabet) to complement.
    complement_idx : ndarray[uint32]
        Index specifying how to reorder elements of alphabet to get complement.
    """
    if len(ohe_arr.shape) == 3:
        return ohe_arr[:, ::-1, complement_idx]
    elif len(ohe_arr.shape) == 5:
        return ohe_arr[:, ::-1, :, :, complement_idx]
    else:
        raise ValueError(
            f"Input array has unexpected shape: {ohe_arr.shape}. Expected either 3-d or 5-d array."
        )
