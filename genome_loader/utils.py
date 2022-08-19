import numpy as np
import pandas as pd
from numpy.typing import NDArray


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


def read_bed2(bed_file):
    """Read a BED2 file (i.e. just chrom & start) as a pandas.DataFrame"""
    bed = pd.read_csv(
        bed_file,
        header=None,
        names=["chrom", "start"],
        sep='\t',
        dtype={"chrom": 'category', "start": np.uint32},
    )
    return bed
