import logging
from pathlib import Path
from subprocess import CalledProcessError, run
from textwrap import dedent
from typing import Union

import numpy as np
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
        ohe_axis_idx = ohe_axis
    shape = tuple(dim for i, dim in enumerate(ohe_arr.shape) if i != ohe_axis_idx)
    # (regs samples ploidy length)
    return alphabet[idx].reshape(shape)


def order_as(a1: ArrayLike, a2: ArrayLike) -> NDArray[np.integer]:
    """Get indices that would order ar1 as ar2, assuming all elements of a1 are in a2."""
    _, idx1, idx2 = np.intersect1d(a1, a2, assume_unique=True, return_indices=True)
    return idx1[idx2]


def get_complement_idx(
    comp_dict: dict[bytes, bytes], alphabet: NDArray[np.bytes_]
) -> NDArray[np.integer]:
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


def run_shell(cmd: str, **kwargs):
    try:
        status = run(dedent(cmd).strip(), check=True, shell=True, **kwargs)
    except CalledProcessError as e:
        logging.error(e.stdout)
        logging.error(e.stderr)
        raise e
    return status
