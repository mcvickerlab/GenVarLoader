from __future__ import annotations

import logging
import warnings
from pathlib import Path
from textwrap import dedent
from typing import Optional, Union

import h5py
import numpy as np
import polars as pl
from numpy.typing import NDArray

from genome_loader.utils import (
    DNA_COMPLEMENT,
    IndexType,
    PathType,
    read_bed,
    rev_comp_byte,
    rev_comp_ohe,
)

logger = logging.getLogger(__name__)


class ReferenceGenomeLoader:
    """A reference genome."""

    embedding: str
    ref_genome_path: PathType
    contigs: NDArray[np.str_]
    spec: Optional[NDArray[np.bytes_]] = None

    def __init__(self, ref_genome_h5: PathType) -> None:
        """A reference genome with optional VCF for getting consensus sequences."""

        self.ref_genome_path = Path(ref_genome_h5)

        with self._open() as ref_genome:
            self.embedding: str = ref_genome.attrs["id"]
            logger.info(f"Genome is {self.embedding} embedded.")
            self._ref_dtype = ref_genome[next(iter(ref_genome.keys()))][
                self.embedding
            ].dtype
            self.contigs = np.array(list(ref_genome.keys()), dtype="U")

            if ref_genome.attrs.get("encode_spec", None) is not None:
                self.spec = ref_genome.attrs["encode_spec"].astype("S")
                assert self.spec is not None
                self._check_reverse_is_complement(self.spec, DNA_COMPLEMENT)  # type: ignore

    def _open(self):
        return h5py.File(self.ref_genome_path)

    def _check_reverse_is_complement(
        self, alphabet: NDArray[np.bytes_], complement_map: dict[bytes, bytes]
    ):
        _alphabet = alphabet[:-1] if b"N" in alphabet else alphabet
        rev_alpha = _alphabet[::-1]
        for a, r in zip(_alphabet, rev_alpha):
            if complement_map[a] != r:
                raise ValueError("Reverse of alphabet does not yield the complement.")

    def __repr__(self) -> str:
        out = dedent(
            f"""
            GenomeLoader with...
            reference genome: {self.ref_genome_path}
            embedding: {self.embedding}
            """
        ).strip()
        if self.embedding == "onehot":
            assert self.spec is not None
            out += f"\nalphabet: {''.join(self.spec.astype('U'))}"
        return out

    def sel(
        self,
        chroms: NDArray[np.str_],
        starts: NDArray[np.int32],
        length: Union[int, np.uint32],
        strands: Optional[NDArray[np.str_]] = None,
        sorted_chroms: bool = False,
        pad_val: Union[bytes, str] = "N",
    ) -> Union[NDArray[np.bytes_], NDArray[np.uint8]]:
        """Select regions from genome using uniform length bed coordinates.

        The type of embedding depends on the HDF5 genome e.g. sequence (bytes), onehot. See `self.embedding`.
        Pass `sample` to get consensus (variant-aware) haplotype sequences, otherwise reference sequence is returned.

        Parameters
        ----------
        chroms : ndarray[str]
            Chromosomes for each region.
            Shape: (regions)
        starts : ndarray[uint32]
            Start position of each region. Should be genomic coordinates i.e. 1-indexed.
            Note that `starts` will be cast to uint32 if input is not already.
            Shape: (regions)
        length : uint32
            Length of all regions. A scalar value.
        strands: ndarray[str], default None
            Strand of query regions, if negative stranded will return reverse complement.
            If None, assumes all regions are on positive strand.
        sorted_chroms : bool, default False
            Whether query chromosomes are already sorted.
        pad_val : str, default "N"
            A single character to pad out-of-bound regions by.

        Returns
        -------
        ndarray[bytes | uint8]
            Array of regions. The "alphabet" dimension will be present if the reference is
            onehot encoded.
            Shape: (regions length [alphabet])

        Examples
        --------
        >>> gl = ReferenceGenomeLoader(ref_file_ohe_acgtn)
        >>> chroms = np.array(['21', '20', '20'])
        >>> starts = np.arange(3, dtype='u4')
        >>> length = np.uint32(5)
        >>> ref_seqs = gl.sel(chroms, starts, length)
        >>> ref_seqs.shape
        (3, 5, 5)
        """
        # input normalization
        if starts.dtype.type != np.int32:
            starts = starts.astype("i4")
            warnings.warn("Starts dtype was not int32, casting.")
        if not isinstance(length, np.uint32):
            length = np.uint32(length)
            warnings.warn("Length dtype was not np.uint32, casting.")
        ends = starts + length

        # get pad value
        if isinstance(pad_val, str):
            pad_val = bytes(pad_val, "utf-8")
        if self.embedding == "sequence":
            pad_arr = np.array([pad_val], dtype="|S1")
        elif self.embedding == "onehot":
            assert self.spec is not None
            pad_arr = np.zeros_like(self.spec, dtype="u1")
            if pad_val not in self.spec:
                warnings.warn("Pad value is not in spec, will pad with 0 vector.")
            else:
                pad_arr[self.spec == pad_val] = np.uint8(1)
            pad_arr = pad_arr[None, :]

        logger.info("Validating ref genome args")
        self._sel_validate_ref_genome_args(chroms, length)

        if strands is None:
            strands_flags: NDArray[np.int8] = np.ones(len(chroms), dtype=np.int8)
        else:
            strands_flags = np.ones_like(strands, dtype=np.int8)
            strands_flags[strands == "-"] = np.int8(-1)

        with self._open() as ref_genome:
            logger.info("Slicing GenomeLoader")
            out = self._sel_slice(
                chroms,
                starts,
                ends,
                strands_flags,
                length,
                ref_genome,
                pad_arr,
                sorted_chroms,
            )

        return out

    def _sel_validate_ref_genome_args(
        self,
        chroms: NDArray[np.str_],
        length: np.uint32,
    ):
        if length < 1:
            raise ValueError("Length must be greater than 0.")
        if not np.isin(chroms, self.contigs).all():
            raise ValueError("Chromosome not in reference genome.")

    def _sel_slice(
        self,
        chroms: NDArray[np.str_],
        starts: NDArray[np.int32],
        ends: NDArray[np.int32],
        strands_flags: NDArray[np.int8],
        length: np.uint32,
        ref_genome: h5py.File,
        pad_arr: Union[NDArray[np.bytes_], NDArray[np.uint8]],
        sorted_chroms: bool = False,
    ) -> Union[NDArray[np.bytes_], NDArray[np.uint8]]:
        # requires low level wrangling with variable length arrays
        # current implementation relies on numba but awkward array may provide even more speed
        # see experiments in sbox.ipynb with awkward arrays
        # - David Laub

        # TODO: consider opportunities to speed up if there are many different contigs
        if not sorted_chroms:
            sort_idx = chroms.argsort()
            chroms = chroms[sort_idx]
            starts = starts[sort_idx]
            ends = ends[sort_idx]
        out_ls = []  # will be sorted if input chroms are sorted
        logger.info(f"Unique chroms: {np.unique(chroms)}")
        for chrom in np.unique(chroms):  # guaranteed to proceed in a sorted order
            logger.info(f"Slicing chrom: {chrom}")
            chrom_starts = starts[chroms == chrom]
            chrom_ends = ends[chroms == chrom]
            ref_chrom, ref_start, rel_starts = self._sel_padded_min_ref_chrom(
                ref_genome[chrom], chrom_starts, chrom_ends, pad_arr
            )
            coords = (
                np.tile(np.arange(length, dtype=np.uint32), (len(chrom_starts), 1))
                + rel_starts[:, None]
            )
            chrom_out = ref_chrom[coords]
            out_ls.append(chrom_out)
        sorted_out = np.concatenate(out_ls)

        # rearrange into original order
        if not sorted_chroms:
            out = sorted_out[sort_idx.argsort()]
        else:
            out = sorted_out

        logger.info("Reverse complementing")
        if self.embedding == "sequence":
            out[strands_flags == -1] = rev_comp_byte(
                out[strands_flags == -1], DNA_COMPLEMENT
            )
        elif self.embedding == "onehot":
            assert self.spec is not None
            out[strands_flags == -1] = rev_comp_ohe(
                out[strands_flags == -1], b"N" in self.spec
            )

        return out

    def _sel_padded_min_ref_chrom(
        self,
        ref_chrom_h5: h5py.Group,
        starts: NDArray[np.int32],
        ends: NDArray[np.int32],
        pad_arr: Union[NDArray[np.bytes_], NDArray[np.uint8]],
    ):
        ref_start: int = starts.min()
        ref_end: int = ends.max()
        rel_starts: NDArray[np.int32] = starts - ref_start

        # pad for out-of-bound
        chrom_length: int = ref_chrom_h5.attrs["length"]
        n_past_chrom = max(ref_end - chrom_length, 0)
        n_before_chrom = max(-ref_start, 0)
        real_ref_end = min(ref_end, chrom_length)
        real_ref_start = max(ref_start, 0)
        # ref_chrom shape: (length) or (length alphabet)
        # pad_val shape: (1) or (alphabet)
        ref_chrom: Union[NDArray[np.bytes_], NDArray[np.uint8]]
        ref_chrom = ref_chrom_h5[self.embedding][real_ref_start:real_ref_end]
        ref_chrom = np.concatenate(
            [
                pad_arr.repeat(n_before_chrom, 0),
                ref_chrom,
                pad_arr.repeat(n_past_chrom, 0),
            ],
            axis=0,
        )
        return ref_chrom, ref_start, rel_starts

    def sel_from_bed(
        self,
        bed: Union[pl.DataFrame, PathType],
        length: int,
        region_idx: Optional[IndexType] = None,
        pad_val: Union[bytes, str] = "N",
        sorted_chroms: bool = False,
    ) -> Union[NDArray[np.bytes_], NDArray[np.uint8]]:
        """Get sequences specified by a bed file.

        Parameters
        ----------
        bed_file : PathType
            BED-like file defining at minimum the chromosome, start, and strand of each region.
            Must have a header defining column names.
        length : int
            length of all regions
        region_idx : int, slice, list[int], ndarray[int]
            index of regions from bed_file to write i.e. select a subset of regions
        samples : Optional[list[str]], optional
            list of samples to get consensus regions for, otherwise gets all, defaults to None
        sorted_chroms : bool, default false
            Whether query chromosomes are sorted.
        cache : bool, default false
            Whether to cache the bed file in memory. Avoids reading the bed again when sequentially fetching bed regions.

        Returns
        -------
        ndarray[bytes | uint8]
            Array of regions. The "alphabet" dimension will be present if the reference is
            onehot encoded.
            Shape: (regions length [alphabet])
        """

        if isinstance(bed, str) or isinstance(bed, Path):
            _bed: pl.DataFrame = read_bed(bed, region_idx).collect()
        else:
            _bed = bed

        chroms: NDArray[np.str_] = _bed["chrom"].to_numpy().astype("U")
        starts: NDArray[np.int32] = _bed["start"].to_numpy()
        strands: NDArray[np.str_] = _bed["strand"].to_numpy().astype("U")

        out = self.sel(
            chroms,
            starts,
            np.uint32(length),
            strands,
            sorted_chroms,
            pad_val,
        )

        return out
