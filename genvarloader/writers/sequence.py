import gc
import logging
import math
from time import perf_counter
from typing import Optional, Set, Union

import numpy as np
import zarr
from numcodecs import Blosc
from pysam import FastaFile

from genvarloader.types import PathType, SequenceAlphabet, SequenceEncoding
from genvarloader.utils import bytes_to_ohe


def fasta_to_zarr(
    fasta_path: PathType,
    out_path: PathType,
    alphabet: SequenceAlphabet,
    encodings: Union[SequenceEncoding, Set[SequenceEncoding]],
    contigs: Optional[Set[str]] = None,
    ignore_case: bool = False,
    compression_level: int = 0,
):

    if not isinstance(encodings, set):
        encodings = {encodings}
    if len(encodings) == 0:
        raise ValueError("Need at least one encoding.")

    logging.info("Starting to write Zarr from FASTA.")

    zrr = zarr.open_group(str(out_path))
    for encoding in encodings:
        zrr.create_group(encoding.value)
    zrr.attrs["alphabet"] = alphabet.string
    compressor = Blosc(clevel=compression_level)

    with FastaFile(str(fasta_path)) as fasta:

        # check contigs and write their lengths
        contigs_available = fasta.references
        if contigs is None:
            _contigs = contigs_available
        else:
            unknown_contigs = contigs - set(contigs_available)
            if len(unknown_contigs) > 0:
                raise ValueError("Got contigs not in FASTA:", unknown_contigs)
            _contigs = contigs
        zrr.attrs["lengths"] = {c: fasta.get_reference_length(c) for c in _contigs}

        t_start_io = perf_counter()

        for contig in _contigs:
            t_start_contig = perf_counter()
            seq = fasta.fetch(contig)
            if ignore_case:
                seq = seq.upper()
            seq_arr = np.frombuffer(seq.encode(), "|S1").copy()
            seq_arr[~np.isin(seq_arr, alphabet.array)] = b"N"
            if SequenceEncoding.BYTES in encodings:
                zrr["bytes"].create_dataset(  # type: ignore
                    contig,
                    data=seq_arr.view("u1"),
                    compressor=compressor,
                )
            if SequenceEncoding.ONEHOT in encodings:
                ohe_arr = bytes_to_ohe(seq_arr, alphabet)
                zrr["onehot"].create_dataset(  # type: ignore
                    contig,
                    data=ohe_arr,
                    compressor=compressor,
                )
                del ohe_arr
            del seq_arr
            logging.info(
                f"Wrote contig {contig} in {perf_counter() - t_start_contig:.2f} seconds."
            )
            gc.collect()

    zarr.consolidate_metadata(str(out_path))

    logging.info(f"Finished in {perf_counter() - t_start_io:.2f} seconds.")
