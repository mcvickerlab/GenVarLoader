import numpy as np
import polars as pl
import pyarrow as pa
from attrs import define
from seqpro._ragged import Ragged


@define
class RaggedAlleles(Ragged[np.bytes_]):
    """Ragged array of alleles.

    Create RaggedAlleles from a polars Series of strings:
    >>> alleles = RaggedAlleles.from_polars(pl.Series(["A", "AC", "G"]))

    Create RaggedAlleles from offsets and alleles:
    >>> offsets = np.array([0, 1, 3, 4], np.uint64)
    >>> alleles = np.frombuffer(b"AACG", "|S1")
    >>> alleles = RaggedAlleles.from_offsets(alleles, alleles)
    """

    @classmethod
    def from_polars(cls, alleles: pl.Series):
        offsets = np.empty(len(alleles) + 1, np.int64)
        offsets[0] = 0
        offsets[1:] = alleles.str.len_bytes().cast(pl.Int64).cum_sum().to_numpy()
        flat_alleles = np.frombuffer(alleles.str.join().to_numpy()[0].encode(), "S1")
        shape = len(alleles)
        return cls.from_offsets(flat_alleles, shape, offsets)

    def to_polars(self):
        n_alleles = len(self)
        offset_buffer = pa.py_buffer(self.offsets)
        allele_buffer = pa.py_buffer(self.data)
        string_arr = pa.LargeStringArray.from_buffers(
            n_alleles, offset_buffer, allele_buffer
        )
        return pl.Series(string_arr)
