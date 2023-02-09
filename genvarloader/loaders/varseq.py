import asyncio
from typing import Optional, Tuple, Union, cast

import numba
import numpy as np
from icecream import ic
from numpy.typing import NDArray

from genvarloader.loaders import AsyncLoader
from genvarloader.loaders.types import Queries
from genvarloader.loaders.variants import Variants
from genvarloader.types import ALPHABETS, SequenceEncoding
from genvarloader.utils import bytes_to_ohe, rev_comp_byte


@numba.njit(nogil=True, parallel=True)
def apply_variants(
    seqs: NDArray[np.bytes_],
    starts: NDArray[np.integer],
    variants: NDArray[np.bytes_],
    positions: NDArray[np.integer],
    offsets: NDArray[np.unsignedinteger],
):
    # shapes:
    # seqs (i l)
    # starts (i)
    # variants (v)
    # positions (v)
    # offsets (i+1)

    for i in numba.prange(len(seqs)):
        i_vars = variants[offsets[i] : offsets[i + 1]]
        i_pos = positions[offsets[i] : offsets[i + 1]] - starts[i]
        seq = seqs[i]
        seq[i_pos] = i_vars


class VarSequence:
    def __init__(
        self,
        sequence: AsyncLoader,
        variants: Variants,
    ) -> None:
        self.sequence = sequence
        self.variants = variants

    def sel(
        self,
        queries: Queries,
        length: int,
        **kwargs,
    ) -> Union[NDArray[np.bytes_], NDArray[np.uint8]]:
        """Get sequences with sample's variants applied to them.

        Parameters
        ----------
        queries : Queries
        length : int
        **kwargs : dict
            encoding : 'bytes' or 'onehot', required
                How to encode the sequences.

        Returns
        -------
        seqs : ndarray[bytes | uint8]
            Sequences with variants applied to them.
        """
        seqs = asyncio.run(self.async_sel(queries, length, **kwargs))
        return seqs

    async def async_sel(
        self,
        queries: Queries,
        length: int,
        **kwargs,
    ) -> Union[NDArray[np.bytes_], NDArray[np.uint8]]:
        """Get sequences with sample's variants applied to them.

        Parameters
        ----------
        queries : Queries
        length : int
        **kwargs : dict
            sorted : bool, default False
            encoding : 'bytes' or 'onehot', required
                How to encode the sequences.

        Returns
        -------
        seqs : ndarray[bytes | uint8]
            Sequences with variants applied to them.
        """
        if "strand" not in queries:
            queries["strand"] = "+"
            queries["strand"] = queries.strand.astype("category")

        encoding = SequenceEncoding(kwargs.get("encoding"))
        positive_stranded_queries = cast(Queries, queries.assign(strand="+"))
        positive_stranded_queries["strand"] = positive_stranded_queries.strand.astype(
            "category"
        )

        # apply variants as bytes to reduce how much data is moving around
        # S1 is the same size as uint8
        res: Optional[
            Tuple[NDArray[np.bytes_], NDArray[np.integer], NDArray[np.unsignedinteger]]
        ]
        seqs, res = await asyncio.gather(
            *[
                self.sequence.async_sel(
                    positive_stranded_queries, length, encoding="bytes"
                ),
                self.variants.async_sel(queries, length),
            ]
        )
        seqs = cast(NDArray[np.bytes_], seqs)

        if res is not None:
            variants, positions, offsets = res
            apply_variants(seqs, queries.start.to_numpy(), variants, positions, offsets)
            # starts = queries.start.to_numpy()
            # for i in range(len(seqs)):
            #     i_vars = variants[offsets[i] : offsets[i + 1]]
            #     i_pos = positions[offsets[i] : offsets[i + 1]] - starts[i]
            #     seqs[i, i_pos] = i_vars

        to_rev_comp = cast(NDArray[np.bool_], (queries["strand"] == "-").values)
        if to_rev_comp.any():
            seqs[to_rev_comp] = rev_comp_byte(
                seqs[to_rev_comp], alphabet=ALPHABETS["DNA"]
            )

        if encoding is SequenceEncoding.ONEHOT:
            seqs = cast(NDArray[np.uint8], bytes_to_ohe(seqs, alphabet=ALPHABETS["DNA"]))  # type: ignore

        return seqs


# TODO: fix this so it works lol, important for benchmarking
'''class FastaVarSequence:
    def __init__(self, sample_sheet: PathType, encoding: Sequence.ENCODING) -> None:
        """Loader for obtaining variant sequences from fasta files that already have
        the variant sequences.

        Parameters
        ----------
        sample_sheet : str, Path
            A sample sheet mapping samples & haplotypes to fastas with variants applied to them.
        encoding : 'bytes' or 'onehot'
            Use 'bytes' to get an array of bytes and 'onehot' to get one hot encoded sequence.
        """
        _sample_sheet = pl.read_csv(sample_sheet)
        required_columns = ["sample", "haplotype", "variant_fasta"]
        validate_sample_sheet(_sample_sheet, required_columns)
        sample_haplotype = pl.concat_str(
            [pl.col("sample"), pl.col("haplotype")], sep="_"
        ).alias("sample_haplotype")
        _sample_sheet = _sample_sheet.with_column(sample_haplotype)
        self.sample_hap_to_varfa: dict[str, str] = dict(
            zip(_sample_sheet["sample_haplotype"], _sample_sheet["variant_fasta"])
        )
        self._dtype = np.uint8 if encoding == "onehot" else "|S1"
        self.encoding = encoding
        self.fasta_cache: dict[str, FastaSequence] = {}

    def sel(
        self,
        contigs: NDArray[np.str_],
        starts: NDArray[np.integer],
        length: int,
        strands: NDArray[np.str_],
        samples: NDArray[np.str_],
        ploid_idx: NDArray[np.integer],
        sorted=False,
    ) -> Union[NDArray[np.bytes_], NDArray[np.uint8]]:
        raise NotImplementedError
        # queries = pl.DataFrame(
        #     [
        #         pl.Series("contig", contigs, pl.Utf8),
        #         pl.Series("start", starts, pl.Int32),
        #         pl.Series("strand", strands, pl.Utf8),
        #         pl.Series("sample", samples, pl.Utf8),
        #         pl.Series("ploid_idx", ploid_idx, pl.UInt16),
        #     ]
        # ).with_row_count("idx")

        seqs = np.empty((len(contigs), length), dtype=self._dtype)

        npi.group_by()

        # each sample and haplotype is a different fasta, open each only once
        for group in queries.groupby(["sample", "ploid_idx"]):
            sample = group[0, "sample"]
            ploid_idx = group[0, "ploid_idx"]
            sample_contigs = group["contig"].to_numpy().astype("U")
            sample_starts = group["start"].to_numpy()
            sample_strands = group["strand"].to_numpy().astype("U")
            idx = group["idx"].to_numpy()
            s_p = f"{sample}_{ploid_idx}"
            fasta_path = self.sample_hap_to_varfa[s_p]
            if fasta_path not in self.fasta_cache:
                self.fasta_cache[fasta_path] = FastaSequence(fasta_path, self.encoding)
            fasta = self.fasta_cache[fasta_path]
            seqs[idx] = fasta.sel(sample_contigs, sample_starts, length, sample_strands)

        gc.collect()
        return seqs

    def close(self):
        for fasta in self.fasta_cache.values():
            fasta.fasta.close()

    def get_torch_collator(self, queries: PathType, length: int):
        if not _TORCH_AVAILABLE:
            raise ImportError("Using torch collators requires PyTorch.")
        return self.TorchCollator(self, queries, length)

    class TorchCollator:
        def __init__(
            self, fastavarseq: FastaVarSequence, queries: PathType, length: int
        ) -> None:
            self.fastavarseq = fastavarseq
            self.queries = parse_queries(queries)
            self.length = length

        def __call__(self, batch_indices: list[int]):
            batch = self.queries[batch_indices]
            contigs = batch["contig"].to_numpy().astype("U")
            starts = batch["start"].to_numpy()
            strands = batch["strand"].to_numpy().astype("U")
            samples = batch["sample"].to_numpy().astype("U")
            ploid_idx = batch["ploid_idx"].to_numpy()
            seqs = self.fastavarseq.sel(
                contigs, starts, self.length, strands, samples, ploid_idx
            )
            out = {"seqs": torch.as_tensor(seqs)}
            other_series = batch.select(
                pl.exclude(["contig", "start", "strand", "sample", "ploid_idx"])
            ).to_dict(as_series=True)
            other_cols = {
                k: torch.as_tensor(v.to_numpy()) for k, v in other_series.items()
            }
            out.update(other_cols)
            return out
'''
