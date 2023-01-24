from __future__ import annotations

import gc
from typing import Union, cast

import numba
import numpy as np
from numpy.typing import NDArray

from genome_loader.gloader.experimental import Queries
from genome_loader.gloader.experimental.sequence import FastaSequence, Sequence
from genome_loader.gloader.experimental.variants import Variants, VCFVariants
from genome_loader.utils import (
    ALPHABETS,
    DNA_COMPLEMENT,
    PathType,
    bytes_to_ohe,
    rev_comp_byte,
    run_shell,
    validate_sample_sheet,
)


class VarSequence:
    def __init__(
        self,
        sequence: Sequence,
        variants: Variants,
        missing_value: Variants.MISSING_VALUE = "reference",
    ) -> None:
        self.sequence = sequence
        self.variants = variants
        self.missing_value = missing_value

    def sel(
        self,
        queries: Queries,
        length: int,
        **kwargs,
    ) -> Union[NDArray[np.bytes_], NDArray[np.uint8]]:
        sorted = kwargs.get("sorted", False)
        # apply variants to sequences to reduce how much data is moving around
        # S1 is the same size as uint8
        old_encoding = self.sequence.encoding
        self.sequence.encoding = "bytes"
        positive_stranded_queries = cast(Queries, queries.assign(strand="+"))

        seqs = self.sequence.sel(positive_stranded_queries, length, sorted=sorted)
        res = self.variants.sel(
            queries, length, missing_value=self.missing_value, sorted=sorted
        )

        if res is not None:
            variants, positions, offsets = res
            apply_variants(seqs, queries.start.to_numpy(), variants, positions, offsets)

        rev_comp_idx = np.flatnonzero(queries.strand == "-")
        if len(rev_comp_idx) > 0:
            seqs[rev_comp_idx] = rev_comp_byte(
                seqs[rev_comp_idx], complement_map=DNA_COMPLEMENT
            )

        if old_encoding == "onehot":
            seqs = bytes_to_ohe(seqs, alphabet=ALPHABETS["DNA"])  # type: ignore

        self.sequence.encoding = old_encoding
        gc.collect()
        return seqs

    def sel_pyfaidx(self, contigs, starts, length, strands, samples, ploid_idx):
        # TODO: use pyfaidx.FastaVariant here
        # Mostly for benchmarking, I expect it to be slow since it iterates through each
        # allele of each sample and haplotype in a Python for loop
        raise NotImplementedError

    def sel_bcftools(self, contigs, starts, length, strands, samples, ploid_idx):
        if not (
            isinstance(self.sequence, FastaSequence)
            and isinstance(self.variants, VCFVariants)
        ):
            raise TypeError(
                "Reference and variant loaders must be Fasta and VCF to use the bcftools implementation."
            )
        region_strs = (
            f"{chrom}:{start+1}-{start+length+1}"
            for chrom, start in zip(contigs, starts)
        )
        _seqs = []
        rev_comp_idx = []
        for i, (region_str, strand, sample, ploid) in enumerate(
            zip(region_strs, strands, samples, ploid_idx)
        ):
            vcf_path = self.variants.sample_to_vcf[sample]
            cmd = f"samtools faidx {self.sequence.path} {region_str} | bcftools consensus -H {ploid+1} {vcf_path}"
            status = run_shell(cmd, capture_output=True)
            seq = "".join(status.stdout.split("\n")[1:])
            if strand == "-":
                rev_comp_idx.append(i)
            _seqs.append(seq)

        seqs = np.array(_seqs, dtype="|S1")

        rev_comp_idx = np.array(rev_comp_idx)
        seqs[rev_comp_idx] = rev_comp_byte(seqs[rev_comp_idx], DNA_COMPLEMENT)

        if self.sequence.encoding == "onehot":
            seqs = bytes_to_ohe(seqs, alphabet=ALPHABETS["DNA"])

        return seqs


@numba.njit(nogil=True, parallel=True)
def apply_variants(
    seqs: NDArray[np.bytes_],
    starts: NDArray[np.integer],
    variants: NDArray[np.bytes_],
    positions: NDArray[np.integer],
    offsets: NDArray[np.unsignedinteger],
):
    # seqs (i l)
    # starts (i)
    # variants (v)
    # positions (v)
    # offsets (i+1)

    for i in numba.prange(len(seqs)):
        i_vars = variants[offsets[i] : offsets[i + 1]]
        i_pos = positions[offsets[i] : offsets[i + 1]] - starts[i]
        seqs[i, i_pos] = i_vars


# TODO: fix this so this works lol, important for benchmarking
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
