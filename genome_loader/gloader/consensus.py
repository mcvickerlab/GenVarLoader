from __future__ import annotations

import logging
import warnings
from pathlib import Path
from textwrap import dedent
from typing import Dict, Optional, Union

import dask
import dask.array as da
import einops as ein
import h5py
import numba
import numpy as np
import polars as pl
import xarray as xr
import zarr
from numcodecs import Blosc
from numcodecs.abc import Codec
from numpy.typing import NDArray
from tqdm import tqdm

from genome_loader.utils import (
    DNA_COMPLEMENT,
    IndexType,
    PathType,
    df_to_zarr,
    get_complement_idx,
    order_as,
    read_bed,
    rev_comp_byte,
    rev_comp_ohe,
    zarr_to_df,
)        


class ConsensusGenomeLoader:
    """A reference genome with optional VCF for getting consensus sequences."""

    embedding: str
    ref_genome_path: PathType
    ref_contigs: NDArray[np.str_]
    vcf_path: PathType
    spec: Optional[NDArray[np.bytes_]] = None

    def __init__(self, ref_genome_h5: PathType, vcf_zarr: PathType) -> None:
        """A reference genome with optional VCF for getting consensus sequences."""

        self.ref_genome_path = Path(ref_genome_h5)

        with self._open() as ref_genome:
            self.embedding: str = ref_genome.attrs["id"]
            logging.info(f"Genome is {self.embedding} embedded.")
            self._ref_dtype = ref_genome[next(iter(ref_genome.keys()))][
                self.embedding
            ].dtype
            self.ref_contigs = np.array(list(ref_genome.keys()), dtype="U")

            if ref_genome.attrs.get("encode_spec", None) is not None:
                self.spec = ref_genome.attrs["encode_spec"].astype("S")
                assert self.spec is not None
                self._check_reverse_is_complement(self.spec, DNA_COMPLEMENT)  # type: ignore
                if b"N" not in self.spec and vcf_zarr is not None:
                    warnings.warn(
                        "N is not in the reference genome alphabet so unknown genotypes will default to match the reference."
                    )
                self.complement_idx = get_complement_idx(DNA_COMPLEMENT, self.spec)

        self.vcf_path = vcf_zarr
        self._vcf = xr.open_dataset(
            vcf_zarr,
            engine="zarr",
            chunks={},
            concat_characters=False,
            drop_variables=[
                "variant_filter",
                "variant_id",
                "variant_id_mask",
                "variant_quality",
                "call_genotype_mask",
                "call_genotype_phased",
            ],
        )

        ref_contigs, _, ref_to_vcf_contig_idx = np.intersect1d(
            self.ref_contigs,
            self._vcf.attrs["contigs"],
            assume_unique=True,
            return_indices=True,
        )

        # change to 0-index
        self._vcf["variant_position"] = self._vcf["variant_position"] - np.int32(1)
        self._vcf["variant_allele"] = self._vcf["variant_allele"].astype("S")
        # map contig name to index in var_contig_idx
        self._contig_str_to_vcf_contig_idx: Dict[str, int] = dict(
            zip(ref_contigs, ref_to_vcf_contig_idx)
        )
        # variant contig index: where contig by index first occurs in sorted list of variants
        self._var_contig_idx: NDArray[np.int64] = np.searchsorted(
            self._vcf.variant_contig.load(), np.arange(len(self._vcf.attrs["contigs"]))
        )

    def _open(self):
        return h5py.File(self.ref_genome_path)

    def _check_reverse_is_complement(
        self, alphabet: NDArray[np.bytes_], complement_map: dict[bytes, bytes]
    ):
        if b"N" in alphabet:
            _alphabet = alphabet[:-1]
        else:
            _alphabet = alphabet
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
        out += f"\nVCF: {self.vcf_path}"
        return out

    def sel(
        self,
        contigs: NDArray[np.str_],
        starts: NDArray[np.int32],
        length: Union[int, np.uint32],
        samples: Optional[
            Union[list[str], NDArray[np.str_], NDArray[np.uint32]]
        ] = None,
        strands: Optional[NDArray[np.str_]] = None,
        sorted_contigs: bool = False,
        pad_val: Union[bytes, str] = "N",
        ploid_idx: Optional[NDArray[np.uint32]] = None,
    ) -> Union[NDArray[np.bytes_], NDArray[np.uint8]]:
        """Select regions from genome using uniform length bed coordinates.

        The type of embedding depends on the HDF5 genome e.g. sequence (bytes), onehot. See `self.embedding`.
        Pass `sample` to get consensus (variant-aware) haplotype sequences, otherwise reference sequence is returned.

        Parameters
        ----------
        contigs : ndarray[str]
            Contig names for each region.
            Shape: (regions)
        starts : ndarray[int32]
            Start position of each region. Should be genomic coordinates i.e. 1-indexed.
            Shape: (regions)
        length : int, uint32
            Length of all regions. A scalar value.
        samples : list[str], ndarray[str | uint32], default None
            An array of unique sample names. If integers, will select by index. If None, gets all.
        strands: ndarray[str], default None
            Strand (- or +) of query regions, if negative stranded will return reverse complement.
            If None, assumes all regions are on positive strand.
        sorted_contigs : bool, default False
            Whether query contigs are already sorted.
        pad_val : str, default "N"
            A single character to pad out-of-bound regions by.
        ploid_idx : ndarray[uint32], default None
            Array of which ploids to select. e.g. [0, 1] is the 1st and 2nd ploids/copies. If None, gets all.

        Returns
        -------
        ndarray[bytes | uint8]
            Array of regions. The "alphabet" dimension will be present if the reference is
            onehot encoded.
            Shape: (regions samples ploidy length [alphabet])

        Examples
        --------
        >>> gl = ConsensusGenomeLoader(ref_file_ohe_acgtn, vcf_file)
        >>> contigs = np.array(['21', '20', '20'])
        >>> starts = np.arange(3, dtype='u4')
        >>> length = np.uint32(5)
        >>> samples = ['OCI-AML5', 'NCI-H660']
        >>> consensus_seqs = gl.sel(contigs, starts, length, samples)
        >>> consensus_seqs.shape
        (3, 5, 4, 2, 5) # (regions, samples, ploidy, length, alphabet)
        """
        # input normalization
        if starts.dtype.type != np.int32:
            starts = starts.astype("i4")
            warnings.warn("Starts dtype was not int32, casting.")
        if not isinstance(length, np.uint32):
            if length < 1:
                raise ValueError(f"Negative length: {length}.")
            length = np.uint32(length)
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

        logging.info("Validating ref genome args")
        self._sel_validate_ref_genome_args(contigs, length)

        # validate samples and get sample_idx
        if samples is not None:
            logging.info("Validating sample args")
            if isinstance(samples, np.ndarray) and samples.dtype.kind == "u":
                if not np.isin(samples, np.arange(self._vcf.dims["samples"])).all():
                    raise ValueError("Got sample indices not in the VCF.")
                sample_idx: NDArray[np.uint32] = samples  # type: ignore
            else:
                self._sel_validate_sample_args(samples)  # type: ignore
                sample_idx = order_as(self._vcf["sample_id"], samples)
            valid_embeddings = {"sequence", "onehot"}
            if self.embedding not in valid_embeddings:
                raise ValueError(
                    f"Invalid embedding for getting variant-aware sequence, must be one of: {valid_embeddings}"
                )
        else:
            sample_idx = np.arange(self._vcf.dims["samples"], dtype="u4")

        # get default ploid_idx
        if ploid_idx is None:
            _ploid_idx: NDArray[np.uint32] = np.arange(
                self._vcf.dims["ploidy"], dtype="u4"
            )
        else:
            _ploid_idx = ploid_idx

        # get strands_flags
        if strands is None:
            strands_flags: NDArray[np.int8] = np.ones(len(contigs), dtype=np.int8)
        else:
            strands_flags = np.ones_like(strands, dtype=np.int8)
            strands_flags[strands == "-"] = np.int8(-1)

        with self._open() as ref_genome, dask.config.set(
            **{"array.slicing.split_large_chunks": True}
        ):
            logging.info("Slicing GenomeLoader")
            out = self._sel_slice(
                contigs,
                starts,
                ends,
                strands_flags,
                length,
                ref_genome,
                pad_arr,
                sample_idx,
                _ploid_idx,
                sorted_contigs,
            )

        return out

    def _sel_validate_ref_genome_args(
        self,
        contigs: NDArray[np.str_],
        length: np.uint32,
    ):
        if length < 1:
            raise ValueError("Length must be greater than 0.")
        if not np.isin(contigs, self.ref_contigs).all():
            raise ValueError("Contig not in reference genome.")

    def _sel_validate_sample_args(self, samples: Union[list[str], NDArray[np.str_]]):
        if len(set(samples)) != len(samples):
            raise ValueError("Got duplicate samples, which must be unique.")
        elif not np.isin(samples, self._vcf["sample_id"]).all():
            raise ValueError("Got sample that is not in the VCF.")

    def _sel_slice(
        self,
        contigs: NDArray[np.str_],
        starts: NDArray[np.int32],
        ends: NDArray[np.int32],
        strands_flags: NDArray[np.int8],
        length: np.uint32,
        ref_genome: h5py.File,
        pad_arr: Union[NDArray[np.bytes_], NDArray[np.uint8]],
        sample_idx: NDArray[np.uint32],
        ploid_idx: NDArray[np.uint32],
        sorted_contigs: bool = False,
    ) -> Union[NDArray[np.bytes_], NDArray[np.uint8]]:
        # requires low level wrangling with variable length arrays
        # current implementation relies on numba but awkward array may provide even more speed
        # see experiments in sbox.ipynb with awkward arrays
        # - David Laub

        # TODO: consider opportunities to speed up if there are many different contigs
        if not sorted_contigs:
            sort_idx = contigs.argsort()
            contigs = contigs[sort_idx]
            starts = starts[sort_idx]
            ends = ends[sort_idx]
        out_ls = []  # will be sorted if input contigs are sorted
        logging.info(f"Unique contigs: {np.unique(contigs)}")
        chrom: str
        for chrom in np.unique(contigs):  # guaranteed to proceed in a sorted order
            logging.info(f"Slicing chrom: {chrom}")
            chrom_starts = starts[contigs == chrom]
            chrom_ends = ends[contigs == chrom]
            ref_chrom, ref_start, rel_starts = self._sel_padded_min_ref_chrom(
                ref_genome[chrom], chrom_starts, chrom_ends, pad_arr
            )
            # (r s p l [a])
            chrom_out = self._sel_slice_chrom_genos(
                sample_idx,
                self._contig_str_to_vcf_contig_idx[chrom],
                chrom_starts,
                chrom_ends,
                length,
                ref_chrom,
                ref_start,
                rel_starts,
                ploid_idx,
            )
            out_ls.append(chrom_out)
        sorted_out = np.concatenate(out_ls)

        # rearrange into original order
        if not sorted_contigs:
            out = sorted_out[sort_idx.argsort()]
        else:
            out = sorted_out

        logging.info("Reverse complementing")
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
        ref_start: np.int32 = starts.min()
        ref_end: np.int32 = ends.max()
        rel_starts: NDArray[np.int32] = starts - ref_start

        # pad for out-of-bound
        chrom_length = ref_chrom_h5.attrs["length"]
        n_past_chrom = np.maximum(ref_end - chrom_length, 0)
        n_before_chrom = np.maximum(-ref_start, 0)
        real_ref_end: int = np.clip(
            ref_end, a_min=None, a_max=ref_chrom_h5.attrs["length"]
        )
        real_ref_start: int = np.clip(ref_start, a_min=0, a_max=None)
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

    def _sel_slice_chrom_genos(
        self,
        sample_idx: NDArray[np.uint32],
        contig_idx: int,
        starts: NDArray[np.int32],
        ends: NDArray[np.int32],
        length: np.uint32,
        ref_chrom: Union[NDArray[np.bytes_], NDArray[np.uint8]],
        ref_start: np.int32,
        rel_starts: NDArray[np.int32],
        ploid_idx: NDArray[np.uint32],
    ):
        assert self.spec is not None
        genos, variants_per_region, var_pos = self._sel_genos_bytes(
            contig_idx, sample_idx, starts, ends, ploid_idx
        )
        offsets = np.array(
            [0, *variants_per_region.cumsum(), genos.shape[-1]], dtype="u4"
        )
        del variants_per_region
        if self.embedding == "sequence":
            logging.info("Calling numba helper")
            out = _sel_bytes_helper(
                rel_starts,
                length,
                offsets,
                genos.view("i1"),
                var_pos,
                ref_chrom.view("i1"),
                ref_start,
            )
            return out.view("|S1")
        elif self.embedding == "onehot":
            for i, nuc in enumerate(self.spec):
                genos[genos == nuc] = i
            genos = genos.astype("u1")
            # (s p v a)
            ohe_genos: NDArray[np.uint8] = np.eye(5, dtype="u1")[genos]
            logging.info("Calling numba helper")
            out = _sel_ohe_helper(
                rel_starts, length, offsets, ohe_genos, var_pos, ref_chrom, ref_start
            )
            return out

    def _sel_genos_bytes(
        self,
        contig_idx: int,
        sample_idx: NDArray[np.uint32],
        starts: NDArray[np.int32],
        ends: NDArray[np.int32],
        ploid_idx: NDArray[np.uint32],
    ) -> tuple[NDArray[np.bytes_], NDArray[np.int32], NDArray[np.int32]]:
        """Get genotypes of given samples within the specified coordinates as bytes (e.g. nucleotides)."""
        logging.info("Slicing sample genotypes")

        unknown_to_N_flag = False
        if self.embedding == "sequence":
            unknown_to_N_flag = True
        elif self.embedding == "onehot":
            assert self.spec is not None
            if b"N" in self.spec:
                unknown_to_N_flag = True

        offset = self._var_contig_idx[contig_idx]
        # variant positions for the contig of interest
        if contig_idx == len(self._var_contig_idx) - 1:
            var_pos_contig = self._vcf.variant_position[
                self._var_contig_idx[contig_idx] :
            ]
        else:
            var_pos_contig = self._vcf.variant_position[
                self._var_contig_idx[contig_idx] : self._var_contig_idx[contig_idx + 1]
            ]
        var_pos_contig = var_pos_contig.load()
        v_starts = np.searchsorted(var_pos_contig, starts).astype("i4") + offset
        v_ends = np.searchsorted(var_pos_contig, ends).astype("i4") + offset
        variants_per_region: NDArray[np.int32] = v_ends - v_starts
        del var_pos_contig
        geno_idx_ls = []
        for start, end in zip(v_starts, v_ends):
            if start == end:
                continue
            geno_idx_ls.append(
                self._vcf.call_genotype.loc[start:end, sample_idx, ploid_idx]
            )
        # no variants in any region!
        if len(geno_idx_ls) == 0:
            # (s p v)
            genos: NDArray[np.bytes_] = np.empty(
                (len(sample_idx), len(ploid_idx), 0), dtype="|S1"
            )
            # (v)
            var_pos: NDArray[np.int32] = np.empty((0), dtype="i4")
            return genos, variants_per_region, var_pos
        geno_idx = xr.concat(geno_idx_ls, dim="variants")
        # del geno_idx_ls
        # A genotype coded as -1 is "unknown" in VCF spec. Map these to 0 and get ref alleles
        # from indices, then replace alleles that were unknown with "N".
        if unknown_to_N_flag:
            unk_mask = (geno_idx == -1).to_numpy()
        geno_idx = geno_idx.where(lambda x: x != -1, 0)
        geno_ls = []
        curr = 0
        for start, end in zip(v_starts, v_ends):
            if start == end:
                continue
            geno_ls.append(
                self._vcf.variant_allele.loc[
                    start:end, geno_idx[curr : curr + end - start].load()
                ].astype("S")
            )
            curr += end - start
        # (v s p)
        # del geno_idx
        genos = xr.concat(geno_ls, dim="variants").to_numpy()
        # del geno_ls
        if unknown_to_N_flag:
            genos[unk_mask] = b"N"
        var_pos_ls = []
        for start, end in zip(v_starts, v_ends):
            var_pos_ls.append(self._vcf.variant_position.loc[start:end])
        var_pos = xr.concat(var_pos_ls, "variants").to_numpy()
        genos = ein.rearrange(genos, "v s p -> s p v")
        # (s p v), (r), (v)
        return genos, variants_per_region, var_pos

    def sel_from_bed(
        self,
        bed: Union[pl.DataFrame, PathType],
        length: int,
        region_idx: Optional[IndexType] = None,
        samples: Optional[
            Union[list[str], NDArray[np.str_], NDArray[np.uint32]]
        ] = None,
        ploid_idx: Optional[NDArray[np.uint32]] = None,
        pad_val: Union[bytes, str] = "N",
        sorted_contigs: bool = False,
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
        samples : list[str], ndarray[str | uint32], default None
            An array of unique sample names. If integers, will select by index. If None, gets all.
        sorted_contigs : bool, default false
            Whether query contigs are sorted.
        cache : bool, default false
            Whether to cache the bed file in memory. Avoids reading the bed again when sequentially fetching bed regions.

        Returns
        -------
        ndarray[bytes | uint8]
            Array of regions. The "alphabet" dimension will be present if the reference is
            onehot encoded.
            Shape: (regions samples ploidy length [alphabet])
        """
        if isinstance(bed, str) or isinstance(bed, Path):
            _bed: pl.DataFrame = read_bed(bed, region_idx).collect()
        else:
            if region_idx is None:
                region_idx = np.arange(len(bed))
            _bed = (
                bed.with_row_count("index")
                .join(pl.DataFrame(region_idx, columns=["index"]), on="index")
                .drop("index")
            )

        chroms: NDArray[np.str_] = _bed["chrom"].to_numpy().astype("U")
        starts: NDArray[np.int32] = _bed["start"].to_numpy()
        strands: NDArray[np.str_] = _bed["strand"].to_numpy().astype("U")

        out = self.sel(
            chroms,
            starts,
            np.uint32(length),
            samples,
            strands,
            sorted_contigs,
            pad_val,
            ploid_idx,
        )

        return out

    def into_fixedlengthconsensus(
        self,
        out_file: PathType,
        bed_file: PathType,
        length: int,
        region_idx: Optional[IndexType] = None,
        samples: Optional[Union[list[str], NDArray[np.str_]]] = None,
        ploid_idx: Optional[NDArray[np.uint32]] = None,
        pad_val: Union[bytes, str] = "N",
        max_memory: Optional[int] = None,
        compressor: Codec = Blosc("lz4", shuffle=-1),
        sorted_contigs: bool = False,
        n_workers=Optional[int],
        threads_per_worker=Optional[int],
    ) -> FixedLengthConsensus:
        """Construct FixedLengthConsensus from a BED-like file.
        Will write the regions to a Zarr store specified by `out_file`.

        Parameters
        ----------
        out_file : PathType
            path to write Zarr store
        bed_file : PathType
            BED-like file defining at minimum the chromosome, start, and strand of each region.
            Must have a header defining column names.
        length : int
            length of all regions
        region_idx : int, list[int], ndarray[int]
            index of regions from bed_file to write i.e. select a subset of regions
        samples : Optional[list[str]], optional
            list of samples to get consensus regions for, otherwise gets all, defaults to None
        max_memory : int, optional
            approximate maximum memory to use in bytes, doesn't account for overhead or intermediate objects, defaults to -1
        compressor : Codec, optional
            compressor to use when writing Zarr, defaults to Blosc("lz4", shuffle=-1)

        Returns
        -------
        FixedLengthConsensus
        """
        return FixedLengthConsensus.from_gl_bed(
            self,
            out_file,
            bed_file,
            length,
            region_idx,
            samples,
            ploid_idx,
            pad_val,
            max_memory,
            compressor,
            sorted_contigs,
            n_workers,
            threads_per_worker,
        )


@numba.jit(
    "u1[:, :, :, :, :](i4[:], u4, u4[:], u1[:, :, :, :], i4[:], u1[:, :], i4)",
    parallel=True,
    nogil=True,
)
def _sel_ohe_helper(
    rel_starts: NDArray[np.int32],
    length: np.uint32,
    offsets: NDArray[np.uint32],
    ohe_genos: NDArray[np.uint8],
    var_pos: NDArray[np.int32],
    ohe_ref_chrom: NDArray[np.uint8],
    ref_start: np.int32,
) -> NDArray[np.uint8]:
    """Iterate over OHE variant regions and put each into the OHE reference.
    Since each can have a different # of variants this is not broadcastable.
    Reverse complement negative stranded sequences.
    """
    n_regions = len(rel_starts)
    n_samp, n_ploid, n_vars, n_alpha = ohe_genos.shape
    out = np.empty((n_regions, n_samp, n_ploid, length, n_alpha), dtype="u1")
    for i in numba.prange(n_regions):
        rel_start, offset_start, offset_end = (
            rel_starts[i],
            offsets[i],
            offsets[i + 1],
        )
        end = rel_start + length
        # (l a)
        ref_region = ohe_ref_chrom[rel_start:end]
        # (s p l a)
        region_out = out[i]
        for s in numba.prange(n_samp):
            for p in numba.prange(n_ploid):
                # (l a) = (l a)
                region_out[s, p] = ref_region
        # (v in region)
        rel_var_pos = var_pos[offset_start:offset_end] - rel_start - ref_start
        # (s p (v in r) a) = (s p (v in r) a)
        region_out[..., rel_var_pos, :] = ohe_genos[..., offset_start:offset_end, :]
    return out


@numba.njit(
    "char[:, :, :, :](i4[:], u4, u4[:], char[:, :, :], i4[:], char[:], i4)",
    parallel=True,
    nogil=True,
)
def _sel_bytes_helper(
    rel_starts: NDArray[np.int32],
    length: NDArray[np.uint32],
    offsets: NDArray[np.uint32],
    genos: NDArray[np.int8],
    var_pos: NDArray[np.int32],
    ref_chrom: NDArray[np.int8],
    ref_start: NDArray[np.int32],
) -> NDArray[np.int8]:
    """Iterate over byte variant regions and put each into the byte reference.
    Since each can have a different # of variants this is not broadcastable.
    """
    n_regions = len(rel_starts)
    n_samp, n_ploid, n_vars = genos.shape
    out = np.empty((n_regions, n_samp, n_ploid, length), dtype="i1")
    for i in numba.prange(n_regions):
        rel_start, offset_start, offset_end = (
            rel_starts[i],
            offsets[i],
            offsets[i + 1],
        )
        end = rel_start + length
        # (l)
        ref_region = ref_chrom[rel_start:end]
        region_out = out[i]
        for s in numba.prange(n_samp):
            for p in numba.prange(n_ploid):
                # (l) = (l)
                region_out[s, p] = ref_region
        # (v in r)
        rel_var_pos = var_pos[offset_start:offset_end] - rel_start - ref_start
        # (s p (v in r)) = (s p (v in r))
        region_out[..., rel_var_pos] = genos[..., offset_start:offset_end]
    return out


class FixedLengthConsensus:
    """Zarr wrapper to conveniently index fixed length consensus regions and track metadata e.g. bed coordinates.

    To index reference sequences, use `GenomeLoader.sel()` instead.

    Attributes
    ----------
    regions : zarr.Array
    bed : polars.DataFrame
    embedding : str
    length : np.uint32
    samples : ndarray[str]
    ref_genome_path : PathType
    vcf_path : PathType

    Notes
    -----
    Zarr tree:
    /
    ├── regions: (regions samples ploidy length [alphabet]), bytes or uint8
    │       Shape depends on store's embedding and whether it contains consensus sequences.
    │       See the `samples` and `vcf` groups and the `alphabet` attribute.
    ├── bed: Group
    │   ├── chrom: (regions), np.dtype('U')
    │   ├── start: (regions), uint8
    │   ├── strand: (regions), np.dtype('U')
    │   └── [other]: (regions)
    │           Any other columns included in the bed file
    ├── samples: Array of samples, str
    ├── [alphabet]: If onehot encoded, the alphabet used, np.dtype('|S1')
    └── attrs
        ├── ref_genome: str, Path
        │       Path to the reference genome used to get consensus regions
        ├── vcf: str, Path
        │       Path to the VCF used to get consensus regions
        ├── embedding: str, "sequence" or "onehot"
        └── length: uint32
    """

    def __init__(self, zarr_file: PathType):
        """Construct FixedLengthConsensus from a pre-existing Zarr store.

        Parameters
        ----------
        zarr_file : PathType
            Zarr store
        """
        self._file: zarr.Group = zarr.open(zarr_file)

        self.regions: zarr.Array = self._file["regions"]
        self.bed = zarr_to_df(self._file["bed"]).with_row_count("index")
        self.samples: NDArray[np.str_] = self._file["samples"][:]

        self.embedding: str = self._file.attrs["embedding"]
        self.length: np.uint32 = self._file.attrs["length"]
        self.ref_genome_path: PathType = self._file.attrs["ref_genome"]
        self.vcf_path: PathType = self._file.attrs["vcf"]

        if self.embedding == "onehot":
            self.alphabet: NDArray[np.bytes_] = self._file["alphabet"][:]

    @classmethod
    def from_gl_bed(
        self,
        gl: ConsensusGenomeLoader,
        out_file: PathType,
        bed_file: PathType,
        length: int,
        region_idx: Optional[IndexType] = None,
        samples: Optional[Union[list[str], NDArray[np.str_]]] = None,
        ploid_idx: Optional[NDArray[np.uint32]] = None,
        pad_val: Union[bytes, str] = "N",
        max_memory: Optional[int] = None,
        compressor: Codec = Blosc("lz4", shuffle=-1),
        sorted_contigs: bool = False,
        n_workers=Optional[int],
        threads_per_worker=Optional[int],
    ) -> FixedLengthConsensus:
        """Construct FixedLengthConsensus from a BED-like file.
        Will write the regions to a Zarr store specified by `out_file`.

        Parameters
        ----------
        gl: GenomeLoader
            instance of a GenomeLoader to get consensus sequences from
        out_file : PathType
            path to write Zarr store
        bed_file : PathType
            BED-like file defining at minimum the chromosome, start, and strand of each region.
            Must have a header defining column names.
        length : int
            length of all regions
        region_idx : int, list[int], ndarray[int]
            index of regions from bed_file to write i.e. select a subset of regions
        samples : Optional[list[str]], optional
            list of samples to get consensus regions for, otherwise gets all, defaults to None
        max_memory : int, optional
            approximate maximum memory to use in bytes, doesn't account for overhead or intermediate objects, defaults to -1
        compressor : Codec, optional
            compressor to use when writing Zarr, defaults to Blosc("lz4", shuffle=-1)

        Returns
        -------
        FixedLengthConsensus
        """

        # raise NotImplementedError("Writing Zarr to file is intractably slow since it hasn't been parallelized yet.")

        logging.info("Opening Zarr store.")
        z: zarr.Group = zarr.group(out_file, overwrite=True)
        z.attrs["embedding"] = gl.embedding
        z.attrs["ref_genome"] = str(gl.ref_genome_path)
        z.attrs["vcf"] = str(gl.vcf_path)
        z.attrs["length"] = np.uint32(length)

        logging.info("Reading BED-like file.")
        bed: pl.DataFrame = read_bed(bed_file, region_idx).collect()

        logging.info("Writing BED-like file.")
        df_to_zarr(bed, z.require_group("bed", overwrite=True), compressor)

        chroms: NDArray[np.str_] = bed["chrom"].to_numpy().astype("U")
        starts: NDArray[np.int32] = bed["start"].to_numpy()
        strands: NDArray[np.str_] = bed["strand"].to_numpy().astype("U")

        if samples is None:
            samples = gl._vcf["sample_id"].to_numpy()
        assert samples is not None
        z.create_dataset("samples", data=np.array(samples), dtype=str)

        if ploid_idx is None:
            ploid_idx = np.arange(gl._vcf.dims["ploidy"])

        if gl.embedding == "sequence":
            region_dtype = np.dtype("S1")  # type: ignore
            regions_shape: tuple = (len(bed), len(samples), len(ploid_idx), length)
        elif gl.embedding == "onehot":
            assert gl.spec is not None
            z.create_dataset("alphabet", data=gl.spec)
            region_dtype = np.dtype("u1")  # type: ignore
            regions_shape = (
                len(bed),
                len(samples),
                len(ploid_idx),
                length,
                len(gl.spec),
            )
        tot_bytes = np.prod(regions_shape) * region_dtype.itemsize

        # ignores memory requirements of overhead and intermediates
        # most memory overhead is probably from loading spanning region of each ref chromosome
        # TODO: chunk by samples and/or haplotypes if bytes_per_region > max_memory
        if max_memory is not None:
            if tot_bytes // len(bed) < max_memory:
                chunk_by = "region"
                bytes_per_chunk = tot_bytes // len(bed)
            # elif tot_bytes//(len(bed)*len(samples)) < max_memory:
            #     chunk_by = 'region_sample'
            #     bytes_per_chunk = tot_bytes//(len(bed)*len(samples))
            # elif tot_bytes//(len(bed)*len(samples)*2) < max_memory:
            #     chunk_by = 'region_sample_ploidy'
            #     bytes_per_chunk = tot_bytes//(len(bed)*len(samples)*2)
            else:
                raise NotImplementedError("Chunking by samples or haplotypes.")
                raise ValueError(
                    f"Max memory specified ({max_memory}) is insufficient to construct a consensus for a single (region + sample + haplotype)."
                )
            regs_per_chunk = max_memory // bytes_per_chunk
            n_chunks = int(-(-len(bed) // regs_per_chunk))  # ceil
            if gl.embedding == "sequence":
                chunks: tuple = (n_chunks, 1, None, None)
            elif gl.embedding == "onehot":
                chunks = (n_chunks, 1, 1, None, None)
            z_regions: zarr.Array = z.create_dataset(
                "regions",
                shape=regions_shape,
                dtype=region_dtype,
                compressor=compressor,
                chunks=chunks,
            )
            with tqdm(
                total=len(bed),
                unit="regions",
            ) as pbar:
                start_idx = 0
                while start_idx < len(bed):
                    end_idx = min(start_idx + regs_per_chunk, len(bed))
                    out = gl.sel(
                        chroms[start_idx:end_idx],
                        starts[start_idx:end_idx],
                        np.uint32(length),
                        samples,
                        strands,
                        sorted_contigs,
                        pad_val,
                        ploid_idx,
                    )
                    logging.info("Writing chunk to file.")
                    da.from_array(out, chunks=(1, *chunks[1:])).to_zarr(
                        z_regions, region=slice(start_idx, end_idx)
                    )
                    pbar.update(end_idx - start_idx)
                    start_idx = end_idx
        else:
            if gl.embedding == "sequence":
                chunks = (1, 1, None, None)
            elif gl.embedding == "onehot":
                chunks = (1, 1, 1, None, None)
            z_regions = z.create_dataset(
                "regions",
                shape=regions_shape,
                dtype=region_dtype,
                compressor=compressor,
                chunks=chunks,
            )
            out = gl.sel(
                chroms,
                starts,
                np.uint32(length),
                samples,
                strands,
                sorted_contigs,
                pad_val,
                ploid_idx,
            )
            logging.info("Writing regions to file.")
            da.from_array(out, chunks=chunks).to_zarr(z_regions)

        return FixedLengthConsensus(out_file)

    def sel(
        self,
        chroms: Optional[NDArray[np.str_]] = None,
        starts: Optional[NDArray[np.uint32]] = None,
        strands: Optional[NDArray[np.str_]] = None,
        region_idx: Optional[IndexType] = None,
        samples: Optional[Union[list[str], NDArray[np.str_], IndexType]] = None,
        haplotype_idx: Optional[IndexType] = None,
    ) -> Union[NDArray[np.bytes_], NDArray[np.uint8]]:
        """Select regions, samples, and haplotypes.

        Must provide exclusively either (`chroms`, `starts`, and `strands`) or `regions` (i.e. region indices) or none of them,
        which will select all regions. Regions will correspond to the BED2 file (`self.bed`).
        If no samples or haplotypes are provided, selects all of them.
        All indices must be 0-indexed i.e. the first haplotype is index 0.

        To index reference sequences, use `GenomeLoader.sel()` instead.

        Parameters
        ----------
        chroms : ndarray[str], defaults to None
            Array of chromosomes
        starts : ndarray[int], defaults to None
            Array of start indices
        strands : ndarray[str], defaults to None
            Array of strands ("+" or "-"). Negative strands are reverse complemented.
        region_idx : int, slice, ndarray[int], optional
            Index of bed regions (0-indexed) to select, defaults to all regions
        samples : int, slice, ndarray[int], list[str], ndarray[str], optional
            Indices or a list of samples, defaults to all samples
        haplotype_idx : int, slice, ndarray[int], optional
            Index of haplotypes (0-indexed), defaults to all haplotypes

        Returns
        -------
        Union[NDArray[np.bytes_], NDArray[np.uint8]]
            Selected regions of the genome

        Examples
        --------
        Suppose a FixedLengthConsensus has length 200, 10 samples, two chromosomes, and
        is onehot encoded with alphabet "ACGTN".

        >>> flc = FixedLengthConsensus(zarr_file)
        >>> chroms = np.array(['1', '3', '1'])
        >>> starts = np.array([123, 456, 789])
        >>> samples = ['OCI-AML5', 'NCI-H660']
        >>> haplotypes = [1, 0]
        >>> seqs = flc.sel(chroms, starts)
        >>> seqs.shape
        (3, 200, 10, 2, 5) # (regions, length, samples, ploidy, alphabet)
        """

        coord_is_none = {
            "chroms": chroms is None,
            "starts": starts is None,
            "strands": strands is None,
        }
        present_coords = [coord for coord, none in coord_is_none.items() if not none]
        missing_coords = [coord for coord, none in coord_is_none.items() if none]
        if present_coords and missing_coords:
            raise ValueError(f"Got {present_coords} but not {missing_coords}")
        elif not missing_coords and region_idx is not None:
            raise ValueError(
                "Got both (chroms, starts, strands) and region_idx, specify just one of the two."
            )
        elif not present_coords and region_idx is None:
            region_idx = slice(None)
        elif not missing_coords:
            with pl.StringCache():
                _chroms = pl.Series("chrom", chroms, dtype=pl.Categorical)
                _starts = pl.Series("start", starts, dtype=pl.Int32)
                _strands = pl.Series("strand", strands, dtype=pl.Categorical)
                q = pl.DataFrame([_chroms, _starts, _strands])
                region_idx = (
                    self.bed.with_columns(
                        pl.col(["chrom", "strand"]).cast(pl.Categorical)
                    )
                    .join(q, on=["chrom", "start", "strand"])
                    .with_row_count("index")
                    .index.to_numpy()
                )
            if len(region_idx) != len(q):  # type: ignore
                raise ValueError(
                    "Got query regions that are not represented in the file."
                )

        # sample index
        sample_idx: IndexType
        if samples is None:
            sample_idx = slice(None)
        else:
            if not np.isin(samples, self.samples).all():
                raise ValueError(
                    "Got samples that are not in this FixedLengthConsensus."
                )
            if isinstance(samples, list) or (
                isinstance(samples, np.ndarray) and samples.dtype.kind == "U"
            ):
                sample_idx = order_as(self.samples, samples)  # type: ignore
            else:
                sample_idx = samples  # type: ignore

        if haplotype_idx is None:
            haplotype_idx = slice(None)

        if self.embedding == "sequence":
            return self.regions.oindex[region_idx, :, sample_idx, haplotype_idx]
        else:
            return self.regions.oindex[region_idx, :, sample_idx, haplotype_idx, :]
