from __future__ import annotations

import logging
import warnings
from pathlib import Path
from textwrap import dedent
from typing import Optional, Union

import dask.array as da
import h5py
import numba
import numpy as np
import pandas as pd
import sgkit as sg
import xarray as xr
import zarr
import numcodecs
from numcodecs import Blosc
from numcodecs.abc import Codec
from numpy.typing import NDArray

from .utils import read_bed2

PathType = Union[str, Path]
IndexType = Union[int, slice, NDArray[np.int_], NDArray[np.uint]]


class FixedLengthConsensus:
    """Zarr wrapper to conveniently index fixed length consensus regions and track metadata e.g. bed coordinates.

    To index reference sequences, use `GenomeLoader.sel()` instead.

    Attributes
    ----------
    regions : zarr.Array
    bed : pandas.DataFrame
    embedding : str
    length : np.uint32
    samples : ndarray[str]
    ref_genome_path : PathType
    vcf_path : PathType

    Notes
    -----
    Zarr tree:
    /
    ├── regions: (regions length samples ploidy [alphabet]), uint8
    │       Shape depends on store's embedding and whether it contains consensus sequences.
    │       See the `samples` and `vcf` groups and the `alphabet` attribute.
    ├── chroms: (regions), np.dtype('U')
    ├── starts: (regions), uint8
    ├── samples: An array of samples, str
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
        chrom = pd.Series(self._file["chroms"][:], dtype="category", name="chrom")
        start = pd.Series(self._file["starts"][:], dtype=np.uint32, name="start")
        self.bed = pd.concat([chrom, start], axis=1)
        self.embedding: str = self._file.attrs["embedding"]
        self.length: np.uint32 = self._file.attrs["length"]
        self.samples: NDArray[np.str_] = self._file["samples"][:]
        self.ref_genome_path: PathType = self._file.attrs["ref_genome"]
        self.vcf_path: PathType = self._file.attrs["vcf"]

        if self.embedding == "onehot":
            self.alphabet: NDArray[np.bytes_] = self._file["alphabet"][:]

    @classmethod
    def from_ref_vcf_bed(
        cls,
        ref_file: PathType,
        vcf_file: PathType,
        bed_file: PathType,
        out_file: PathType,
        length: int,
        samples: Optional[list[str]] = None,
        max_memory: Optional[int] = None,
        compressor: Codec = Blosc("lz4", shuffle=-1),
    ) -> FixedLengthConsensus:
        """Construct FixedLengthConsensus from a reference.h5, vcf.zarr, and BED2 file.
        Will write the regions to a Zarr store specified by `out_file`.

        Parameters
        ----------
        ref_file : PathType
            reference genome file
        bed_file : PathType
            BED2 file defining the chromosome and start of each region
        length : int
            length of all regions
        out_file : PathType
            path to output Zarr store
        vcf_file : PathType
            VCF file
        samples : Optional[list[str]], optional
            list of samples to get consensus regions for, otherwise gets all, defaults to None
        max_memory : int, optional
            approximate maximum memory to use in bytes, doesn't account for overhead or intermediate objects, defaults to -1
        compressor : Codec, optional
            compressor to use when writing Zarr, defaults to Blosc("lz4", shuffle=-1)

        Returns
        -------
        FixedLengthConsensus
            an instance of FixedLengthConsensus

        """
        gl = GenomeLoader(ref_file, vcf_file)

        z: zarr.Group = zarr.open(out_file, "w")
        z.attrs["embedding"] = gl.embedding
        z.attrs["ref_genome"] = str(ref_file)
        z.attrs["vcf"] = str(vcf_file)
        z.attrs["length"] = np.uint32(length)

        bed = read_bed2(bed_file)
        chrom = bed["chrom"].to_numpy().astype("U")
        z.create_dataset(
            "chroms",
            data=chrom,
            compressor=compressor,
            filters=[
                numcodecs.Categorize(
                    bed["chrom"].cat.categories, chrom.dtype, astype="u2"
                )
            ],
        )
        z.create_dataset("starts", data=bed["start"].to_numpy(), compressor=compressor)

        if gl.embedding == "sequence":
            rep_size = 1
            region_dtype = np.dtype("S1")
        elif gl.embedding == "onehot":
            z.create_dataset("alphabet", data=gl.spec)  # type: ignore
            rep_size = len(gl.spec)  # type: ignore
            region_dtype = np.dtype("u1")

        if samples is None:
            samples = gl._vcf.sample_id.to_numpy()  # type: ignore
        z.create_dataset("samples", data=np.array(samples), dtype=str)
        region_shape: tuple = (len(bed), length, len(samples), 2, rep_size)
        chunks: tuple = (
            None,
            1,
            None,
            None,
            1,
        )  # will almost always be accessing full length and full encoding

        z.create_dataset(
            "regions",
            shape=region_shape,
            dtype=region_dtype,
            compressor=compressor,
            chunks=chunks,
        )
        # ignores memory requirements of overhead and intermediates
        if max_memory is not None:
            bytes_per_region = np.prod(
                region_shape[1:]
            )  # all representations are 1 byte
            regs_per_chunk = max_memory // bytes_per_region
            n_chunks, final_chunk = divmod(len(bed), regs_per_chunk)
            for i in range(n_chunks):
                start_idx = i * regs_per_chunk
                end_idx = start_idx + regs_per_chunk
                chroms = bed.loc[start_idx:end_idx, "chrom"].to_numpy()
                starts = bed.loc[start_idx:end_idx, "start"].to_numpy()
                z["regions"][start_idx:end_idx] = gl.sel(
                    chroms, starts, np.uint32(length), samples
                )
            if final_chunk > 0:
                start_idx = i * regs_per_chunk
                end_idx = start_idx + final_chunk
                chroms = bed.loc[start_idx:end_idx, "chrom"].to_numpy()
                starts = bed.loc[start_idx:end_idx, "start"].to_numpy()
                z["regions"][start_idx:end_idx] = gl.sel(
                    chroms, starts, np.uint32(length), samples
                )
        else:
            chroms = bed["chrom"].to_numpy()
            starts = bed["start"].to_numpy()
            z["regions"] = gl.sel(chroms, starts, np.uint32(length), samples)

        return cls(out_file)

    def sel(
        self,
        chroms: Optional[NDArray[np.str_]] = None,
        starts: Optional[NDArray[np.uint32]] = None,
        region_idx: Optional[IndexType] = None,
        samples: Optional[Union[list[str], NDArray[np.str_], IndexType]] = None,
        haplotype_idx: Optional[IndexType] = None,
    ) -> Union[NDArray[np.bytes_], NDArray[np.uint8]]:
        """Select regions, samples, and haplotypes.

        Must provide exclusively either (`chroms` and `starts`) or `regions` (i.e. region indices) or none of them,
        which will select all regions. Regions will correspond to the BED2 file (`self.bed`).
        If no samples or haplotypes are provided, selects all of them.
        All indices must be 0-indexed i.e. the first haplotype is index 0.

        To index reference sequences, use `GenomeLoader.sel()` instead.

        Parameters
        ----------
        chroms : ndarray[str], optional
            array of chromosomes, defaults to None
        starts : ndarray[int], optional
            array of start indices, defaults to None
        region_idx : int, slice, ndarray[int], optional
            index of bed regions (0-indexed) to select, defaults to all regions
        samples : int, slice, ndarray[int], list[str], ndarray[str], optional
            indices or a list of samples, defaults to all samples
        haplotype_idx : int, slice, ndarray[int], optional
            index of haplotypes (0-indexed), defaults to all haplotypes

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

        if chroms is not None and starts is None:
            raise ValueError("Got chroms but no starts.")
        elif chroms is None and starts is not None:
            raise ValueError("Got starts but no chroms.")
        elif chroms is not None and starts is not None and region_idx is not None:
            raise ValueError(
                "Got both (chroms, starts) and region_idx, specify just one of the two."
            )
        elif chroms is None and starts is None and region_idx is None:
            region_idx = slice(None)
        elif chroms is not None and starts is not None:
            q = pd.DataFrame([chroms, starts], columns=["chrom", "start"])
            region_idx = self.bed.merge(q, on=["chrom", "start"]).index.to_numpy()
            if len(region_idx) != len(chroms):  # type: ignore
                raise ValueError(
                    "Got query regions that are not represented in the file."
                )

        # sample index
        sample_idx: IndexType
        if samples is None:
            sample_idx = slice(None)
        elif isinstance(samples, list):
            _, self_idx, query_idx = np.intersect1d(
                self.samples, samples, return_indices=True, assume_unique=True
            )
            sample_idx = self_idx[query_idx]
        elif isinstance(samples, np.ndarray):
            if samples.dtype.kind == "U":
                _, self_idx, query_idx = np.intersect1d(  # type: ignore
                    self.samples, samples, return_indices=True, assume_unique=True
                )
                sample_idx = self_idx[query_idx]
            else:
                sample_idx = samples  # type: ignore
        else:
            sample_idx = samples

        if haplotype_idx is None:
            haplotype_idx = slice(None)

        if self.embedding == "sequence":
            return self.regions.oindex[region_idx, :, sample_idx, haplotype_idx]
        else:
            return self.regions.oindex[region_idx, :, sample_idx, haplotype_idx, :]


class GenomeLoader:
    """A reference genome with optional VCF for getting consensus sequences."""

    embedding: str
    ref_genome_path: PathType
    contigs: NDArray[np.str_]
    vcf_path: Optional[PathType] = None
    spec: Optional[NDArray[np.bytes_]] = None

    def __init__(
        self, ref_genome_h5: PathType, vcf_zarr: Optional[PathType] = None
    ) -> None:
        """A reference genome with optional VCF for getting consensus sequences."""

        self.ref_genome_path = Path(ref_genome_h5)

        with self._open() as ref_genome:
            self.embedding: str = ref_genome.attrs["id"]
            logging.info(f"Genome is {self.embedding} embedded.")
            self._ref_dtype = ref_genome[next(iter(ref_genome.keys()))][
                self.embedding
            ].dtype
            self.contigs = np.array(list(ref_genome.keys()))

            if vcf_zarr is not None:
                self.vcf_path = vcf_zarr
                self._vcf: Optional[xr.Dataset] = sg.load_dataset(Path(vcf_zarr))
                self._cast_vcf_objects_to_bytes()
                self._0_idx_vcf_positions()
                var_contigs = da.array(self._vcf.attrs["contigs"])[
                    self._vcf.variant_contig
                ]
                if not da.isin(var_contigs, self.contigs).all().compute():
                    raise ValueError(
                        "VCF has contigs that are not in reference genome."
                    )
            else:
                self._vcf = None

            if ref_genome.attrs.get("encode_spec", None) is not None:
                self.spec = ref_genome.attrs["encode_spec"].astype("S")
                if b"N" not in self.spec and vcf_zarr is not None:  # type: ignore
                    warnings.warn(
                        "N is not in the reference genome alphabet so unknown genotypes will default to match the reference."
                    )

    def _open(self):
        return h5py.File(self.ref_genome_path)

    def __repr__(self) -> str:
        out = dedent(
            f"""
            GenomeLoader with...
            reference genome: {self.ref_genome_path}
            embedding: {self.embedding}
            """
        ).strip()
        if self.embedding == "onehot":
            out += f"\nalphabet: {''.join(self.spec.astype('U'))}"  # type: ignore
        if self.vcf_path is not None:
            out += f"\nVCF: {self.vcf_path}"
        return out

    def _cast_vcf_objects_to_bytes(self):
        self._vcf["variant_allele"] = self._vcf.variant_allele.astype("S")  # type: ignore
        self._vcf["variant_id"] = self._vcf.variant_id.astype("S")  # type: ignore

    def _0_idx_vcf_positions(self):
        self._vcf["variant_position"] = self._vcf.variant_position.astype(  # type: ignore
            "u4"
        ) - np.uint32(
            1
        )

    def sel(
        self,
        chroms: NDArray[np.str_],
        starts: NDArray[np.uint32],
        length: np.uint32,
        samples: Optional[Union[list[str], NDArray[np.str_]]] = None,
        sorted_chroms: bool = False,
    ) -> Union[NDArray[np.bytes_], NDArray[np.uint8]]:
        """Select sequences from genome using uniform length bed coordinates.

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
        samples : Sequence[str], default None
            A list of unique samples (no duplicates!).

        Returns
        -------
        Union[NDArray[np.bytes_], NDArray[np.uint8]]
            Array of sequences.
        """
        # input normalization and validation
        length = np.uint32(length)  # type: ignore
        if starts.dtype.type != np.uint32:
            starts = starts.astype("u4")
            warnings.warn("Starts dtype was not uint32, casting.")
        starts = starts - np.uint32(
            1
        )  # VCF positions are 1-indexed but h5 genome is 0-indexed
        ends = starts + length
        with self._open() as ref_genome:
            self._sel_validate_ref_genome_args(chroms, starts, ends, length, ref_genome)
            if samples is not None:
                self._sel_validate_sample_args(samples)
                sample_idx = self._sel_sample_idx(samples)
                val_embeddings = {"sequence", "onehot"}
                if self.embedding not in val_embeddings:
                    raise ValueError(
                        f"Invalid embedding for getting variant-aware sequence, must be one of: {val_embeddings}"
                    )
            else:
                sample_idx = None
            return self._sel_slice(
                chroms, starts, ends, length, ref_genome, sample_idx, sorted_chroms
            )

    def _sel_sample_idx(self, samples: Union[list[str], NDArray[np.str_]]):
        _, s_id_idx, s_query_idx = np.intersect1d(self._vcf.sample_id, samples, assume_unique=True, return_indices=True)  # type: ignore
        return s_id_idx[s_query_idx]

    def _sel_validate_ref_genome_args(
        self,
        chroms: NDArray[np.str_],
        starts: NDArray[np.uint32],
        ends: NDArray[np.uint32],
        length: np.uint32,
        ref_genome: h5py.File,
    ):
        if length < 1:
            raise ValueError("Length must be greater than 0.")
        if (starts < 0).any():
            raise IndexError("Region has start < 0.")
        if not np.isin(chroms, self.contigs).all():
            raise ValueError("Chromosome not in reference genome.")
        for chrom in chroms:
            if (ends[chroms == chrom] > ref_genome[chrom].attrs["length"]).any():
                raise IndexError("Region goes outside bounds of contig.")

    def _sel_validate_sample_args(self, samples: Union[list[str], NDArray[np.str_]]):
        if len(set(samples)) != len(samples):
            raise ValueError("Got duplicate samples, which must be unique.")
        if self._vcf is None:
            raise ValueError("Cannot select sample without a VCF.")
        elif self._vcf is not None and not np.isin(samples, self._vcf.sample_id).all():
            raise ValueError("Got sample that is not in the VCF.")

    def _sel_slice(
        self,
        chroms: NDArray[np.str_],
        starts: NDArray[np.uint32],
        ends: NDArray[np.uint32],
        length: np.uint32,
        ref_genome: h5py.File,
        sample_idx: Optional[NDArray[np.uint32]] = None,
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
        for chrom in np.unique(chroms):  # guaranteed to proceed in a sorted order
            chrom_starts = starts[chroms == chrom]
            chrom_ends = ends[chroms == chrom]
            if sample_idx is not None:
                chrom_out = self._sel_slice_chrom_genos(
                    sample_idx, chrom, chrom_starts, chrom_ends, length, ref_genome
                )
            else:
                # get minimal range of reference to give to numba
                # note ref is 0-indexed but coordinates 1-indexed
                ref_start = chrom_starts.min()
                ref_end = chrom_ends.max()
                ref_chrom = ref_genome[chrom][self.embedding][ref_start:ref_end]
                rel_starts = chrom_starts - ref_start
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
        return out

    def _sel_slice_chrom_genos(
        self,
        sample_idx: NDArray[np.uint32],
        chrom: str,
        starts: NDArray[np.uint32],
        ends: NDArray[np.uint32],
        length: np.uint32,
        ref_genome: h5py.File,
    ):
        contig_idx = self._sel_contig_idx(chrom)
        genos, variants_per_region, variant_idx = self._sel_genos_bytes(
            contig_idx, sample_idx, starts, ends
        )
        offsets = np.array(
            [0, *variants_per_region.cumsum(), genos.shape[0]], dtype="u4"
        )
        var_pos = self._vcf.variant_position[variant_idx].astype("u4").to_numpy()  # type: ignore

        # load minimal range of reference in memory to give to numba
        ref_start = starts.min()
        ref_end = ends.max()
        rel_starts = starts - ref_start
        if self.embedding == "sequence":
            ref_chrom = ref_genome[chrom][self.embedding][ref_start:ref_end]
            # TODO: maybe need to cast to bytes from int8
            out = _sel_helper_bytes(
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
            ohe_ref_chrom = ref_genome[chrom][self.embedding][ref_start:ref_end]
            for i, nuc in enumerate(self.spec):  # type: ignore
                genos[genos == nuc] = i
            genos = genos.astype("u1")
            ohe_genos = np.eye(5, dtype="u1")[genos]
            return _sel_helper_ohe(
                rel_starts,
                length,
                offsets,
                ohe_genos,
                var_pos,
                ohe_ref_chrom,
                ref_start,
            )

    def _sel_contig_idx(self, chrom: str):
        return np.nonzero(
            np.isin(np.asarray(self._vcf.attrs["contigs"]), chrom, assume_unique=True)  # type: ignore
        )[0]

    def _sel_genos_bytes(
        self,
        contig_idx: NDArray[np.uint32],
        sample_idx: NDArray[np.uint32],
        starts: NDArray[np.uint32],
        ends: NDArray[np.uint32],
    ) -> tuple[NDArray[np.bytes_], NDArray[np.uint32], NDArray[np.uint32]]:
        """Get genotypes of given samples within the specified coordinates as bytes (e.g. nucleotides)."""

        unknown_to_N_flag = False
        if self.embedding == "sequence":
            unknown_to_N_flag = True
        elif self.embedding == "onehot":
            if b"N" in self.spec:  # type: ignore
                unknown_to_N_flag = True

        variant_mask = (
            (self._vcf.variant_contig.isin(contig_idx)).expand_dims(  # type: ignore
                {"idx": len(starts)}, 1
            )
            & (
                starts
                <= self._vcf.variant_position.expand_dims({"idx": len(starts)}, 1)  # type: ignore
            )
            & (self._vcf.variant_position.expand_dims({"idx": len(starts)}, 1) < ends)  # type: ignore
        )
        variant_idx, region_idx = da.compute(da.nonzero(variant_mask))[0]
        variants_per_region = np.diff(
            region_idx.searchsorted(np.arange(len(starts)))
        ).astype("u4")
        geno_idx = self._vcf.call_genotype.loc[variant_idx, sample_idx]  # type: ignore
        if unknown_to_N_flag:
            unk_mask = (geno_idx == -1).to_numpy()
        geno_idx = geno_idx.where(lambda x: x != -1, 0)
        # (v s p)
        genos = self._vcf.variant_allele.loc[  # type: ignore
            variant_idx, geno_idx.as_numpy()
        ].to_numpy()
        if unknown_to_N_flag:
            genos[unk_mask] = b"N"
        # (v s p), (r), (v)
        return genos, variants_per_region, variant_idx


@numba.njit(
    "u1[:, :, :, :, :](u4[:], u4, u4[:], u1[:, :, :, :], u4[:], u1[:, :], u4)",
    parallel=True,
    nogil=True,
)
def _sel_helper_ohe(
    rel_starts, length, offsets, ohe_genos, var_pos, ohe_ref_chrom, ref_start
) -> NDArray[np.uint8]:
    """Iterate over OHE variant regions and put each into the OHE reference.
    Since each can have a different # of variants this is not broadcastable.
    """
    n_regions = len(rel_starts)
    n_vars, n_samp, n_ploid, n_alpha = ohe_genos.shape
    out = np.empty((n_regions, length, n_samp, n_ploid, n_alpha), dtype="u1")
    for i in numba.prange(n_regions):
        rel_start, offset_start, offset_end = (
            rel_starts[i],
            offsets[i],
            offsets[i + 1],
        )
        end = rel_start + length
        rel_var_pos = var_pos[offset_start:offset_end] - rel_start - ref_start
        ref_region = ohe_ref_chrom[rel_start:end]
        region_out = out[i]
        for s in numba.prange(n_samp):
            for p in numba.prange(n_ploid):
                region_out[:, s, p, :] = ref_region
        region_out[rel_var_pos] = ohe_genos[offset_start:offset_end]
    return out


# TODO: maybe need to cast to bytes from int8
@numba.njit(
    "char[:, :, :, :](u4[:], u4, u4[:], char[:, :, :], u4[:], char[:], u4)",
    parallel=True,
    nogil=True,
)
def _sel_helper_bytes(
    rel_starts, length, offsets, genos, var_pos, ref_chrom, ref_start
) -> NDArray[np.int8]:
    """Iterate over byte variant regions and put each into the byte reference.
    Since each can have a different # of variants this is not broadcastable.
    """
    n_regions = len(rel_starts)
    n_vars, n_samp, n_ploid = genos.shape
    out = np.empty((n_regions, length, n_samp, n_ploid), dtype="i1")
    for i in numba.prange(n_regions):
        rel_start, offset_start, offset_end = (
            rel_starts[i],
            offsets[i],
            offsets[i + 1],
        )
        end = rel_start + length
        rel_var_pos = var_pos[offset_start:offset_end] - rel_start - ref_start
        ref_region = ref_chrom[rel_start:end]
        region_out = out[i]
        for s in numba.prange(n_samp):
            for p in numba.prange(n_ploid):
                region_out[:, s, p] = ref_region
        region_out[rel_var_pos] = genos[offset_start:offset_end]
    return out
