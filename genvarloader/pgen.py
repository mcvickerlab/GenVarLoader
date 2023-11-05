from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numba as nb
import numpy as np
import polars as pl
from numpy.typing import NDArray

from .types import DenseGenotypes, Variants, VLenAlleles

try:
    import pgenlib

    PGENLIB_INSTALLED = True
except ImportError:
    PGENLIB_INSTALLED = False


class Pgen(Variants):
    def __init__(
        self,
        paths: Union[str, Path, Dict[str, Union[str, Path]]],
        samples: Optional[List[str]] = None,
    ) -> None:
        """Reads genotypes from PGEN files. Currently does not support multi-allelic
        sites, but does support **split** multi-allelic sites. This can be done by
        running `bcftools norm -a --atom-overlaps . -m - <file.vcf>` and creating the
        PGEN with the `--vcf-half-call r` option.

        Parameters
        ----------
        path : Union[str, Path]
            Path to any of the PGEN files (.pgen, .pvar, .psam) or their prefix.
        samples : Optional[List[str]], optional
            Which samples to include, by default all samples.

        Notes
        -----
        Writes a copy of the .pvar file as an Arrow file (.gvl.arrow) to speed up
        loading (by 25x or more for larger files.)
        """
        if not PGENLIB_INSTALLED:
            raise ImportError("Pgenlib must be installed to read PGEN files.")

        # pgenlib is exclusively diploid
        self.ploidy = 2
        # unknown genotypes are set to -9
        self.UNKNOWN = -9

        if isinstance(paths, (str, Path)):
            _paths = Path(paths)
            self.pgen_paths = {"_all": _paths.with_suffix(".pgen")}
            self.split_by_contig = False
        elif isinstance(paths, dict):
            _paths = {
                contig: Path(path).with_suffix(".pgen")
                for contig, path in paths.items()
            }
            self.pgen_paths = _paths
            self.split_by_contig = True
        _first_path = self.pgen_paths[next(iter(self.pgen_paths))]

        try:
            psam_samples = pl.read_csv(
                _first_path.with_suffix(".psam"), separator="\t", columns=["IID"]
            )["IID"].to_numpy()
        except pl.ColumnNotFoundError:
            psam_samples = pl.read_csv(
                _first_path.with_suffix(".psam"), separator="\t", columns=["#IID"]
            )["#IID"].to_numpy()

        if samples is not None:
            _samples, sample_idx, _ = np.intersect1d(
                psam_samples, samples, return_indices=True
            )
            if len(_samples) != len(samples):
                raise ValueError("Got samples that are not in the pgen file.")
            self.samples = _samples
            self.sample_idx = sample_idx.astype(np.uint32)
        else:
            self.samples = psam_samples
            self.sample_idx = np.arange(len(psam_samples), dtype=np.uint32)

        self.n_samples = len(self.samples)

        self.positions: Dict[str, NDArray[np.int32]] = {}
        self.size_diffs: Dict[str, NDArray[np.int32]] = {}
        self.ends: Dict[str, NDArray[np.int32]] = {}
        # end_to_var_idx maps from relative end idxs to absolute variant idxs
        self.end_to_var_idx: Dict[str, NDArray[np.int32]] = {}
        # no multi-allelics
        self.ref: Dict[str, VLenAlleles] = {}
        self.alt: Dict[str, VLenAlleles] = {}

        self.contig_offsets: Dict[str, int] = {}

        for contig, path in self.pgen_paths.items():
            pvar_path = path.with_suffix(".pvar")
            pvar_arrow_path = path.with_suffix(".gvl.arrow")
            ends_arrow_path = path.with_suffix(".ends.gvl.arrow")
            # exists and was modified more recently than .pvar
            if (
                pvar_arrow_path.exists()
                and pvar_arrow_path.stat().st_mtime > pvar_path.stat().st_mtime
                and ends_arrow_path.exists()
                and ends_arrow_path.stat().st_mtime > pvar_path.stat().st_mtime
            ):
                pvar = pl.read_ipc(pvar_arrow_path)
                ends = pl.read_ipc(ends_arrow_path)
            else:
                with open(pvar_path, "r") as f:
                    skip_rows = 0
                    while f.readline().startswith("##"):
                        skip_rows += 1

                pvar = pl.read_csv(
                    pvar_path,
                    separator="\t",
                    skip_rows=skip_rows,
                    columns=["#CHROM", "POS", "REF", "ALT"],
                    dtypes={"#CHROM": pl.Utf8, "POS": pl.Int32},
                )

                if (pvar["ALT"].str.contains(",")).any():
                    raise RuntimeError(
                        f"""PGEN file {path} contains multi-allelic variants which are 
                        not yet supported by GenVarLoader. Split your multi-allelic 
                        variants with `bcftools norm -a --atom-overlaps . -m - 
                        <file.vcf>` then remake the PGEN file with the `--vcf-half-call 
                        r` option."""
                    )

                pvar = pvar.with_columns(
                    POS=pl.col("POS") - 1,
                    ILEN=(
                        pl.col("ALT").str.len_bytes().cast(pl.Int32)
                        - pl.col("REF").str.len_bytes().cast(pl.Int32)
                    ),
                )

                # ends in reference coordiantes, 0-based **inclusive**
                # end_to_var_idx is a mapping from variants sorted by end to the
                # earliest positioned variant that has an end >= the end of the variant
                # at that index e.g. if the v0 has end 200, and v1 has end 100, then \
                # ends would be sorted as [v1, v0] and end_to_var_idx[0] = 1 and
                # end_to_var_idx[1] = 1
                ends = (
                    pvar.with_row_count("VAR_IDX")
                    .with_columns(
                        END=pl.col("POS") - pl.col("ILEN").clip_max(0)  #! end-inclusive
                    )
                    # variants are sorted by pos
                    .filter(pl.col("POS") == pl.col("POS").min().over("#CHROM", "END"))
                    .select("#CHROM", "END", "VAR_IDX")
                )
                ends = (
                    ends.group_by("#CHROM", maintain_order=True)
                    .agg(
                        pl.all().sort_by("END"),
                    )
                    .explode(pl.exclude("#CHROM"))
                    .group_by("#CHROM", maintain_order=True)
                    .agg(
                        pl.all(),
                        pl.col("VAR_IDX")
                        .reverse()
                        .rolling_min(ends.height, min_periods=1)
                        .reverse()
                        .alias("END_TO_VAR_IDX"),
                    )
                    .explode(pl.exclude("#CHROM"))
                    .select("#CHROM", "END", "END_TO_VAR_IDX")
                )
                pvar.write_ipc(pvar_arrow_path)
                ends.write_ipc(ends_arrow_path)

            if contig == "_all":
                last_offset = 0
                for _contig, partition in pvar.partition_by(
                    "#CHROM", maintain_order=True, as_dict=True
                ).items():
                    self.contig_offsets[_contig] = last_offset + partition.height
                    last_offset = self.contig_offsets[_contig]
                    self.positions[_contig] = partition["POS"].to_numpy()
                    # no multi-allelics
                    self.ref[_contig] = VLenAlleles.from_polars(partition["REF"])
                    self.alt[_contig] = VLenAlleles.from_polars(partition["ALT"])
                    self.size_diffs[_contig] = partition["ILEN"].to_numpy()

                for _contig, partition in ends.partition_by(
                    "#CHROM", as_dict=True
                ).items():
                    self.ends[_contig] = partition["END"].to_numpy()
                    self.end_to_var_idx[_contig] = partition[
                        "END_TO_VAR_IDX"
                    ].to_numpy()

                # make all contigs map to the same pgen file
                pgen_path = self.pgen_paths["_all"]
                self.pgen_paths = {contig: pgen_path for contig in self.contig_offsets}
            else:
                self.contig_offsets[contig] = 0
                self.positions[contig] = pvar["POS"].to_numpy()
                self.size_diffs[contig] = pvar["ILEN"].to_numpy()
                self.ends[contig] = ends["END"].to_numpy()
                self.end_to_var_idx[contig] = ends["END_TO_VAR_IDX"].to_numpy()
                # no multi-allelics
                self.ref[contig] = VLenAlleles.from_polars(pvar["REF"])
                self.alt[contig] = VLenAlleles.from_polars(pvar["ALT"])

        self.contigs = list(self.contig_offsets.keys())
        self.contig_starts_with_chr = self.infer_contig_prefix(self.contigs)

    def _pgen(self, contig: str, sample_idx: Optional[NDArray[np.uint32]]):
        if sample_idx is not None:
            sample_idx = np.sort(sample_idx)
        return pgenlib.PgenReader(
            bytes(self.pgen_paths[contig]), sample_subset=sample_idx
        )

    def read(
        self, contig: str, starts: NDArray[np.int64], ends: NDArray[np.int64], **kwargs
    ) -> List[Optional[DenseGenotypes]]:
        samples = kwargs.get("sample", None)
        if samples is None:
            n_samples = self.n_samples
            pgen_idx, query_idx = self.sample_idx, None
        else:
            n_samples = len(samples)
            pgen_idx, query_idx = self.get_sample_idx(samples)

        contig = self.normalize_contig_name(contig)

        out: List[Optional[DenseGenotypes]] = [None] * len(starts)

        # contig is not present in PGEN, has no variants
        if contig not in self.contigs:
            return out

        _s_idxs = np.searchsorted(self.ends[contig], starts)
        # absolute idxs
        s_idxs = self.end_to_var_idx[contig][_s_idxs] + self.contig_offsets[contig]
        e_idxs = (
            np.searchsorted(self.positions[contig], ends) + self.contig_offsets[contig]
        )

        # absolute idxs
        min_s_idx = s_idxs.min()
        max_e_idx = e_idxs.max()

        if min_s_idx == max_e_idx:
            return out

        # get alleles
        genotypes = np.empty(
            (max_e_idx - min_s_idx, n_samples * self.ploidy), dtype=np.int32
        )
        with self._pgen(contig, pgen_idx) as f:
            # (v s*2)
            f.read_alleles_range(min_s_idx, max_e_idx, genotypes)

        genotypes = genotypes.astype(np.int8)
        # (s*2 v)
        genotypes = genotypes.swapaxes(0, 1)
        # (s 2 v)
        genotypes = np.stack([genotypes[::2], genotypes[1::2]], axis=1)

        # re-order samples to be in query order
        if query_idx is not None:
            genotypes = genotypes[query_idx]

        # contig relative idxs
        rel_s_idxs = s_idxs - self.contig_offsets[contig]
        rel_e_idxs = e_idxs - self.contig_offsets[contig]
        for i, (s_idx, e_idx) in enumerate(zip(rel_s_idxs, rel_e_idxs)):
            # genotype query relative idxs
            rel_s_idx = 0
            rel_e_idx = e_idx - s_idx
            if s_idx == e_idx:
                out[i] = None
            else:
                out[i] = DenseGenotypes(
                    positions=self.positions[contig][s_idx:e_idx],
                    size_diffs=self.size_diffs[contig][s_idx:e_idx],
                    ref=self.ref[contig][s_idx:e_idx],
                    alt=self.alt[contig][s_idx:e_idx],
                    genotypes=genotypes[..., rel_s_idx:rel_e_idx],
                )

        return out

    def read_for_haplotype_construction(
        self,
        contig: str,
        starts: NDArray[np.int64],
        ends: NDArray[np.int64],
        target_length: int,
        **kwargs,
    ) -> Tuple[List[Optional[DenseGenotypes]], NDArray[np.int64]]:
        """Read genotypes for haplotype construction. This is a special case of `read` 
        where variants beyond the query regions are included to ensure that haplotypes 
        of `target_length` can be constructed. This is necessary because deletions can
        shorten the haplotype, so variants downstream of `end` may be needed to add more
        sequence to the haplotype.

        Parameters
        ----------
        contig : str
            Name of the contig/chromosome.
        starts : NDArray[int32]
            Start coordinates, 0-based.
        ends : int, NDArray[int32]
            End coordinates, 0-based exclusive.
        target_length : int
            Target length of the reconstructed haplotypes.

        Returns
        -------
        List[Optional[DenseGenotypes]]
            Genotypes for each query region.
        NDArray[np.int64]
            New ends for querying the reference genome such that enough sequence is 
            available to get haplotypes of `target_length`.
        """ """"""
        samples = kwargs.get("sample", None)
        if samples is None:
            n_samples = self.n_samples
            pgen_idx, query_idx = self.sample_idx, None
        else:
            n_samples = len(samples)
            pgen_idx, query_idx = self.get_sample_idx(samples)

        ploid = kwargs.get("ploid", None)
        if ploid is None:
            ploid = np.arange(self.ploidy)
        else:
            ploid = np.asarray(ploid)

        starts, ends = np.asarray(starts, dtype=np.int64), np.asarray(
            ends, dtype=np.int64
        )

        contig = self.normalize_contig_name(contig)

        out: List[Optional[DenseGenotypes]] = [None] * len(starts)

        # contig is not present in PGEN, has no variants
        if contig not in self.contigs:
            return out, ends

        _s_idxs = np.searchsorted(self.ends[contig], starts)
        s_idxs = self.end_to_var_idx[contig][_s_idxs]

        max_ends, e_idxs = get_ends_and_idxs(
            starts,
            s_idxs - self.contig_offsets[contig],
            self.positions[contig],
            self.size_diffs[contig],
            ends - starts,
        )
        # make idxs absolute
        e_idxs += self.contig_offsets[contig]

        with self._pgen(contig, pgen_idx) as f:
            for i, (s_idx, e_idx) in enumerate(zip(s_idxs, e_idxs)):
                # no variants in query regions
                if s_idx == e_idx:
                    out[i] = None
                    continue

                genotypes = np.empty(
                    (e_idx - s_idx, n_samples * self.ploidy), dtype=np.int32
                )
                f.read_alleles_range(s_idx, e_idx, genotypes)
                genotypes = genotypes.astype(np.int8)
                # (s*2 v)
                genotypes = genotypes.swapaxes(0, 1)
                # (s 2 v)
                genotypes = np.stack([genotypes[::2], genotypes[1::2]], axis=1)

                # re-order samples to be in query order
                if query_idx is not None:
                    genotypes = genotypes[query_idx]

                out[i] = DenseGenotypes(
                    positions=self.positions[contig][s_idx:e_idx],
                    size_diffs=self.size_diffs[contig][s_idx:e_idx],
                    ref=self.ref[contig][s_idx:e_idx],
                    alt=self.alt[contig][s_idx:e_idx],
                    genotypes=genotypes[:, ploid],
                )

        return (out, max_ends)

    def get_sample_idx(self, samples):
        _samples, _pgen_idx, query_idx = np.intersect1d(
            self.samples, samples, return_indices=True
        )
        if len(_samples) != len(samples):
            unrecognized_samples = set(samples) - set(_samples)
            raise ValueError(
                f"""Got samples that are not in the pgen file: 
                {unrecognized_samples}."""
            )
        pgen_idx: NDArray[np.uint32] = self.sample_idx[_pgen_idx]
        return pgen_idx, query_idx


@nb.njit(nogil=True, cache=True)
def get_ends_and_idxs(
    starts: NDArray[np.int64],
    start_idxs: NDArray[np.uint32],
    positions: NDArray[np.int32],
    size_diffs: NDArray[np.int32],
    target_lengths: NDArray[np.integer],
) -> Tuple[NDArray[np.int64], NDArray[np.uint32]]:
    """Note: this operates on relative idxs."""
    max_ends = starts + target_lengths
    end_idxs = start_idxs.copy()

    if len(positions) == 1:
        var_pos = positions[0]
        var_diff = size_diffs[0]
        max_ends[:] = starts + target_lengths
        if var_diff < 0:
            max_ends -= var_diff
        end_idxs[:] = start_idxs + 1
        return max_ends, end_idxs

    for i in nb.prange(len(start_idxs)):
        # positive, how much deleted
        total_del = 0
        start = starts[i]
        ref_pos = start
        length = 0
        var_idx = start_idxs[i]
        target_length = target_lengths[i]

        # no variants
        if var_idx >= len(positions):
            continue

        var_diff = size_diffs[var_idx]
        prev_del_pos = positions[var_idx]
        prev_del_end = prev_del_pos
        prev_del_diff = 0

        # no variants in span of start + target_length
        if prev_del_pos >= start + target_length:
            continue

        for var_idx in range(start_idxs[i], len(positions)):
            var_pos = positions[var_idx]
            var_diff = size_diffs[var_idx]
            # pos + 1 nt of ref - var_diff (neg for dels)
            var_end = var_pos + 1 - var_diff

            if var_pos < start:
                # could get a del_end <= start because variants are sorted by
                # pos->ends. even tho start_idx is the first end to span start,
                # variants that do not span start can come after it
                var_diff = min(0, start - var_end)

            if var_diff < 0:
                # split multiallelic, use the largest del at the site
                # they are sorted so the last one is largest
                if var_pos == prev_del_pos:
                    prev_del_diff = var_diff
                    prev_del_pos = var_pos
                    prev_del_end = var_end
                # overlapping deletion
                elif var_pos < prev_del_end:
                    # ignore del if it's larger (i.e. shorter)
                    if var_diff >= prev_del_diff:
                        continue
                    # use this deletion instead
                    else:
                        prev_del_diff = var_diff
                        prev_del_pos = var_pos
                        prev_del_end = var_end
                # prev_del is the largest del from the prev position
                elif var_pos >= prev_del_end:
                    # increment total_del by var_diff (negative for dels)
                    total_del -= prev_del_diff
                    # update prev_del
                    prev_del_diff = var_diff
                    prev_del_pos = var_pos
                    prev_del_end = var_end

                length = var_pos - start - total_del
                if length >= target_length:
                    break
            else:
                length = var_pos - start - total_del
                if length > target_length:
                    # stop at this variant
                    var_idx -= 1
                    break

        total_del -= prev_del_diff
        ref_pos = prev_del_end
        length = ref_pos - start - total_del
        ref_pos += target_length - length

        max_ends[i] = ref_pos
        end_idxs[i] = var_idx + 1

    return max_ends, end_idxs
