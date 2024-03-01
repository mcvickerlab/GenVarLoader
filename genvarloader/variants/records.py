import re
from pathlib import Path
from textwrap import dedent
from typing import Dict, Iterable, List, Optional, Tuple, Union, overload

import numba as nb
import numpy as np
import polars as pl
from attrs import define
from loguru import logger
from numpy.typing import ArrayLike, NDArray
from typing_extensions import Self

try:
    import cyvcf2

    CYVCF2_INSTALLED = True
except ImportError:
    CYVCF2_INSTALLED = False


@define
class VLenAlleles:
    """Variable length alleles.

    Create VLenAlleles from a polars Series of strings:
    >>> alleles = VLenAlleles.from_polars(pl.Series(["A", "AC", "G"]))

    Create VLenAlleles from offsets and alleles:
    >>> offsets = np.array([0, 1, 3, 4], np.uint32)
    >>> alleles = np.frombuffer(b"AACG", "|S1")
    >>> alleles = VLenAlleles(offsets, alleles)

    Get a single allele:
    >>> alleles[0]
    b'A'

    Get a slice of alleles:
    >>> alleles[1:]
    VLenAlleles(offsets=array([0, 2, 3]), alleles=array([b'AC', b'G'], dtype='|S1'))
    """

    offsets: NDArray[np.uint32]
    alleles: NDArray[np.bytes_]

    @overload
    def __getitem__(self, idx: int) -> NDArray[np.bytes_]:
        ...

    @overload
    def __getitem__(self, idx: slice) -> "VLenAlleles":
        ...

    def __getitem__(self, idx: Union[int, slice, np.integer]):
        if isinstance(idx, (int, np.integer)):
            return self.get_idx(idx)
        elif isinstance(idx, slice):
            return self.get_slice(idx)

    def get_idx(self, idx: Union[int, np.integer]):
        if idx >= len(self) or idx < -len(self):
            raise IndexError("Index out of range.")
        if idx < 0:
            idx = len(self) + idx
        return self.alleles[self.offsets[idx] : self.offsets[idx + 1]]

    def get_slice(self, slc: slice):
        start: Optional[int]
        stop: Optional[int]
        start, stop = slc.start, slc.stop
        if start is None:
            start = 0
        elif start < 0:
            start += len(self)

        # handle empty result
        if start >= len(self) or (stop is not None and stop <= start):
            return VLenAlleles(np.empty(0, np.uint32), np.empty(0, "|S1"))

        if stop is None:
            stop = len(self)
        elif stop < 0:
            stop += len(self)
        stop += 1
        new_offsets = self.offsets[start:stop].copy()
        _start, _stop = new_offsets[0], new_offsets[-1]
        new_alleles = self.alleles[_start:_stop]
        new_offsets -= self.offsets[start]
        return VLenAlleles(new_offsets, new_alleles)

    def __len__(self):
        """Number of alleles."""
        return len(self.offsets) - 1

    @property
    def nbytes(self):
        return self.offsets.nbytes + self.alleles.nbytes

    @classmethod
    def from_polars(cls, alleles: pl.Series):
        offsets = np.r_[np.uint32(0), alleles.str.len_bytes().cum_sum().to_numpy()]
        flat_alleles = np.frombuffer(
            alleles.str.concat("").to_numpy()[0].encode(), "S1"
        )
        return cls(offsets, flat_alleles)

    @staticmethod
    def concat(*vlen_alleles: "VLenAlleles"):
        if len(vlen_alleles) == 0:
            return VLenAlleles(np.array([0], np.uint32), np.array([], "|S1"))
        elif len(vlen_alleles) == 1:
            return vlen_alleles[0]

        offset_ends = np.array([v.offsets[-1] for v in vlen_alleles[:-1]]).cumsum(
            dtype=np.uint32
        )
        offsets = np.concatenate(
            [
                vlen_alleles[0].offsets,
                *(
                    v.offsets + last_offset_end
                    for v, last_offset_end in zip(vlen_alleles[1:], offset_ends)
                ),
            ]
        )
        alleles = np.concatenate([v.alleles for v in vlen_alleles])
        return VLenAlleles(offsets, alleles)


@define
class RecordInfo:
    positions: NDArray[np.int32]
    size_diffs: NDArray[np.int32]
    refs: VLenAlleles
    alts: VLenAlleles
    start_idxs: NDArray[np.int32]
    end_idxs: NDArray[np.int32]
    offsets: NDArray[np.uint32]


@define
class Records:
    contigs: List[str]
    contig_offsets: Dict[str, int]

    ## sorted by starts ##
    v_starts: Dict[str, NDArray[np.int32]]
    # difference in length between ref and alt, sorted by start
    v_diffs: Dict[str, NDArray[np.int32]]
    # no multi-allelics
    ref: Dict[str, VLenAlleles]
    alt: Dict[str, VLenAlleles]
    ######################

    ## sorted by ends ##
    v_ends: Dict[str, NDArray[np.int32]]
    e2s_idx: Dict[str, NDArray[np.int32]]
    v_diffs_sorted_by_ends: Dict[str, NDArray[np.int32]]
    max_del_q: Dict[str, NDArray[np.intp]]
    ####################

    @property
    def n_variants(self) -> int:
        return sum(len(v) for v in self.v_starts.values())

    @classmethod
    def from_vcf(cls, vcf: Union[str, Path, Dict[str, Path]]) -> Self:
        if not CYVCF2_INSTALLED:
            raise ImportError(
                "cyvcf2 is not installed. Please install it with `pip install cyvcf2`"
            )

        if isinstance(vcf, (str, Path)):
            vcf = {"_all": Path(vcf)}

        if "_all" in vcf:
            multi_contig_source = True
        else:
            multi_contig_source = False

        vcf_suffix = re.compile(r"\.[vb]cf(\.gz)?$")
        arrow_paths = {
            c: p.parent / vcf_suffix.sub(".gvl.arrow", p.name) for c, p in vcf.items()
        }

        if cls.gvl_arrow_exists(vcf, arrow_paths):
            return cls.from_gvl_arrow(arrow_paths)

        if multi_contig_source:
            start_df = cls.read_vcf(vcf["_all"])
            start_dfs = start_df.partition_by("#CHROM", as_dict=True)
        else:
            start_dfs: Dict[str, pl.DataFrame] = {}
            for contig, path in vcf.items():
                start_dfs[contig] = cls.read_vcf(path)

        start_dfs, end_dfs = cls.process_start_df(start_dfs)

        if multi_contig_source:
            path = vcf["_all"]
            pl.concat(start_dfs.values()).write_ipc(
                path.parent / vcf_suffix.sub(".gvl.arrow", path.name)
            )
            pl.concat(end_dfs.values()).write_ipc(
                path.parent / vcf_suffix.sub(".gvl.ends.arrow", path.name)
            )
        else:
            for s_df, e_df, path in zip(
                start_dfs.values(), end_dfs.values(), vcf.values()
            ):
                s_df.write_ipc(path.parent / vcf_suffix.sub(".gvl.arrow", path.name))
                e_df.write_ipc(
                    path.parent / vcf_suffix.sub(".gvl.ends.arrow", path.name)
                )

        return cls.from_var_df(start_dfs, end_dfs)

    @staticmethod
    def read_vcf(vcf_path: Path):
        if not CYVCF2_INSTALLED:
            raise ImportError(
                "cyvcf2 is not installed. Please install it with `pip install cyvcf2`"
            )

        vcf = cyvcf2.VCF(str(vcf_path))
        n_variants = vcf.num_records
        chroms = np.empty(n_variants, dtype=np.object_)
        positions = np.empty(n_variants, dtype=np.int32)
        refs = np.empty(n_variants, dtype=np.object_)
        alts = np.empty(n_variants, dtype=np.object_)
        for i, v in enumerate(vcf):
            chroms[i] = v.CHROM
            positions[i] = v.POS
            refs[i] = v.REF
            # TODO: punt multi-allelics. also punt missing ALT?
            alt = v.ALT
            if len(alt) != 1:
                raise RuntimeError(
                    f"""VCF file {vcf_path} contains multi-allelic or overlappings
                    variants which are not yet supported by GenVarLoader. Normalize 
                    the VCF with `bcftools norm -f <reference.fa>
                    -a --atom-overlaps . -m - <file.vcf>`"""
                )
            alts[i] = alt[0]
        return pl.DataFrame(
            {
                "#CHROM": chroms.astype(str),
                "POS": positions,
                "REF": refs.astype(str),
                "ALT": alts.astype(str),
            }
        )

    @classmethod
    def from_pvar(cls, pvar: Union[str, Path, Dict[str, Path]]) -> Self:
        if isinstance(pvar, (str, Path)):
            pvar = {"_all": Path(pvar)}

        if "_all" in pvar:
            multi_contig_source = True
        else:
            multi_contig_source = False

        arrow_paths = {c: p.with_suffix(".gvl.arrow") for c, p in pvar.items()}

        if cls.gvl_arrow_exists(pvar, arrow_paths):
            return cls.from_gvl_arrow(arrow_paths)

        if multi_contig_source:
            start_df = cls.read_pvar(pvar["_all"])
            start_dfs = start_df.partition_by("#CHROM", as_dict=True)
        else:
            start_dfs: Dict[str, pl.DataFrame] = {}
            for contig, path in pvar.items():
                start_dfs[contig] = cls.read_pvar(path)

        start_dfs, end_dfs = cls.process_start_df(start_dfs)

        if multi_contig_source:
            path = pvar["_all"]
            pl.concat(start_dfs.values()).write_ipc(path.with_suffix(".gvl.arrow"))
            pl.concat(end_dfs.values()).write_ipc(path.with_suffix(".gvl.ends.arrow"))
        else:
            for s_df, e_df, path in zip(
                start_dfs.values(), end_dfs.values(), pvar.values()
            ):
                s_df.write_ipc(path.with_suffix(".gvl.arrow"))
                e_df.write_ipc(path.with_suffix(".gvl.ends.arrow"))

        return cls.from_var_df(start_dfs, end_dfs)

    @staticmethod
    def read_pvar(pvar_path: Path):
        with open(pvar_path, "r") as f:
            skip_rows = 0
            while f.readline().startswith("##"):
                skip_rows += 1

        logger.info("Reading .pvar file...")
        pvar = pl.read_csv(
            pvar_path,
            separator="\t",
            skip_rows=skip_rows,
            columns=["#CHROM", "POS", "REF", "ALT"],
            dtypes={"#CHROM": pl.Utf8, "POS": pl.Int32},
        )
        logger.info("Finished reading .pvar file.")
        if (pvar["ALT"].str.contains(",")).any():
            raise RuntimeError(
                f"""PGEN file {pvar_path} contains multi-allelic variants which are 
                not yet supported by GenVarLoader. Split your multi-allelic 
                variants with `bcftools norm -f <reference.fa> -a
                --atom-overlaps . -m - <file.vcf>` then remake the PGEN file
                with the `--vcf-half-call r` option."""
            )
        return pvar

    @staticmethod
    def gvl_arrow_exists(
        sources: Union[Path, Dict[str, Path]], arrow: Union[Path, Dict[str, Path]]
    ) -> bool:
        # TODO: check if files were created after the last breaking change to the gvl.arrow format
        # check if the files exist
        if isinstance(arrow, Path):
            assert isinstance(sources, Path)
            if not arrow.exists():
                return False
            if not arrow.stat().st_mtime >= sources.stat().st_mtime:
                return False
        else:
            assert isinstance(sources, dict)
            for contig, p in arrow.items():
                if not p.exists():
                    return False
                if not p.stat().st_mtime > sources[contig].stat().st_mtime:
                    return False

        return True

    @classmethod
    def from_gvl_arrow(cls, arrow: Union[str, Path, Dict[str, Path]]) -> Self:
        if isinstance(arrow, (str, Path)):
            arrow = {"_all": Path(arrow)}

        if "_all" in arrow:
            multi_contig_source = True
        else:
            multi_contig_source = False

        arrow_paths: Dict[str, Path] = {}
        for c, p in arrow.items():
            if p.suffix == ".gvl":
                arrow_paths[c] = p.with_suffix(".gvl.arrow")
            elif p.suffixes[-2:] == [".gvl", ".arrow"]:
                arrow_paths[c] = p
            else:
                raise ValueError(
                    f"Arrow file {p} does not have the .gvl suffix. Please provide the correct path."
                )

        if multi_contig_source:
            path = arrow_paths["_all"]
            start_dfs = pl.read_ipc(path).partition_by("#CHROM", as_dict=True)
            end_dfs = pl.read_ipc(
                path.parent / path.name.replace(".gvl.arrow", ".gvl.ends.arrow")
            ).partition_by("#CHROM", as_dict=True)
        else:
            start_dfs: Dict[str, pl.DataFrame] = {}
            end_dfs: Dict[str, pl.DataFrame] = {}
            for contig, path in arrow_paths.items():
                start_dfs[contig] = pl.read_ipc(path)
                end_dfs[contig] = pl.read_ipc(
                    path.parent / path.name.replace(".gvl.arrow", ".gvl.ends.arrow")
                )

        return cls.from_var_df(start_dfs, end_dfs)

    @staticmethod
    def process_start_df(start_dfs: Dict[str, pl.DataFrame]):
        """_summary_

        Parameters
        ----------
        start_dfs : Dict[str, pl.DataFrame]
            Each dataframe should have columns: POS, REF, ALT

        Returns
        -------
        start_dfs : Dict[str, pl.DataFrame]
        end_dfs : Dict[str, pl.DataFrame]
        """
        end_dfs: Dict[str, pl.DataFrame] = {}
        for contig, df in start_dfs.items():
            df = df.with_columns(
                POS=pl.col("POS") - 1,  #! change to 0-indexed
                ILEN=(
                    pl.col("ALT").str.len_bytes().cast(pl.Int32)
                    - pl.col("REF").str.len_bytes().cast(pl.Int32)
                ),
            )
            start_dfs[contig] = df

            ends = (
                df.select(
                    "#CHROM",
                    "ILEN",
                    END=pl.col("POS")
                    - pl.col("ILEN").clip(upper_bound=0),  #! end-inclusive
                )
                .with_row_count("VAR_IDX")
                .select(
                    pl.all().sort_by("END"),
                    # make E2S_IDX relative to each contig
                    pl.int_range(0, pl.count(), dtype=pl.UInt32)
                    .sort_by("END")
                    .reverse()
                    .rolling_min(df.height, min_periods=1)
                    .reverse()
                    .alias("E2S_IDX"),
                )
                .select("#CHROM", "END", "ILEN", "VAR_IDX", "E2S_IDX")
                .sort("END")
                .join(df.select("POS").with_row_count("VAR_IDX"), on="VAR_IDX")
            )

            _starts = ends["POS"].to_numpy()
            _ends = ends["END"].to_numpy()
            was_ends = np.empty(len(_ends) + 1, dtype=_ends.dtype)
            was_ends[0] = 0
            #! convert to end-exclusive, + 1
            was_ends[1:] = _ends + 1
            md_q = np.searchsorted(was_ends, _starts, side="right") - 1
            ends = ends.with_columns(MD_Q=md_q).select(
                "#CHROM", "END", "MD_Q", "ILEN", "E2S_IDX"
            )

            end_dfs[contig] = ends

        return start_dfs, end_dfs

    @classmethod
    def from_var_df(
        cls, start_dfs: Dict[str, pl.DataFrame], end_dfs: Dict[str, pl.DataFrame]
    ) -> Self:
        ## sorted by starts ##
        v_starts: Dict[str, NDArray[np.int32]] = {}
        # difference in length between ref and alt, sorted by start
        v_diffs: Dict[str, NDArray[np.int32]] = {}
        # no multi-allelics
        ref: Dict[str, VLenAlleles] = {}
        alt: Dict[str, VLenAlleles] = {}
        ######################

        ## sorted by ends ##
        v_ends: Dict[str, NDArray[np.int32]] = {}
        v_diffs_sorted_by_ends: Dict[str, NDArray[np.int32]] = {}
        e2s_idx: Dict[str, NDArray[np.int32]] = {}
        max_del_q: Dict[str, NDArray[np.int32]] = {}
        ####################

        contig_offset = 0
        contig_offsets: Dict[str, int] = {}
        for contig, s_df in start_dfs.items():
            e_df = end_dfs[contig]
            contig_offsets[contig] = contig_offset
            contig_offset += s_df.height
            v_starts[contig] = s_df["POS"].to_numpy()
            v_diffs[contig] = s_df["ILEN"].to_numpy()
            v_ends[contig] = e_df["END"].to_numpy()
            v_diffs_sorted_by_ends[contig] = e_df["ILEN"].to_numpy()
            e2s_idx[contig] = np.empty(e_df.height + 1, dtype=np.uint32)
            e2s_idx[contig][:-1] = e_df["E2S_IDX"].to_numpy()
            e2s_idx[contig][-1] = e_df.height
            max_del_q[contig] = e_df["MD_Q"].to_numpy()

            # no multi-allelics
            ref[contig] = VLenAlleles.from_polars(s_df["REF"])
            alt[contig] = VLenAlleles.from_polars(s_df["ALT"])

        return cls(
            contigs=list(start_dfs.keys()),
            contig_offsets=contig_offsets,
            v_starts=v_starts,
            v_diffs=v_diffs,
            ref=ref,
            alt=alt,
            v_ends=v_ends,
            v_diffs_sorted_by_ends=v_diffs_sorted_by_ends,
            e2s_idx=e2s_idx,
            max_del_q=max_del_q,
        )

    def vars_in_range(
        self,
        contig: str,
        starts: ArrayLike,
        ends: ArrayLike,
    ) -> Optional[RecordInfo]:
        starts = np.atleast_1d(np.asarray(starts, dtype=int))
        ends = np.atleast_1d(np.asarray(ends, dtype=int))

        _s_idxs = np.searchsorted(self.v_ends[contig], starts)
        # make idxs absolute
        s_idxs = self.e2s_idx[contig][_s_idxs] + self.contig_offsets[contig]
        e_idxs = (
            np.searchsorted(self.v_starts[contig], ends) + self.contig_offsets[contig]
        )

        if s_idxs.min() == e_idxs.max():
            return None

        n_var_per_region = e_idxs - s_idxs
        offsets = np.empty(len(n_var_per_region) + 1, dtype=np.uint32)
        offsets[0] = 0
        offsets[1:] = np.cumsum(n_var_per_region)

        rel_s_idxs = s_idxs - self.contig_offsets[contig]
        rel_e_idxs = e_idxs - self.contig_offsets[contig]

        positions = np.concatenate(
            [self.v_starts[contig][s:e] for s, e in zip(rel_s_idxs, rel_e_idxs)]
        )
        size_diffs = np.concatenate(
            [self.v_diffs[contig][s:e] for s, e in zip(rel_s_idxs, rel_e_idxs)]
        )
        ref = VLenAlleles.concat(
            *(self.ref[contig][s:e] for s, e in zip(rel_s_idxs, rel_e_idxs))
        )
        alt = VLenAlleles.concat(
            *(self.alt[contig][s:e] for s, e in zip(rel_s_idxs, rel_e_idxs))
        )

        return RecordInfo(
            positions=positions,
            size_diffs=size_diffs,
            refs=ref,
            alts=alt,
            start_idxs=s_idxs,
            end_idxs=e_idxs,
            offsets=offsets,
        )

    def vars_in_range_for_haplotype_construction(
        self,
        contig: str,
        starts: ArrayLike,
        ends: ArrayLike,
    ) -> Tuple[Optional[RecordInfo], NDArray[np.int32]]:
        starts = np.atleast_1d(np.asarray(starts, dtype=int))
        ends = np.atleast_1d(np.asarray(ends, dtype=int))

        _s_idxs = np.searchsorted(self.v_ends[contig], starts)

        max_ends, _e_idxs = get_max_ends_and_idxs(
            self.v_ends[contig],
            self.v_diffs_sorted_by_ends[contig],
            self.max_del_q[contig],
            _s_idxs,
            ends,
        )

        s_idxs = self.e2s_idx[contig][_s_idxs]
        e_idxs = self.e2s_idx[contig][_e_idxs]

        # make idxs absolute
        s_idxs += self.contig_offsets[contig]
        e_idxs += self.contig_offsets[contig]

        if s_idxs.min() == e_idxs.max():
            return None, ends.astype(np.int32)

        np.concatenate(
            [np.arange(s, e, dtype=np.uint32) for s, e in zip(s_idxs, e_idxs)]
        )
        n_var_per_region = e_idxs - s_idxs
        offsets = np.empty(len(n_var_per_region) + 1, dtype=np.uint32)
        offsets[0] = 0
        np.cumsum(n_var_per_region, out=offsets[1:])

        rel_s_idxs = s_idxs - self.contig_offsets[contig]
        rel_e_idxs = e_idxs - self.contig_offsets[contig]

        positions = np.concatenate(
            [self.v_starts[contig][s:e] for s, e in zip(rel_s_idxs, rel_e_idxs)]
        )
        size_diffs = np.concatenate(
            [self.v_diffs[contig][s:e] for s, e in zip(rel_s_idxs, rel_e_idxs)]
        )
        ref = VLenAlleles.concat(
            *(self.ref[contig][s:e] for s, e in zip(rel_s_idxs, rel_e_idxs))
        )
        alt = VLenAlleles.concat(
            *(self.alt[contig][s:e] for s, e in zip(rel_s_idxs, rel_e_idxs))
        )

        v_info = RecordInfo(
            positions=positions,
            size_diffs=size_diffs,
            refs=ref,
            alts=alt,
            start_idxs=s_idxs,
            end_idxs=e_idxs,
            offsets=offsets,
        )

        return v_info, max_ends

    def normalize_contig_name(
        self, contig: str, contigs: Iterable[str]
    ) -> Optional[str]:
        """Normalize the contig name to adhere to the convention of the underlying file.
        i.e. remove or add "chr" to the contig name.

        Parameters
        ----------
        contig : str

        Returns
        -------
        str
            Normalized contig name.
        """
        for c in contigs:
            # exact match, remove chr, add chr
            if contig == c or contig[3:] == c or f"chr{contig}" == c:
                return c
        return None

    def __repr__(self):
        return dedent(
            f"""
            {self.__class__.__name__}
            contigs: {self.contigs}
            n_variants: {self.n_variants})
            """
        ).strip()


@nb.njit(nogil=True, cache=True)
def get_max_ends_and_idxs(
    v_ends: NDArray[np.int32],
    v_diffs: NDArray[np.int32],
    nearest_nonoverlapping: NDArray[np.intp],
    start_idxs: NDArray[np.intp],
    query_ends: NDArray[np.int64],
) -> Tuple[NDArray[np.int32], NDArray[np.intp]]:
    max_ends: NDArray[np.int32] = np.empty(len(start_idxs), dtype=np.int32)
    end_idxs: NDArray[np.intp] = np.empty(len(start_idxs), dtype=np.intp)
    for r in nb.prange(len(start_idxs)):
        s = start_idxs[r]
        if s == len(v_ends):  # no variants in this region
            max_ends[r] = query_ends[r]
            end_idxs[r] = s
            continue

        w = -v_diffs[s:]  # flip sign so deletions have positive weight

        # to adjust q from [0, j) to [i, j)
        # (q[i:] - i).clip(0)
        q = (nearest_nonoverlapping[s:] - s).clip(0)
        max_end, end_idx = weighted_activity_selection(v_ends[s:], w, q, query_ends[r])
        max_ends[r] = max_end
        end_idxs[r] = s + end_idx
    return max_ends, end_idxs


@nb.njit(nogil=True, cache=True)
def weighted_activity_selection(
    v_ends: NDArray[np.int32],
    w: NDArray[np.int32],
    q: NDArray[np.intp],
    query_end: int,
) -> Tuple[int, int]:
    """Implementation of the [weighted activity selection problem](https://en.wikipedia.org/wiki/Activity_selection_problem)
    to compute the maximum length of deletions that can occur for each region. This is
    used to adjust the end coordinates for reference sequence queries and include all
    variants for that are needed for haplotype construction.

    Parameters
    ----------
    v_ends : NDArray[np.int64]
        Shape: (variants). End coordinates of variants, 0-based inclusive.
    w : NDArray[np.int64]
        Shape: (variants). Weights of activities (i.e. deletion lengths).
    q : NDArray[np.intp]
        Shape: (variants). Nearest variant i such that i < j and variants are non-overlapping, q[j] = i.
    query_end : int
        Shape: (regions). End of query region.

    Returns
    -------
    max_ends : NDArray[np.int32]
        Shape: (regions). Maximum end coordinate for each query region.
    end_idxs : NDArray[np.intp]
        Shape: (regions). Index of the variant with the maximum end coordinate for each
        query region.

    Notes
    -----
    For the weighted activity selection problem, each deletion corresponds
    to an activity with weight equal to the length of the deletion. The goal is to
    compute the maximum total weight of deletions for each query region.

    Psuedocode from (Princeton slides)[https://www.cs.princeton.edu/~wayne/cs423/lectures/dynamic-programming-4up.pdf]:
    Given starts :math: `s_1, ..., s_n`, ends :math: `e_1, ..., e_n`, and weights
    :math: `w_1, ..., w_n`.
    Note that ends are sorted, :math: `e_1 <= ... <= e_n`.
    Let :math: `q_j` = largest index :math: `i < j` such that activity :math: `i` is
    compatible with :math: `j`.
    Let opt(j) = value of solution to the problem consisting of activities 1 to j
    Then,
        opt(0) = 0
    and
        opt(j) = max(w_j + opt(q_j), opt(j - 1))
    """
    n_vars = len(w)
    max_del: NDArray[np.int32] = np.empty(n_vars + 1, dtype=np.int32)
    max_del[0] = 0
    for j in range(1, n_vars + 1):
        max_del[j] = max(max_del[q[j - 1]] + w[j - 1], max_del[j - 1])
        v_del_end = v_ends[j - 1] - max_del[j] + 1  # + 1, v_ends is end-inclusive
        # if:
        # this variant more than satisfies query length
        # last variant doesn't span v_del_end
        if v_del_end > query_end and j > 1 and v_ends[j - 2] <= v_del_end:
            # then add the max deletion length up to but not including this variant
            # to the query end, and return the index of this variant for slicing
            return query_end + max_del[j - 1], j - 1
    return query_end + max_del[-1], n_vars
