import re
import subprocess
from pathlib import Path
from textwrap import dedent
from typing import Dict, List, Literal, Optional, Union, cast, overload

import numpy as np
import polars as pl
import pyarrow as pa
from attrs import define
from loguru import logger
from numpy.typing import ArrayLike, NDArray
from tqdm.auto import tqdm
from typing_extensions import Self, assert_never

from .._utils import _lengths_to_offsets, _normalize_contig_name, _offsets_to_lengths

try:
    import cyvcf2

    CYVCF2_INSTALLED = True
except ImportError:
    CYVCF2_INSTALLED = False

__all__ = []


@define
class VLenAlleles:
    """Variable length alleles.

    Create VLenAlleles from a polars Series of strings:
    >>> alleles = VLenAlleles.from_polars(pl.Series(["A", "AC", "G"]))

    Create VLenAlleles from offsets and alleles:
    >>> offsets = np.array([0, 1, 3, 4], np.uint64)
    >>> alleles = np.frombuffer(b"AACG", "|S1")
    >>> alleles = VLenAlleles(offsets, alleles)

    Get a single allele:
    >>> alleles[0]
    b'A'

    Get a slice of alleles:
    >>> alleles[1:]
    VLenAlleles(offsets=array([0, 2, 3]), alleles=array([b'AC', b'G'], dtype='|S1'))
    """

    offsets: NDArray[np.int64]
    alleles: NDArray[np.bytes_]

    @overload
    def __getitem__(self, idx: int) -> NDArray[np.bytes_]: ...

    @overload
    def __getitem__(self, idx: slice) -> "VLenAlleles": ...

    def __getitem__(self, idx: Union[int, np.integer, slice]):
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
            return VLenAlleles(np.empty(0, np.int64), np.empty(0, "|S1"))

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
        offsets = np.empty(len(alleles) + 1, np.int64)
        offsets[0] = 0
        offsets[1:] = alleles.str.len_bytes().cast(pl.Int64).cum_sum().to_numpy()
        flat_alleles = np.frombuffer(
            alleles.str.concat("").to_numpy()[0].encode(), "S1"
        )
        return cls(offsets, flat_alleles)

    def to_polars(self):
        n_alleles = len(self)
        offset_buffer = pa.py_buffer(self.offsets)
        allele_buffer = pa.py_buffer(self.alleles)
        string_arr = pa.LargeStringArray.from_buffers(
            n_alleles, offset_buffer, allele_buffer
        )
        return pl.Series(string_arr)

    @staticmethod
    def concat(*vlen_alleles: "VLenAlleles"):
        if len(vlen_alleles) == 0:
            return VLenAlleles(np.array([0], np.int64), np.array([], "|S1"))
        elif len(vlen_alleles) == 1:
            return vlen_alleles[0]

        nuc_per_allele = np.concatenate(
            [_offsets_to_lengths(v.offsets) for v in vlen_alleles]
        )
        offsets = _lengths_to_offsets(nuc_per_allele, np.int64)
        alleles = np.concatenate([v.alleles for v in vlen_alleles])
        return VLenAlleles(offsets, alleles)


@define
class RecordInfo:
    """Information about sets of records corresponding to range queries.

    Attributes
    ----------
    positions : NDArray[np.int32]
        Shape: (n_variants)
        Positions of variants.
    size_diffs : NDArray[np.int32]
        Shape: (n_variants)
        Difference in length between ref and alt.
    refs : VLenAlleles
        Effective shape: (n_variants)
        Reference alleles.
    alts : VLenAlleles
        Effective shape: (n_variants)
        Alternate alleles.
    start_idxs : NDArray[np.int32]
        Shape: (n_queries)
        Start indices of the records for each query.
    end_idxs : NDArray[np.int32]
        Shape: (n_queries)
        End indices of the records for each query.
    offsets : NDArray[np.int32]
        Shape: (n_queries + 1).
        Offsets for the records such that records[offsets[i]:offsets[i+1]] are the records
        for the i-th query.
    """

    positions: NDArray[np.int32]  # (n_variants)
    size_diffs: NDArray[np.int32]  # (n_variants)
    refs: VLenAlleles
    alts: VLenAlleles
    start_idxs: NDArray[np.int32]  # (n_queries)
    end_idxs: NDArray[np.int32]  # (n_queries)
    offsets: NDArray[np.int64]  # (n_queries + 1)

    @property
    def n_queries(self):
        return len(self.start_idxs)

    @classmethod
    def empty(cls, n_queries: int):
        return cls(
            positions=np.empty(n_queries, np.int32),
            size_diffs=np.empty(n_queries, np.int32),
            refs=VLenAlleles(
                np.zeros(n_queries + 1, np.int64), np.empty(n_queries, "S1")
            ),
            alts=VLenAlleles(
                np.zeros(n_queries + 1, np.int64), np.empty(n_queries, "S1")
            ),
            start_idxs=np.empty(n_queries, np.int32),
            end_idxs=np.empty(n_queries, np.int32),
            offsets=np.zeros(n_queries + 1, np.int64),
        )

    @staticmethod
    def concat(
        *record_infos: "RecordInfo", how: Literal["separate", "merge"] = "separate"
    ):
        if len(record_infos) == 0:
            return RecordInfo.empty(0)
        elif len(record_infos) == 1:
            return record_infos[0]

        positions = np.concatenate([r.positions for r in record_infos])
        size_diffs = np.concatenate([r.size_diffs for r in record_infos])
        refs = VLenAlleles.concat(*(r.refs for r in record_infos))
        alts = VLenAlleles.concat(*(r.alts for r in record_infos))
        start_idxs = np.concatenate([r.start_idxs for r in record_infos])
        end_idxs = np.concatenate([r.end_idxs for r in record_infos])
        v_per_query = np.concatenate(
            [_offsets_to_lengths(r.offsets) for r in record_infos]
        )
        if how == "separate":
            offsets = _lengths_to_offsets(v_per_query)
        elif how == "merge":
            offsets = np.array([0, v_per_query.sum()], np.int64)
        else:
            assert_never(how)
        return RecordInfo(
            positions=positions,
            size_diffs=size_diffs,
            refs=refs,
            alts=alts,
            start_idxs=start_idxs,
            end_idxs=end_idxs,
            offsets=offsets,
        )


@define
class Records:
    contigs: List[str]
    contig_offsets: Dict[str, int]
    """Offsets for each contig corresponding to the underlying file(s). Thus, if a single file backs
    a Records instance, then the contig offsets will be non-zero and correspond to the first variant
    that is on that contig. If contig-specific files back the Records instance, then the contig offsets
    will all be zero."""

    v_starts: Dict[str, NDArray[np.int32]]
    """0-based positions of variants, sorted by start."""
    v_diffs: Dict[str, NDArray[np.int32]]
    """Difference in length |REF| - |ALT|, sorted by start"""
    # no multi-allelics
    ref: Dict[str, VLenAlleles]
    alt: Dict[str, VLenAlleles]
    ######################

    ## sorted by ends ##
    v_ends: Dict[str, NDArray[np.int32]]
    """0-based, end *inclusive* ends of variants, sorted by end."""
    e2s_idx: Dict[str, NDArray[np.int32]]
    """End to start index (sorted by starts). After searching for overlapping variants with Records.v_ends,
    this maps the found indices to indices of variants sorted by starts."""
    v_diffs_sorted_by_ends: Dict[str, NDArray[np.int32]]
    """Difference in length |REF| - |ALT|, sorted by end."""
    max_del_q: Dict[str, NDArray[np.intp]]
    """Pending deprecation. Maximum possible deletion length for a query including variants from index 0 to i."""
    ####################

    @property
    def n_variants(self) -> int:
        return sum(len(v) for v in self.v_starts.values())

    @classmethod
    def from_vcf(cls, vcf: Union[str, Path, Dict[str, Path]]) -> Self:
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
            start_dfs = {
                cast(str, c[0]): v
                for c, v in start_df.partition_by(
                    "#CHROM", as_dict=True, maintain_order=True
                ).items()
            }
            del start_df
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

        vcf = cyvcf2.VCF(str(vcf_path))  # pyright: ignore
        try:
            n_variants = cast(int, vcf.num_records)
        except ValueError as no_idx_err:
            logger.info(f"VCF file {vcf_path} appears to be unindexed, indexing...")
            try:
                subprocess.run(
                    ["bcftools", "index", vcf_path], check=True, capture_output=True
                )
            except subprocess.CalledProcessError as bcftools_err:
                logger.error(f"Failed to index VCF file {vcf_path}.")
                logger.error(f"stdout:\n{bcftools_err.stdout.decode()}")
                logger.error(f"stderr:\n{bcftools_err.stderr.decode()}")
                raise bcftools_err from no_idx_err
            vcf = cyvcf2.VCF(str(vcf_path))  # pyright: ignore
            n_variants = cast(int, vcf.num_records)
        chroms: List[Optional[str]] = [None] * n_variants
        positions = np.empty(n_variants, dtype=np.int32)
        refs: List[Optional[str]] = [None] * n_variants
        alts: List[Optional[str]] = [None] * n_variants
        sv_or_unknown = False
        with tqdm(
            total=n_variants, desc=f"Scanning variants from {vcf_path.name}"
        ) as pbar:
            for i, v in enumerate(vcf):
                chroms[i] = cast(str, v.CHROM)
                positions[i] = cast(int, v.POS)  # 1-indexed
                if v.is_sv or v.var_type == "unknown":
                    if not sv_or_unknown:
                        logger.warning(
                            f"VCF file {vcf_path} contains structural or unknown variants. These variants will be ignored."
                        )
                        sv_or_unknown = True
                    # use placeholder that makes ilen = 0 so it doesn't affect anything downstream
                    alt = "N" * len(v.REF)
                else:
                    alt = cast(List[str], v.ALT)
                refs[i] = cast(str, v.REF)
                # TODO: multi-allelics. Also, missing ALT (aka the * allele)?
                if len(alt) != 1:
                    raise RuntimeError(
                        f"""VCF file {vcf_path} contains multi-allelic or overlappings
                        variants which are not yet supported by GenVarLoader. Normalize 
                        the VCF with `bcftools norm -f <reference.fa>
                        -a --atom-overlaps . -m - <file.vcf>`"""
                    )
                alts[i] = alt[0]
                pbar.update()
        return pl.DataFrame(
            {
                "#CHROM": chroms,
                "POS": positions,
                "REF": refs,
                "ALT": alts,
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
            start_dfs = {
                cast(str, c[0]): v
                for c, v in start_df.partition_by(
                    "#CHROM", as_dict=True, maintain_order=True
                ).items()
            }
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
    def read_pvar(pvar_path: Path) -> pl.DataFrame:
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
            schema_overrides={"#CHROM": pl.Utf8, "POS": pl.Int32},
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
        # check if the files exist and are more recent than the sources
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
            start_dfs = {
                cast(str, c[0]): v
                for c, v in pl.read_ipc(path)
                .partition_by("#CHROM", as_dict=True)
                .items()
            }
            end_dfs = {
                cast(str, c[0]): v
                for c, v in pl.read_ipc(
                    path.parent / path.name.replace(".gvl.arrow", ".gvl.ends.arrow")
                )
                .partition_by("#CHROM", as_dict=True)
                .items()
            }
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
                .with_row_index("VAR_IDX")
                .select(
                    pl.all().sort_by("END"),
                    # make E2S_IDX relative to each contig
                    pl.int_range(0, pl.len(), dtype=pl.Int32)
                    .sort_by("END")
                    .reverse()
                    .rolling_min(df.height, min_samples=1)
                    .reverse()
                    .alias("E2S_IDX"),
                )
                .select("#CHROM", "END", "ILEN", "VAR_IDX", "E2S_IDX")
                .sort("END")
                .join(df.select("POS").with_row_index("VAR_IDX"), on="VAR_IDX")
            )

            _starts = ends["POS"].to_numpy()
            _ends = ends["END"].to_numpy()
            was_ends = np.empty(len(_ends) + 1, dtype=_ends.dtype)
            was_ends[0] = 0
            #! convert to end-exclusive, + 1
            was_ends[1:] = _ends + 1
            md_q = np.searchsorted(was_ends, _starts, side="right") - 1
            ends = ends.with_columns(MD_Q=md_q).select(  # type: ignore
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
            e2s_idx[contig] = np.empty(e_df.height + 1, dtype=np.int32)
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
    ) -> tuple[NDArray[np.int32], NDArray[np.int32]] | None:
        starts = np.atleast_1d(np.asarray(starts, dtype=np.int32))
        ends = np.atleast_1d(np.asarray(ends, dtype=np.int32))

        _contig = _normalize_contig_name(contig, self.contigs)
        if _contig is None:
            return

        rel_s_idxs = self.find_relative_start_idx(_contig, starts)
        rel_e_idxs = self.find_relative_end_idx(_contig, ends)

        if rel_s_idxs.min() == rel_e_idxs.max():
            return

        # make idxs absolute
        s_idxs = rel_s_idxs + self.contig_offsets[_contig]
        e_idxs = rel_e_idxs + self.contig_offsets[_contig]

        return s_idxs, e_idxs

    def find_relative_start_idx(
        self, contig: str, starts: NDArray[np.int32]
    ) -> NDArray[np.int32]:
        s_idx_by_end = np.searchsorted(self.v_ends[contig], starts)
        rel_s_idx = self.e2s_idx[contig][s_idx_by_end]
        return rel_s_idx

    def find_relative_end_idx(
        self, contig: str, ends: NDArray[np.int32]
    ) -> NDArray[np.int32]:
        rel_e_idx = np.searchsorted(self.v_starts[contig], ends)
        return rel_e_idx

    def __repr__(self):
        return dedent(
            f"""
            {self.__class__.__name__}
            contigs: {self.contigs}
            n_variants: {self.n_variants}
            """
        ).strip()
