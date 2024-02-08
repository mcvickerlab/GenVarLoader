from pathlib import Path
from typing import Iterable, List, Optional, Union, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray

from .types import SparseAlleles, Variants

try:
    import tiledbvcf

    TILEDBVCF_INSTALLED = True
except ImportError:
    TILEDBVCF_INSTALLED = False


class TileDB_VCF(Variants):
    def __init__(
        self, path: Union[str, Path], ploidy: int, samples: Optional[List[str]] = None
    ) -> None:
        """Read variants from a TileDB-VCF store.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the TileDB-VCF store.
        ploidy : int
            Ploidy of the genotypes, e.g. humans are diploid so ploidy = 2.
        samples : Optional[List[str]], optional
            Names of the samples to read, by default all samples available are read.
        """
        if not TILEDBVCF_INSTALLED:
            raise ImportError(
                "TileDB-VCF must be installed to read TileDB-VCF datasets."
            )

        self.path = path
        self.ds = tiledbvcf.Dataset(str(path))
        self.PLOIDY = ploidy
        if samples is None:
            self.samples = self.ds.samples()
        else:
            self.samples = samples
        self.n_samples = len(self.samples)
        self.contig_starts_with_chr = None

    def read(
        self, contig: str, start: int, end: int, **kwargs
    ) -> Optional[SparseAlleles]:
        samples: Iterable[str]
        samples = kwargs.get("sample", None)
        if samples is None:
            samples = self.samples

        ploid: Iterable[int]
        ploid = kwargs.get("ploid", None)
        if ploid is None:
            ploid = range(self.PLOIDY)

        region = f"{contig}:{start+1}-{end}"
        df = self.ds.read_arrow(
            ["pos_start", "alleles", "fmt_GT", "sample_name"],
            regions=[region],
            samples=samples,
        )
        df = cast(pl.DataFrame, pl.from_arrow(df))

        # infer contig prefix
        if self.contig_starts_with_chr is None and df.height > 0:
            self.contig_starts_with_chr = self.infer_contig_prefix([contig])
        elif self.contig_starts_with_chr is None and df.height == 0:
            if contig.startswith("chr"):
                contig = contig[3:]
                starts_with_chr = False
            else:
                contig = "chr" + contig
                starts_with_chr = True
            region = f"{contig}:{start+1}-{end}"
            df = self.ds.read_arrow(
                ["pos_start", "alleles", "fmt_GT", "sample_name"],
                regions=[region],
                samples=samples,
            )
            df = cast(pl.DataFrame, pl.from_arrow(df))
            if df.height > 0:
                self.contig_starts_with_chr = starts_with_chr

        if df.height == 0:
            return

        df = (
            df.filter(~pl.col("fmt_GT").list.contains(-1))
            .with_row_count()
            .explode("fmt_GT")
            .groupby("row_nr")
            .agg(
                pl.exclude("alleles", "fmt_GT").first(),
                pl.col("alleles").list.get(pl.col("fmt_GT")),
            )
            .drop("row_nr")
        )

        alleles = (
            pl.col("alleles").list.get(int(p)).alias(f"allele_{p}") for p in ploid
        )
        with pl.StringCache():
            pl.Series(samples, dtype=pl.Categorical)
            df = (
                df.sort(pl.col("sample_name").cast(pl.Categorical), "pos_start")
                .with_columns(*alleles)
                .drop("alleles")
            )

        counts = (
            df.select(pl.col("sample_name").rle())
            .unnest("sample_name")["lengths"]
            .cast(pl.UInt32)
            .to_numpy()
        )
        counts = cast(NDArray[np.uint32], counts)
        offsets = np.zeros(len(counts) + 1, dtype=counts.dtype)
        counts.cumsum(out=offsets[1:])
        positions = df["pos_start"].to_numpy() - 1  # convert to 0-based
        alleles = [f"allele_{p}" for p in ploid]
        alleles = df.select(alleles).to_numpy().astype("S1").T
        # offsets (samples + 1)
        # positions (variants)
        # alleles (ploid, variants)
        return SparseAlleles(offsets, positions, alleles)
