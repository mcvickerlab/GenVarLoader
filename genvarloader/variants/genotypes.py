import re
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Union, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from tqdm.auto import tqdm
from typing_extensions import Self

from .records import Records

try:
    import pgenlib

    PGENLIB_INSTALLED = True
except ImportError:
    PGENLIB_INSTALLED = False

try:
    import tensorstore as ts
    import zarr

    ZARR_TENSORSTORE_INSTALLED = True
except ImportError:
    ZARR_TENSORSTORE_INSTALLED = False

try:
    import cyvcf2

    CYVCF2_INSTALLED = True
except ImportError:
    CYVCF2_INSTALLED = False


class Genotypes(Protocol):
    paths: Dict[str, Path]
    chunked: bool
    samples: NDArray[np.str_]
    ploidy: int

    def read(
        self,
        contig: str,
        start_idxs: NDArray[np.int32],
        end_idxs: NDArray[np.int32],
        sample_idx: Optional[NDArray[np.intp]] = None,
        haplotype_idx: Optional[NDArray[np.intp]] = None,
    ) -> NDArray[np.int8]:
        """Read genotypes from a contig from index i to j, 0-based exclusive.

        Parameters
        ----------
        contig : str
            Name of the contig/chromosome.
        start_idxs : NDArray[np.intp]
            Start indices, 0-based.
        end_idxs : NDArray[np.intp]
            End indices, 0-based exclusive.
        sample_idx : NDArray[np.intp], optional
            Indices of the samples to include. Must be unique.
        haplotype_idx : NDArray[np.intp], optional
            Indices of the haplotypes to include. Must be unique.

        Returns
        -------
        genotypes : NDArray[np.int8]
            Shape: (samples ploidy variants). Genotypes for each query region.
        """
        ...

    @property
    def n_samples(self) -> int:
        return len(self.samples)


class FromRecsGenos(Protocol):
    @classmethod
    def from_recs_genos(
        cls,
        records: Records,
        genotypes: Genotypes,
        paths: Optional[Dict[str, Path]] = None,
        mem: int = int(1e9),
        overwrite: bool = False,
        **kwargs,
    ) -> Self:
        ...


class PgenGenos(Genotypes):
    chunked = False
    ploidy = 2

    def __init__(self, paths: Dict[str, Path], sample_names: NDArray[np.str_]) -> None:
        if not PGENLIB_INSTALLED:
            raise ImportError(
                "pgenlib must be installed to use PGEN files for genotypes."
            )
        if len(paths) == 0:
            raise ValueError("No paths provided.")
        n_samples = None
        n_variants = 0
        for p in paths.values():
            if not p.exists():
                raise FileNotFoundError(f"PGEN file {p} not found.")
            if n_samples is None:
                n_samples = pgenlib.PgenReader(bytes(p)).get_raw_sample_ct()
            n_variants += pgenlib.PgenReader(bytes(p)).get_variant_ct()
        self.paths = paths
        self.n_samples: int = n_samples  # type: ignore
        self.n_variants = n_variants
        self.samples = sample_names
        self.handle = None

    def _pgen(self, contig: str, sample_idx: Optional[NDArray[np.uint32]]):
        return pgenlib.PgenReader(bytes(self.paths[contig]), sample_subset=sample_idx)

    def read(
        self,
        contig: str,
        start_idxs: NDArray[np.integer],
        end_idxs: NDArray[np.integer],
        sample_idx: Optional[NDArray[np.integer]] = None,
        haplotype_idx: Optional[NDArray[np.integer]] = None,
    ) -> NDArray[np.int8]:
        if self.handle is None:
            self.handle = self._pgen(contig, None)

        if sample_idx is None:
            n_samples = self.n_samples
            pgen_idx = None
            sample_sorter = None
        else:
            n_samples = len(sample_idx)  # noqa: F841
            sample_sorter = np.argsort(sample_idx)
            pgen_idx = sample_idx[sample_sorter].astype(np.uint32)  # noqa: F841

        n_vars = (end_idxs - start_idxs).sum()
        # (v s*2)
        genotypes = np.empty((n_vars, self.n_samples * self.ploidy), dtype=np.int32)

        for i, (s, e) in enumerate(zip(start_idxs, end_idxs)):
            if s == e:
                continue
            rel_s = s - start_idxs[i]
            rel_e = e - start_idxs[i]
            self.handle.read_alleles_range(
                s, e, allele_int32_out=genotypes[rel_s:rel_e]
            )

        # (v s*2)
        genotypes = genotypes.astype(np.int8)
        # (s*2 v)
        genotypes = genotypes.swapaxes(0, 1)
        # (s 2 v)
        genotypes = np.stack([genotypes[::2], genotypes[1::2]], axis=1)

        if sample_idx is not None:
            genotypes = genotypes[sample_idx]

        if haplotype_idx is not None:
            genotypes = genotypes[:, haplotype_idx]

        # re-order samples to be in query order
        # if sample_sorter is not None and (np.arange(n_samples) != sample_sorter).any():
        #     genotypes = genotypes[sample_sorter]

        return genotypes


class ZarrGenos(Genotypes, FromRecsGenos):
    chunked = True
    driver = "zarr"

    def __init__(self, paths: Union[Path, Dict[str, Path]]) -> None:
        if not ZARR_TENSORSTORE_INSTALLED:
            raise ImportError(
                "Zarr and TensorStore must be installed to use chunked array caches like Zarr and N5."
            )
        if isinstance(paths, Path):
            paths = {"_all": paths}

        if "_all" in paths:
            one_source = True
        else:
            one_source = False

        if not all(p.suffix == ".zarr" for p in paths.values()):
            paths = self.convert_paths_to_zarr(paths)
            if not all(p.exists() for p in paths.values()):
                raise FileNotFoundError(f"Zarr file(s) not found: {paths}")

        self.paths = paths

        first_path = next(iter(paths.values()))
        if one_source:
            tstore = self._open_tstore(next(iter(paths)))
            contigs = zarr.open_array(first_path).attrs["contigs"]
            self.tstores = {c: tstore for c in contigs}
        else:
            self.tstores = {contig: self._open_tstore(contig) for contig in paths}

        self.samples = np.asarray(zarr.open_array(first_path).attrs["samples"])
        # (s p v)
        self.ploidy = self.tstores[next(iter(self.tstores))].shape[1]

    @staticmethod
    def convert_paths_to_zarr(paths: Dict[str, Path]) -> Dict[str, Path]:
        extension = re.compile(r"\.\w+(\.gz)?$")
        paths = {
            c: p.parent / extension.sub(".gvl.genos.zarr", p.name)
            if p.suffix != ".zarr"
            else p
            for c, p in paths.items()
        }
        return paths

    @classmethod
    def from_recs_genos(
        cls,
        records: Records,
        genotypes: Genotypes,
        paths: Optional[Dict[str, Path]] = None,
        mem=int(1e9),
        overwrite: bool = False,
        chunk_shape=None,
    ) -> Self:
        if not ZARR_TENSORSTORE_INSTALLED:
            raise ImportError(
                "Zarr and TensorStore must be installed to use chunked array caches like Zarr and N5."
            )

        n_samples = len(genotypes.samples)
        ploidy = genotypes.ploidy
        contigs = records.contigs
        contig_offsets = records.contig_offsets
        n_var_per_contig = {c: len(a) for c, a in records.v_starts.items()}

        if chunk_shape is None:
            chunk_shape = (100, ploidy, int(1e5))

        if paths is None:
            # suffix, possibly with .gz
            # examples: .pgen, .vcf, .vcf.gz, .bcf
            extension = re.compile(r"\.\w+(\.gz)?$")
            paths = {
                c: p.parent / extension.sub(".gvl.genos.zarr", p.name)
                for c, p in genotypes.paths.items()
            }

        if not overwrite and any(p.exists() for p in paths.values()):
            raise FileExistsError("Zarr file(s) already exist.")

        if "_all" in paths:
            one_source = True
        else:
            one_source = False

        first_path = next(iter(paths.values()))
        c_n_variants = np.array(list(n_var_per_contig.values()))
        var_per_chunk = mem // (n_samples * ploidy)
        if one_source:
            n_variants = sum(n_var_per_contig.values())
            ts_open_kwargs = {
                "spec": {
                    "driver": cls.driver,
                    "kvstore": {"driver": "file", "path": str(first_path)},
                },
                "dtype": np.int8,
                "shape": (n_samples, ploidy, n_variants),
                "create": True,
                "delete_existing": True,
                "chunk_layout": ts.ChunkLayout(  # pyright: ignore[reportAttributeAccessIssue]
                    chunk_shape=chunk_shape
                ),
            }
            ts_handle = ts.open(  # pyright: ignore[reportAttributeAccessIssue]
                **ts_open_kwargs
            ).result()
            arr = zarr.open_array(first_path)
            arr.attrs["contigs"] = records.contigs
            arr.attrs["samples"] = genotypes.samples.tolist()

        for contig, c_n_vars in zip(contigs, c_n_variants):
            c_offset = contig_offsets[contig]

            if not one_source:
                path = genotypes.paths[contig]
                ts_open_kwargs = {
                    "spec": {
                        "driver": cls.driver,
                        "kvstore": {"driver": "file", "path": str(path)},
                    },
                    "dtype": np.int8,
                    "shape": (n_samples, ploidy, c_n_vars),
                    "create": True,
                    "delete_existing": True,
                    "chunk_layout": ts.ChunkLayout(  # pyright: ignore[reportAttributeAccessIssue]
                        chunk_shape=chunk_shape
                    ),
                }
                ts_handle = ts.open(  # pyright: ignore[reportAttributeAccessIssue]
                    **ts_open_kwargs
                ).result()
                arr = zarr.open_array(path)
                arr.attrs["contigs"] = [contig]
                arr.attrs["samples"] = genotypes.samples.tolist()

            # write all genotypes to cache in chunks of up to 1 GB of memory, using up
            # to (8 + 1) * chunk size = 9 GB working space
            # [c_offset, c_offset + c_n_vars)
            # for contig files, c_offset always = 0
            # for one file, c_offset = cumsum of previous offsets
            # c_n_vars is the number of variants in the contig
            idxs = np.arange(
                c_offset, c_offset + c_n_vars + var_per_chunk, var_per_chunk
            )
            idxs[-1] = c_offset + c_n_vars
            for s_idx, e_idx in tqdm(
                zip(idxs[:-1], idxs[1:]),
                total=len(idxs) - 1,
                desc=f"Writing contig {contig}",
            ):
                genos = genotypes.read(contig, np.array([s_idx]), np.array([e_idx]))
                ts_handle[  # pyright: ignore[reportUnboundVariable]
                    :, :, s_idx:e_idx
                ] = genos

        return cls(paths)

    def _open_tstore(self, contig: str):
        ts_open_kwargs = {
            "spec": {
                "driver": self.driver,
                "kvstore": {"driver": "file", "path": str(self.paths[contig])},
            },
            "open": True,
            "read": True,
        }

        return ts.open(  # pyright: ignore[reportAttributeAccessIssue]
            **ts_open_kwargs
        ).result()

    def read(
        self,
        contig: str,
        start_idxs: ArrayLike,
        end_idxs: ArrayLike,
        sample_idx: Optional[ArrayLike] = None,
        haplotype_idx: Optional[ArrayLike] = None,
    ) -> NDArray[np.int8]:
        start_idxs = np.atleast_1d(np.asarray(start_idxs, dtype=int))
        end_idxs = np.atleast_1d(np.asarray(end_idxs, dtype=int))

        # let tensorstore have the full query available to hopefully parallelize
        # reading variable length slices of genotypes
        sub_genos = [None] * len(start_idxs)
        _tstore = self.tstores[contig]

        if sample_idx is not None:
            sample_idx = np.atleast_1d(np.asarray(sample_idx, dtype=int))
            _tstore = _tstore[sample_idx]
        if haplotype_idx is not None:
            haplotype_idx = np.atleast_1d(np.asarray(haplotype_idx, dtype=int))
            _tstore = _tstore[:, haplotype_idx]

        for i, (s, e) in enumerate(zip(start_idxs, end_idxs)):
            # no variants in query regions
            if s == e:
                continue
            sub_genos[i] = _tstore[..., s:e]

        genotypes = ts.concat(  # pyright: ignore[reportAttributeAccessIssue]
            sub_genos, axis=-1
        )[
            ts.d[0].translate_to[0]  # pyright: ignore[reportAttributeAccessIssue]
        ]

        genotypes = cast(NDArray[np.int8], genotypes.read().result())

        return genotypes


class MemmapGenos(Genotypes, FromRecsGenos):
    chunked = False

    def __init__(self, memmaps: Dict[str, Path]) -> None:
        # TODO: need to store sample names as associated metadata
        raise NotImplementedError
        self.paths = memmaps
        self.memmaps = {
            c: np.memmap(p, dtype=np.int8, mode="r") for c, p in memmaps.items()
        }

    def read(
        self,
        contig: str,
        start_idxs: NDArray[np.integer],
        end_idxs: NDArray[np.integer],
        sample_idx: Optional[NDArray[np.integer]] = None,
        haplotype_idx: Optional[NDArray[np.integer]] = None,
    ) -> NDArray[np.int8]:
        raise NotImplementedError
        if sample_idx is None:
            _sample_idx = slice(None)
        else:
            _sample_idx = sample_idx
        if haplotype_idx is None:
            _haplotype_idx = slice(None)
        else:
            _haplotype_idx = haplotype_idx
        genos_ls: List[NDArray[np.int8]] = []
        for s_idx, e_idx in zip(start_idxs, end_idxs):
            # (s p v)
            genos_ls.append(
                self.memmaps[contig][_sample_idx, _haplotype_idx, s_idx:e_idx]
            )
        genos = np.concatenate(genos_ls, axis=-1)
        return genos

    @classmethod
    def from_recs_genos(
        cls,
        records: Records,
        genotypes: Genotypes,
        paths: Optional[Dict[str, Path]] = None,
        mem=int(1e9),
        overwrite: bool = False,
    ) -> Self:
        raise NotImplementedError
        n_samples = len(genotypes.samples)
        ploidy = genotypes.ploidy
        contig_offsets = records.contig_offsets
        n_var_per_contig = {c: len(a) for c, a in records.v_starts.items()}

        # TODO: get default paths if not provided
        memmaps: Dict[str, np.memmap] = {}
        for contig, path in paths.items():
            c_offset = contig_offsets[contig]
            c_n_vars = n_var_per_contig[contig]
            memmaps[contig] = np.memmap(
                path, dtype=np.int8, mode="w+", shape=(n_samples, ploidy, c_n_vars)
            )
            idxs = np.arange(c_offset, c_offset + c_n_vars + mem, mem)
            idxs[-1] = c_offset + c_n_vars
            for s_idx, e_idx in tqdm(
                zip(idxs[:-1], idxs[1:]),
                total=len(idxs) - 1,
                desc=f"Writing contig {contig}",
            ):
                memmaps[contig][..., s_idx:e_idx] = genotypes.read(
                    contig, np.array([s_idx]), np.array([e_idx])
                )[:]
                memmaps[contig].flush()
        return cls(paths)


class VCFGenos(Genotypes):
    chunked = False
    ploidy = 2

    def __init__(
        self, vcfs: Union[Path, Dict[str, Path]], contig_offsets: Dict[str, int]
    ) -> None:
        if not CYVCF2_INSTALLED:
            raise ImportError(
                "cyvcf2 must be installed to use VCF files for genotypes."
            )
        if isinstance(vcfs, Path):
            vcfs = {"_all": vcfs}
        self.paths = vcfs
        samples = None
        for p in self.paths.values():
            if not p.exists():
                raise FileNotFoundError(f"VCF file {p} not found.")
            if samples is None:
                samples = np.array(cyvcf2.VCF(str(p)).samples)
        self.samples = samples  # type: ignore
        self.handles: Optional[Dict[str, cyvcf2.VCF]] = None
        self.contig_offsets = contig_offsets

    def read(
        self,
        contig: str,
        start_idxs: ArrayLike,
        end_idxs: ArrayLike,
        sample_idx: Optional[ArrayLike] = None,
        haplotype_idx: Optional[ArrayLike] = None,
    ) -> NDArray[np.int8]:
        start_idxs = np.atleast_1d(np.asarray(start_idxs, dtype=int))
        end_idxs = np.atleast_1d(np.asarray(end_idxs, dtype=int))

        if self.handles is None and "_all" not in self.paths:
            self.handles = {
                c: cyvcf2.VCF(str(p), lazy=True) for c, p in self.paths.items()
            }
        elif self.handles is None:
            handle = cyvcf2.VCF(str(next(iter(self.paths.values()))), lazy=True)
            self.handles = {c: handle for c in handle.seqnames}

        if sample_idx is None:
            _sample_idx = slice(None)
        else:
            _sample_idx = np.atleast_1d(np.asarray(sample_idx, dtype=int))

        if haplotype_idx is None:
            _haplotype_idx = slice(None)
        else:
            _haplotype_idx = np.atleast_1d(np.asarray(haplotype_idx, dtype=int))

        # (s p v)
        genos = np.empty(
            (len(self.samples), self.ploidy, (end_idxs - start_idxs).sum()),
            dtype=np.int8,
        )
        geno_idx = 0
        offset = self.contig_offsets[contig]
        for i, v in enumerate(self.handles[contig](contig), start=offset):
            for s_idx, e_idx in zip(start_idxs, end_idxs):
                if i >= s_idx and i < e_idx:
                    # (s p) = (s p)
                    genos[..., geno_idx] = v.genotype.array()[:, :2]
                    geno_idx += 1
                if i >= e_idx:
                    break

        genos = genos[_sample_idx, _haplotype_idx]
        # cyvcf2 encoding: 0, 1, -1 => gvl/pgen encoding: 0, 1, -9
        genos[genos == -1] = -9

        return genos
