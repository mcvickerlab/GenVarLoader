import re
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple, Union, cast, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from tqdm.auto import tqdm
from typing_extensions import Self

from genvarloader.utils import get_rel_starts

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
    chunked: bool
    samples: NDArray[np.str_]
    ploidy: int

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    def read(
        self,
        contig: str,
        start_idxs: ArrayLike,
        end_idxs: ArrayLike,
        sample_idx: Optional[ArrayLike] = None,
        haplotype_idx: Optional[ArrayLike] = None,
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

    def read_for_length(
        self,
        contig: str,
        start_idxs: ArrayLike,
        length: int,
        ilens: NDArray[np.int32],
        sample_idx: Optional[ArrayLike] = None,
        haplotype_idx: Optional[ArrayLike] = None,
    ): ...


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
    ) -> Self: ...


@runtime_checkable
class VIdxGenos(Protocol):
    """Vectorized indexing for genotypes."""

    def vidx(
        self,
        contigs: ArrayLike,
        start_idxs: ArrayLike,
        end_idxs: ArrayLike,
        sample_idxs: ArrayLike,
        haplotype_idxs: ArrayLike,
    ) -> NDArray[np.int8]:
        """Read data akin to vectorized indexing. Output shape will be (n_variants,)."""
        ...


class PgenGenos(Genotypes):
    chunked = False
    ploidy = 2

    def __init__(
        self, paths: Union[str, Path, Dict[str, Path]], sample_names: NDArray[np.str_]
    ) -> None:
        if not PGENLIB_INSTALLED:
            raise ImportError(
                "pgenlib must be installed to use PGEN files for genotypes."
            )
        if isinstance(paths, (str, Path)):
            paths = {"_all": Path(paths)}

        if len(paths) == 0:
            raise ValueError("No paths provided.")
        n_samples = None
        n_variants = 0
        for p in paths.values():
            if not p.exists():
                raise FileNotFoundError(f"PGEN file {p} not found.")
            if n_samples is None:
                n_samples = pgenlib.PgenReader(bytes(p)).get_raw_sample_ct()  # pyright: ignore
            n_variants += pgenlib.PgenReader(bytes(p)).get_variant_ct()  # pyright: ignore
        self.paths = paths
        self.n_samples: int = n_samples  # type: ignore
        self.n_variants = n_variants
        self.samples = sample_names
        self.handle = None

    def _pgen(self, contig: str, sample_idx: Optional[NDArray[np.uint32]]):
        return pgenlib.PgenReader(bytes(self.paths[contig]), sample_subset=sample_idx)  # pyright: ignore

    def read(
        self,
        contig: str,
        start_idxs: ArrayLike,
        end_idxs: ArrayLike,
        sample_idx: Optional[ArrayLike] = None,
        haplotype_idx: Optional[ArrayLike] = None,
    ) -> NDArray[np.int8]:
        if self.handle is None:
            self.handle = self._pgen(contig, None)

        start_idxs = np.atleast_1d(np.asarray(start_idxs, dtype=np.int32))
        end_idxs = np.atleast_1d(np.asarray(end_idxs, dtype=np.int32))

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
            _sample_idx = np.atleast_1d(np.asarray(sample_idx, dtype=np.intp))
        else:
            _sample_idx = slice(None)

        if haplotype_idx is not None:
            _haplotype_idx = np.atleast_1d(np.asarray(haplotype_idx, dtype=np.intp))
        else:
            _haplotype_idx = slice(None)

        genotypes = genotypes[_sample_idx, _haplotype_idx]
        # re-order samples to be in query order
        # if sample_sorter is not None and (np.arange(n_samples) != sample_sorter).any():
        #     genotypes = genotypes[sample_sorter]

        return genotypes

    def read_for_length(
        self,
        contig: str,
        start_idxs: ArrayLike,
        length: int,
        ilens: NDArray[np.int32],
        sample_idx: ArrayLike | None = None,
        haplotype_idx: ArrayLike | None = None,
    ):
        raise NotImplementedError

    def __del__(self) -> None:
        if self.handle is not None:
            self.handle.close()


"""
Implementation note regarding using tensorstore with num_workers > 0
https://github.com/google/tensorstore/issues/61
TL;DR TensorStore is not Send.
"""


class ZarrGenos(Genotypes, FromRecsGenos, VIdxGenos):
    chunked = True
    driver = "zarr"

    def __init__(self, paths: Union[str, Path, Dict[str, Path]]) -> None:
        if not ZARR_TENSORSTORE_INSTALLED:
            raise ImportError(
                "Zarr and TensorStore must be installed to use chunked array caches like Zarr and N5."
            )
        if isinstance(paths, (str, Path)):
            paths = {"_all": Path(paths)}

        if not all(p.suffix == ".zarr" for p in paths.values()):
            paths = self.convert_paths_to_zarr(paths)
            if not all(p.exists() for p in paths.values()):
                raise FileNotFoundError(f"Zarr file(s) not found: {paths}")

        self.paths = paths

        first_path = next(iter(self.paths.values()))
        z = zarr.open_array(first_path)  # pyright: ignore[reportPossiblyUnboundVariable]
        self.samples = np.asarray(z.attrs["samples"])
        # (s p v)
        self.ploidy = z.shape[1]
        self.tstores = None

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
        paths: Optional[Union[str, Path, Dict[str, Path]]] = None,
        mem=int(1e9),
        overwrite: bool = False,
        chunk_shape: Optional[Tuple[int, int, int]] = None,
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
            chunk_shape = (10, ploidy, int(5e4))

        if paths is None:
            geno_paths = getattr(genotypes, "paths", None)
            if geno_paths is None:
                raise ValueError("No paths provided and no paths attribute found.")
            # suffix, possibly with .gz
            # examples: .pgen, .vcf, .vcf.gz, .bcf
            extension = re.compile(r"\.\w+(\.gz)?$")
            paths = {
                c: p.parent / extension.sub(".gvl.genos.zarr", p.name)
                for c, p in geno_paths.items()
            }
        elif isinstance(paths, (str, Path)):
            paths = {"_all": Path(paths)}

        if not overwrite and any(p.exists() for p in paths.values()):
            raise FileExistsError("Zarr file(s) already exist.")

        if "_all" in paths:
            one_source = True
        else:
            one_source = False

        first_path = next(iter(paths.values()))
        c_n_variants = np.array(list(n_var_per_contig.values()), dtype=np.int32)
        var_per_chunk = mem // (n_samples * ploidy)
        n_chunks = sum(np.ceil(c_n_variants / var_per_chunk).astype(int))
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
                "chunk_layout": ts.ChunkLayout(  # pyright: ignore[reportPossiblyUnboundVariable,reportAttributeAccessIssue]
                    chunk_shape=chunk_shape
                ),
            }
            ts_handle = ts.open(  # pyright: ignore[reportPossiblyUnboundVariable,reportAttributeAccessIssue]
                **ts_open_kwargs
            ).result()
            arr = zarr.open_array(first_path)  # pyright: ignore[reportPossiblyUnboundVariable]
            arr.attrs["contigs"] = records.contigs
            arr.attrs["samples"] = genotypes.samples.tolist()

        with tqdm(total=n_chunks, unit="chunk") as pbar:
            for contig, c_n_vars in zip(contigs, c_n_variants):
                pbar.set_description(f"Reading contig {contig}")
                c_offset = contig_offsets[contig]

                if not one_source:
                    path = paths[contig]
                    ts_open_kwargs = {
                        "spec": {
                            "driver": cls.driver,
                            "kvstore": {"driver": "file", "path": str(path)},
                        },
                        "dtype": np.int8,
                        "shape": (n_samples, ploidy, c_n_vars),
                        "create": True,
                        "delete_existing": True,
                        "chunk_layout": ts.ChunkLayout(  # pyright: ignore[reportPossiblyUnboundVariable,reportAttributeAccessIssue]
                            chunk_shape=chunk_shape
                        ),
                    }
                    ts_handle = ts.open(  # pyright: ignore[reportPossiblyUnboundVariable,reportAttributeAccessIssue]
                        **ts_open_kwargs
                    ).result()
                    arr = zarr.open_array(path)  # pyright: ignore[reportPossiblyUnboundVariable]
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
                for s_idx, e_idx in zip(idxs[:-1], idxs[1:]):
                    genos = genotypes.read(contig, np.array([s_idx]), np.array([e_idx]))
                    pbar.set_description(f"Writing contig {contig}")
                    ts_handle[  # pyright: ignore[reportPossiblyUnboundVariable]
                        :, :, s_idx:e_idx
                    ] = genos
                    pbar.update()

        return cls(paths)

    def _init_tstores(self):
        if "_all" in self.paths:
            one_source = True
        else:
            one_source = False

        first_path = next(iter(self.paths.values()))
        if one_source:
            tstore = self._open_tstore(next(iter(self.paths)))
            contigs = zarr.open_array(first_path).attrs["contigs"]  # pyright: ignore[reportPossiblyUnboundVariable]
            return {c: tstore for c in contigs}
        else:
            return {contig: self._open_tstore(contig) for contig in self.paths}

    def _open_tstore(self, contig: str):
        ts_open_kwargs = {
            "spec": {
                "driver": self.driver,
                "kvstore": {"driver": "file", "path": str(self.paths[contig])},
            },
            "open": True,
            "read": True,
        }

        return ts.open(  # pyright: ignore[reportPossiblyUnboundVariable,reportAttributeAccessIssue]
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
        if self.tstores is None:
            self.tstores = self._init_tstores()

        start_idxs = np.atleast_1d(np.asarray(start_idxs, dtype=int))
        end_idxs = np.atleast_1d(np.asarray(end_idxs, dtype=int))

        # let tensorstore have the full query available to hopefully parallelize
        # reading variable length slices of genotypes
        sub_genos = [None] * len(start_idxs)
        tstore = self.tstores[contig]

        if sample_idx is not None:
            sample_idx = np.atleast_1d(np.asarray(sample_idx, dtype=int))
            tstore = tstore[sample_idx]
        if haplotype_idx is not None:
            haplotype_idx = np.atleast_1d(np.asarray(haplotype_idx, dtype=int))
            tstore = tstore[:, haplotype_idx]

        for i, (s, e) in enumerate(zip(start_idxs, end_idxs)):
            # no variants in query regions
            if s == e:
                continue
            sub_genos[i] = tstore[..., s:e]

        genotypes = ts.concat(  # pyright: ignore[reportPossiblyUnboundVariable,reportAttributeAccessIssue]
            sub_genos, axis=-1
        )[
            ts.d[0].translate_to[0]  # pyright: ignore[reportPossiblyUnboundVariable,reportAttributeAccessIssue]
        ]

        genotypes = cast(NDArray[np.int8], genotypes.read().result())

        return genotypes

    def read_for_length(
        self,
        contig: str,
        start_idxs: ArrayLike,
        length: int,
        ilens: NDArray[np.int32],
        sample_idx: ArrayLike | None = None,
        haplotype_idx: ArrayLike | None = None,
    ):
        raise NotImplementedError


class NumpyGenos(Genotypes, FromRecsGenos):
    chunked = False

    def __init__(
        self,
        arrays: Dict[str, NDArray[np.int8]],
        contig_offsets: Dict[str, int],
        samples: ArrayLike,
    ) -> None:
        self.contigs = list(arrays)
        self.arrays = arrays
        self.samples = np.array(samples, dtype=np.str_)
        self.ploidy = arrays[self.contigs[0]].shape[1]
        self.contig_offsets = contig_offsets

    def read(
        self,
        contig: str,
        start_idxs: ArrayLike,
        end_idxs: ArrayLike,
        sample_idx: Optional[ArrayLike] = None,
        haplotype_idx: Optional[ArrayLike] = None,
    ) -> NDArray[np.int8]:
        offset = self.contig_offsets[contig]
        start_idxs = np.atleast_1d(np.asarray(start_idxs, dtype=int)) - offset
        end_idxs = np.atleast_1d(np.asarray(end_idxs, dtype=int)) - offset

        if sample_idx is None:
            _sample_idx = slice(None)
            n_samples = len(self.samples)
        else:
            _sample_idx = np.atleast_1d(np.asarray(sample_idx, dtype=int))
            n_samples = len(_sample_idx)

        if haplotype_idx is None:
            _haplotype_idx = slice(None)
            ploidy = self.ploidy
        else:
            _haplotype_idx = np.atleast_1d(np.asarray(haplotype_idx, dtype=int))
            ploidy = len(_haplotype_idx)

        genos = np.empty(
            (n_samples, ploidy, (end_idxs - start_idxs).sum()),
            dtype=np.int8,
        )
        rel_starts = get_rel_starts(start_idxs, end_idxs)
        rel_ends = rel_starts + (end_idxs - start_idxs)
        for s, e, r_s, r_e in zip(start_idxs, end_idxs, rel_starts, rel_ends):
            if s == e:
                continue
            genos[..., r_s:r_e] = self.arrays[contig][_sample_idx, _haplotype_idx, s:e]

        return genos

    def read_for_length(
        self,
        contig: str,
        start_idxs: ArrayLike,
        length: int,
        ilens: NDArray[np.int32],
        sample_idx: ArrayLike | None = None,
        haplotype_idx: ArrayLike | None = None,
    ):
        raise NotImplementedError

    def __getitem__(
        self,
        contigs: ArrayLike,
        start_idxs: ArrayLike,
        end_idxs: ArrayLike,
        sample_idxs: ArrayLike,
        haplotype_idxs: ArrayLike,
    ) -> NDArray[np.int8]:
        contigs = np.atleast_1d(np.asarray(contigs, dtype=np.str_))
        start_idxs = np.atleast_1d(np.asarray(start_idxs, dtype=np.intp))
        end_idxs = np.atleast_1d(np.asarray(end_idxs, dtype=np.intp))
        sample_idxs = np.atleast_1d(np.asarray(sample_idxs, dtype=np.intp))
        haplotype_idxs = np.atleast_1d(np.asarray(haplotype_idxs, dtype=np.intp))

        genotypes = np.empty((end_idxs - start_idxs).sum(), dtype=np.int8)
        rel_starts = get_rel_starts(start_idxs, end_idxs)
        rel_ends = rel_starts + (end_idxs - start_idxs)
        for c, s, e, sp, h, r_s, r_e in zip(
            contigs,
            start_idxs,
            end_idxs,
            sample_idxs,
            haplotype_idxs,
            rel_starts,
            rel_ends,
        ):
            arr = self.arrays[c]
            offset = self.contig_offsets[c]
            genotypes[r_s:r_e] = arr[sp, h, s - offset : e - offset]

        return genotypes

    @classmethod
    def from_recs_genos(cls, records: Records, genotypes: Genotypes) -> Self:
        arrays = {}
        for contig in records.contigs:
            n_variants = len(records.v_starts[contig])
            arrays[contig] = genotypes.read(contig, 0, n_variants)
        return cls(arrays, records.contig_offsets, genotypes.samples)


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
        start_idxs: ArrayLike,
        end_idxs: ArrayLike,
        sample_idx: Optional[ArrayLike] = None,
        haplotype_idx: Optional[ArrayLike] = None,
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

    def read_for_length(
        self,
        contig: str,
        start_idxs: ArrayLike,
        length: int,
        ilens: NDArray[np.int32],
        sample_idx: ArrayLike | None = None,
        haplotype_idx: ArrayLike | None = None,
    ):
        raise NotImplementedError

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
        self, vcfs: Union[str, Path, Dict[str, Path]], contig_offsets: Dict[str, int]
    ) -> None:
        if not CYVCF2_INSTALLED:
            raise ImportError(
                "cyvcf2 must be installed to use VCF files for genotypes."
            )

        if isinstance(vcfs, (str, Path)):
            vcfs = {"_all": Path(vcfs)}

        self.paths = vcfs
        samples = None
        for p in self.paths.values():
            if not p.exists():
                raise FileNotFoundError(f"VCF file {p} not found.")
            if samples is None:
                f = cyvcf2.VCF(str(p))  # type: ignore
                samples = np.array(f.samples)
                f.close()
                del f
        self.samples = samples  # type: ignore
        self.handles: Optional[Dict[str, "cyvcf2.VCF"]] = None
        self.contig_offsets = contig_offsets

    def close(self):
        if self.handles is not None:
            for handle in self.handles.values():
                handle.close()

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
                c: cyvcf2.VCF(str(p), lazy=True)  # type: ignore
                for c, p in self.paths.items()
            }
        elif self.handles is None:
            first_path = next(iter(self.paths.values()))
            handle = cyvcf2.VCF(str(first_path), lazy=True)  # type: ignore
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
        n_variants = (end_idxs - start_idxs).sum()
        genos = np.empty(
            (len(self.samples), self.ploidy, n_variants),
            dtype=np.int8,
        )

        if genos.size == 0:
            return genos

        # (n_queries)
        geno_idxs = get_rel_starts(start_idxs, end_idxs)
        finish_idxs = np.empty_like(geno_idxs)
        finish_idxs[:-1] = geno_idxs[1:]
        finish_idxs[-1] = n_variants
        offset = self.contig_offsets[contig]
        for i, v in enumerate(self.handles[contig](contig), start=offset):
            # (n_queries)
            overlapping_query_intervals = (i >= start_idxs) & (i < end_idxs)
            if overlapping_query_intervals.any():
                # (n_valid)
                place_idx = geno_idxs[overlapping_query_intervals]
                # increment idxs for next iteration
                geno_idxs[overlapping_query_intervals] += 1
                genos[..., place_idx] = v.genotype.array()[:, :2, None]
            if (geno_idxs == finish_idxs).all():
                break

        genos = genos[_sample_idx, _haplotype_idx]
        # cyvcf2 encoding: 0, 1, -1 => gvl/pgen encoding: 0, 1, -9
        genos[genos == -1] = -9

        return genos

    def read_for_length(
        self,
        contig: str,
        start_idxs: Optional[ArrayLike],
        init_end_idxs: Optional[ArrayLike],
        length: int,
        ilens: NDArray[np.int32],
        sample_idx: Optional[ArrayLike] = None,
        haplotype_idx: Optional[ArrayLike] = None,
    ):
        raise NotImplementedError
        start_idxs = np.atleast_1d(np.asarray(start_idxs, dtype=np.int32))
        init_end_idxs = np.atleast_1d(np.asarray(init_end_idxs, dtype=np.int32))

        if self.handles is None:
            self.handles = self.init_handles()

        if sample_idx is None:
            _sample_idx = slice(None)
        else:
            _sample_idx = np.atleast_1d(np.asarray(sample_idx, dtype=int))

        if haplotype_idx is None:
            _haplotype_idx = slice(None)
        else:
            _haplotype_idx = np.atleast_1d(np.asarray(haplotype_idx, dtype=int))

        n_variants = (init_end_idxs - start_idxs).sum()
        genos = np.empty(
            (len(self.samples), self.ploidy, n_variants),
            dtype=np.int8,
        )

        if genos.size == 0:
            return genos

        geno_idxs = get_rel_starts(start_idxs, init_end_idxs)
        finish_idxs = np.empty_like(geno_idxs)
        finish_idxs[:-1] = geno_idxs[1:]
        finish_idxs[-1] = n_variants
        offset = self.contig_offsets[contig]
        for i, v in enumerate(self.handles[contig](contig), start=offset):
            # (n_queries)
            overlapping_query_intervals = (i >= start_idxs) & (i < init_end_idxs)
            if overlapping_query_intervals.any():
                # (n_valid)
                place_idx = geno_idxs[overlapping_query_intervals]
                # increment idxs for next iteration
                geno_idxs[overlapping_query_intervals] += 1
                genos[..., place_idx] = v.genotype.array()[:, :2, None]
            if (geno_idxs == finish_idxs).all():
                break

    def init_handles(self) -> Dict[str, "cyvcf2.VCF"]:
        if self.handles is None and "_all" not in self.paths:
            handles = {
                c: cyvcf2.VCF(str(p), lazy=True)  # pyright: ignore
                for c, p in self.paths.items()
            }
        elif self.handles is None:
            handle = cyvcf2.VCF(str(next(iter(self.paths.values()))), lazy=True)  # pyright: ignore
            handles = {c: handle for c in handle.seqnames}
        else:
            handles = self.handles
        return handles

    def __del__(self) -> None:
        if self.handles is not None:
            for handle in self.handles.values():
                handle.close()
