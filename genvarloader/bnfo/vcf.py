from pathlib import Path
from typing import List, Optional, Sequence, Union, cast

import numpy as np

from .types import DenseAlleles, Variants

try:
    import cyvcf2

    CYVCF2_INSTALLED = True
except ImportError:
    CYVCF2_INSTALLED = False


class VCF(Variants):
    def __init__(
        self,
        path: Union[str, Path],
        samples: Optional[List[str]] = None,
        threads: int = 1,
    ) -> None:
        if not CYVCF2_INSTALLED:
            raise ImportError("cyvcf2 must be installed to read VCF.")
        self.ploidy = 2

        self.path = Path(path)
        if not self.path.exists():
            raise ValueError("File does not exist.")

        if samples is None:
            samples = cast(List[str], self._vcf().samples)
        else:
            raise NotImplementedError

        self.samples = samples
        self.n_samples = len(samples)
        self.threads = threads

    def _vcf(self, samples: Optional[Sequence[str]] = None):
        return cyvcf2.VCF(str(self.path), threads=self.threads, samples=samples)

    def read(
        self, contig: str, start: int, end: int, **kwargs
    ) -> Optional[DenseAlleles]:
        samples = kwargs.get("samples", None)

        if samples is not None:
            raise NotImplementedError

        vcf = self._vcf(samples)
        region = f"{contig}:{start+1}-{end}"
        v_info = [(v.start, np.array(v.gt_bases)) for v in vcf(region) if v.is_snp]
        vcf.close()

        if len(v_info) == 0:
            return

        # TODO Need to reshape alleles? Type conversion can't fail.
        positions = np.array([v[0] for v in v_info]).astype(np.int32)
        alleles = np.stack([v[1] for v in v_info]).astype("S1")
        raise NotImplementedError
        return DenseAlleles(positions, alleles)
