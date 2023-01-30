from pathlib import Path

import h5py

ref_h5 = Path(
    "/cellar/shared/carterlab/genomes/homo_sapiens/ensembl_grch38.p13_v107/grch38_ACGTN.h5"
)
out_h5 = Path("grch38.20.21.ohe.ACGTN.h5")
with h5py.File(ref_h5) as src, h5py.File(out_h5, "w") as dest:
    dest.attrs.update(src.attrs)
    chroms = ["20", "21"]
    for chrom in chroms:
        src.copy(chrom, dest)
