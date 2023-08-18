import polars as pl
from tqdm.auto import tqdm

from genvarloader.bnfo import GVL, Fasta, FastaVariants, TileDB_VCF

ref = "/cellar/users/dlaub/projects/ML4GLand/SeqData/data/vcf/Homo_sapiens_assembly38.fasta.gz"
vcf = "/cellar/users/dlaub/projects/ML4GLand/SeqData/data/vcf/ccle.tdb"
fasta = Fasta("seq", ref)
tdb = TileDB_VCF(vcf, 2, samples=["OCI-AML5", "NCI-H660"])
varseq = FastaVariants("varseq", fasta, tdb)

bed = (
    pl.from_arrow(
        tdb.ds.read_arrow(regions=["chr20:1-99999999999", "chr21:1-99999999999"])
    )
    .select("contig", "pos_start")
    .unique()
    .select(
        chrom="contig", chromStart=pl.col("pos_start") - 1, chromEnd=pl.col("pos_start")
    )
)

gvloader = GVL(
    varseq,
    bed=bed.tail(1),
    batch_dims=["sample", "ploid"],
    fixed_length=int(1.5e5),
    batch_size=4,
    max_memory_gb=2,
)

for batch in tqdm(gvloader):
    continue
