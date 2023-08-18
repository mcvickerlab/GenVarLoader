import polars as pl
from tqdm.auto import tqdm

from genvarloader.bnfo import GVL, Fasta, FastaVariants, TileDB_VCF

ref = "/cellar/users/dlaub/projects/ML4GLand/SeqData/data/vcf/Homo_sapiens_assembly38.fasta.gz"
vcf = "/cellar/users/dlaub/projects/ML4GLand/SeqData/data/vcf/ccle.tdb"
fasta = Fasta("seq", ref)
tdb = TileDB_VCF(vcf, 2, samples=["OCI-AML5"])
varseq = FastaVariants("varseq", fasta, tdb)

bed = (
    pl.from_arrow(tdb.ds.read_arrow(regions=["chr20:1-99999999"]))
    .select("contig", "pos_start")
    .unique()
    .with_columns(chromEnd=pl.col("pos_start") + 1)
    .rename({"contig": "chrom", "pos_start": "chromStart"})
)

gvloader = GVL(
    varseq,
    bed=bed,
    fixed_length=int(1e4),
    batch_size=4,
    max_memory_gb=2,
    batch_dims=["sample"],
)
for batch in tqdm(gvloader, total=len(gvloader)):
    print(batch["varseq"])
