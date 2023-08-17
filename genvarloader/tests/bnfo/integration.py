import polars as pl

from genvarloader.bnfo import GVL, Fasta, FastaVariants, TileDB_VCF

ref = "/cellar/users/dlaub/projects/ML4GLand/SeqData/data/vcf/Homo_sapiens_assembly38.fasta.gz"
vcf = "/cellar/users/dlaub/projects/ML4GLand/SeqData/data/vcf/ccle.tdb"
fasta = Fasta("seq", ref)
tdb = TileDB_VCF(vcf, 2, samples=["OCI-AML5"])
varseq = FastaVariants("varseq", fasta, tdb)

bed = pl.DataFrame(
    {
        "chrom": ["chr20", "chr20"],
        "chromStart": [96320, 158814],
        "chromEnd": [96321, 158815],
    }
)

gvloader = GVL(varseq)
dl = gvloader.iter_batches(bed, fixed_length=3, batch_size=1, max_memory_gb=2)
for batch in dl:
    print(batch["varseq"])
