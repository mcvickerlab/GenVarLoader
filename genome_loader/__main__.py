import typer

from genome_loader.cli import coverage
from genome_loader.cli.sequence import fasta_to_zarr_cli
from genome_loader.cli.variants import vcfs_merge_filter_to_zarr

app = typer.Typer(
    name="GenVarLoader", help="""Write files to "GenVarLoader-ready" data structures."""
)

fasta_to_zarr_cli = app.command("fasta")(fasta_to_zarr_cli)
vcfs_merge_filter_to_zarr = app.command("vcf")(vcfs_merge_filter_to_zarr)
app.add_typer(coverage.app, name="coverage")

if __name__ == "__main__":
    app()
