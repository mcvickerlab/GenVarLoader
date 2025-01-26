from pathlib import Path

import pyBigWig
import typer


def main(n_samples: int = 2):
    data_dir = Path(__file__).resolve().parent
    for i in range(n_samples):
        path = data_dir / f"sample_{i}.bw"
        with pyBigWig.open(str(path), "w") as bw:
            bw.addHeader([("chr1", 2000), ("chr2", 1000)], maxZooms=0)
            bw.addEntries(["chr1", "chr1"], [1, 100], [5, 105], [1.0, 2.0])
            bw.addEntries(["chr2", "chr2"], [1, 100], [5, 105], [1.0, 2.0])


if __name__ == "__main__":
    typer.run(main)
