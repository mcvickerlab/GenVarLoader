"""Split a py-spy --format raw (folded) stack file into Python vs native
self-time by LEAF frame. A leaf frame is Python iff it contains '.py:'.

  python split_folded.py <folded.txt>
"""
import sys
from collections import Counter


def is_python(frame: str) -> bool:
    return ".py:" in frame or frame.endswith(".py")


def main(path):
    py = nat = 0
    leaves = Counter()
    classed = {}
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            stack, _, cnt = line.rpartition(" ")
            try:
                n = int(cnt)
            except ValueError:
                continue
            leaf = stack.split(";")[-1]
            leaves[leaf] += n
            classed[leaf] = "python" if is_python(leaf) else "native"
            if is_python(leaf):
                py += n
            else:
                nat += n
    tot = py + nat
    if tot == 0:
        print("no samples parsed"); return
    print(f"python_pct={100 * py / tot:.1f} native_pct={100 * nat / tot:.1f} total_samples={tot}")
    print("top-15 leaf frames (self-time):")
    for leaf, n in leaves.most_common(15):
        print(f"  {100 * n / tot:5.1f}%  [{classed[leaf]:6s}]  {leaf}")


if __name__ == "__main__":
    main(sys.argv[1])
