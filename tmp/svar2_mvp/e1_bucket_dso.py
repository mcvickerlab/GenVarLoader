"""Bucket a `perf report --sort=dso --no-children --stdio` self-time dump into the
E1 attribution classes. Reads that dump on stdin.

  <abs-perf> report --stdio --sort=dso --no-children -i data.perf | python e1_bucket_dso.py
"""

import re
import sys

buckets = {
    "native-gvl": 0.0,
    "native-genoray": 0.0,
    "numpy-conv": 0.0,
    "python-interp": 0.0,
    "other": 0.0,
}
for line in sys.stdin:
    if line.lstrip().startswith("#") or not line.strip():
        continue
    m = re.match(r"\s*([0-9.]+)%\s+(.*)", line)
    if not m:
        continue
    pct = float(m.group(1))
    dso = m.group(2).strip().lower()
    if "genvarloader" in dso:
        b = "native-gvl"
    elif "genoray" in dso or "_core.cpython" in dso:
        b = "native-genoray"
    elif "multiarray" in dso or "umath" in dso or "/numpy" in dso:
        b = "numpy-conv"
    elif dso.startswith("python") or "libpython" in dso:
        b = "python-interp"
    else:
        b = "other"
    buckets[b] += pct

tot = sum(buckets.values()) or 1.0
for k in ("native-gvl", "native-genoray", "numpy-conv", "python-interp", "other"):
    print(f"  {buckets[k]:6.1f}%  {k}")
nat = buckets["native-gvl"] + buckets["native-genoray"]
print(
    f"  ---- native(gvl+genoray)={nat:.1f}%  numpy-conv={buckets['numpy-conv']:.1f}%  "
    f"python-interp={buckets['python-interp']:.1f}%  (sum {tot:.1f}%)"
)
