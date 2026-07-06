"""cProfile + pyinstrument over the live read (Python-layer attribution).

  python tmp/svar2_mvp/prof_python.py <mode> <cohort> <K>

cProfile ranks Python functions by cumulative time; pyinstrument gives a
low-overhead statistical wall-clock call tree as a cross-check (cProfile's own
per-call overhead can distort tiny hot loops)."""
import cProfile
import io
import pstats
import sys

from prof_getitem import make_call


def main(mode, cohort, K):
    call = make_call(mode, cohort)
    call()  # warm

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(K):
        call()
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(30)
    print(f"### cProfile {mode} {cohort} (K={K}), sort=cumulative\n")
    print("```\n" + s.getvalue() + "```\n")

    from pyinstrument import Profiler
    p = Profiler(interval=0.0005)
    p.start()
    for _ in range(K):
        call()
    p.stop()
    print(f"### pyinstrument {mode} {cohort} (K={K})\n")
    print("```\n" + p.output_text(unicode=False, color=False, show_all=False) + "```\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
