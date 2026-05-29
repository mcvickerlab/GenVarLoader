"""Read results.csv → results_plot.png (3 outputs × 4 axes small multiples).

Each panel: x = the column's axis values, y = instances_per_s, one line per
mode, holding the other three axes at their midpoint. mode=None is absent from
the buffer_bytes column (baseline ignores the buffer budget).

Run (matplotlib lives in the `notebook` pixi env):
    pixi run -e notebook python experiments/dataloader/plot_results.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C

HERE = Path(__file__).resolve().parent
RESULTS_CSV = HERE / "results.csv"
OUT_PNG = HERE / "results_plot.png"

# column axis -> (csv column, fan values, {pinned other axis: midpoint})
AXES = {
    "threads": ("threads", C.THREADS_FAN,
                {"region_length": C.REGION_MID, "batch_size": C.BATCH_MID}),
    "region_length": ("region_length", C.REGION_FAN,
                      {"threads": C.THREADS_MID, "batch_size": C.BATCH_MID}),
    "batch_size": ("batch_size", C.BATCH_FAN,
                   {"threads": C.THREADS_MID, "region_length": C.REGION_MID}),
    "buffer_bytes": ("buffer_bytes", C.BUFFER_FAN,
                     {"threads": C.THREADS_MID, "region_length": C.REGION_MID,
                      "batch_size": C.BATCH_MID}),
}

MODE_STYLE = {
    "": ("None", "tab:gray", "o"),
    "buffered": ("buffered", "tab:blue", "s"),
    "double_buffered": ("double_buffered", "tab:green", "^"),
}


def _panel_series(df: pl.DataFrame, axis_col: str, pins: dict, mode_key: str):
    """Return (x, y) sorted by x for one mode on one panel."""
    # The CSV stores mode=None (baseline) as an empty field, which read_csv with
    # null_values=[""] loads as null — so match it via is_null(), not == "".
    if mode_key == "":
        sub = df.filter(pl.col("mode").is_null())
    else:
        sub = df.filter(pl.col("mode") == mode_key)
    for col, mid in pins.items():
        sub = sub.filter(pl.col(col) == mid)
    # baseline has no buffer_bytes; the buffer panel pins it for new modes only
    if axis_col == "buffer_bytes" and mode_key == "":
        return [], []
    if axis_col != "buffer_bytes":
        # new modes pin buffer at midpoint; baseline rows have null buffer
        if mode_key == "":
            sub = sub.filter(pl.col("buffer_bytes").is_null())
        else:
            sub = sub.filter(pl.col("buffer_bytes") == C.BUFFER_MID)
    sub = sub.sort(axis_col)
    return sub[axis_col].to_list(), sub["instances_per_s"].to_list()


def main() -> None:
    if not RESULTS_CSV.exists():
        print(f"Missing {RESULTS_CSV} — run bench.py first.")
        return
    # buffer_bytes is empty for baseline rows -> read as nullable int
    df = pl.read_csv(RESULTS_CSV, null_values=[""])

    axis_names = list(AXES)
    fig, axs = plt.subplots(
        len(C.OUTPUTS), len(axis_names),
        figsize=(4 * len(axis_names), 3 * len(C.OUTPUTS)),
        constrained_layout=True, squeeze=False,
    )

    for r, output in enumerate(C.OUTPUTS):
        out_df = df.filter(pl.col("with_seqs") == output)
        for c, axis_name in enumerate(axis_names):
            ax = axs[r][c]
            axis_col, _, pins = AXES[axis_name]
            for mode_key, (label, color, marker) in MODE_STYLE.items():
                x, y = _panel_series(out_df, axis_col, pins, mode_key)
                if x:
                    ax.plot(x, y, marker=marker, color=color, label=label)
            if axis_name == "buffer_bytes":
                ax.set_xscale("log", base=2)
            if r == 0:
                ax.set_title(axis_name)
            if c == 0:
                ax.set_ylabel(f"{output}\ninstances/s")
            ax.grid(True, alpha=0.3)
    # Anchor the legend to a panel that carries all three series (the
    # buffer_bytes column has no None baseline).
    axs[0][0].legend(fontsize=8)

    fig.suptitle(
        "DataLoader throughput: mode comparison across knobs "
        "(other axes pinned at midpoint)",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Saved {OUT_PNG}")
    plt.close(fig)

    # --- optional MiB_per_s variant (uncomment to render bandwidth view) ---
    # Re-run with "MiB_per_s" substituted for "instances_per_s" in
    # _panel_series and OUT_PNG -> results_plot_bandwidth.png.


if __name__ == "__main__":
    main()
