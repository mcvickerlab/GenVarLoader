"""Read results_polars_bio.csv + results_pyranges1.csv and produce bar plots.

Run with any env that has matplotlib + polars:
    pixi run -e dev python experiments/bench_table_overlap/plot_results.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

HERE = Path(__file__).parent
CSV_FILES = [HERE / "results_polars_bio.csv", HERE / "results_pyranges1.csv"]


# label: backend__method
def label(row: dict) -> str:
    return f"{row['backend']}\n{row['method']}"


def main():
    import numpy as np

    frames = []
    for p in CSV_FILES:
        if p.exists():
            frames.append(pl.read_csv(p))
        else:
            print(f"Missing {p} — run the corresponding bench script first.")
    if not frames:
        return
    df = pl.concat(frames).drop_nulls(["time_s", "peak_MB"])
    df = df.with_columns(
        pl.concat_str([pl.col("backend"), pl.lit("\n"), pl.col("method")]).alias(
            "label"
        )
    )

    # Aggregate per (case, label): median + std across trials
    agg = (
        df.group_by(["case", "label", "n_regions", "n_samples", "ipp"])
        .agg(
            pl.col("time_s").median().alias("t_med"),
            pl.col("time_s").std().alias("t_std"),
            pl.col("peak_MB").median().alias("p_med"),
            pl.col("peak_MB").std().alias("p_std"),
        )
        .sort(["case", "label"])
    )

    cases = agg["case"].unique().sort().to_list()
    n_cases = len(cases)
    fig, axes = plt.subplots(
        2, n_cases, figsize=(5 * n_cases, 8), constrained_layout=True
    )
    if n_cases == 1:
        axes = [[axes[0]], [axes[1]]]

    for col, case in enumerate(cases):
        sub = agg.filter(pl.col("case") == case).sort("label")
        labels = sub["label"].to_list()
        t_med = np.array(sub["t_med"].to_list())
        t_std = np.array(sub["t_std"].fill_null(0).to_list())
        p_med = np.array(sub["p_med"].to_list())
        p_std = np.array(sub["p_std"].fill_null(0).to_list())
        x = np.arange(len(labels))

        ax_t = axes[0][col]
        ax_m = axes[1][col]

        ax_t.bar(x, t_med, yerr=t_std, capsize=4, color="steelblue", edgecolor="white")
        ax_t.set_xticks(x)
        ax_t.set_xticklabels(labels, fontsize=8)
        ax_t.set_ylabel("time (s)")
        ax_t.set_title(
            f"{case}\n(n_reg={sub['n_regions'][0]}, n_s={sub['n_samples'][0]}, ipp={sub['ipp'][0]})"
        )
        for xi, v in zip(x, t_med):
            ax_t.text(xi, v * 1.01, f"{v:.3f}", ha="center", va="bottom", fontsize=7)

        ax_m.bar(x, p_med, yerr=p_std, capsize=4, color="darkorange", edgecolor="white")
        ax_m.set_xticks(x)
        ax_m.set_xticklabels(labels, fontsize=8)
        ax_m.set_ylabel("peak memory (MB)")
        for xi, v in zip(x, p_med):
            ax_m.text(xi, v * 1.01, f"{v:.0f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle(
        "Table.count_intervals backend comparison (median ± std, 20 trials)",
        fontsize=13,
        fontweight="bold",
    )

    out = HERE / "results_plot.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
