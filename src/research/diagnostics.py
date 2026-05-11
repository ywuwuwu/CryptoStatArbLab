\
"""
Diagnostics + plots for sweep results.

Usage:
  python -m src.research.diagnostics --results runs/latest/results.parquet --out runs/latest/figures
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_hist(df: pd.DataFrame, col: str, out_path: str) -> None:
    x = df[col].dropna().values
    if len(x) == 0:
        return
    plt.figure()
    plt.hist(x, bins=40)
    plt.title(f"Histogram: {col}")
    plt.xlabel(col)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_scatter(df: pd.DataFrame, x: str, y: str, out_path: str) -> None:
    if x not in df.columns or y not in df.columns:
        return
    d = df[[x, y]].dropna()
    if len(d) == 0:
        return
    plt.figure()
    plt.scatter(d[x], d[y], s=12)
    plt.title(f"{y} vs {x}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_heatmap(df: pd.DataFrame, x: str, y: str, val: str, out_path: str) -> None:
    if not all(c in df.columns for c in [x, y, val]):
        return
    d = df[[x, y, val]].dropna()
    if len(d) == 0:
        return
    pivot = d.pivot_table(index=y, columns=x, values=val, aggfunc="mean")
    plt.figure()
    plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.title(f"Heatmap: {val} (mean) by {x} and {y}")
    plt.xticks(range(pivot.shape[1]), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(pivot.shape[0]), pivot.index)
    plt.colorbar(label=val)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def channel_breakout_diagnostics(df: pd.DataFrame, out_dir: str) -> None:
    x = "signal.entry_lookback"
    y = "signal.exit_lookback"
    if not all(c in df.columns for c in [x, y]):
        return

    heatmaps = [
        ("net_sharpe", "heatmap_net_sharpe.png"),
        ("gross_sharpe", "heatmap_gross_sharpe.png"),
        ("avg_turnover", "heatmap_turnover.png"),
        ("breakout_frequency", "heatmap_breakout_frequency.png"),
    ]
    for value_col, filename in heatmaps:
        if value_col in df.columns:
            plot_heatmap(df, x=x, y=y, val=value_col, out_path=os.path.join(out_dir, filename))

    feature_cols = [
        x,
        y,
        "breakout_frequency",
        "avg_channel_width",
        "median_channel_width",
        "avg_channel_position",
        "avg_distance_to_upper_channel",
        "avg_distance_to_lower_channel",
        "avg_turnover",
        "gross_sharpe",
        "net_sharpe",
        "error",
    ]
    cols = [c for c in feature_cols if c in df.columns]
    if cols:
        df.loc[:, cols].sort_values([x, y]).to_csv(
            os.path.join(out_dir, "feature_summary.csv"),
            index=False,
        )


def calendar_breakdowns(timeseries_path: str, out_dir: str) -> None:
    if not os.path.exists(timeseries_path):
        return
    ts = pd.read_parquet(timeseries_path)
    if "net_return" not in ts.columns or not isinstance(ts.index, pd.DatetimeIndex):
        return

    r = ts["net_return"].dropna()
    if r.empty:
        return

    cal = pd.DataFrame({"net_return": r})
    cal["is_weekend"] = cal.index.dayofweek >= 5
    cal["hour"] = cal.index.hour

    weekday = (
        cal.groupby("is_weekend")["net_return"]
        .agg(["count", "mean", "std", "sum"])
        .rename(index={False: "weekday", True: "weekend"})
    )
    hourly = cal.groupby("hour")["net_return"].agg(["count", "mean", "std", "sum"])

    weekday.to_csv(os.path.join(out_dir, "weekday_weekend_breakdown.csv"))
    hourly.to_csv(os.path.join(out_dir, "hour_of_day_breakdown.csv"))


def make_plots(results_path: str, out_dir: str) -> None:
    _ensure_dir(out_dir)
    df = pd.read_parquet(results_path)

    # Common plots
    if "net_sharpe" in df.columns:
        plot_hist(df, "net_sharpe", os.path.join(out_dir, "hist_net_sharpe.png"))
    if "gross_sharpe" in df.columns:
        plot_hist(df, "gross_sharpe", os.path.join(out_dir, "hist_gross_sharpe.png"))
    if "avg_turnover" in df.columns and "net_sharpe" in df.columns:
        plot_scatter(df, "avg_turnover", "net_sharpe", os.path.join(out_dir, "turnover_vs_net_sharpe.png"))

    # Try heatmaps for typical sweeps
    if "lookback_bars" in df.columns and "bps_per_turnover" in df.columns and "net_sharpe" in df.columns:
        plot_heatmap(df, x="lookback_bars", y="bps_per_turnover", val="net_sharpe",
                     out_path=os.path.join(out_dir, "heatmap_lookback_vs_cost_net_sharpe.png"))
    if "lookback_bars" in df.columns and "z_entry" in df.columns and "net_sharpe" in df.columns:
        plot_heatmap(df, x="lookback_bars", y="z_entry", val="net_sharpe",
                     out_path=os.path.join(out_dir, "heatmap_lookback_vs_zentry_net_sharpe.png"))

    channel_breakout_diagnostics(df, out_dir)

    # Save a quick leaderboard
    if "net_sharpe" in df.columns:
        leaderboard = df.dropna(subset=["net_sharpe"]).sort_values("net_sharpe", ascending=False).head(20)
        leaderboard.to_csv(os.path.join(out_dir, "leaderboard_top20.csv"), index=False)

    if "net_sharpe" in df.columns:
        scoreboard = df.sort_values("net_sharpe", ascending=False, na_position="last")
        scoreboard.to_csv(os.path.join(out_dir, "scoreboard.csv"), index=False)

    timeseries_path = os.path.join(os.path.dirname(results_path), "timeseries.parquet")
    calendar_breakdowns(timeseries_path, out_dir)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="Path to results.parquet")
    p.add_argument("--out", required=True, help="Output directory for figures")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    make_plots(args.results, args.out)
    print(f"Saved figures to: {args.out}")


if __name__ == "__main__":
    main()
