\
"""
Research runner:
- reads YAML config
- loads panel data
- builds signal(s)
- runs single backtest or parameter sweep
- saves results under runs/{experiment}_{timestamp}/

Usage:
  python -m src.research.sweeps --config configs/momentum.yaml --mode single
  python -m src.research.sweeps --config configs/momentum.yaml --mode sweep
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import yaml

from ..data.clean import data_qc_report, load_ohlcv_panel
from ..data.universe import UniverseSpec, select_topn_by_dollar_volume
from ..signals.momentum import (
    ChannelBreakoutSpec,
    channel_breakout_features,
    channel_breakout_signal,
    MomentumSpec,
    momentum_signal,
)
from ..signals.reversal import (
    BetaResidualReversalSpec,
    ReversalSpec,
    VolumeFilterSpec,
    beta_residual_reversal_signal,
    reversal_signal,
)
from ..backtest.engine import BacktestSpec, run_backtest
from ..portfolio.risk import VolTargetSpec


def _timestamp_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _symlink_latest(run_dir: str, latest_path: str) -> None:
    # Create/replace runs/latest symlink (or copy on Windows-like systems)
    try:
        if os.path.islink(latest_path) or os.path.exists(latest_path):
            if os.path.islink(latest_path):
                os.unlink(latest_path)
            else:
                shutil.rmtree(latest_path)
        os.symlink(os.path.abspath(run_dir), latest_path)
    except Exception:
        # fallback: copy (not ideal but works)
        if os.path.exists(latest_path):
            shutil.rmtree(latest_path)
        shutil.copytree(run_dir, latest_path)


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        kk = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=kk + "."))
        else:
            out[kk] = v
    return out


def _product_grid(grid: Dict[str, list[Any]]) -> list[Dict[str, Any]]:
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    combos = []
    for prod in itertools.product(*vals):
        combos.append(dict(zip(keys, prod)))
    return combos


def _scoreboard(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "net_sharpe" in out.columns and "stability_proxy" not in out.columns:
        out["stability_proxy"] = out["net_sharpe"] - out["net_sharpe"].median(skipna=True)

    required = [
        "net_sharpe",
        "net_ann_return",
        "net_ann_vol",
        "net_max_drawdown",
        "net_worst_month",
        "avg_turnover",
        "net_beta",
        "stability_proxy",
    ]
    param_cols = [
        c
        for c in out.columns
        if c not in required and not c.startswith(("gross_", "net_")) and c not in {"error"}
    ]
    cols = param_cols + [c for c in required if c in out.columns] + [
        c for c in ["avg_cost_drag", "total_cost_drag", "error"] if c in out.columns
    ]
    board = out.loc[:, cols]
    if "net_sharpe" in board.columns:
        board = board.sort_values("net_sharpe", ascending=False, na_position="last")
    return board


def _set_by_dot_path(cfg: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor = cfg
    for part in parts[:-1]:
        next_value = cursor.setdefault(part, {})
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot set {path}: {part} is not a mapping")
        cursor = next_value
    cursor[parts[-1]] = value


def _is_invalid_combo(cfg: Dict[str, Any]) -> Optional[str]:
    sig = cfg.get("signal", {}) or {}
    if sig.get("name") == "channel_breakout":
        entry = int(sig.get("entry_lookback", 0))
        exit_ = int(sig.get("exit_lookback", 0))
        if exit_ >= entry:
            return "skipped: exit_lookback must be smaller than entry_lookback"
    return None


def _build_universe(panel: pd.DataFrame, cfg: Dict[str, Any]) -> Optional[pd.Series]:
    uni_cfg = cfg.get("universe", {})
    spec = UniverseSpec(
        enabled=bool(uni_cfg.get("enabled", True)),
        top_n=int(uni_cfg.get("top_n", 50)),
        dv_window=int(uni_cfg.get("dv_window", 168)),
        min_history_bars=int(uni_cfg.get("min_history_bars", 200)),
        lag_bars=int(uni_cfg.get("lag_bars", 1)),
    )
    return select_topn_by_dollar_volume(panel, spec=spec)


def _build_signal(panel: pd.DataFrame, sig_cfg: Dict[str, Any]) -> pd.Series:
    name = sig_cfg.get("name")
    if name == "momentum":
        spec = MomentumSpec(
            lookback_bars=int(sig_cfg.get("lookback_bars", 24)),
            lag_bars=int(sig_cfg.get("lag_bars", 1)),
        )
        return momentum_signal(panel, spec)
    if name == "channel_breakout":
        spec = ChannelBreakoutSpec(
            entry_lookback=int(sig_cfg.get("entry_lookback", 72)),
            exit_lookback=int(sig_cfg.get("exit_lookback", 24)),
            lag_bars=int(sig_cfg.get("lag_bars", 1)),
        )
        return channel_breakout_signal(panel, spec)
    if name in {"reversal", "shock_fade"}:
        vf_cfg = sig_cfg.get("volume_filter", {}) or {}
        vf = VolumeFilterSpec(
            enabled=bool(vf_cfg.get("enabled", False)),
            vol_window=int(vf_cfg.get("vol_window", 288)),
            z_max=float(vf_cfg.get("z_max", -0.25)),
        )
        spec = ReversalSpec(
            lookback_bars=int(sig_cfg.get("lookback_bars", 3)),
            lag_bars=int(sig_cfg.get("lag_bars", 1)),
            z_entry=float(sig_cfg.get("z_entry", 1.5)),
            volume_filter=vf,
        )
        return reversal_signal(panel, spec)
    if name == "beta_residual_fade":
        beta_cfg = sig_cfg.get("beta", {}) or {}
        data_bench = sig_cfg.get("benchmark_symbol")
        spec = BetaResidualReversalSpec(
            lookback_bars=int(sig_cfg.get("lookback_bars", 1)),
            lag_bars=int(sig_cfg.get("lag_bars", 1)),
            z_entry=float(sig_cfg.get("z_entry", 1.5)),
            benchmark_symbol=str(
                beta_cfg.get("benchmark_symbol", data_bench or "BTCUSDT")
            ),
            beta_window=int(beta_cfg.get("window", 288)),
        )
        return beta_residual_reversal_signal(panel, spec)
    raise ValueError(f"Unknown signal name: {name}")


def _combine_signals(panel: pd.DataFrame, signals_cfg: list[Dict[str, Any]]) -> pd.Series:
    # Weighted linear combination; weights must sum to 1 (we normalize if not).
    sigs = []
    weights = []
    for s in signals_cfg:
        w = float(s.get("weight", 1.0))
        sig = _build_signal(panel, s)
        sigs.append(sig)
        weights.append(w)
    wsum = sum(weights)
    weights = [w / wsum if wsum != 0 else 1.0 / len(weights) for w in weights]
    combo = None
    for sig, w in zip(sigs, weights):
        combo = sig.mul(w) if combo is None else combo.add(sig.mul(w), fill_value=0.0)
    return combo.rename("signal")


def _build_backtest_spec(cfg: Dict[str, Any]) -> BacktestSpec:
    port = cfg.get("portfolio", {}) or {}
    vt_cfg = port.get("vol_target", {}) or {}
    vt = VolTargetSpec(
        enabled=bool(vt_cfg.get("enabled", False)),
        target_ann_vol=float(vt_cfg.get("target_ann_vol", 0.15)),
        vol_window=int(vt_cfg.get("vol_window", 336)),
        max_leverage=float(vt_cfg.get("max_leverage", 3.0)),
    )

    exec_cfg = cfg.get("execution", {}) or {}
    bps = float(exec_cfg.get("bps_per_turnover", 20.0))

    data_cfg = cfg.get("data", {}) or {}
    bench = data_cfg.get("benchmark_symbol", None)

    return BacktestSpec(
        interval=str(data_cfg.get("interval")),
        long_quantile=float(port.get("long_quantile", 0.2)),
        short_quantile=float(port.get("short_quantile", 0.2)),
        gross_leverage=float(port.get("gross_leverage", 1.0)),
        max_abs_weight=float(port.get("max_abs_weight", 0.05)),
        neutralize=str(port.get("neutralize", "dollar")),
        vol_target=vt,
        hold_bars=int(port.get("hold_bars", 1)),
        bps_per_turnover=bps,
        benchmark_symbol=bench,
    )


def _feature_metrics(panel: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    sig = cfg.get("signal", {}) or {}
    if sig.get("name") != "channel_breakout":
        return {}
    features = channel_breakout_features(
        panel,
        ChannelBreakoutSpec(
            entry_lookback=int(sig.get("entry_lookback", 72)),
            exit_lookback=int(sig.get("exit_lookback", 24)),
            lag_bars=int(sig.get("lag_bars", 1)),
        ),
    )
    return {
        "breakout_frequency": float(features["long_breakout"].mean()),
        "avg_channel_width": float(features["channel_width"].mean()),
        "median_channel_width": float(features["channel_width"].median()),
        "avg_channel_position": float(features["channel_position"].mean()),
        "avg_distance_to_upper_channel": float(features["distance_to_upper_channel"].mean()),
        "avg_distance_to_lower_channel": float(features["distance_to_lower_channel"].mean()),
    }


def _walk_forward_slices(n: int, train: int, test: int, embargo: int) -> list[tuple[slice, slice]]:
    """
    Rolling walk-forward with fixed train/test lengths:
      [train][embargo][test] repeated by stepping forward by test length.

    Returns list of (train_slice, test_slice) on integer indices.
    """
    out = []
    start = 0
    while True:
        train_start = start
        train_end = train_start + train
        test_start = train_end + embargo
        test_end = test_start + test
        if test_end > n:
            break
        out.append((slice(train_start, train_end), slice(test_start, test_end)))
        start = start + test
    return out


def _run_one(cfg: Dict[str, Any], run_dir: str) -> Dict[str, Any]:
    data_cfg = cfg["data"]
    panel = load_ohlcv_panel(
        root=data_cfg["root"],
        interval=data_cfg["interval"],
        venue=data_cfg.get("venue", "spot"),
        symbols=data_cfg.get("symbols", None),
        exclude_symbols=bool(data_cfg.get("exclude_symbols", True)),
    )
    feature_metrics = _feature_metrics(panel, cfg)
    _ensure_dir(run_dir)
    data_qc_report(panel, interval=data_cfg["interval"]).to_csv(
        os.path.join(run_dir, "data_qc.csv"),
        index=False,
    )

    universe = _build_universe(panel, cfg)

    # signal or signals list
    if "signals" in cfg:
        signal = _combine_signals(panel, cfg["signals"])
    else:
        signal = _build_signal(panel, cfg["signal"])

    bt_spec = _build_backtest_spec(cfg)
    res = run_backtest(panel=panel, signal=signal, universe=universe, spec=bt_spec)

    # Save artifacts
    # timeseries
    ts = pd.concat([res.gross_returns, res.net_returns, res.turnover], axis=1)
    ts.to_parquet(os.path.join(run_dir, "timeseries.parquet"))
    # weights (can be large)
    res.weights.to_frame().to_parquet(os.path.join(run_dir, "weights.parquet"))
    # metrics
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(res.metrics, f, indent=2, sort_keys=True)

    return {**res.metrics, **feature_metrics}


def run_single(cfg: Dict[str, Any], run_dir: str) -> pd.DataFrame:
    metrics = _run_one(cfg, run_dir)
    row = _flatten(cfg)
    row.update(metrics)
    return pd.DataFrame([row])


def run_sweep(cfg: Dict[str, Any], run_dir: str) -> pd.DataFrame:
    sweep_cfg = cfg.get("sweep", {}) or {}
    grid = (sweep_cfg.get("grid", {}) or {})
    if not grid:
        raise ValueError("No sweep.grid found in config. Use --mode single or add sweep.grid")

    combos = _product_grid(grid)
    rows = []

    for i, combo in enumerate(combos):
        cfg_i = copy.deepcopy(cfg)

        for key, value in combo.items():
            if "." in key:
                _set_by_dot_path(cfg_i, key, value)
            elif key in {"lookback_bars", "z_entry"} and "signal" in cfg_i:
                cfg_i["signal"][key] = value
            elif key in {"long_quantile", "short_quantile", "hold_bars"}:
                cfg_i.setdefault("portfolio", {})
                cfg_i["portfolio"][key] = value
            elif key == "top_n":
                cfg_i.setdefault("universe", {})
                cfg_i["universe"][key] = value
            elif key == "bps_per_turnover":
                cfg_i.setdefault("execution", {})
                cfg_i["execution"][key] = value
            else:
                cfg_i[key] = value

        invalid_reason = _is_invalid_combo(cfg_i)
        if invalid_reason is not None:
            rows.append({**combo, "error": invalid_reason, "skipped": True})
            continue

        tag = f"trial_{i:04d}"
        trial_dir = os.path.join(run_dir, "trials", tag)
        _ensure_dir(trial_dir)

        try:
            metrics = _run_one(cfg_i, trial_dir)
            row = {**combo, **metrics}
            rows.append(row)
        except Exception as e:
            rows.append({**combo, "error": str(e)})

    df = pd.DataFrame(rows)
    if "net_sharpe" in df.columns:
        df["stability_proxy"] = df["net_sharpe"] - df["net_sharpe"].median(skipna=True)
    df.to_parquet(os.path.join(run_dir, "results.parquet"), index=False)
    _scoreboard(df).to_csv(os.path.join(run_dir, "scoreboard.csv"), index=False)

    # pick best by net sharpe (if available)
    if "net_sharpe" in df.columns:
        best = df.dropna(subset=["net_sharpe"]).sort_values("net_sharpe", ascending=False).head(1)
        if len(best) > 0:
            best_row = best.iloc[0].to_dict()
            with open(os.path.join(run_dir, "best.json"), "w", encoding="utf-8") as f:
                json.dump(best_row, f, indent=2, sort_keys=True)

    return df


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--mode", choices=["single", "sweep"], default="single")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp = cfg.get("experiment", {}) or {}
    exp_name = exp.get("name", "experiment")
    run_root = exp.get("run_dir", "runs")
    tag = _timestamp_tag()
    run_dir = os.path.join(run_root, f"{exp_name}_{tag}")
    _ensure_dir(run_dir)

    # Save a copy of config
    with open(os.path.join(run_dir, "config.snapshot.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # Run
    if args.mode == "single":
        df = run_single(cfg, run_dir)
        df.to_parquet(os.path.join(run_dir, "results.parquet"), index=False)
    else:
        df = run_sweep(cfg, run_dir)

    # Latest link
    latest_path = os.path.join(run_root, "latest")
    _symlink_latest(run_dir, latest_path)

    print(f"Saved run to: {run_dir}")
    print(df.head())


if __name__ == "__main__":
    main()
