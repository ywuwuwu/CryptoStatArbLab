\
"""
Backtest engine: unconstrained long/short backtests with turnover + bps costs.

Design:
- signals are computed and then converted to weights
- weights at time t are applied to returns at time t+1 (1-bar delay)
- turnover = 0.5 * sum |w_t - w_{t-1}|
- costs are turnover * bps

Outputs:
- gross_returns (Series)
- net_returns (Series)
- turnover (Series)
- weights (Series long format)
- metrics (dict)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from ..portfolio.weights import LSWeightSpec, construct_long_short_weights
from ..portfolio.risk import VolTargetSpec, vol_target_weights
from .costs import turnover as compute_turnover, apply_bps_costs
from .metrics import compute_metrics


@dataclass
class BacktestSpec:
    interval: str
    long_quantile: float = 0.2
    short_quantile: float = 0.2
    gross_leverage: float = 1.0
    max_abs_weight: float = 0.05
    neutralize: str = "dollar"
    vol_target: VolTargetSpec = VolTargetSpec(enabled=False)
    hold_bars: int = 1
    bps_per_turnover: float = 20.0
    benchmark_symbol: Optional[str] = None


@dataclass
class BacktestResult:
    gross_returns: pd.Series
    net_returns: pd.Series
    turnover: pd.Series
    weights: pd.Series
    metrics: Dict[str, Any]


def _asset_returns_wide(panel: pd.DataFrame) -> pd.DataFrame:
    prices = panel["close"].unstack("symbol").sort_index()
    return prices.pct_change(1, fill_method=None)


def _apply_rebalance_schedule(weights_wide: pd.DataFrame, hold_bars: int) -> pd.DataFrame:
    """
    Rebalance target weights every hold_bars rows and carry positions between
    scheduled rebalances.
    """
    if hold_bars <= 1 or weights_wide.empty:
        return weights_wide

    schedule = pd.Series(False, index=weights_wide.index)
    schedule.iloc[::hold_bars] = True
    return weights_wide.where(schedule, other=np.nan).ffill().fillna(0.0)


def run_backtest(
    panel: pd.DataFrame,
    signal: pd.Series,
    universe: Optional[pd.Series],
    spec: BacktestSpec,
) -> BacktestResult:
    # 1) Convert signal -> weights
    w_spec = LSWeightSpec(
        long_quantile=spec.long_quantile,
        short_quantile=spec.short_quantile,
        gross_leverage=spec.gross_leverage,
        max_abs_weight=spec.max_abs_weight,
        neutralize=spec.neutralize,
    )
    weights = construct_long_short_weights(signal=signal, universe=universe, spec=w_spec)
    weights_wide = weights.unstack("symbol").sort_index()

    # 2) Asset returns
    asset_ret = _asset_returns_wide(panel)

    # align columns
    asset_ret = asset_ret.reindex(columns=weights_wide.columns)
    asset_ret = asset_ret.replace([np.inf, -np.inf], np.nan)
    weights_wide = weights_wide.reindex(asset_ret.index).fillna(0.0)

    # 3) Apply hold period / rebalance schedule before any risk scaling.
    weights_wide = _apply_rebalance_schedule(weights_wide, hold_bars=int(spec.hold_bars))

    # 4) Optional vol targeting (scales weights)
    vt = spec.vol_target
    if vt is not None and vt.enabled:
        weights_wide = vol_target_weights(
            weights_wide=weights_wide,
            asset_returns_wide=asset_ret,
            interval=spec.interval,
            spec=vt,
        )

    # 5) Gross portfolio returns
    gross = (weights_wide.shift(1).fillna(0.0) * asset_ret.fillna(0.0)).sum(axis=1).rename("gross_return")

    # 6) Turnover + costs
    to = compute_turnover(weights_wide.fillna(0.0))
    net = apply_bps_costs(gross_returns=gross, turnover=to, bps_per_turnover=spec.bps_per_turnover)

    # 7) Benchmark returns for alpha/beta
    bench = None
    if spec.benchmark_symbol is not None and spec.benchmark_symbol in panel.index.get_level_values("symbol"):
        bench_prices = panel["close"].xs(spec.benchmark_symbol, level="symbol").sort_index()
        bench = bench_prices.pct_change(1, fill_method=None).rename("benchmark_return")

    # 8) Metrics
    m_gross = compute_metrics(gross, interval=spec.interval, benchmark_returns=bench, prefix="gross_")
    m_net = compute_metrics(net, interval=spec.interval, benchmark_returns=bench, prefix="net_")
    m_to = {
        "avg_turnover": float(to.mean()),
        "p95_turnover": float(to.quantile(0.95)),
        "avg_cost_drag": float((gross - net).mean()),
        "total_cost_drag": float((gross - net).sum()),
    }
    metrics = {**m_gross, **m_net, **m_to}

    return BacktestResult(
        gross_returns=gross,
        net_returns=net,
        turnover=to,
        weights=weights_wide.stack().rename("weight"),
        metrics=metrics,
    )
