\
"""
Risk management: volatility targeting for portfolio weights.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VolTargetSpec:
    enabled: bool = True
    target_ann_vol: float = 0.15
    vol_window: int = 336
    max_leverage: float = 3.0


def periods_per_year_from_interval(interval: str) -> float:
    m = interval.strip().lower()
    if m.endswith("m"):
        mins = int(m[:-1])
        return (60 / mins) * 24 * 365
    if m.endswith("h"):
        hrs = int(m[:-1])
        return (24 / hrs) * 365
    if m.endswith("d"):
        days = int(m[:-1])
        return 365 / days
    raise ValueError(f"Unsupported interval: {interval}")


def compute_portfolio_returns(weights_wide: pd.DataFrame, asset_returns_wide: pd.DataFrame) -> pd.Series:
    """
    weights applied with 1-bar delay to avoid lookahead: pnl_t = sum_i w_{t-1,i} * r_{t,i}
    """
    w_lag = weights_wide.shift(1).reindex(asset_returns_wide.index).fillna(0.0)
    r = asset_returns_wide.fillna(0.0)
    return (w_lag * r).sum(axis=1).rename("gross_return")


def vol_target_weights(
    weights_wide: pd.DataFrame,
    asset_returns_wide: pd.DataFrame,
    interval: str,
    spec: VolTargetSpec,
) -> pd.DataFrame:
    """
    Scale weights to target an annualized volatility.
    """
    if not spec.enabled:
        return weights_wide

    ppy = periods_per_year_from_interval(interval)
    gross = compute_portfolio_returns(weights_wide, asset_returns_wide)
    realized = gross.rolling(spec.vol_window, min_periods=spec.vol_window).std(ddof=0) * np.sqrt(ppy)
    leverage = (spec.target_ann_vol / realized.replace(0.0, np.nan)).clip(upper=spec.max_leverage)

    # Use yesterday's leverage to avoid lookahead
    leverage = leverage.shift(1).fillna(1.0)

    return weights_wide.mul(leverage, axis=0)
