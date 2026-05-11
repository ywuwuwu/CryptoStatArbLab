\
"""
Regime / activity indicators (volume z-score, realized volatility, market-wide activity).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def prices_wide(panel: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    return panel[price_col].unstack("symbol").sort_index()


def volume_wide(panel: pd.DataFrame) -> pd.DataFrame:
    return panel["volume"].unstack("symbol").sort_index()


def returns_wide(panel: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    p = prices_wide(panel, price_col=price_col)
    return p.pct_change(1)


def volume_zscore(panel: pd.DataFrame, window: int = 288, lag_bars: int = 1) -> pd.DataFrame:
    vol = volume_wide(panel)
    mu = vol.rolling(window=window, min_periods=window).mean()
    sd = vol.rolling(window=window, min_periods=window).std(ddof=0)
    z = (vol - mu) / sd.replace(0.0, np.nan)
    return z.shift(lag_bars)


def realized_vol(panel: pd.DataFrame, window: int = 288, lag_bars: int = 1) -> pd.DataFrame:
    r = returns_wide(panel)
    rv = r.rolling(window=window, min_periods=window).std(ddof=0)
    return rv.shift(lag_bars)


def market_activity_index(
    panel: pd.DataFrame,
    dv_window: int = 288,
    rv_window: int = 288,
    lag_bars: int = 1,
) -> pd.Series:
    """
    Simple market-wide activity index (higher = more activity/info flow).

    Combines z-scored total dollar volume and z-scored cross-sectional realized volatility.
    """
    prices = prices_wide(panel)
    vols = volume_wide(panel)
    dv = (prices * vols).sum(axis=1)

    dv_mu = dv.rolling(dv_window, min_periods=dv_window).mean()
    dv_sd = dv.rolling(dv_window, min_periods=dv_window).std(ddof=0)
    dv_z = (dv - dv_mu) / dv_sd.replace(0.0, np.nan)

    rv = realized_vol(panel, window=rv_window, lag_bars=0)
    rv_cs = rv.mean(axis=1)
    rv_mu = rv_cs.rolling(rv_window, min_periods=rv_window).mean()
    rv_sd = rv_cs.rolling(rv_window, min_periods=rv_window).std(ddof=0)
    rv_z = (rv_cs - rv_mu) / rv_sd.replace(0.0, np.nan)

    act = (dv_z + rv_z).shift(lag_bars).rename("market_activity")
    return act
