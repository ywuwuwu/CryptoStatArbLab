\
"""
Short-horizon reversal signals (optionally filtered by low-activity / low-information proxy).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .beta_hedge import BetaHedgeSpec, beta_hedged_residual_returns
from .regimes import volume_zscore


@dataclass(frozen=True)
class VolumeFilterSpec:
    enabled: bool = False
    vol_window: int = 288
    z_max: float = -0.25   # only trade if volume z <= z_max (low activity)


@dataclass(frozen=True)
class ReversalSpec:
    lookback_bars: int = 3
    lag_bars: int = 1
    z_entry: float = 1.5
    price_col: str = "close"
    volume_filter: VolumeFilterSpec = VolumeFilterSpec()


@dataclass(frozen=True)
class BetaResidualReversalSpec:
    lookback_bars: int = 1
    lag_bars: int = 1
    z_entry: float = 1.5
    benchmark_symbol: str = "BTCUSDT"
    beta_window: int = 288
    price_col: str = "close"


def reversal_signal(panel: pd.DataFrame, spec: ReversalSpec) -> pd.Series:
    """
    A simple reversal signal:
      - compute short-horizon returns over lookback_bars
      - convert to cross-sectional z-score per timestamp
      - take the opposite direction: signal = -z
      - apply entry threshold abs(z) >= z_entry, else 0
      - optional filter for low-activity volume z-score
    """
    prices = panel[spec.price_col].unstack("symbol").sort_index()
    r = prices.pct_change(spec.lookback_bars)

    # Cross-sectional z-score each timestamp
    mu = r.mean(axis=1)
    sd = r.std(axis=1, ddof=0).replace(0.0, np.nan)
    z = r.sub(mu, axis=0).div(sd, axis=0)

    sig = -z
    sig = sig.where(z.abs() >= spec.z_entry, other=0.0)

    # Optional low-activity filter
    if spec.volume_filter.enabled:
        vz = volume_zscore(panel, window=spec.volume_filter.vol_window, lag_bars=0)
        sig = sig.where(vz <= spec.volume_filter.z_max, other=0.0)

    # Lag to avoid lookahead (trade at t+1)
    sig = sig.shift(spec.lag_bars)

    return sig.stack().rename("signal")


def beta_residual_reversal_signal(
    panel: pd.DataFrame,
    spec: BetaResidualReversalSpec,
) -> pd.Series:
    """
    Fade short-horizon residual moves after removing rolling BTC beta.

    The beta estimate is lagged inside beta_hedged_residual_returns, and the
    final signal is lagged again before trading.
    """
    resid, _beta = beta_hedged_residual_returns(
        panel,
        BetaHedgeSpec(
            benchmark_symbol=spec.benchmark_symbol,
            window=spec.beta_window,
            lag_bars=spec.lag_bars,
            price_col=spec.price_col,
        ),
    )
    shock = resid.rolling(spec.lookback_bars, min_periods=spec.lookback_bars).sum()

    mu = shock.mean(axis=1)
    sd = shock.std(axis=1, ddof=0).replace(0.0, np.nan)
    z = shock.sub(mu, axis=0).div(sd, axis=0)

    sig = -z
    sig = sig.where(z.abs() >= spec.z_entry, other=0.0)
    sig = sig.shift(spec.lag_bars)

    return sig.stack().rename("signal")
