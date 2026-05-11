\
"""
Cross-sectional momentum signals.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MomentumSpec:
    lookback_bars: int = 24
    lag_bars: int = 1
    price_col: str = "close"


@dataclass(frozen=True)
class ChannelBreakoutSpec:
    entry_lookback: int = 72
    exit_lookback: int = 24
    lag_bars: int = 1
    price_col: str = "close"


def momentum_signal(panel: pd.DataFrame, spec: MomentumSpec) -> pd.Series:
    """
    Compute cross-sectional momentum signal based on lookback returns.

    Returns a Series indexed by (timestamp, symbol).
    """
    prices = panel[spec.price_col].unstack("symbol").sort_index()
    rets = prices.pct_change(spec.lookback_bars)

    # Lag to avoid lookahead: you can only trade on info available at t (execute at t+1)
    rets = rets.shift(spec.lag_bars)

    sig = rets.stack().rename("signal")
    return sig


def channel_breakout_features(panel: pd.DataFrame, spec: ChannelBreakoutSpec) -> pd.DataFrame:
    """
    Compute lag-safe channel breakout features in long format.

    Rolling channels are shifted by spec.lag_bars before comparing against
    the current close, so the current bar cannot define its own channel.
    """
    if spec.exit_lookback >= spec.entry_lookback:
        raise ValueError("exit_lookback must be smaller than entry_lookback")

    close = panel[spec.price_col].unstack("symbol").sort_index()
    prior_high = (
        close.rolling(spec.entry_lookback, min_periods=spec.entry_lookback)
        .max()
        .shift(spec.lag_bars)
    )
    prior_low = (
        close.rolling(spec.exit_lookback, min_periods=spec.exit_lookback)
        .min()
        .shift(spec.lag_bars)
    )

    width = prior_high - prior_low
    position = close.sub(prior_low).div(width.replace(0.0, np.nan))
    long_breakout = close > prior_high
    short_breakout = close < prior_low
    distance_upper = prior_high.sub(close).div(close.replace(0.0, np.nan))
    distance_lower = close.sub(prior_low).div(close.replace(0.0, np.nan))

    features = {
        "prior_rolling_high": prior_high,
        "prior_rolling_low": prior_low,
        "channel_width": width,
        "channel_position": position,
        "long_breakout": long_breakout.astype(float),
        "short_breakout": short_breakout.astype(float),
        "distance_to_upper_channel": distance_upper,
        "distance_to_lower_channel": distance_lower,
    }

    return pd.concat({name: df.stack() for name, df in features.items()}, axis=1)


def channel_breakout_signal(panel: pd.DataFrame, spec: ChannelBreakoutSpec) -> pd.Series:
    """
    Long-only channel breakout regime signal.

    Enter when close breaks above the prior entry channel high. Stay long until
    close breaks below the shorter prior exit channel low.
    """
    features = channel_breakout_features(panel, spec)
    long_breakout = features["long_breakout"].unstack("symbol").sort_index().fillna(0.0).astype(bool)
    short_breakout = features["short_breakout"].unstack("symbol").sort_index().fillna(0.0).astype(bool)

    position = pd.DataFrame(False, index=long_breakout.index, columns=long_breakout.columns)
    held = pd.Series(False, index=long_breakout.columns)
    for timestamp in position.index:
        held = (held | long_breakout.loc[timestamp]) & ~short_breakout.loc[timestamp]
        position.loc[timestamp] = held

    signal = position.astype(float).where(position)
    return signal.stack().rename("signal")
