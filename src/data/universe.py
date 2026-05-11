\
"""
Universe selection: dynamic liquidity universe (rolling dollar volume top-N).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class UniverseSpec:
    enabled: bool = True
    top_n: int = 50
    dv_window: int = 168
    min_history_bars: int = 200
    lag_bars: int = 1


def compute_dollar_volume(panel: pd.DataFrame) -> pd.Series:
    """
    panel: MultiIndex(timestamp, symbol) with columns close, volume.
    returns: Series with same index of dollar volume = close * volume
    """
    return (panel["close"] * panel["volume"]).rename("dollar_volume")


def select_topn_by_dollar_volume(panel: pd.DataFrame, spec: UniverseSpec) -> pd.Series:
    """
    Returns a boolean Series aligned to panel.index (timestamp, symbol):
      True if symbol is in top-N by rolling mean dollar volume at that timestamp.
    """
    if not spec.enabled:
        return pd.Series(True, index=panel.index, name="in_universe")

    dv = compute_dollar_volume(panel)
    dv_wide = dv.unstack("symbol").sort_index()

    # Rolling average dollar volume, lagged to avoid using current bar's dv
    dv_roll = dv_wide.rolling(window=spec.dv_window, min_periods=spec.dv_window).mean()
    dv_roll = dv_roll.shift(spec.lag_bars)

    # Select top N per timestamp
    ranks = dv_roll.rank(axis=1, ascending=False, method="first")
    in_uni_wide = ranks <= spec.top_n

    # Warmup: force False until min_history_bars are present
    if spec.min_history_bars is not None and spec.min_history_bars > 0:
        warmup_mask = np.arange(len(in_uni_wide)) < spec.min_history_bars
        in_uni_wide.iloc[warmup_mask, :] = False

    in_uni = in_uni_wide.stack().rename("in_universe")
    # Align to panel index (in case panel has missing symbols/timestamps)
    in_uni = in_uni.reindex(panel.index).fillna(False).astype(bool)
    return in_uni
