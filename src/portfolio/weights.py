\
"""
Convert signals into long/short portfolio weights.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constraints import apply_basic_constraints


@dataclass(frozen=True)
class LSWeightSpec:
    long_quantile: float = 0.2
    short_quantile: float = 0.2
    gross_leverage: float = 1.0
    max_abs_weight: float = 0.05
    neutralize: str = "dollar"  # "dollar" or "none"


def construct_long_short_weights(
    signal: pd.Series,
    universe: pd.Series | None,
    spec: LSWeightSpec,
) -> pd.Series:
    """
    Build equal-weighted long-short weights based on cross-sectional signal ranks.

    signal: Series indexed by (timestamp, symbol)
    universe: boolean Series indexed by (timestamp, symbol). If None, assume all True.

    Returns:
      weights: Series indexed by (timestamp, symbol)
    """
    sig_wide = signal.unstack("symbol").sort_index()
    if universe is not None:
        uni_wide = universe.unstack("symbol").reindex(sig_wide.index).reindex(columns=sig_wide.columns)
        sig_wide = sig_wide.where(uni_wide.fillna(False), other=np.nan)

    # Rank signal cross-sectionally (pct rank in [0,1])
    ranks = sig_wide.rank(axis=1, pct=True, method="first")

    long_mask = ranks >= (1.0 - spec.long_quantile)
    short_mask = ranks <= spec.short_quantile

    # Equal weights within long/short; handle rows with no selections.
    n_long = long_mask.sum(axis=1).replace(0, np.nan)
    n_short = short_mask.sum(axis=1).replace(0, np.nan)

    w_long = long_mask.div(n_long, axis=0).fillna(0.0)
    w_short = -short_mask.div(n_short, axis=0).fillna(0.0)

    w = (w_long + w_short)

    # Apply constraints (dollar neutral, max weight, gross leverage)
    w = apply_basic_constraints(
        w,
        gross_leverage=spec.gross_leverage,
        max_abs_weight=spec.max_abs_weight,
        neutralize=spec.neutralize,
    )

    return w.stack().rename("weight")
