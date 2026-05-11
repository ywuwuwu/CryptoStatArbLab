\
"""
Transaction cost utilities.
"""
from __future__ import annotations

import pandas as pd


def turnover(weights_wide: pd.DataFrame) -> pd.Series:
    """
    One-way turnover per bar:
      0.5 * sum_i |w_t - w_{t-1}|
    (0.5 is standard for L/S portfolios; counts buy+sell as one rebalance.)
    The first row is measured versus zero exposure, so initial allocation
    costs are included.
    """
    w = weights_wide.fillna(0.0)
    dw = w.sub(w.shift(1).fillna(0.0)).abs()
    to = 0.5 * dw.sum(axis=1)
    return to.rename("turnover")


def apply_bps_costs(gross_returns: pd.Series, turnover: pd.Series, bps_per_turnover: float) -> pd.Series:
    """
    Costs in returns units: cost = turnover * (bps / 10_000)
    """
    cost = turnover.reindex(gross_returns.index).fillna(0.0) * (bps_per_turnover / 10_000.0)
    net = gross_returns - cost
    return net.rename("net_return")
