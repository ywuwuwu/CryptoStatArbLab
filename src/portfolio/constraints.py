\
"""
Portfolio constraints utilities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def neutralize_dollar(weights: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce sum(weights)=0 at each timestamp.
    """
    w = weights.copy()
    row_sum = w.sum(axis=1)
    # subtract mean weight across active names
    n = w.notna().sum(axis=1).replace(0, np.nan)
    adj = row_sum / n
    w = w.sub(adj, axis=0)
    return w


def clip_max_abs_weight(weights: pd.DataFrame, max_abs_weight: float) -> pd.DataFrame:
    if max_abs_weight is None:
        return weights
    return weights.clip(lower=-max_abs_weight, upper=max_abs_weight)


def scale_to_gross_leverage(weights: pd.DataFrame, gross_leverage: float) -> pd.DataFrame:
    """
    Scale each row so sum(abs(weights)) == gross_leverage (if possible).
    """
    w = weights.copy()
    gross = w.abs().sum(axis=1).replace(0.0, np.nan)
    scale = gross_leverage / gross
    w = w.mul(scale, axis=0)
    return w


def apply_basic_constraints(
    weights: pd.DataFrame,
    gross_leverage: float = 1.0,
    max_abs_weight: float | None = None,
    neutralize: str = "dollar",
) -> pd.DataFrame:
    w = weights.copy()
    w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if neutralize == "dollar":
        w = neutralize_dollar(w)

    if max_abs_weight is not None:
        w = clip_max_abs_weight(w, max_abs_weight=max_abs_weight)

    w = scale_to_gross_leverage(w, gross_leverage=gross_leverage)
    w = w.fillna(0.0)
    return w
