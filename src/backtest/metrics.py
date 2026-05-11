\
"""
Performance evaluation metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd


def periods_per_year(interval: str) -> float:
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


def max_drawdown(returns: pd.Series) -> float:
    curve = (1.0 + returns.fillna(0.0)).cumprod()
    peak = curve.cummax()
    dd = curve / peak - 1.0
    return float(dd.min())


def worst_month(returns: pd.Series) -> float:
    if not isinstance(returns.index, pd.DatetimeIndex):
        return float("nan")
    monthly = (1.0 + returns.fillna(0.0)).resample("ME").prod() - 1.0
    if monthly.empty:
        return float("nan")
    return float(monthly.min())


def alpha_beta(strategy: pd.Series, benchmark: pd.Series, ppy: float) -> tuple[float, float, float]:
    """
    Compute beta = cov/var and alpha (annualized) of strategy vs benchmark.
    Also returns R^2.
    """
    df = pd.concat([strategy, benchmark], axis=1).dropna()
    if df.shape[0] < 10:
        return float("nan"), float("nan"), float("nan")

    s = df.iloc[:, 0].values
    b = df.iloc[:, 1].values
    b_var = np.var(b)
    if b_var <= 1e-18:
        return float("nan"), float("nan"), float("nan")

    cov = np.cov(s, b, ddof=0)[0, 1]
    beta = cov / b_var
    resid = s - beta * b
    alpha = np.mean(resid) * ppy

    # R^2
    ss_res = np.sum((s - (alpha / ppy + beta * b)) ** 2)
    ss_tot = np.sum((s - np.mean(s)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else float("nan")
    return float(alpha), float(beta), float(r2)


def compute_metrics(
    returns: pd.Series,
    interval: str,
    benchmark_returns: Optional[pd.Series] = None,
    prefix: str = "",
) -> Dict[str, Any]:
    """
    returns: per-bar portfolio returns
    """
    ppy = periods_per_year(interval)

    r = returns.dropna().astype(float)
    if len(r) < 5:
        return {f"{prefix}n_obs": int(len(r))}

    ann_ret = r.mean() * ppy
    ann_vol = r.std(ddof=0) * np.sqrt(ppy)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-18 else float("nan")
    mdd = max_drawdown(r)

    out: Dict[str, Any] = {
        f"{prefix}n_obs": int(len(r)),
        f"{prefix}ann_return": float(ann_ret),
        f"{prefix}ann_vol": float(ann_vol),
        f"{prefix}sharpe": float(sharpe),
        f"{prefix}max_drawdown": float(mdd),
        f"{prefix}worst_month": worst_month(r),
        f"{prefix}hit_rate": float((r > 0).mean()),
        f"{prefix}skew": float(r.skew()),
        f"{prefix}kurtosis": float(r.kurtosis()),
    }

    if benchmark_returns is not None:
        a, b, r2 = alpha_beta(r, benchmark_returns.reindex(r.index), ppy=ppy)
        out[f"{prefix}alpha"] = a
        out[f"{prefix}beta"] = b
        out[f"{prefix}r2"] = r2

    return out
