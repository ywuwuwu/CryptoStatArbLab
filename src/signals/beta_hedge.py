\
"""
Rolling beta estimation and residual (beta-hedged) returns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BetaHedgeSpec:
    benchmark_symbol: str = "BTCUSDT"
    window: int = 336  # e.g., 14 days for 1h bars
    lag_bars: int = 1
    price_col: str = "close"


def beta_hedged_residual_returns(panel: pd.DataFrame, spec: BetaHedgeSpec) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute rolling beta of each asset vs benchmark and residual returns.

    Returns:
      residual_returns_wide: DataFrame index timestamp, columns symbols
      beta_wide: DataFrame index timestamp, columns symbols

    Notes:
      - Uses rolling cov/var for speed and robustness.
      - Lags beta by spec.lag_bars to avoid lookahead.
    """
    prices = panel[spec.price_col].unstack("symbol").sort_index()
    rets = prices.pct_change(1)

    if spec.benchmark_symbol not in rets.columns:
        raise ValueError(f"Benchmark symbol {spec.benchmark_symbol} not found in panel.")

    bench = rets[spec.benchmark_symbol]
    bench_var = bench.rolling(spec.window, min_periods=spec.window).var(ddof=0)

    beta = {}
    for sym in rets.columns:
        if sym == spec.benchmark_symbol:
            continue
        cov = rets[sym].rolling(spec.window, min_periods=spec.window).cov(bench)
        beta[sym] = cov / bench_var.replace(0.0, np.nan)

    beta_wide = pd.DataFrame(beta, index=rets.index).shift(spec.lag_bars)
    # Residual returns for non-benchmark symbols
    resid = rets[beta_wide.columns].sub(beta_wide.mul(bench, axis=0), axis=0)

    return resid, beta_wide
