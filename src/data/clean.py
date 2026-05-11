\
"""
Data cleaning + IO utilities.

This project standardizes OHLCV into:

Columns:
  timestamp (UTC pandas datetime64[ns, UTC])
  open, high, low, close, volume (float)

Files:
  - Prefer Parquet (fast). If parquet dependencies are missing, fall back to CSV.
"""
from __future__ import annotations

import os
import re
from typing import Optional, Iterable

import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def interval_to_pandas_freq(interval: str) -> str:
    # Binance style intervals
    m = interval.strip().lower()
    if m.endswith("m"):
        return f"{int(m[:-1])}min"
    if m.endswith("h"):
        return f"{int(m[:-1])}h"
    if m.endswith("d"):
        return f"{int(m[:-1])}D"
    if m.endswith("w"):
        return f"{int(m[:-1])}W"
    raise ValueError(f"Unsupported interval: {interval}")


def standardize_klines_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Binance kline dataframe (raw columns) to standardized OHLCV.

    Expected columns in df_raw include:
      open_time, open, high, low, close, volume, ...
    """
    df = df_raw.copy()
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return df


def clean_ohlcv(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Reindex to a full timestamp grid (keeps NaNs for missing bars),
    enforce dtypes, sort, and drop duplicates.
    """
    if df.empty:
        return df

    freq = interval_to_pandas_freq(interval)
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.drop_duplicates(subset=["timestamp"], keep="last")
    out = out.sort_values("timestamp")
    out = out.set_index("timestamp")

    full_index = pd.date_range(out.index.min(), out.index.max(), freq=freq, tz="UTC")
    out = out.reindex(full_index)

    # Keep timestamp as a column too
    out.index.name = "timestamp"
    out = out.reset_index()

    # Numeric coercion again after reindex
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def save_ohlcv_file(df: pd.DataFrame, path: str) -> None:
    """
    Save standardized OHLCV file to parquet if possible, else CSV.
    """
    ensure_dir(os.path.dirname(path))
    try:
        df.to_parquet(path, index=False)
    except Exception:
        # fall back to CSV
        csv_path = os.path.splitext(path)[0] + ".csv"
        df.to_csv(csv_path, index=False)


def load_ohlcv_file(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    # Try parquet first, then csv
    try:
        return pd.read_parquet(path)
    except Exception:
        csv_path = os.path.splitext(path)[0] + ".csv"
        return pd.read_csv(csv_path)


def infer_symbols(root: str, venue: str, interval: str) -> list[str]:
    """
    Infer symbols from data directory root/{venue}/{interval}/*.parquet|*.csv
    """
    base = os.path.join(root, venue, interval)
    if not os.path.isdir(base):
        return []
    syms: set[str] = set()
    for fn in os.listdir(base):
        if fn.endswith(".parquet") or fn.endswith(".csv"):
            syms.add(os.path.splitext(fn)[0])
    return sorted(syms)


STABLECOIN_BASES = {
    "BUSD",
    "DAI",
    "FDUSD",
    "TUSD",
    "USDC",
    "USDD",
    "USDP",
    "USDS",
    "USDT",
}
LEVERAGED_SUFFIXES = (
    "BULL",
    "BEAR",
    "UP",
    "DOWN",
    "HALF",
    "HEDGE",
    "2L",
    "2S",
    "3L",
    "3S",
    "4L",
    "4S",
    "5L",
    "5S",
)


def base_asset(symbol: str) -> str:
    s = symbol.upper()
    for quote in ("USDT", "USDC", "BUSD", "USD", "BTC", "ETH"):
        if s.endswith(quote) and len(s) > len(quote):
            return s[: -len(quote)]
    return s


def is_excluded_symbol(symbol: str) -> bool:
    """
    Exclude stablecoin bases, leveraged/inverse token wrappers, and duplicate-like
    denomination variants such as 1000SHIBUSDT.
    """
    base = base_asset(symbol)
    if base in STABLECOIN_BASES:
        return True
    if base.startswith(("1000", "10000", "100000", "1M")):
        return True
    if any(base.endswith(suffix) for suffix in LEVERAGED_SUFFIXES):
        return True
    if re.search(r"(BULL|BEAR|UP|DOWN)\d*$", base):
        return True
    return False


def filter_excluded_symbols(symbols: Iterable[str]) -> list[str]:
    return [sym for sym in symbols if not is_excluded_symbol(sym)]


def data_qc_report(
    panel: pd.DataFrame,
    interval: str,
    extreme_return_threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Per-symbol data quality report for cleaned OHLCV panels.
    """
    rows = []
    freq = interval_to_pandas_freq(interval)
    for sym, df in panel.reset_index().groupby("symbol", sort=True):
        ts = pd.to_datetime(df["timestamp"], utc=True)
        expected = pd.date_range(ts.min(), ts.max(), freq=freq, tz="UTC") if len(ts) else []
        close = pd.to_numeric(df["close"], errors="coerce")
        volume = pd.to_numeric(df["volume"], errors="coerce")
        returns = close.pct_change(1, fill_method=None)

        rows.append(
            {
                "symbol": sym,
                "start": ts.min(),
                "end": ts.max(),
                "rows": int(len(df)),
                "expected_rows": int(len(expected)),
                "missing_bars": int(close.isna().sum()),
                "duplicate_timestamps": int(ts.duplicated().sum()),
                "zero_volume_bars": int((volume.fillna(0.0) <= 0.0).sum()),
                "extreme_return_bars": int(
                    returns.replace([float("inf"), float("-inf")], pd.NA)
                    .abs()
                    .gt(extreme_return_threshold)
                    .sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def load_ohlcv_panel(
    root: str,
    interval: str,
    venue: str,
    symbols: Optional[Iterable[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    clean: bool = True,
    exclude_symbols: bool = True,
) -> pd.DataFrame:
    """
    Load OHLCV for many symbols and return a panel DataFrame:

      index: MultiIndex(timestamp, symbol)
      columns: open, high, low, close, volume

    start/end are optional ISO date strings (YYYY-MM-DD or full timestamps).
    """
    if symbols is None:
        symbols = infer_symbols(root=root, venue=venue, interval=interval)
    symbols = list(symbols)
    if exclude_symbols:
        symbols = filter_excluded_symbols(symbols)

    if not symbols:
        raise FileNotFoundError(
            f"No symbols found. Expected files under {os.path.join(root, venue, interval)}"
        )

    dfs = []
    for sym in symbols:
        p_parquet = os.path.join(root, venue, interval, f"{sym}.parquet")
        p_csv = os.path.join(root, venue, interval, f"{sym}.csv")
        path = p_parquet if os.path.exists(p_parquet) else p_csv
        if not os.path.exists(path):
            continue
        df = load_ohlcv_file(path)
        if "timestamp" not in df.columns:
            raise ValueError(f"{path} missing 'timestamp' column")
        if clean:
            df = clean_ohlcv(df, interval=interval)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["symbol"] = sym
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError("Found no readable OHLCV files.")

    panel = pd.concat(dfs, ignore_index=True)
    panel = panel.set_index(["timestamp", "symbol"]).sort_index()

    # Optional time filter
    if start is not None:
        panel = panel.loc[pd.Timestamp(start, tz="UTC") :]
    if end is not None:
        panel = panel.loc[: pd.Timestamp(end, tz="UTC")]

    # Ensure expected columns exist
    need = ["open", "high", "low", "close", "volume"]
    missing = [c for c in need if c not in panel.columns]
    if missing:
        raise ValueError(f"Missing columns in panel: {missing}")

    panel[need] = panel[need].apply(pd.to_numeric, errors="coerce")
    return panel
