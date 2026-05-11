\
"""
Data downloader.

Supports:
- spot:   https://api.binance.com/api/v3/klines
- futures (USDT-M): https://fapi.binance.com/fapi/v1/klines
- kraken spot: https://api.kraken.com/0/public/OHLC

No API key required for klines.

Outputs per-symbol files into:
  {root}/{venue}/{interval}/{symbol}.parquet

If parquet writer dependencies are missing, falls back to CSV.
"""
from __future__ import annotations

import argparse
import time
import os
from dataclasses import dataclass
from typing import Iterable, Optional, Literal

import pandas as pd
import requests
from tqdm import tqdm

from .clean import ensure_dir, save_ohlcv_file, standardize_klines_df


Venue = Literal["spot", "futures"]
Provider = Literal["binance", "kraken"]


BINANCE_ENDPOINTS = {
    "spot": "https://api.binance.com/api/v3/klines",
    "futures": "https://fapi.binance.com/fapi/v1/klines",
}


@dataclass(frozen=True)
class DownloadSpec:
    symbols: list[str]
    interval: str
    start: str  # YYYY-MM-DD
    end: str    # YYYY-MM-DD
    venue: Venue
    out_root: str
    provider: Provider = "binance"
    sleep_s: float = 0.15
    overwrite: bool = False


def _to_ms(dt_str: str) -> int:
    # Interpret date strings as UTC midnight.
    ts = pd.Timestamp(dt_str, tz="UTC")
    return int(ts.value // 1_000_000)  # ns -> ms


def _to_seconds(dt_str: str) -> int:
    return _to_ms(dt_str) // 1000


def _kraken_interval_minutes(interval: str) -> int:
    m = interval.strip().lower()
    if m.endswith("m"):
        return int(m[:-1])
    if m.endswith("h"):
        return int(m[:-1]) * 60
    if m.endswith("d"):
        return int(m[:-1]) * 1440
    raise ValueError(f"Unsupported Kraken interval: {interval}")


def _kraken_pair(symbol: str) -> str:
    # Kraken uses XBT instead of BTC and XDG internally for DOGE, but common
    # request aliases are accepted for most pairs. Keep output filenames as the
    # user's requested symbol.
    s = symbol.upper()
    if s.startswith("BTC"):
        return "XBT" + s[3:]
    return s


def fetch_binance_klines(
    session: requests.Session,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    venue: Venue,
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Fetch klines [start_ms, end_ms) with pagination (max 1000 rows per request).
    Returns a standardized OHLCV dataframe with columns:
        timestamp, open, high, low, close, volume
    """
    url = BINANCE_ENDPOINTS[venue]
    rows: list[list] = []
    curr = start_ms

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": curr,
            "endTime": end_ms,
            "limit": limit,
        }
        r = session.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        if not data:
            break

        rows.extend(data)

        last_open_time = int(data[-1][0])
        # Binance returns klines including last_open_time; step forward by 1ms
        curr = last_open_time + 1

        # If we got fewer than limit rows, we are done.
        if len(data) < limit:
            break

        # Respect rate limits.
        time.sleep(0.05)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df_raw = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "n_trades",
            "taker_buy_base_vol",
            "taker_buy_quote_vol",
            "ignore",
        ],
    )
    df = standardize_klines_df(df_raw)
    return df


def fetch_kraken_ohlc(
    session: requests.Session,
    symbol: str,
    interval: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Fetch Kraken OHLC data.

    Kraken's public OHLC endpoint returns only up to the most recent 720
    entries for the requested interval. Older history cannot be retrieved from
    this endpoint even when since is supplied.
    """
    url = "https://api.kraken.com/0/public/OHLC"
    params = {
        "pair": _kraken_pair(symbol),
        "interval": _kraken_interval_minutes(interval),
        "since": _to_seconds(start),
    }
    r = session.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    errors = payload.get("error", [])
    if errors:
        raise RuntimeError(f"Kraken API error for {symbol}: {errors}")

    result = payload.get("result", {})
    pair_keys = [k for k in result.keys() if k != "last"]
    if not pair_keys:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    rows = result[pair_keys[0]]
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "vwap", "volume", "count"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp")

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)]
    df = df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return df


def download_symbol(spec: DownloadSpec, symbol: str) -> str:
    start_ms = _to_ms(spec.start)
    end_ms = _to_ms(spec.end)
    out_dir = os.path.join(spec.out_root, spec.venue, spec.interval)
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{symbol}.parquet")

    if (not spec.overwrite) and os.path.exists(out_path):
        return out_path

    with requests.Session() as session:
        if spec.provider == "binance":
            df = fetch_binance_klines(
                session=session,
                symbol=symbol,
                interval=spec.interval,
                start_ms=start_ms,
                end_ms=end_ms,
                venue=spec.venue,
            )
        elif spec.provider == "kraken":
            if spec.venue != "spot":
                raise ValueError("Kraken downloader supports spot only")
            df = fetch_kraken_ohlc(
                session=session,
                symbol=symbol,
                interval=spec.interval,
                start=spec.start,
                end=spec.end,
            )
        else:
            raise ValueError(f"Unknown provider: {spec.provider}")

    # Save
    save_ohlcv_file(df, out_path)
    return out_path


def download(spec: DownloadSpec) -> list[str]:
    paths: list[str] = []
    if spec.provider == "kraken":
        print("[INFO] Kraken OHLC returns only the most recent 720 candles per request.")
    for sym in tqdm(spec.symbols, desc=f"Downloading {spec.provider} {spec.venue} {spec.interval}"):
        try:
            p = download_symbol(spec, sym)
            paths.append(p)
            time.sleep(spec.sleep_s)
        except Exception as e:
            print(f"[WARN] Failed {sym}: {e}")
    return paths


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["binance", "kraken"], default="binance")
    p.add_argument("--symbols", nargs="+", required=True, help="Symbols like BTCUSDT ETHUSDT ...")
    p.add_argument("--interval", required=True, help="Binance interval: 1m, 5m, 1h, 1d, ...")
    p.add_argument("--start", required=True, help="YYYY-MM-DD (UTC)")
    p.add_argument("--end", required=True, help="YYYY-MM-DD (UTC), exclusive-ish")
    p.add_argument("--venue", choices=["spot", "futures"], default="spot")
    p.add_argument("--out", default="data", help="Output root directory")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--sleep", type=float, default=0.15, help="Sleep between symbols (seconds)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    spec = DownloadSpec(
        symbols=list(args.symbols),
        interval=args.interval,
        start=args.start,
        end=args.end,
        venue=args.venue,
        out_root=args.out,
        provider=args.provider,
        overwrite=bool(args.overwrite),
        sleep_s=float(args.sleep),
    )
    paths = download(spec)
    print(f"Saved {len(paths)} files under {spec.out_root}/")


if __name__ == "__main__":
    main()
