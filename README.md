# CryptoStatArbLab

A research-grade project for **statistical arbitrage in cryptocurrencies** (momentum + reversal), designed to look and feel like a **quant hedge fund research repo**:
- data ingestion + QC
- dynamic liquidity universe
- signal library (momentum, reversal, regimes, beta-hedged residuals)
- execution-aware backtesting (turnover + costs)
- walk-forward research sweeps + diagnostics

This repo is intentionally **reproducible** and **interview-ready**: it emphasizes validation hygiene, lagging, no lookahead, and net-of-cost reporting.

---

## Quickstart

### 1) Create env + install deps
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or:
```bash
make venv install
```

### 2) Run tests
```bash
make test
```

### 3) Download data
```bash
make data_momentum
```

If Binance returns HTTP `451`, use Kraken for a recent-data study:

```bash
python -m src.data.download \
  --provider kraken \
  --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT ADAUSDT DOGEUSDT AVAXUSDT LINKUSDT LTCUSDT \
  --interval 1h \
  --start 2026-01-01 \
  --end 2026-12-31 \
  --venue spot \
  --out data \
  --overwrite
```

Kraken's public OHLC endpoint only returns the most recent 720 candles per request, so use it for recent unseen-data checks or short feature studies, not full multi-year history.

For reversal research, download the same universe at `5m` into `data/spot/5m/`.

### 4) Run scope-safe sweeps
```bash
make sweep_momentum
make sweep_reversal
```

Outputs are written under `runs/{experiment}_{timestamp}/`, with `runs/latest` pointing to the newest run.

Key artifacts:
- `results.parquet`: full sweep results
- `scoreboard.csv`: headline net-of-cost metrics
- `data_qc.csv`: missing bars, duplicates, extreme returns, zero-volume bars
- `trials/*/timeseries.parquet`: per-trial gross/net returns and turnover

### 5) Plot diagnostics
```bash
make diagnostics
```

Diagnostics include net Sharpe distributions, turnover-vs-Sharpe plots, sweep heatmaps where applicable, weekday/weekend breakdowns, hour-of-day breakdowns, and a leaderboard.

### Daily maintenance

Use [`docs/daily_routine.md`](docs/daily_routine.md) for the daily update, test, research, and GitHub push workflow. Raw data, generated runs, and the Medium draft are intentionally ignored by Git.

---

## Repo structure

```
CryptoStatArbLab/
  src/
    data/        download.py clean.py universe.py
    signals/     momentum.py reversal.py regimes.py beta_hedge.py
    portfolio/   weights.py constraints.py risk.py
    backtest/    engine.py costs.py metrics.py
    research/    sweeps.py diagnostics.py
  configs/       momentum.yaml reversal.yaml
  notebooks/     01_eda.ipynb 02_results_review.ipynb
  reports/       report.md figures/
  tests/
  Makefile
  README.md
```

---

## Notes & design choices

- **No lookahead**: signals at time _t_ are executed at _t+1_.
- **Dynamic universe**: top-N by rolling dollar volume, lagged before trading.
- **Universe exclusions**: stablecoin bases, leveraged/inverse wrappers, and duplicate-like denomination symbols are filtered by default.
- **Costs**: market-order bps model tied to one-way turnover, swept at 10 and 20 bps.
- **Rebalance discipline**: `portfolio.hold_bars` controls scheduled rebalances and includes initial allocation turnover.
- **Headline reporting**: results are evaluated net of costs first.
- **Channel breakout first pass**: `configs/momentum_channel_breakout.yaml` is currently a BTCUSDT-only 1h time-series momentum feature study using `data_train` for 2021-2023 exploration. Treat 2024 `data_holdout` as unseen data after choosing a stable parameter region.

---

