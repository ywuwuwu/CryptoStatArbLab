# BTCUSDT Channel-Breakout Feature Study

## Abstract
This first channel-breakout study is a BTCUSDT-only 1h time-series momentum analysis, not yet a cross-sectional stat-arb result. The purpose is to understand the input features, turnover, cost drag, and stability of the channel-breakout surface before expanding to more symbols. The exploration period is 2021-01-01 through 2023-12-31 using `data_train`. The pre-holdout candidate was frozen as `entry_lookback=168`, `exit_lookback=72`, with 20 bps per one-way turnover as the headline cost assumption. The unseen holdout test was then run once on calendar year 2024 using `data_holdout`, without changing the candidate after seeing results.

## Data
- Symbol: BTCUSDT only.
- Interval: 1h bars.
- Exploration data root: `data_train`.
- Exploration period: 2021-01-01 00:00 UTC to 2023-12-31 23:00 UTC.
- Holdout data root: `data_holdout`.
- Holdout period reserved for next step: 2024-01-01 to 2025-01-01.
- Source files are stored under `data_train/spot/1h/BTCUSDT.parquet` and `data_holdout/spot/1h/BTCUSDT.parquet`.

## Data Hygiene
Training QC from the latest sweep:

| symbol | rows | expected rows | missing bars | duplicate timestamps | zero-volume bars | extreme return bars |
|---|---:|---:|---:|---:|---:|---:|
| BTCUSDT | 26,280 | 26,280 | 14 | 0 | 16 | 0 |

Holdout QC:

| symbol | rows | expected rows | missing bars | duplicate timestamps | zero-volume bars | extreme return bars |
|---|---:|---:|---:|---:|---:|---:|
| BTCUSDT | 8,784 | 8,784 | 0 | 0 | 0 | 0 |

The training set is usable for feature exploration. The missing and zero-volume bars are small relative to the full sample, but they should remain disclosed. The 2024 holdout set is clean by these QC checks. Asset and benchmark returns are computed with `pct_change(fill_method=None)`, so missing prices are not silently forward-filled into returns.

## Methodology
The signal is a long-only channel breakout:
- Compute a prior rolling high over `entry_lookback`.
- Compute a prior rolling low over `exit_lookback`.
- Shift all rolling channels by `lag_bars=1` before comparing against the current close.
- Enter long when close breaks above the prior rolling high.
- Exit when close breaks below the shorter prior rolling low.
- Invalid grid cells where `exit_lookback >= entry_lookback` are skipped.

The sweep grid was:
- `entry_lookback`: 24, 48, 72, 120, 168
- `exit_lookback`: 6, 12, 24, 48, 72
- costs: 10 and 20 bps per one-way turnover

The backtest reports net-of-cost results as the headline. Vol targeting is enabled at 15% annual volatility with max leverage 3.0. This is a feature study, so the first objective is a stable, interpretable surface rather than maximum train Sharpe.

## Training Outputs
Training run:

`runs/btcusdt_channel_breakout_train_20260504_042439`

Generated artifacts:
- `runs/btcusdt_channel_breakout_train_20260504_042439/results.parquet`
- `runs/btcusdt_channel_breakout_train_20260504_042439/scoreboard.csv`
- `runs/btcusdt_channel_breakout_train_20260504_042439/best.json`
- training heatmaps and feature plots under that run's `figures/` directory

The sweep produced 50 rows: 38 valid backtests and 12 correctly skipped invalid parameter combinations.

## Training Results
Across valid training cells:

| metric | value |
|---|---:|
| mean net Sharpe | -0.080 |
| median net Sharpe | 0.036 |
| max net Sharpe | 0.451 |
| mean gross Sharpe | 0.440 |
| median gross Sharpe | 0.510 |
| median average turnover | 0.0073 |
| median breakout frequency | 0.0336 |

Interpretation: the feature has visible gross signal, but net performance is fragile after costs. The median gross Sharpe is positive, while the median net Sharpe is only slightly above zero. This means cost drag and turnover are central to the research conclusion.

## Heatmap Read
Net Sharpe by entry and exit lookback showed that short entry windows were weak net of costs. The most stable region appeared around longer entry lookbacks, especially `entry_lookback=168`.

Mean net Sharpe surface:

| exit \\ entry | 24 | 48 | 72 | 120 | 168 |
|---:|---:|---:|---:|---:|---:|
| 6 | -0.826 | -0.709 | -0.055 | -0.203 | -0.106 |
| 12 | -0.482 | -0.383 | 0.260 | 0.052 | 0.288 |
| 24 | n/a | -0.549 | -0.084 | 0.295 | 0.346 |
| 48 | n/a | n/a | -0.478 | 0.098 | 0.351 |
| 72 | n/a | n/a | n/a | 0.289 | 0.373 |

Turnover declined as exit lookback lengthened:

| exit \\ entry | 24 | 48 | 72 | 120 | 168 |
|---:|---:|---:|---:|---:|---:|
| 6 | 0.0164 | 0.0136 | 0.0125 | 0.0116 | 0.0102 |
| 12 | 0.0107 | 0.0090 | 0.0084 | 0.0082 | 0.0073 |
| 24 | n/a | 0.0059 | 0.0056 | 0.0057 | 0.0052 |
| 48 | n/a | n/a | 0.0040 | 0.0043 | 0.0041 |
| 72 | n/a | n/a | n/a | 0.0036 | 0.0035 |

Breakout frequency was controlled by entry lookback, as expected:

| entry lookback | breakout frequency |
|---:|---:|
| 24 | 0.0859 |
| 48 | 0.0583 |
| 72 | 0.0458 |
| 120 | 0.0336 |
| 168 | 0.0276 |

Interpretation: rarer breakouts were more cost-resilient. The feature is not simply better because it trades more. In fact, the higher-frequency short-entry region had worse net results.

## Frozen Pre-Holdout Candidate
Candidate selected before running holdout:

| parameter | value |
|---|---:|
| `entry_lookback` | 168 |
| `exit_lookback` | 72 |
| headline cost | 20 bps per one-way turnover |

Training metrics for the frozen candidate:

| cost bps | net Sharpe | gross Sharpe | net ann return | net ann vol | max drawdown | worst month | avg turnover | breakout frequency | total cost drag |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.443 | 0.586 | 0.097 | 0.219 | -0.281 | -0.154 | 0.0035 | 0.0276 | 0.093 |
| 20 | 0.301 | 0.586 | 0.066 | 0.220 | -0.297 | -0.160 | 0.0035 | 0.0276 | 0.186 |

Rationale: this is not the highest 10 bps training cell. It was selected because it is a lower-turnover point inside the stable `entry_lookback=168` region and remains positive net of 20 bps costs. This is better aligned with the research goal than choosing the maximum training Sharpe.

## Holdout Test
Step 7 evaluated the frozen candidate on unseen 2024 data.

What was done:
- Created `configs/momentum_channel_breakout_holdout.yaml`.
- Kept `entry_lookback=168`, `exit_lookback=72`, `lag_bars=1`.
- Kept the headline execution assumption at 20 bps per one-way turnover.
- Changed only the data root from `data_train` to `data_holdout`.
- Ran a single backtest, not a parameter sweep.

Why this was done:
- The purpose of holdout is to evaluate the preselected candidate, not to search again.
- Running a grid on holdout would invite cherry-picking.
- Keeping the exact same signal and cost settings preserves the train/holdout discipline.

Expected result before running:
- If the feature was stable, holdout net Sharpe should remain positive or at least not collapse relative to training.
- Turnover and breakout frequency should be broadly comparable, though not identical.
- Drawdown and worst month should remain within a plausible range for a BTC trend-following strategy.
- If holdout failed, the correct conclusion would be that the training surface was not stable enough.

Holdout run:

`runs/btcusdt_channel_breakout_holdout_2024_20260511_032738`

Holdout artifacts:
- `runs/btcusdt_channel_breakout_holdout_2024_20260511_032738/results.parquet`
- `runs/btcusdt_channel_breakout_holdout_2024_20260511_032738/metrics.json`
- `runs/btcusdt_channel_breakout_holdout_2024_20260511_032738/timeseries.parquet`
- `runs/btcusdt_channel_breakout_holdout_2024_20260511_032738/weights.parquet`
- `runs/btcusdt_channel_breakout_holdout_2024_20260511_032738/data_qc.csv`
- holdout diagnostics under `runs/btcusdt_channel_breakout_holdout_2024_20260511_032738/figures/`

Holdout metrics:

| metric | holdout 2024 |
|---|---:|
| net Sharpe | 1.317 |
| gross Sharpe | 1.599 |
| net annual return | 0.281 |
| net annual volatility | 0.214 |
| max drawdown | -0.160 |
| worst month | -0.084 |
| average turnover | 0.0034 |
| breakout frequency | 0.0428 |
| total cost drag | 0.0596 |
| net BTC beta | 0.194 |

Train vs holdout comparison for the frozen 20 bps candidate:

| metric | train 2021-2023 | holdout 2024 | change |
|---|---:|---:|---:|
| net Sharpe | 0.301 | 1.317 | +1.015 |
| gross Sharpe | 0.586 | 1.599 | +1.013 |
| net annual return | 0.066 | 0.281 | +0.215 |
| net annual volatility | 0.220 | 0.214 | -0.006 |
| max drawdown | -0.297 | -0.160 | +0.138 |
| worst month | -0.160 | -0.084 | +0.076 |
| average turnover | 0.0035 | 0.0034 | -0.0001 |
| breakout frequency | 0.0276 | 0.0428 | +0.0152 |
| total cost drag | 0.186 | 0.060 | -0.127 |
| net BTC beta | 0.126 | 0.194 | +0.068 |

Interpretation:
- The holdout did not collapse. It improved materially versus the training result.
- Net Sharpe rose from 0.301 in training to 1.317 in 2024 holdout.
- Turnover stayed almost unchanged, which is important because the holdout improvement was not caused by taking many more trades.
- Breakout frequency rose from 2.76% to 4.28%, meaning BTC broke out more often in 2024 under the same channel definition.
- Max drawdown and worst month were less severe in holdout than training.
- Cost drag was lower in holdout despite higher breakout frequency, because realized turnover stayed low.
- This is supportive evidence for the frozen candidate, but it is still only one held-out year.

Calendar diagnostics:

| segment | count | mean net return | sum net return |
|---|---:|---:|---:|
| weekday | 6,288 | 0.000034 | 0.217 |
| weekend | 2,496 | 0.000026 | 0.066 |

Interpretation: both weekday and weekend segments were positive. Most total PnL came from weekdays because there are more weekday bars, but weekends did not appear to be a failure zone.

Top positive hour-of-day buckets by summed net return:

| UTC hour | sum net return |
|---:|---:|
| 22 | 0.092 |
| 17 | 0.087 |
| 9 | 0.042 |
| 18 | 0.038 |
| 3 | 0.037 |

Worst hour-of-day buckets by summed net return:

| UTC hour | sum net return |
|---:|---:|
| 12 | -0.058 |
| 19 | -0.044 |
| 5 | -0.039 |
| 21 | -0.037 |

Interpretation: hour-of-day performance is uneven, but this should be treated as descriptive diagnostics only. It is not a new regime filter and should not be optimized at this stage.

## Step 8 Evaluation: Train vs Holdout
Step 8 compares the frozen candidate across the 2021-2023 exploration period and the 2024 holdout period.

What was done:
- Loaded the frozen candidate's training row from `runs/btcusdt_channel_breakout_train_20260504_042439/results.parquet`.
- Loaded the single-row holdout result from `runs/btcusdt_channel_breakout_holdout_2024_20260511_032738/results.parquet`.
- Compared net Sharpe, gross Sharpe, annual return, volatility, drawdown, worst month, turnover, breakout frequency, total cost drag, hit rate, and BTC beta.
- Read the train and holdout `timeseries.parquet` and `weights.parquet` files to compare active exposure and monthly return behavior.

Why this was done:
- A holdout result is only useful if it is compared against the selection-period behavior.
- The goal is not only to ask whether 2024 was profitable, but whether the mechanism looked similar enough to trust as out-of-sample evidence.
- Turnover and breakout frequency are especially important because a higher holdout Sharpe caused by very different trading behavior would be less convincing.

Expected result before analysis:
- A robust candidate should not require materially higher turnover in holdout.
- Breakout frequency may vary by market regime, but should remain plausible for a 168-hour channel.
- Net Sharpe should not collapse after costs.
- Drawdown and worst month should not become dramatically worse than training.
- If holdout performance is much better than training, the improvement should be treated as favorable evidence, not as a new performance baseline.

Detailed comparison:

| metric | train 2021-2023 | holdout 2024 | evaluation |
|---|---:|---:|---|
| net Sharpe | 0.301 | 1.317 | Improved materially out of sample. Positive evidence, but one year only. |
| gross Sharpe | 0.586 | 1.599 | Gross signal was stronger in 2024. |
| net annual return | 0.066 | 0.281 | Higher holdout return with similar volatility. |
| net annual volatility | 0.220 | 0.214 | Very similar risk scale. |
| max drawdown | -0.297 | -0.160 | Holdout drawdown was less severe. |
| worst month | -0.160 | -0.084 | Holdout tail month was less severe. |
| net hit rate | 0.157 | 0.187 | Improved, but still low because the strategy is often flat and trend-following payoffs are asymmetric. |
| average turnover | 0.0035 | 0.0034 | Stable. This is one of the strongest signs the holdout behavior is comparable. |
| p95 turnover | 0.0028 | 0.0030 | Stable upper-tail turnover. |
| breakout frequency | 0.0276 | 0.0428 | More breakouts occurred in 2024, but still low-frequency. |
| active bar fraction | 0.308 | 0.370 | Holdout spent more time in position, consistent with more breakouts. |
| entries / exits | 71 / 71 | 24 / 24 | Annualized entry count is similar: about 24 per year in both periods. |
| positive / negative months | 18 / 18 | 8 / 4 | Holdout had a better monthly balance. |
| best month | 0.181 | 0.136 | Training had the single best month. |
| worst month | -0.160 | -0.084 | Holdout had a milder worst month. |
| total cost drag | 0.186 | 0.060 | Lower in holdout; part of this is because holdout is one year vs three training years. |
| net BTC beta | 0.126 | 0.194 | Exposure to BTC direction increased but remained far below buy-and-hold beta. |

Evaluation:
- The holdout result is coherent with the training thesis. The strategy remained low-turnover, low-frequency, and net profitable after 20 bps costs.
- The improvement in 2024 did not come from a turnover explosion. Average turnover was slightly lower in holdout than training.
- The active bar fraction rose from 30.8% to 37.0%, which is consistent with the higher breakout frequency in 2024.
- Annualized entry behavior was comparable: 71 entries over three training years versus 24 entries in the one-year holdout.
- The 2024 monthly profile was better than training: 8 positive months and 4 negative months, compared with an even 18/18 split in training.
- The result supports the frozen candidate, but it should not be oversold. A single strong holdout year can reflect a favorable BTC trend regime.

What needs improvement:
- Add ETHUSDT and SOLUSDT as separate single-symbol studies before claiming broader robustness.
- Add a simple buy-and-hold BTC benchmark table. The current report has beta metrics, but it should explicitly compare the strategy to passive BTC over the same train and holdout windows.
- Add an exposure-adjusted comparison, such as return per active bar or return per unit of exposure, because the strategy is flat much of the time.
- Add monthly return plots or tables to the diagnostics output rather than manually inspecting monthly series.
- Add a drawdown curve figure for the frozen candidate in both train and holdout.
- Consider a small sensitivity band around the frozen region in future robustness work, but do not reselect BTC parameters based on 2024.
- Fix the reporting distinction between total cost drag over different sample lengths and annualized/average cost drag. Total cost drag is not directly comparable between three-year train and one-year holdout.
- Later, test on additional post-2024 data when available. The current holdout is clean but short.

## Limitations
- The 2024 holdout is only one calendar year.
- BTCUSDT-only results should not be described as cross-sectional stat arb.
- The strategy still has material drawdowns, with the frozen candidate showing around -30% max drawdown at 20 bps in training.
- The data has small missing-bar and zero-volume counts in training.
- Slippage/impact beyond the bps cost model is not modeled.
- Hour-of-day diagnostics are descriptive only and were not used for parameter selection.
- No live trading, OMS, limit-order modeling, or machine learning is included.

## Next Steps
1. Do not change the BTCUSDT candidate after seeing the 2024 holdout.
2. Repeat the same separate-symbol study for ETHUSDT and SOLUSDT.
3. Compare each symbol's train/holdout net Sharpe, turnover, breakout frequency, drawdown, and worst month.
4. Only after separate-symbol validation, run a multi-symbol portfolio version.
5. Keep reporting net-of-cost results as the headline.
