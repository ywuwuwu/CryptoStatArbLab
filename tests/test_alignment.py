\
import pandas as pd
import numpy as np

from src.portfolio.risk import compute_portfolio_returns
from src.backtest.costs import turnover, apply_bps_costs


def test_compute_portfolio_returns_is_lagged():
    # Two assets, 4 timestamps
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    w = pd.DataFrame(
        {
            "A": [1.0, 0.0, 0.0, 0.0],
            "B": [0.0, 1.0, 0.0, 0.0],
        },
        index=idx,
    )
    r = pd.DataFrame(
        {
            "A": [0.0, 0.10, 0.10, 0.10],
            "B": [0.0, 0.01, 0.01, 0.01],
        },
        index=idx,
    )

    # If lagged: pnl at t1 uses weights at t0 -> 1*A => 0.10
    pnl = compute_portfolio_returns(w, r)
    assert abs(pnl.loc[idx[1]] - 0.10) < 1e-12
    # At t2 uses weights at t1 -> 1*B => 0.01
    assert abs(pnl.loc[idx[2]] - 0.01) < 1e-12


def test_asset_returns_do_not_forward_fill_missing_prices():
    from src.backtest.engine import _asset_returns_wide

    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC", name="timestamp")
    mi = pd.MultiIndex.from_product([idx, ["BTCUSDT"]], names=["timestamp", "symbol"])
    panel = pd.DataFrame(
        {
            "close": [100.0, None, 110.0, 121.0],
            "volume": 1.0,
        },
        index=mi,
    )

    returns = _asset_returns_wide(panel)

    assert pd.isna(returns.loc[idx[1], "BTCUSDT"])
    assert pd.isna(returns.loc[idx[2], "BTCUSDT"])
    assert abs(returns.loc[idx[3], "BTCUSDT"] - 0.10) < 1e-12


def test_turnover_and_costs():
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    w = pd.DataFrame({"A": [1.0, 0.0, 0.0], "B": [0.0, 1.0, 1.0]}, index=idx)
    to = turnover(w)
    # Initial allocation is measured from zero exposure.
    assert abs(to.loc[idx[0]] - 0.5) < 1e-12
    # turnover at t1: 0.5*(|0-1|+|1-0|) = 1.0
    assert abs(to.loc[idx[1]] - 1.0) < 1e-12

    gross = pd.Series([0.0, 0.10, 0.0], index=idx)
    net = apply_bps_costs(gross, to, bps_per_turnover=20)  # 20 bps * 1 turnover => 0.002
    assert abs(net.loc[idx[1]] - (0.10 - 0.002)) < 1e-12


def test_hold_bars_carries_weights_between_rebalances():
    from src.backtest.engine import BacktestSpec, run_backtest
    from src.portfolio.risk import VolTargetSpec

    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    symbols = ["A", "B", "C"]
    mi = pd.MultiIndex.from_product([idx, symbols], names=["timestamp", "symbol"])
    panel = pd.DataFrame(
        {
            "close": np.tile([100.0, 100.0, 100.0], len(idx)),
            "volume": 1.0,
        },
        index=mi,
    )

    signal_wide = pd.DataFrame(
        {
            "A": [3.0, 1.0, 1.0, 3.0, 3.0],
            "B": [2.0, 3.0, 3.0, 2.0, 2.0],
            "C": [1.0, 2.0, 2.0, 1.0, 1.0],
        },
        index=idx,
    )
    signal_wide.index.name = "timestamp"
    signal_wide.columns.name = "symbol"
    signal = signal_wide.stack().rename("signal")

    res = run_backtest(
        panel=panel,
        signal=signal,
        universe=None,
        spec=BacktestSpec(
            interval="1h",
            long_quantile=0.34,
            short_quantile=0.34,
            gross_leverage=1.0,
            max_abs_weight=1.0,
            neutralize="dollar",
            vol_target=VolTargetSpec(enabled=False),
            hold_bars=2,
            bps_per_turnover=0.0,
        ),
    )

    weights = res.weights.unstack("symbol").reindex(idx).fillna(0.0)
    pd.testing.assert_series_equal(weights.loc[idx[1]], weights.loc[idx[0]], check_names=False)
    assert not weights.loc[idx[2]].equals(weights.loc[idx[1]])


def test_dot_path_sweep_sets_nested_config_values():
    from src.research.sweeps import _set_by_dot_path

    cfg = {"signal": {"lookback_bars": 12}, "portfolio": {"hold_bars": 1}}
    _set_by_dot_path(cfg, "signal.lookback_bars", 72)
    _set_by_dot_path(cfg, "portfolio.hold_bars", 24)

    assert cfg["signal"]["lookback_bars"] == 72
    assert cfg["portfolio"]["hold_bars"] == 24


def test_build_backtest_spec_returns_configured_spec():
    from src.research.sweeps import _build_backtest_spec

    spec = _build_backtest_spec(
        {
            "data": {"interval": "1h", "benchmark_symbol": "BTCUSDT"},
            "portfolio": {
                "long_quantile": 0.2,
                "short_quantile": 0.0,
                "hold_bars": 3,
                "neutralize": "none",
                "vol_target": {"enabled": False},
            },
            "execution": {"bps_per_turnover": 20},
        }
    )

    assert spec.interval == "1h"
    assert spec.long_quantile == 0.2
    assert spec.short_quantile == 0.0
    assert spec.hold_bars == 3
    assert spec.bps_per_turnover == 20


def test_beta_residual_reversal_fades_idiosyncratic_shock():
    from src.signals.reversal import BetaResidualReversalSpec, beta_residual_reversal_signal

    idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC", name="timestamp")
    returns = pd.DataFrame(
        {
            "BTCUSDT": [0.0] + [0.01] * 9,
            "AAAUSDT": [0.0] + [0.02] * 5 + [0.07] + [0.02] * 3,
            "BBBUSDT": [0.0] + [0.02] * 5 + [-0.03] + [0.02] * 3,
        },
        index=idx,
    )
    prices = 100.0 * (1.0 + returns).cumprod()
    prices.columns.name = "symbol"

    panel = prices.stack().rename("close").to_frame()
    panel["open"] = panel["close"]
    panel["high"] = panel["close"]
    panel["low"] = panel["close"]
    panel["volume"] = 1.0

    signal = beta_residual_reversal_signal(
        panel,
        BetaResidualReversalSpec(
            lookback_bars=1,
            lag_bars=1,
            z_entry=0.0,
            benchmark_symbol="BTCUSDT",
            beta_window=3,
        ),
    ).unstack("symbol")

    assert signal.loc[idx[7], "AAAUSDT"] < 0.0
    assert signal.loc[idx[7], "BBBUSDT"] > 0.0
    assert "BTCUSDT" not in signal.columns


def test_symbol_exclusion_rules_filter_stablecoins_leverage_and_duplicates():
    from src.data.clean import filter_excluded_symbols

    symbols = ["BTCUSDT", "ETHUSDT", "USDCUSDT", "ETHUPUSDT", "BTCDOWNUSDT", "1000SHIBUSDT"]

    assert filter_excluded_symbols(symbols) == ["BTCUSDT", "ETHUSDT"]


def test_data_qc_report_counts_missing_zero_volume_and_extreme_returns():
    from src.data.clean import data_qc_report

    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC", name="timestamp")
    mi = pd.MultiIndex.from_product([idx, ["AAAUSDT"]], names=["timestamp", "symbol"])
    panel = pd.DataFrame(
        {
            "open": [100.0, 100.0, 300.0, 1000.0],
            "high": [100.0, 100.0, 300.0, 1000.0],
            "low": [100.0, 100.0, 300.0, 1000.0],
            "close": [100.0, None, 300.0, 1000.0],
            "volume": [1.0, 0.0, 1.0, 1.0],
        },
        index=mi,
    )

    qc = data_qc_report(panel, interval="1h").set_index("symbol")

    assert qc.loc["AAAUSDT", "missing_bars"] == 1
    assert qc.loc["AAAUSDT", "zero_volume_bars"] == 1
    assert qc.loc["AAAUSDT", "extreme_return_bars"] == 1


def test_diagnostics_calendar_outputs_smoke(tmp_path):
    from src.research.diagnostics import make_plots

    idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    results = pd.DataFrame(
        {
            "net_sharpe": [1.0],
            "gross_sharpe": [1.2],
            "avg_turnover": [0.3],
        }
    )
    timeseries = pd.DataFrame(
        {
            "gross_return": 0.001,
            "net_return": [0.001, -0.001] * 24,
            "turnover": 0.1,
        },
        index=idx,
    )

    results_path = tmp_path / "results.parquet"
    timeseries_path = tmp_path / "timeseries.parquet"
    out_dir = tmp_path / "figures"
    results.to_parquet(results_path)
    timeseries.to_parquet(timeseries_path)

    make_plots(str(results_path), str(out_dir))

    assert (out_dir / "weekday_weekend_breakdown.csv").exists()
    assert (out_dir / "hour_of_day_breakdown.csv").exists()
    assert (out_dir / "scoreboard.csv").exists()


def test_momentum_signal_is_lagged_before_trading():
    from src.signals.momentum import MomentumSpec, momentum_signal

    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC", name="timestamp")
    prices = pd.DataFrame({"AAAUSDT": [100.0, 110.0, 121.0, 133.1]}, index=idx)
    prices.columns.name = "symbol"
    panel = prices.stack().rename("close").to_frame()
    panel["volume"] = 1.0

    sig = (
        momentum_signal(panel, MomentumSpec(lookback_bars=1, lag_bars=1))
        .unstack("symbol")
        .reindex(idx)
    )

    assert pd.isna(sig.loc[idx[1], "AAAUSDT"])
    assert abs(sig.loc[idx[2], "AAAUSDT"] - 0.10) < 1e-12


def test_universe_selection_uses_lagged_dollar_volume():
    from src.data.universe import UniverseSpec, select_topn_by_dollar_volume

    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC", name="timestamp")
    symbols = ["AAAUSDT", "BBBUSDT"]
    mi = pd.MultiIndex.from_product([idx, symbols], names=["timestamp", "symbol"])
    panel = pd.DataFrame(
        {
            "close": 1.0,
            "volume": [
                1.0,
                10.0,
                100.0,
                10.0,
                100.0,
                10.0,
                100.0,
                10.0,
            ],
        },
        index=mi,
    )

    uni = select_topn_by_dollar_volume(
        panel,
        UniverseSpec(enabled=True, top_n=1, dv_window=1, min_history_bars=0, lag_bars=1),
    ).unstack("symbol")

    assert bool(uni.loc[idx[1], "BBBUSDT"])
    assert not bool(uni.loc[idx[1], "AAAUSDT"])


def test_walk_forward_slices_apply_embargo_between_train_and_test():
    from src.research.sweeps import _walk_forward_slices

    slices = _walk_forward_slices(n=20, train=5, test=3, embargo=2)

    train_slice, test_slice = slices[0]
    assert train_slice == slice(0, 5)
    assert test_slice == slice(7, 10)
    assert test_slice.start - train_slice.stop == 2


def test_channel_breakout_channel_uses_prior_bars_only():
    from src.signals.momentum import ChannelBreakoutSpec, channel_breakout_features

    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC", name="timestamp")
    prices = pd.DataFrame({"AAAUSDT": [10.0, 11.0, 100.0, 12.0]}, index=idx)
    prices.columns.name = "symbol"
    panel = prices.stack().rename("close").to_frame()

    features = channel_breakout_features(
        panel,
        ChannelBreakoutSpec(entry_lookback=2, exit_lookback=1, lag_bars=1),
    )

    assert features.loc[(idx[2], "AAAUSDT"), "prior_rolling_high"] == 11.0
    assert features.loc[(idx[2], "AAAUSDT"), "long_breakout"] == 1.0


def test_channel_breakout_triggers_against_prior_rolling_high_only():
    from src.signals.momentum import ChannelBreakoutSpec, channel_breakout_signal

    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC", name="timestamp")
    prices = pd.DataFrame({"AAAUSDT": [10.0, 11.0, 11.0, 11.5, 9.0]}, index=idx)
    prices.columns.name = "symbol"
    panel = prices.stack().rename("close").to_frame()

    signal = (
        channel_breakout_signal(
            panel,
            ChannelBreakoutSpec(entry_lookback=2, exit_lookback=1, lag_bars=1),
        )
        .unstack("symbol")
        .reindex(idx)
    )

    assert pd.isna(signal.loc[idx[2], "AAAUSDT"])
    assert signal.loc[idx[3], "AAAUSDT"] == 1.0
    assert pd.isna(signal.loc[idx[4], "AAAUSDT"])


def test_channel_breakout_invalid_sweep_combo_is_skipped(tmp_path, monkeypatch):
    from src.research import sweeps

    cfg = {
        "data": {"interval": "1h"},
        "signal": {"name": "channel_breakout", "entry_lookback": 24, "exit_lookback": 6},
        "portfolio": {},
        "execution": {},
        "sweep": {
            "grid": {
                "signal.entry_lookback": [24],
                "signal.exit_lookback": [6, 24],
            }
        },
    }

    monkeypatch.setattr(sweeps, "_run_one", lambda cfg_i, run_dir: {"net_sharpe": 1.0})
    result = sweeps.run_sweep(cfg, str(tmp_path))

    assert len(result) == 2
    skipped_mask = result["skipped"].eq(True)
    assert skipped_mask.sum() == 1
    skipped = result[skipped_mask].iloc[0]
    assert skipped["error"] == "skipped: exit_lookback must be smaller than entry_lookback"


def test_kraken_ohlc_parser_filters_and_standardizes():
    from src.data.download import fetch_kraken_ohlc

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "error": [],
                "result": {
                    "XBTUSDT": [
                        [1704067200, "10", "12", "9", "11", "10.5", "3.0", 7],
                        [1704070800, "11", "13", "10", "12", "11.5", "4.0", 8],
                    ],
                    "last": "1704070800",
                },
            }

    class FakeSession:
        def get(self, url, params, timeout):
            assert url == "https://api.kraken.com/0/public/OHLC"
            assert params["pair"] == "XBTUSDT"
            assert params["interval"] == 60
            return FakeResponse()

    df = fetch_kraken_ohlc(
        FakeSession(),
        symbol="BTCUSDT",
        interval="1h",
        start="2024-01-01",
        end="2024-01-01 01:30",
    )

    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert float(df.loc[0, "close"]) == 11.0
