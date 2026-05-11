.PHONY: help venv install test data_momentum data_momentum_kraken backtest_momentum sweep_momentum sweep_reversal diagnostics diag clean

help:
	@echo "Targets:"
	@echo "  venv             Create venv in .venv"
	@echo "  install          Install requirements"
	@echo "  test             Run pytest"
	@echo "  data_momentum    Example: download BTC/ETH/SOL 1h data (spot)"
	@echo "  data_momentum_kraken Download recent Kraken 1h data"
	@echo "  backtest_momentum Run single momentum backtest config"
	@echo "  sweep_momentum   Run momentum parameter sweep"
	@echo "  sweep_reversal   Run reversal parameter sweep"
	@echo "  diagnostics      Plot diagnostics for latest run"
	@echo "  clean            Remove runs/ and cached artifacts"

venv:
	python -m venv .venv

install:
	pip install -r requirements.txt

test:
	python -m pytest

data_momentum:
	python -m src.data.download --symbols BTCUSDT ETHUSDT SOLUSDT --interval 1h --start 2023-01-01 --end 2026-01-01 --venue spot --out data

data_momentum_kraken:
	python -m src.data.download --provider kraken --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT ADAUSDT DOGEUSDT AVAXUSDT LINKUSDT LTCUSDT --interval 1h --start 2026-01-01 --end 2026-12-31 --venue spot --out data --overwrite

backtest_momentum:
	python -m src.research.sweeps --config configs/momentum.yaml --mode single

sweep_momentum:
	python -m src.research.sweeps --config configs/momentum.yaml --mode sweep

sweep_reversal:
	python -m src.research.sweeps --config configs/reversal.yaml --mode sweep

diagnostics:
	python -m src.research.diagnostics --results runs/latest/results.parquet --out runs/latest/figures

diag: diagnostics

clean:
	rm -rf runs .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete
