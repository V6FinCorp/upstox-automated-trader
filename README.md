# Upstox Automated Trader (Data + Signals)

JSON-only configuration driven scanning pipeline that reuses V6FinCorp/scanners for market data and computes EMA/RSI/Bollinger signals with consensus. Broker code is included but order routing is optional.

## Structure

- src/
  - config: JSON config, schema, loader
  - adapters: thin wrapper over scanners repo
  - signals: indicator math + consensus engine
  - risk: simple risk utilities (optional for scanning)
  - reports: writers for CSV/Parquet + summaries
  - core: broker client (optional)
  - utils: logging and timeframe helpers
  - vendor/scanners: upstream scanners repo (add as submodule)
- tests/: unit tests
- scripts/: Windows helpers

## Quickstart

1. Add the scanners repo as submodule (recommended):
   - PowerShell
     - `git submodule add https://github.com/V6FinCorp/scanners src/vendor/scanners`
     - `git submodule update --init --recursive`
2. Copy `src/config/config.example.json` to `src/config/config.json` and adjust.
3. Install dependencies:
   - `pip install -r requirements.txt`
4. Run a one-off scan:
   - `python -m upstox_automated_trader.main` or `python src/main.py`

## Notes
- UI is intentionally excluded. Focus is data fetch + indicator signals.
- Config is JSON-only. YAML is not used.
- Backtest-ready CSV logs are produced by reports/writer.

## Implementation Steps (mapped)
1. Setup & Authentication: `core/auth.py` creates a Broker from `.env`.
2. Config Loader: `config/loader.py` parses JSON and validates with `schema.json`.
3. Data Module: `core/data_feed.py` wraps SDK `get_ohlc()` and live price.
4. Indicator Module: `signals/engine.py` implements EMA, RSI, Bollinger.
5. Signal Module: crossover detection with RSI mid threshold (`TRADE_SIGNAL`).
6. Risk Module: `risk/risk_manager.py` for sizing and `DrawdownTracker`.
7. Execution Module: `core/broker.py` and `core/execution.py` for orders and exits.
8. Logging & Reporting: `utils/logger.py`, `reports/writer.py`, `reports/performance.py` (PDF+JSON).
9. Paper Trading: `cli/paper_trade.py` and `scripts/paper_trade.bat`.

## Example: place order and monitor
```python
from core.auth import broker_from_env
from core.execution import submit_entry, watch_price_and_exit, eod_flatten

b = broker_from_env()
entry = submit_entry(b, instrument="RELIANCE", qty=1, side="BUY")
print("entry:", entry)
res = watch_price_and_exit(b, instrument="RELIANCE", qty=1, entry_side="BUY", stop_price=2450.0, target_price=2520.0)
print("exit:", res)
# At day end:
eod_flatten(b)
```

## Deliverables
- `paper_trade.py` script for sandbox execution.
- `performance_report.pdf` with equity curve and metrics (see `output.performance_pdf`).