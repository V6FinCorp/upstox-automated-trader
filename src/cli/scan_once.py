import json, os
try:
    from ..config.loader import load_config  # package-style
except ImportError:  # script-style fallback
    from config.loader import load_config

def main():
    cfg_path = os.getenv("UAT_CONFIG", os.path.join(os.path.dirname(__file__), "..", "config", "config.json"))
    cfg_path = os.path.abspath(cfg_path)
    if not os.path.exists(cfg_path):
        print("Config not found; using built-in demo config.")
        cfg = {
            "symbols": ["RELIANCE"],
            "days_back": 60,
            "timeframes": ["5mins"],
            "indicators": {
                "rsi": {"length": 14, "overbought": 70, "oversold": 30, "mid": 50},
                "ema": {"fast": 9, "slow": 21},
                "bollinger": {"length": 20, "multiplier": 2.0}
            },
            "output": {"dir": "reports", "performance_pdf": "reports/performance_report.pdf"}
        }
    else:
        cfg = load_config(cfg_path)
    print(f"Loaded config for symbols={cfg.get('symbols')} timeframes={cfg.get('timeframes')}")
    # TODO: call scanners adapter and signals engine; write reports
    try:
        from ..adapters.scanners_adapter import fetch_df
        from ..signals.engine import generate_signals
        from ..reports.writer import write_csv
    except ImportError:
        from adapters.scanners_adapter import fetch_df
        from signals.engine import generate_signals
        from reports.writer import write_csv

    out_dir = os.path.abspath(os.path.join(os.path.dirname(cfg_path), "..", cfg['output']['dir']))
    
    for symbol in cfg['symbols']:
        for tf in cfg['timeframes']:
            df = fetch_df(symbol, cfg['days_back'], tf)
            df_sig = generate_signals(df, cfg)
            out = write_csv(df_sig, out_dir, symbol, tf)
            print(f"Wrote: {out}")

    # Optionally produce a performance PDF from the last written file (demo)
    perf_pdf = cfg.get('output', {}).get('performance_pdf')
    if perf_pdf:
        try:
            from ..cli.paper_trade import simulate_from_csv
            from ..reports.performance import save_performance_pdf
        except ImportError:
            from cli.paper_trade import simulate_from_csv
            from reports.performance import save_performance_pdf

        # pick the last symbol/timeframe file as a demo
        last_csv = out
        try:
            res = simulate_from_csv(last_csv)
            save_performance_pdf(res['equity'], res['metrics'], os.path.abspath(os.path.join(os.path.dirname(cfg_path), "..", perf_pdf)))
        except Exception as e:
            print(f"Performance PDF generation skipped: {e}")

if __name__ == "__main__":
    main()
