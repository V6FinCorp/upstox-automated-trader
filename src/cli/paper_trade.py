import os
import pandas as pd
from datetime import datetime

try:
    from ..reports.performance import equity_curve, save_performance_pdf
except ImportError:
    from reports.performance import equity_curve, save_performance_pdf


def simulate_from_csv(csv_path: str, starting_cash: float = 100000.0, slippage_bps: float = 1.0) -> dict:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"]) if "timestamp" in open(csv_path).read(50) else pd.read_csv(csv_path)
    price = df["close"]
    signal = df.get("TRADE_SIGNAL")
    if signal is None:
        raise ValueError("CSV must contain TRADE_SIGNAL column")

    pos = 0  # -1 short, 0 flat, 1 long
    cash = starting_cash
    shares = 0.0
    eq = []

    for i in range(len(df)):
        s = signal.iloc[i]
        p = float(price.iloc[i])
        if not pd.notna(p):
            eq.append(cash + shares * p * pos)
            continue

        # apply slippage
        p_exec = p * (1 + (slippage_bps/10000) * (1 if s == 'LONG' else -1 if s == 'SHORT' else 0))

        # simple switch: go to target position based on signal
        target = 1 if s == 'LONG' else -1 if s == 'SHORT' else 0
        if target != pos:
            # close existing
            cash += shares * p_exec * pos
            shares = 0
            pos = 0
            # open new
            if target != 0:
                # all-in nominal: use cash/p to size
                shares = cash / p_exec
                cash -= shares * p_exec * target
                pos = target

        equity = cash + shares * p * pos
        eq.append(equity)

    df_eq = pd.DataFrame({"equity": eq})
    metrics = equity_curve(df_eq["equity"], starting_cash)
    return {"equity": df_eq, "metrics": metrics}


def main(csv_path: str, out_pdf: str):
    res = simulate_from_csv(csv_path)
    save_performance_pdf(res["equity"], res["metrics"], out_pdf)
    print(f"Saved performance report: {out_pdf}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m cli.paper_trade <signals_csv> <out_pdf>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
