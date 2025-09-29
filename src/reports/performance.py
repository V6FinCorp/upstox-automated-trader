from __future__ import annotations
import os
import math
import pandas as pd
import json


def equity_curve(equity: pd.Series, starting_cash: float) -> dict:
    eq = equity.fillna(method="ffill").fillna(starting_cash)
    ret = eq.pct_change().fillna(0.0)
    cum_ret = (1 + ret).cumprod() - 1
    peak = eq.cummax()
    dd = (eq / peak - 1.0).fillna(0.0)
    max_dd = dd.min()
    sharpe = (ret.mean() / (ret.std() + 1e-12)) * math.sqrt(252)
    return {
        "final_equity": float(eq.iloc[-1]),
        "return_pct": float(cum_ret.iloc[-1] * 100),
        "max_drawdown_pct": float(max_dd * 100),
        "sharpe": float(sharpe),
    }


def save_performance_pdf(equity_df: pd.DataFrame, metrics: dict, out_pdf: str):
    try:
        import matplotlib.pyplot as plt  # lazy import
    except Exception as e:
        # Skip PDF if matplotlib isn't available
        print(f"Skipping PDF generation (matplotlib not available): {e}")
        return
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig, ax = plt.subplots(2, 1, figsize=(8.27, 11.69), constrained_layout=True)
    equity_df["equity"].plot(ax=ax[0], title="Equity Curve")
    ax[0].set_ylabel("Equity")

    ax[1].axis('off')
    text = "\n".join([
        f"Final Equity: {metrics['final_equity']:.2f}",
        f"Return %: {metrics['return_pct']:.2f}",
        f"Max Drawdown %: {metrics['max_drawdown_pct']:.2f}",
        f"Sharpe (daily->annual): {metrics['sharpe']:.2f}",
    ])
    ax[1].text(0.02, 0.98, text, va='top', ha='left', fontsize=12)

    fig.savefig(out_pdf)
    plt.close(fig)


def save_metrics_json(metrics: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
