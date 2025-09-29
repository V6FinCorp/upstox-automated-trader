def position_size(balance: float, risk_per_trade: float, stop_distance: float) -> int:
    risk_amount = balance * risk_per_trade
    if stop_distance <= 0:
        return 0
    qty = int(risk_amount // stop_distance)
    return max(qty, 0)


class DrawdownTracker:
    def __init__(self, starting_equity: float):
        self.peak = starting_equity
        self.equity = starting_equity
        self.max_dd = 0.0

    def update(self, equity: float) -> float:
        self.equity = equity
        if equity > self.peak:
            self.peak = equity
        dd = (equity / self.peak) - 1.0
        if dd < self.max_dd:
            self.max_dd = dd
        return dd

    def breached(self, max_drawdown_pct: float) -> bool:
        return (self.max_dd * 100.0) <= (-abs(max_drawdown_pct))
