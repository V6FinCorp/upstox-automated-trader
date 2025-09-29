from __future__ import annotations
import time
from typing import Any, Dict, Optional


def submit_entry(
    broker,
    instrument: Any,
    qty: int,
    side: str,
    order_type: str = "MARKET",
    price: Optional[float] = None,
    tag: Optional[str] = None,
) -> Dict[str, Any]:
    return broker.place_order(
        instrument=instrument,
        quantity=qty,
        side=side,
        order_type=order_type,
        price=price,
        tag=tag,
    )


def manage_bracket_like(
    broker,
    entry_order_id: str,
    instrument: Any,
    qty: int,
    entry_side: str,
    stop_price: float,
    target_price: float,
    poll_interval: float = 1.0,
) -> Dict[str, Any]:
    """Fallback bracket: after entry fills, place OCO-like exits by polling order history.

    This is a simplified example: it polls the entry, and once filled, places a stop and a limit target.
    It does not fully implement OCO cancellation between exit legs; extend as needed.
    """
    # Wait for entry fill
    entry = broker.poll_order_until_complete(entry_order_id)
    if str(entry.get("status", "")).lower() not in {"filled", "complete", "completed"}:
        return {"entry": entry, "exit": None}

    exit_side = "SELL" if entry_side.upper() == "BUY" else "BUY"
    # Place stop-loss
    stop = broker.place_order(
        instrument=instrument,
        quantity=qty,
        side=exit_side,
        order_type="SL",
        trigger_price=stop_price,
        tag="STOP",
    )
    # Place target
    target = broker.place_order(
        instrument=instrument,
        quantity=qty,
        side=exit_side,
        order_type="LIMIT",
        price=target_price,
        tag="TARGET",
    )

    # Poll both exits until one completes (naive loop)
    terminal = {"complete", "completed", "filled", "cancelled", "rejected", "canceled"}
    stop_id = str(stop.get("order_id")) if isinstance(stop, dict) else None
    tgt_id = str(target.get("order_id")) if isinstance(target, dict) else None
    exit_result = {"stop": stop, "target": target}
    while True:
        time.sleep(poll_interval)
        if stop_id:
            sh = broker.get_order_history(stop_id)
            s = sh if isinstance(sh, dict) else (sh[-1] if isinstance(sh, list) and sh else {})
            if str(s.get("status", "")).lower() in terminal:
                exit_result["stop_final"] = s
                break
        if tgt_id:
            th = broker.get_order_history(tgt_id)
            t = th if isinstance(th, dict) else (th[-1] if isinstance(th, list) and th else {})
            if str(t.get("status", "")).lower() in terminal:
                exit_result["target_final"] = t
                break
    return {"entry": entry, "exit": exit_result}


def watch_price_and_exit(
    broker,
    instrument: Any,
    qty: int,
    entry_side: str,
    stop_price: float,
    target_price: float,
    poll_interval: float = 1.0,
    timeout_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """Exit on stop or target by polling live prices.

    - For BUY entries: stop if price <= stop, target if price >= target.
    - For SELL entries: stop if price >= stop, target if price <= target.
    Places a market order to close once a condition hits.
    """
    start = time.time()
    exit_side = "SELL" if entry_side.upper() == "BUY" else "BUY"
    result: Dict[str, Any] = {}
    while True:
        p = broker.get_live_price(instrument)
        if p is not None:
            if entry_side.upper() == "BUY":
                if p <= stop_price:
                    result["reason"] = "STOP"
                    result["price"] = p
                    result["order"] = broker.place_order(instrument, qty, exit_side, order_type="MARKET")
                    return result
                if p >= target_price:
                    result["reason"] = "TARGET"
                    result["price"] = p
                    result["order"] = broker.place_order(instrument, qty, exit_side, order_type="MARKET")
                    return result
            else:  # SELL entry
                if p >= stop_price:
                    result["reason"] = "STOP"
                    result["price"] = p
                    result["order"] = broker.place_order(instrument, qty, exit_side, order_type="MARKET")
                    return result
                if p <= target_price:
                    result["reason"] = "TARGET"
                    result["price"] = p
                    result["order"] = broker.place_order(instrument, qty, exit_side, order_type="MARKET")
                    return result
        if timeout_sec is not None and (time.time() - start) > timeout_sec:
            result["reason"] = "TIMEOUT"
            return result
        time.sleep(poll_interval)


def eod_flatten(broker) -> None:
    """End-of-day: cancel all open orders and flatten net positions."""
    broker.cancel_all_open_orders()
    broker.flatten_all_positions()
