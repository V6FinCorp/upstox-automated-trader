from __future__ import annotations
import time
from typing import Any, Dict, Optional
from upstox_api.api import Upstox, Session


class Broker:
    """Thin wrapper over Upstox client with utility helpers.

    Notes:
    - Parameter names vary across SDK versions. This wrapper accepts common fields
      and forwards them using the SDK's expected names where possible.
    - Instrument vs symbol: pass whatever your SDK expects (string or dict). We forward as-is.
    """

    def __init__(self, api_key: str, api_secret: str, access_token: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.upstox_client = self._initialize_client()

    def _initialize_client(self) -> Upstox:
        session = Session(self.api_key, self.api_secret)
        session.set_access_token(self.access_token)
        return Upstox(self.api_key, self.access_token)

    # ------------------------------------------------------------------
    # Basic Orders
    # ------------------------------------------------------------------
    def place_order(
        self,
        instrument: Any,
        quantity: int,
        side: str,
        order_type: str = "MARKET",
        product_type: str = "CNC",
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        validity: str = "DAY",
        tag: Optional[str] = None,
        disclosed_quantity: Optional[int] = None,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Place an order.

        - instrument: symbol/instrument token per SDK
        - side: 'BUY' or 'SELL'
        - order_type: 'MARKET' | 'LIMIT' | 'SL' | 'SL-M' (as supported)
        - product_type: e.g., 'CNC', 'MIS', 'NRML'
        """
        params: Dict[str, Any] = dict(
            instrument=instrument,
            quantity=quantity,
            side=side,
            order_type=order_type,
            product_type=product_type,
            price=price,
            trigger_price=trigger_price,
            validity=validity,
            tag=tag,
            disclosed_quantity=disclosed_quantity,
        )
        # prune None values to avoid SDK validation issues
        clean = {k: v for k, v in params.items() if v is not None}
        if additional_params:
            clean.update(additional_params)
        # Many SDKs expect symbol instead of instrument; mirror param if needed
        if "instrument" in clean and "symbol" not in clean:
            clean["symbol"] = clean["instrument"]
        return self.upstox_client.place_order(**clean)

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        return self.upstox_client.get_order(order_id)

    def get_order_history(self, order_id: Optional[str] = None):
        # Some SDKs support filtering by order_id; if not, we fetch and filter
        try:
            return self.upstox_client.get_order_history(order_id)
        except TypeError:
            all_hist = self.upstox_client.get_order_history()
            if order_id is None:
                return all_hist
            return [h for h in (all_hist or []) if str(h.get("order_id")) == str(order_id)]

    def cancel_order(self, order_id: str):
        return self.upstox_client.cancel_order(order_id)

    def get_balance(self):
        return self.upstox_client.get_balance()

    # ------------------------------------------------------------------
    # Basket / Bracket
    # ------------------------------------------------------------------
    def place_basket_order(self, basket: Any):
        """Place a basket/bracket order if the SDK supports it.

        basket: SDK-specific structure/list of legs.
        """
        if hasattr(self.upstox_client, "place_basket_order"):
            return self.upstox_client.place_basket_order(basket)
        raise NotImplementedError("place_basket_order is not supported by this SDK version")

    # ------------------------------------------------------------------
    # Polling helpers
    # ------------------------------------------------------------------
    def poll_order_until_complete(
        self, order_id: str, timeout_sec: float = 120.0, interval_sec: float = 1.0
    ) -> Dict[str, Any]:
        """Poll order history until terminal state or timeout.

        Terminal states vary; we look for status like 'complete', 'filled', 'cancelled', 'rejected'.
        """
        start = time.time()
        terminal = {"complete", "completed", "filled", "cancelled", "rejected", "canceled"}
        last: Dict[str, Any] | None = None
        while time.time() - start < timeout_sec:
            hist = self.get_order_history(order_id)
            if isinstance(hist, dict):
                last = hist
            elif isinstance(hist, list) and hist:
                last = hist[-1]
            if last:
                status = str(last.get("status", "")).lower()
                if status in terminal:
                    return last
            time.sleep(interval_sec)
        return last or {"order_id": order_id, "status": "timeout"}

    # ------------------------------------------------------------------
    # Live data and EOD helpers (SDK-dependent fallbacks)
    # ------------------------------------------------------------------
    def get_live_price(self, instrument: Any) -> Optional[float]:
        """Return last traded price for instrument using SDK's live feed if available."""
        u = self.upstox_client
        price = None
        if hasattr(u, "get_live_feed"):
            try:
                data = u.get_live_feed(instrument, "full")
                # Upstox formats vary; attempt common keys
                price = (
                    data.get("ltp")
                    or data.get("last_price")
                    or data.get("close")
                    or None
                )
            except Exception:
                price = None
        return float(price) if price is not None else None

    def list_open_orders(self):
        if hasattr(self.upstox_client, "get_orders"):
            try:
                return self.upstox_client.get_orders()
            except Exception:
                return []
        return []

    def list_positions(self):
        if hasattr(self.upstox_client, "get_positions"):
            try:
                return self.upstox_client.get_positions()
            except Exception:
                return []
        return []

    def cancel_all_open_orders(self):
        orders = self.list_open_orders() or []
        for o in orders:
            status = str(o.get("status", "")).lower()
            if status not in {"filled", "complete", "completed", "rejected", "cancelled", "canceled"}:
                oid = o.get("order_id") or o.get("id")
                if oid:
                    try:
                        self.cancel_order(str(oid))
                    except Exception:
                        pass

    def flatten_all_positions(self):
        pos = self.list_positions() or []
        for p in pos:
            qty = int(p.get("net_qty") or p.get("quantity") or 0)
            instrument = p.get("symbol") or p.get("instrument")
            if not instrument or qty == 0:
                continue
            side = "SELL" if qty > 0 else "BUY"
            try:
                self.place_order(instrument=instrument, quantity=abs(qty), side=side, order_type="MARKET")
            except Exception:
                pass
