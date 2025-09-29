from __future__ import annotations
from typing import Any, Optional
import pandas as pd


class DataFeed:
    def __init__(self, broker):
        self.broker = broker
        self.u = broker.upstox_client

    def get_ohlc(self, instrument: Any, interval: str, from_dt: Optional[str] = None, to_dt: Optional[str] = None) -> pd.DataFrame:
        """Wrap SDK get_ohlc.

        interval: e.g., '1minute','5minute','15minute','60minute','1day' depending on SDK.
        """
        if not hasattr(self.u, "get_ohlc"):
            raise NotImplementedError("SDK client does not expose get_ohlc")
        rows = self.u.get_ohlc(instrument, interval, from_dt, to_dt)
        # Normalize into DataFrame
        df = pd.DataFrame(rows)
        # Try to normalize column names
        rename = {
            'timestamp': 'timestamp', 'time': 'timestamp',
            'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close',
            'volume': 'volume', 'vol': 'volume'
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        return df

    def get_live_price(self, instrument: Any) -> Optional[float]:
        return self.broker.get_live_price(instrument)
