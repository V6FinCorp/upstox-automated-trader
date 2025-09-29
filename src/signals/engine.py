import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/length, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def bollinger(close: pd.Series, length: int = 20, mult: float = 2.0):
    ma = close.rolling(length).mean()
    sd = close.rolling(length).std()
    upper = ma + mult * sd
    lower = ma - mult * sd
    return ma, upper, lower


def generate_signals(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    close = df['close']

    # RSI
    rsi_len = cfg['indicators']['rsi']['length']
    rsi_ob = cfg['indicators']['rsi']['overbought']
    rsi_os = cfg['indicators']['rsi']['oversold']
    df['RSI'] = rsi(close, rsi_len)
    df['RSI_signal'] = np.where(df['RSI'] >= rsi_ob, 'SELL', np.where(df['RSI'] <= rsi_os, 'BUY', 'HOLD'))

    # EMA crossover + RSI gating
    fast, slow = cfg['indicators']['ema']['fast'], cfg['indicators']['ema']['slow']
    mid = cfg.get('indicators', {}).get('rsi', {}).get('mid', 50)
    df[f'EMA{fast}'] = ema(close, fast)
    df[f'EMA{slow}'] = ema(close, slow)
    df['EMA_signal'] = np.where(df[f'EMA{fast}'] > df[f'EMA{slow}'], 'BUY', 'SELL')

    # Cross detection: previous relationship vs current
    fast_s = df[f'EMA{fast}']
    slow_s = df[f'EMA{slow}']
    prev_fast_le_slow = fast_s.shift(1) <= slow_s.shift(1)
    prev_fast_ge_slow = fast_s.shift(1) >= slow_s.shift(1)
    cross_up = prev_fast_le_slow & (fast_s > slow_s)
    cross_dn = prev_fast_ge_slow & (fast_s < slow_s)
    df['CROSS_UP'] = cross_up.fillna(False)
    df['CROSS_DOWN'] = cross_dn.fillna(False)
    df['ENTRY_LONG'] = df['CROSS_UP'] & (df['RSI'] > mid)
    df['ENTRY_SHORT'] = df['CROSS_DOWN'] & (df['RSI'] < mid)
    df['TRADE_SIGNAL'] = np.where(df['ENTRY_LONG'], 'LONG', np.where(df['ENTRY_SHORT'], 'SHORT', 'HOLD'))

    # Bollinger
    bb_len, bb_mult = cfg['indicators']['bollinger']['length'], cfg['indicators']['bollinger']['multiplier']
    _, bb_u, bb_l = bollinger(close, bb_len, bb_mult)
    df['BB_upper'], df['BB_lower'] = bb_u, bb_l
    df['BB_signal'] = np.where(close > bb_u, 'SELL', np.where(close < bb_l, 'BUY', 'HOLD'))

    # Consensus (simple majority)
    votes = df[['RSI_signal', 'EMA_signal', 'BB_signal']].apply(lambda r: r.value_counts().idxmax(), axis=1)
    df['CONSENSUS'] = votes
    return df
