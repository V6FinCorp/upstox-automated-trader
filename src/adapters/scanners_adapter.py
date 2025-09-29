import os, sys, pandas as pd

def _ensure_scanners_on_path(scanners_path: str | None = None):
    path = scanners_path or os.environ.get('SCANNERS_REPO_PATH') or os.path.join(os.path.dirname(__file__), '..', 'vendor', 'scanners')
    abs_path = os.path.abspath(path)
    if abs_path not in sys.path:
        sys.path.append(abs_path)


def fetch_df(symbol: str, days_back: int, timeframe: str) -> pd.DataFrame:
    _ensure_scanners_on_path()
    from data_loader import fetch_data_for_symbol  # provided by scanners repo
    csv_path = fetch_data_for_symbol(symbol, days_back)
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp').set_index('timestamp')

    rule_map = {
        '1min': '1T', '3mins': '3T', '5mins': '5T', '10mins': '10T', '15mins': '15T',
        '30mins': '30T', '60mins': '60T', 'day': '1D'
    }
    tf = timeframe.lower()
    if tf in rule_map and tf != '5mins':
        rule = rule_map[tf]
        o = df['open'].resample(rule).first()
        h = df['high'].resample(rule).max()
        l = df['low'].resample(rule).min()
        c = df['close'].resample(rule).last()
        v = df['volume'].resample(rule).sum()
        df = pd.concat([o, h, l, c, v], axis=1).dropna()
        df.columns = ['open', 'high', 'low', 'close', 'volume']
    return df.reset_index()
