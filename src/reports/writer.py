import os

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_csv(df, out_dir: str, symbol: str, timeframe: str):
    ensure_dir(out_dir)
    out = os.path.join(out_dir, f"{symbol}_{timeframe}.csv")
    df.to_csv(out, index=False)
    return out
