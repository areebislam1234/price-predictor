# src/features.py
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "prices.db"


def load_prices(symbol, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT timestamp, price FROM prices WHERE symbol = ? ORDER BY timestamp",
        conn, params=(symbol,)
    )
    conn.close()
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))


def add_features(df):
    out = df.copy()
    ma_7 = out["price"].rolling(7).mean()
    ma_30 = out["price"].rolling(30).mean()
    # Express moving averages as the price's % distance from them, not raw
    # price levels. Raw ma_7/ma_30 are non-stationary -- a stock or coin's
    # price can be several times higher at the end of a 5-year window than
    # at the start, so a split threshold like "ma_30 > 150" learned early
    # on becomes meaningless once price has moved to a different range.
    # The relative version carries the same trend signal but stays
    # comparable across time and price scale.
    out["price_vs_ma7"] = out["price"] / ma_7 - 1
    out["price_vs_ma30"] = out["price"] / ma_30 - 1
    out["daily_return"] = out["price"].pct_change()
    out["volatility"] = out["daily_return"].rolling(7).std()
    out["rsi"] = compute_rsi(out["price"])
    return out.dropna().reset_index(drop=True)