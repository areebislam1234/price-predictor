# src/ingest.py
import sqlite3
import time
from pathlib import Path

import yfinance as yf

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "prices.db"

# CoinGecko's free tier caps historical data at 365 days, which was too
# short for reliable monthly-horizon training. Yahoo Finance has no such
# cap and already covers stocks, so crypto now goes through the same
# source/pipeline as ingest_stocks.py.
# Keys are the friendly names used elsewhere (model.py, diagnose.py);
# values are the Yahoo Finance tickers used to fetch them.
COINS = {
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    "solana": "SOL-USD",
    "cardano": "ADA-USD",
    "dogecoin": "DOGE-USD",
}


def fetch_daily_prices(ticker, period="5y"):
    """Fetch daily closing prices for a Yahoo Finance ticker."""
    hist = yf.Ticker(ticker).history(period=period)
    return [(int(ts.timestamp() * 1000), float(close)) for ts, close in hist["Close"].items()]


def store_prices(prices, symbol, asset_type="crypto", db_path=DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            symbol TEXT NOT NULL,
            asset_type TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            price REAL NOT NULL,
            PRIMARY KEY (symbol, timestamp)
        )
    """)
    rows = [(symbol, asset_type, ts, price) for ts, price in prices]
    conn.executemany("INSERT OR IGNORE INTO prices VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    inserted = conn.total_changes
    conn.close()
    return inserted


def main():
    for symbol, ticker in COINS.items():
        try:
            prices = fetch_daily_prices(ticker)
            n = store_prices(prices, symbol, "crypto")
            print(f"[ingest] {symbol} ({ticker}): fetched {len(prices)} points, {n} new rows stored")
        except Exception as e:
            print(f"[ingest] {symbol} ({ticker}): failed ({e})")
        time.sleep(0.5)


if __name__ == "__main__":
    main()