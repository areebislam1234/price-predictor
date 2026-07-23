# src/ingest_stocks.py
import sqlite3
from pathlib import Path

import yfinance as yf

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "prices.db"
STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]


def fetch_daily_prices(ticker, period="5y"):
    """Fetch daily closing prices for a stock ticker."""
    hist = yf.Ticker(ticker).history(period=period)
    return [(int(ts.timestamp() * 1000), float(close)) for ts, close in hist["Close"].items()]


def store_prices(prices, symbol, asset_type="stock", db_path=DB_PATH):
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
    for ticker in STOCKS:
        try:
            prices = fetch_daily_prices(ticker)
            n = store_prices(prices, ticker, "stock")
            print(f"[ingest_stocks] {ticker}: fetched {len(prices)} points, {n} new rows stored")
        except Exception as e:
            print(f"[ingest_stocks] {ticker}: failed ({e})")


if __name__ == "__main__":
    main()