import io
import pandas as pd
import requests

URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"

def to_yahoo_ticker(t: str) -> str:
    return str(t).strip().replace(".", "-")

def get_nasdaq_listed(limit=None):
    r = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()

    # nasdaqlisted.txt = pipe-separated + в конце футер "File Creation Time"
    df = pd.read_csv(io.StringIO(r.text), sep="|")
    df = df[df["Symbol"].notna()]
    df = df[~df["Symbol"].astype(str).str.contains("File Creation Time", na=False)]

    # фильтры: не ETF, не test issue
    # В этом файле обычно есть колонки ETF и Test Issue
    if "ETF" in df.columns:
        df = df[df["ETF"] == "N"]
    if "Test Issue" in df.columns:
        df = df[df["Test Issue"] == "N"]

    tickers = df["Symbol"].astype(str).map(to_yahoo_ticker).unique().tolist()
    if limit:
        tickers = tickers[:limit]

    return tickers

if __name__ == "__main__":
    tickers = get_nasdaq_listed()
    print("Tickers:", len(tickers))
    print(tickers[:30])

    with open("universe.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(tickers))
    print("Saved -> universe.txt")
