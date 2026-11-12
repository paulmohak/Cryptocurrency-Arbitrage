#!/usr/bin/env python
"""Poll Binance's public spot ticker and append prices to a CSV (timestamp, symbol, price).

Uses the unauthenticated /api/v3/ticker/price endpoint; no API key needed.

    python scripts/price_logger.py --interval 1 --out price_log_binance.csv
Stop with Ctrl+C.
"""
import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

URL = "https://api.binance.com/api/v3/ticker/price"
HEADERS = {"User-Agent": "lob-profitability price logger"}


def fetch(symbols):
    resp = requests.get(
        URL,
        params={"symbols": json.dumps(symbols, separators=(",", ":"))},
        headers=HEADERS,
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "ADAUSDT"])
    p.add_argument("--interval", type=float, default=1.0, help="seconds between polls")
    p.add_argument("--out", default="price_log_binance.csv")
    args = p.parse_args()

    out = Path(args.out)
    write_header = not out.exists() or out.stat().st_size == 0
    with out.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "symbol", "price"])
        while True:
            started = time.monotonic()
            try:
                ts = datetime.now(timezone.utc).isoformat()
                for row in fetch(args.symbols):
                    w.writerow([ts, row["symbol"], row["price"]])
                f.flush()  # make sure rows hit disk even if the script is killed
            except requests.RequestException as exc:
                print(f"request failed: {exc}")
            time.sleep(max(0.0, args.interval - (time.monotonic() - started)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
