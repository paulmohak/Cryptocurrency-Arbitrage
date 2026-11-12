#!/usr/bin/env python
"""CLI for the full pipeline.

Examples
--------
python scripts/run_pipeline.py --data high-frequency-crypto-limit-order-book-data.zip
python scripts/run_pipeline.py --data data/BTC_1sec.csv --horizon 10 --fee-rate 0.0005
python scripts/run_pipeline.py --synthetic        # smoke test, no data needed
"""
import argparse
import json

from lobtrade.pipeline import run_pipeline
from lobtrade.synthetic import generate_synthetic_book


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", help="path to a .csv or the .zip archive")
    p.add_argument("--member", default="BTC_1sec.csv", help="file inside the zip (default BTC_1sec.csv)")
    p.add_argument("--out", default="outputs")
    p.add_argument("--trade-notional", type=float, default=1000.0, help="order size in quote currency")
    p.add_argument("--horizon", type=int, default=5, help="holding period in rows (seconds for 1sec data)")
    p.add_argument("--fee-rate", type=float, default=0.001, help="taker fee per side")
    p.add_argument("--nrows", type=int, default=None, help="only read the first N rows")
    p.add_argument("--synthetic", action="store_true", help="run on generated data instead of --data")
    args = p.parse_args()

    if args.synthetic:
        book = generate_synthetic_book()
        # Synthetic moves are small, so default to a low fee unless the user set one.
        fee = args.fee_rate if args.fee_rate != 0.001 else 0.0001
        metrics = run_pipeline(book=book, out_dir=args.out, trade_notional=args.trade_notional,
                               horizon=args.horizon, fee_rate=fee)
    elif args.data:
        metrics = run_pipeline(args.data, member=args.member, out_dir=args.out,
                               trade_notional=args.trade_notional, horizon=args.horizon,
                               fee_rate=args.fee_rate, nrows=args.nrows)
    else:
        p.error("provide --data or --synthetic")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
