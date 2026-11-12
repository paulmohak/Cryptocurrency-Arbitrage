#!/usr/bin/env python
"""One command that does everything: install, test, run the pipeline, write results.

    python scripts/run_all.py --data high-frequency-crypto-limit-order-book-data.zip
    python scripts/run_all.py --synthetic      # no download needed
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def venv_python():
    """Create a private .venv (once) so we never touch the user's global packages."""
    venv = ROOT / ".venv"
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not py.exists():
        print("Creating virtual environment in .venv ...", flush=True)
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    return str(py)


def run(cmd):
    print("\n>>", " ".join(str(c) for c in cmd), flush=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}  # works even if the install step is skipped
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def results_markdown(m, synthetic):
    s, b, c = m["strategy"], m["baseline_always_trade"], m["config"]
    note = ("_Synthetic data with a planted signal: this only shows the code works, not a real result._\n\n"
            if synthetic else "")
    return (
        f"{note}Setup: {c['labeled_rows']:,} labelled rows, ${c['trade_notional']:,.0f} order, "
        f"{c['horizon_steps']}-step hold, {c['fee_rate_per_side']:.2%} fee per side. "
        f"Test set: {m['test_rows']:,} rows, {m['test_base_rate']:.2%} profitable.\n\n"
        "| Metric | Value |\n|---|---|\n"
        f"| Test ROC AUC | {m['test_roc_auc']:.3f} |\n"
        f"| Test average precision | {m['test_average_precision']:.3f} |\n\n"
        "| Backtest (test set) | Trades | Mean net return | Hit rate | Total net return |\n|---|---|---|---|---|\n"
        f"| Model strategy | {s['n_trades']} | {s['mean_net_return']:.3%} | {s['hit_rate']:.1%} | {s['total_net_return']:.2%} |\n"
        f"| Always trade | {b['n_trades']} | {b['mean_net_return']:.3%} | {b['hit_rate']:.1%} | {b['total_net_return']:.2%} |\n"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", help="path to the Kaggle zip (or a csv)")
    p.add_argument("--member", default="BTC_1sec.csv")
    p.add_argument("--synthetic", action="store_true", help="use generated data instead of --data")
    p.add_argument("--nrows", type=int, help="only read the first N rows (quick test)")
    p.add_argument("--skip-install", action="store_true")
    p.add_argument("--no-venv", action="store_true", help="use the current Python instead of a fresh .venv")
    args = p.parse_args()
    if not args.synthetic and not args.data:
        p.error("give --data <path-to-zip> or use --synthetic")

    py = sys.executable if args.no_venv else venv_python()
    if not args.skip_install:
        run([py, "-m", "pip", "install", "-q", "-e", ".[dev]"])
    run([py, "-m", "pytest", "-q"])

    cmd = [py, "scripts/run_pipeline.py", "--out", "outputs"]
    cmd += ["--synthetic"] if args.synthetic else ["--data", args.data, "--member", args.member]
    if args.nrows:
        cmd += ["--nrows", str(args.nrows)]
    run(cmd)

    metrics = json.loads((ROOT / "outputs" / "metrics.json").read_text())
    md = results_markdown(metrics, args.synthetic)
    (ROOT / "outputs" / "RESULTS.md").write_text(md)

    readme = ROOT / "README.md"
    text = readme.read_text()
    if START in text and END in text and not args.synthetic:
        head, rest = text.split(START, 1)
        _, tail = rest.split(END, 1)
        readme.write_text(f"{head}{START}\n{md}{END}{tail}")
        print("\nREADME results section updated.")
    print("\n" + md + "\nDone. Files are in outputs/")


if __name__ == "__main__":
    main()
