"""End-to-end pipeline: load -> clean -> label -> features -> train -> evaluate -> save."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from .data import clean_book, load_book
from .labeling import DEFAULT_FEE_RATE
from .model import chronological_split, evaluate, make_dataset, train_model


def run_pipeline(
    data_path: Optional[str] = None,
    member: str = "BTC_1sec.csv",
    out_dir: str = "outputs",
    trade_notional: float = 1000.0,
    horizon: int = 5,
    fee_rate: float = DEFAULT_FEE_RATE,
    nrows: Optional[int] = None,
    book: Optional[pd.DataFrame] = None,
    make_plots: bool = True,
) -> dict:
    """Run the full workflow and write metrics.json, model.joblib (and a plot) to `out_dir`.

    Pass either `data_path` (csv/zip) or an already-loaded `book` frame.
    """
    if book is None:
        if data_path is None:
            raise ValueError("provide data_path or book")
        book = load_book(data_path, member=member, nrows=nrows)
    book = clean_book(book)

    ds = make_dataset(book, trade_notional, horizon, fee_rate)
    tr, _, _ = chronological_split(len(ds.y), gap=horizon)
    model = train_model(ds.X.iloc[tr], ds.y.iloc[tr])
    metrics = evaluate(model, ds, horizon)

    metrics["config"] = {
        "trade_notional": trade_notional,
        "horizon_steps": horizon,
        "fee_rate_per_side": fee_rate,
        "rows_after_cleaning": int(len(book)),
        "labeled_rows": int(len(ds.y)),
        "overall_profitable_rate": float(ds.y.mean()),
        "net_return_mean": float(ds.net_return.mean()),
        "net_return_max": float(ds.net_return.max()),
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump(model, out / "model.joblib")

    if make_plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(ds.net_return, bins=120)
        ax.axvline(0, color="red", lw=1)
        ax.set_xlabel(f"net return after fees ({horizon}-step round trip)")
        ax.set_ylabel("count")
        ax.set_title("Distribution of round-trip net returns")
        fig.tight_layout()
        fig.savefig(out / "net_return_hist.png", dpi=130)
        plt.close(fig)
    return metrics
