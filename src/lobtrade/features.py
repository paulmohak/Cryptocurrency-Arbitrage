"""Feature engineering. Every feature at row t uses only rows <= t (no lookahead)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import N_LEVELS


def _sum_levels(book: pd.DataFrame, prefix: str, k: int) -> pd.Series:
    return book[[f"{prefix}_{i}" for i in range(k)]].sum(axis=1)


def _imbalance(a: pd.Series, b: pd.Series) -> pd.Series:
    """(a - b) / (a + b); exactly 0 (neutral) when both sides are empty, e.g. a quiet second."""
    total = (a + b).replace(0.0, np.nan)
    return ((a - b) / total).fillna(0.0)


def build_features(
    book: pd.DataFrame,
    depth_levels: tuple[int, ...] = (1, 5, N_LEVELS),
    return_windows: tuple[int, ...] = (1, 5, 30, 60),
    flow_windows: tuple[int, ...] = (5, 30),
) -> pd.DataFrame:
    """Build a causal feature matrix aligned with `book`'s index."""
    f = pd.DataFrame(index=book.index)
    mid = book["midpoint"].astype(float)
    logmid = np.log(mid)

    # --- price / volatility ---
    f["spread_rel"] = book["spread"] / mid
    for w in return_windows:
        f[f"ret_{w}s"] = logmid.diff(w)
    ret1 = logmid.diff()
    for w in (30, 60):
        f[f"vol_{w}s"] = ret1.rolling(w).std()

    # --- book shape ---
    for k in depth_levels:
        bid = _sum_levels(book, "bids_notional", k)
        ask = _sum_levels(book, "asks_notional", k)
        f[f"imbalance_{k}"] = _imbalance(bid, ask)
        f[f"log_depth_bid_{k}"] = np.log1p(bid)
        f[f"log_depth_ask_{k}"] = np.log1p(ask)
    f["bid_dist_0"] = book["bids_distance_0"]
    f["ask_dist_0"] = book["asks_distance_0"]
    f["bid_width_5"] = book["bids_distance_4"]
    f["ask_width_5"] = book["asks_distance_4"]

    # --- order flow at the current snapshot (top 5 levels) ---
    k = 5
    f["limit_imb_5"] = _imbalance(
        _sum_levels(book, "bids_limit_notional", k), _sum_levels(book, "asks_limit_notional", k)
    )
    f["cancel_imb_5"] = _imbalance(
        _sum_levels(book, "bids_cancel_notional", k), _sum_levels(book, "asks_cancel_notional", k)
    )
    f["market_imb_5"] = _imbalance(
        _sum_levels(book, "bids_market_notional", k), _sum_levels(book, "asks_market_notional", k)
    )

    # --- executed flow (buys / sells columns), rolling ---
    buys = book["buys"].astype(float)
    sells = book["sells"].astype(float)
    f["flow_imb_1"] = _imbalance(buys, sells)
    for w in flow_windows:
        b = buys.rolling(w, min_periods=1).sum()
        s = sells.rolling(w, min_periods=1).sum()
        f[f"flow_imb_{w}s"] = _imbalance(b, s)
        f[f"log_flow_{w}s"] = np.log1p(b + s)

    return f.replace([np.inf, -np.inf], np.nan)
