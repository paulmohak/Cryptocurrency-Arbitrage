"""Synthetic order book generator for tests and smoke runs (not real market data)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FLOW_KINDS, N_LEVELS, SIDES, book_columns


def generate_synthetic_book(
    n: int = 6000, seed: int = 0, vol: float = 4e-4, signal: float = 6e-4
) -> pd.DataFrame:
    """Random-walk book where an order-book imbalance AR(1) weakly drives next-step returns."""
    rng = np.random.default_rng(seed)
    imb = np.zeros(n)
    for t in range(1, n):
        imb[t] = 0.97 * imb[t - 1] + 0.1 * rng.standard_normal()
    imb = np.clip(imb, -0.9, 0.9)

    ret = np.zeros(n)
    ret[1:] = signal * imb[:-1] + vol * rng.standard_normal(n - 1)
    mid = 50000.0 * np.exp(np.cumsum(ret))

    data: dict[str, np.ndarray] = {
        "system_time": pd.date_range("2021-04-07 11:32:42", periods=n, freq="s", tz="UTC"),
        "midpoint": mid,
        "spread": np.full(n, 0.01),
        "buys": rng.exponential(300, n) * (rng.random(n) < 0.3),
        "sells": rng.exponential(300, n) * (rng.random(n) < 0.3),
    }
    levels = np.arange(N_LEVELS)
    for side, sign, skew in (("bids", -1.0, 1.0), ("asks", 1.0, -1.0)):
        base_dist = 1e-7 + 2e-5 * levels**1.5
        for i in levels:
            data[f"{side}_distance_{i}"] = np.full(n, sign * base_dist[i])
            depth = rng.lognormal(7.5, 0.4, n) * (1.0 + skew * imb)
            data[f"{side}_notional_{i}"] = depth
            for kind in FLOW_KINDS:
                data[f"{side}_{kind}_notional_{i}"] = rng.exponential(50, n) * (rng.random(n) < 0.2)
    return pd.DataFrame(data)[book_columns()]
