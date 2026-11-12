"""Profitability labels from simulated market-order round trips.

Round trip at row t:  market BUY now (walk the asks), market SELL `horizon`
rows later (walk the bids), minus a taker fee on each leg.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import N_LEVELS

DEFAULT_FEE_RATE = 0.001  # 0.1% per side; set to your exchange's real taker fee


def vwap_fill(
    book: pd.DataFrame,
    side: str,
    trade_notional: float,
    n_levels: int = N_LEVELS,
) -> tuple[np.ndarray, np.ndarray]:
    """Walk the book for a market order of `trade_notional` (quote currency).

    side='buy' consumes asks, side='sell' consumes bids.

    Returns
    -------
    avg_price : volume-weighted average fill price per row (NaN if nothing fills)
    filled    : notional actually filled per row (<= trade_notional)
    """
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")
    prefix = "asks" if side == "buy" else "bids"

    notional = book[[f"{prefix}_notional_{i}" for i in range(n_levels)]].to_numpy(float)
    dist = book[[f"{prefix}_distance_{i}" for i in range(n_levels)]].to_numpy(float)
    mid = book["midpoint"].to_numpy(float)[:, None]

    # The data stores each level as a relative distance from the midpoint,
    # so the actual price at that level is mid * (1 + distance).
    prices = mid * (1.0 + dist)
    notional = np.clip(np.nan_to_num(notional), 0.0, None)

    # Walk the book level by level:
    #   consumed_before = money already used up by the shallower levels
    #   take            = how much we can still grab at this level
    consumed_before = np.cumsum(notional, axis=1) - notional
    take = np.clip(np.minimum(notional, trade_notional - consumed_before), 0.0, None)

    filled = take.sum(axis=1)
    # VWAP = total money spent / total quantity bought (quantity = notional / price).
    with np.errstate(divide="ignore", invalid="ignore"):
        quantity = (take / prices).sum(axis=1)
        avg_price = np.where(filled > 0, filled / quantity, np.nan)
    return avg_price, filled


def compute_labels(
    book: pd.DataFrame,
    trade_notional: float = 1000.0,
    horizon: int = 5,
    fee_rate: float = DEFAULT_FEE_RATE,
    min_fill_ratio: float = 0.999,
    max_gap_factor: float = 1.5,
) -> pd.DataFrame:
    """Compute the net return of buying at row t and selling at row t + horizon.

    Rows are labeled NaN when
      * the horizon runs past the end of the data,
      * either leg cannot be (almost) fully filled from the visible 15 levels,
      * the timestamp gap to t + horizon is more than `max_gap_factor` times the
        median row spacing * horizon (i.e. the feed had a hole).

    Returns a frame with `buy_price`, `sell_price`, `gross_return`,
    `net_return`, and `profitable` (nullable boolean).
    """
    buy_price, buy_filled = vwap_fill(book, "buy", trade_notional)
    sell_price, sell_filled = vwap_fill(book, "sell", trade_notional)

    buy_price = pd.Series(buy_price, index=book.index)
    buy_ok = pd.Series(buy_filled >= trade_notional * min_fill_ratio, index=book.index)
    # shift(-horizon) pulls the row `horizon` steps in the future back to row t,
    # so each row sees "what would I get if I sold `horizon` seconds from now?"
    sell_price_future = pd.Series(sell_price, index=book.index).shift(-horizon)
    sell_ok_future = pd.Series(sell_filled >= trade_notional * min_fill_ratio, index=book.index).shift(-horizon)

    gross = (sell_price_future - buy_price) / buy_price
    net = gross - 2.0 * fee_rate  # one fee when buying, one when selling

    valid = buy_ok & sell_ok_future.fillna(False).astype(bool)

    # Safety check: if the data has a hole (missing seconds), the "5 rows later"
    # row is not really 5 seconds later, so we throw those labels away.
    if "system_time" in book.columns:
        t = pd.to_datetime(book["system_time"], utc=True)
        step = t.diff().median()
        if pd.notna(step) and step > pd.Timedelta(0):
            elapsed = t.shift(-horizon) - t
            valid &= elapsed <= step * horizon * max_gap_factor

    out = pd.DataFrame(
        {
            "buy_price": buy_price,
            "sell_price": sell_price_future,
            "gross_return": gross.where(valid),
            "net_return": net.where(valid),
        },
        index=book.index,
    )
    out["profitable"] = (out["net_return"] > 0).astype("boolean").where(out["net_return"].notna())
    return out
