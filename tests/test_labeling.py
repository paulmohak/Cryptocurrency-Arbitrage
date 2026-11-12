import numpy as np
import pandas as pd
import pytest

from lobtrade.data import book_columns
from lobtrade.labeling import compute_labels, vwap_fill


def tiny_book(n=3, mids=None):
    """Book with 100 notional at every level; ask prices mid*(1+0.01*(i+1)), bids mirrored."""
    cols = book_columns()
    df = pd.DataFrame(0.0, index=range(n), columns=cols)
    df["system_time"] = pd.date_range("2021-01-01", periods=n, freq="s", tz="UTC")
    df["midpoint"] = mids if mids is not None else 100.0
    for i in range(15):
        df[f"asks_distance_{i}"] = 0.01 * (i + 1)
        df[f"bids_distance_{i}"] = -0.01 * (i + 1)
        df[f"asks_notional_{i}"] = 100.0
        df[f"bids_notional_{i}"] = 100.0
    return df


def test_fill_within_first_level():
    avg, filled = vwap_fill(tiny_book(), "buy", 50.0)
    assert filled[0] == pytest.approx(50.0)
    assert avg[0] == pytest.approx(101.0)


def test_fill_walks_multiple_levels_with_true_vwap():
    avg, filled = vwap_fill(tiny_book(), "buy", 150.0)
    # 100 notional @101, 50 notional @102 -> VWAP = 150 / (100/101 + 50/102)
    expected = 150.0 / (100 / 101.0 + 50 / 102.0)
    assert filled[0] == pytest.approx(150.0)
    assert avg[0] == pytest.approx(expected)


def test_sell_uses_bids():
    avg, _ = vwap_fill(tiny_book(), "sell", 50.0)
    assert avg[0] == pytest.approx(99.0)


def test_partial_fill_and_empty_book():
    book = tiny_book()
    book.loc[0, [f"asks_notional_{i}" for i in range(15)]] = 0.0
    avg, filled = vwap_fill(book, "buy", 50.0)
    assert filled[0] == 0.0 and np.isnan(avg[0])
    _, filled_big = vwap_fill(tiny_book(), "buy", 10_000.0)
    assert filled_big[0] == pytest.approx(1500.0)  # only 15 * 100 visible


def test_label_roundtrip_math():
    book = tiny_book(n=4, mids=[100.0, 100.0, 110.0, 110.0])
    lab = compute_labels(book, trade_notional=50.0, horizon=2, fee_rate=0.001)
    buy = 101.0                 # ask level 0 at t=0
    sell = 110.0 * 0.99         # bid level 0 at t=2
    expected = (sell - buy) / buy - 0.002
    assert lab["net_return"].iloc[0] == pytest.approx(expected)
    assert bool(lab["profitable"].iloc[0]) is True
    # last `horizon` rows have no future -> NaN
    assert lab["net_return"].iloc[2:].isna().all()
    assert lab["profitable"].iloc[2:].isna().all()


def test_unfillable_rows_are_nan():
    book = tiny_book(n=4)
    book.loc[2, [f"bids_notional_{i}" for i in range(15)]] = 0.0
    lab = compute_labels(book, trade_notional=50.0, horizon=2)
    assert np.isnan(lab["net_return"].iloc[0])  # exit leg at t=2 cannot fill


def test_timestamp_gap_invalidates_label():
    book = tiny_book(n=6)
    book.loc[3:, "system_time"] += pd.Timedelta(minutes=10)  # feed hole
    lab = compute_labels(book, trade_notional=50.0, horizon=2)
    assert np.isnan(lab["net_return"].iloc[2])  # t=2 -> t=4 spans the hole
    assert not np.isnan(lab["net_return"].iloc[0])
