"""Loading and cleaning of the high-frequency crypto limit order book dataset.

Expected schema (one row per snapshot, 15 price levels per side):
    system_time, midpoint, spread, buys, sells,
    {bids,asks}_distance_{0..14}          relative distance from the midpoint
    {bids,asks}_notional_{0..14}          resting notional at that level
    {bids,asks}_{cancel,limit,market}_notional_{0..14}   order flow at that level
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Optional, Union

import pandas as pd

N_LEVELS = 15
SIDES = ("bids", "asks")
FLOW_KINDS = ("cancel", "limit", "market")


def book_columns(n_levels: int = N_LEVELS) -> list[str]:
    """Column names of the dataset, in file order (without the 'Unnamed: 0' index column)."""
    cols = ["system_time", "midpoint", "spread", "buys", "sells"]
    for side in SIDES:
        cols += [f"{side}_distance_{i}" for i in range(n_levels)]
        cols += [f"{side}_notional_{i}" for i in range(n_levels)]
        for kind in FLOW_KINDS:
            cols += [f"{side}_{kind}_notional_{i}" for i in range(n_levels)]
    return cols


def load_book(
    path: Union[str, Path],
    member: Optional[str] = "BTC_1sec.csv",
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Read a raw order book CSV, either directly or from inside a zip archive.

    Parameters
    ----------
    path : a .csv file, or a .zip archive containing `member`.
    member : file inside the archive (ignored for plain CSVs).
    nrows : optionally read only the first N rows (handy for quick experiments).
    """
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            if member not in z.namelist():
                raise FileNotFoundError(f"{member!r} not in archive; available: {z.namelist()}")
            with z.open(member) as f:
                return pd.read_csv(f, nrows=nrows)
    return pd.read_csv(path, nrows=nrows)


def drop_stale_prefix(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the leading rows where the midpoint is frozen at its initial value.

    The BTC 1-second file starts with ~16 identical snapshots before the feed
    actually starts updating; those rows carry no information and would create
    artificial constant labels.
    """
    changed = df["midpoint"].ne(df["midpoint"].iloc[0])
    if not changed.any():
        return df
    first_move = changed.to_numpy().argmax()
    return df.iloc[first_move:]


def clean_book(df: pd.DataFrame, drop_stale: bool = True) -> pd.DataFrame:
    """Standardise a raw frame: drop the index column, parse time, sort, dedupe."""
    df = df.copy()
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns="Unnamed: 0")
    df["system_time"] = pd.to_datetime(df["system_time"], utc=True)
    df = df.sort_values("system_time").drop_duplicates(subset="system_time")
    if drop_stale:
        df = drop_stale_prefix(df)
    return df.reset_index(drop=True)
