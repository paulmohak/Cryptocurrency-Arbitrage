import json

import numpy as np
import pytest

from lobtrade.data import clean_book
from lobtrade.features import build_features
from lobtrade.model import backtest, chronological_split, make_dataset
from lobtrade.pipeline import run_pipeline
from lobtrade.synthetic import generate_synthetic_book


@pytest.fixture(scope="module")
def book():
    return clean_book(generate_synthetic_book(n=3000, seed=1))


def test_features_are_causal(book):
    """Changing future rows must not change features computed at earlier rows."""
    cut = 2000
    base = build_features(book)
    altered = book.copy()
    cols = [c for c in altered.columns if c != "system_time"]
    altered.loc[cut:, cols] = altered.loc[cut:, cols] * 1.7 + 3.0
    new = build_features(altered)
    np.testing.assert_allclose(base.iloc[:cut].to_numpy(), new.iloc[:cut].to_numpy(), equal_nan=True)


def test_no_inf_in_features(book):
    assert not np.isinf(build_features(book).to_numpy()).any()


def test_chronological_split_order_and_gap():
    tr, va, te = chronological_split(1000, 0.6, 0.2, gap=5)
    assert tr.stop + 5 == va.start
    assert va.stop + 5 == te.start
    assert te.stop == 1000


def test_chronological_split_validates():
    with pytest.raises(ValueError):
        chronological_split(100, 0.9, 0.2)


def test_backtest_non_overlapping():
    scores = np.ones(10)
    r = np.full(10, 0.01)
    res = backtest(scores, r, threshold=0.5, horizon=5)
    assert res["n_trades"] == 2  # rows 0 and 5
    assert res["total_net_return"] == pytest.approx(0.02)


def test_make_dataset_alignment(book):
    ds = make_dataset(book, trade_notional=1000.0, horizon=5, fee_rate=0.0001)
    assert len(ds.X) == len(ds.y) == len(ds.net_return) == len(ds.time)
    assert not ds.X.isna().any().any()
    assert ds.time.is_monotonic_increasing


def test_pipeline_smoke(tmp_path):
    book = generate_synthetic_book(n=6000, seed=0)
    metrics = run_pipeline(book=book, out_dir=str(tmp_path), fee_rate=0.0001, horizon=5)
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "model.joblib").exists()
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["test_rows"] == metrics["test_rows"] > 0
