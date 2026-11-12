"""Dataset assembly, chronological splitting, training, evaluation and a simple backtest."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from .features import build_features
from .labeling import DEFAULT_FEE_RATE, compute_labels


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series            # 1 if the round trip was profitable after fees
    net_return: pd.Series
    time: pd.Series


def make_dataset(
    book: pd.DataFrame,
    trade_notional: float = 1000.0,
    horizon: int = 5,
    fee_rate: float = DEFAULT_FEE_RATE,
) -> Dataset:
    """Features + labels, with rows lacking either removed (order preserved)."""
    X = build_features(book)
    labels = compute_labels(book, trade_notional, horizon, fee_rate)
    keep = X.notna().all(axis=1) & labels["net_return"].notna()
    return Dataset(
        X=X[keep].reset_index(drop=True),
        y=labels.loc[keep, "profitable"].astype(int).reset_index(drop=True),
        net_return=labels.loc[keep, "net_return"].reset_index(drop=True),
        time=book.loc[keep, "system_time"].reset_index(drop=True),
    )


def chronological_split(
    n: int, train_frac: float = 0.6, val_frac: float = 0.2, gap: int = 0
) -> tuple[slice, slice, slice]:
    """Train / validation / test slices in time order.

    `gap` rows are skipped between segments. Labels look `horizon` rows into the
    future, so use gap >= horizon to stop label overlap leaking across segments.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("need 0 < train_frac, val_frac and train_frac + val_frac < 1")
    a = int(n * train_frac)
    b = int(n * (train_frac + val_frac))
    return slice(0, a), slice(a + gap, b), slice(b + gap, n)


def train_model(X: pd.DataFrame, y: pd.Series, seed: int = 0) -> HistGradientBoostingClassifier:
    """Gradient boosting classifier with class weighting (profitable trades are rare)."""
    if y.nunique() < 2:
        raise ValueError(
            "Training labels contain a single class. With the default 0.1% fee per side a "
            "round trip needs a >0.2% move, which is very rare at 1s resolution; try a lower "
            "fee_rate, a longer horizon, or more data."
        )
    model = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.05, max_iter=200, class_weight="balanced", random_state=seed
    )
    return model.fit(X, y)


def backtest(
    scores: np.ndarray, net_return: np.ndarray, threshold: float, horizon: int
) -> dict:
    """Very simple backtest: trade whenever the model score is above `threshold`.

    Only one position is held at a time, so after entering a trade we skip the
    next `horizon` rows (otherwise we would double-count overlapping trades).
    """
    idx = np.flatnonzero(scores > threshold)
    taken, next_free = [], 0
    for i in idx:
        if i >= next_free:
            taken.append(i)
            next_free = i + horizon
    r = net_return[taken] if taken else np.array([])
    return {
        "n_trades": int(len(taken)),
        "mean_net_return": float(r.mean()) if len(r) else float("nan"),
        "hit_rate": float((r > 0).mean()) if len(r) else float("nan"),
        "total_net_return": float(r.sum()) if len(r) else 0.0,
    }


def pick_threshold(
    scores: np.ndarray, net_return: np.ndarray, horizon: int, min_trades: int = 20
) -> float:
    """Pick the score cut-off that gave the best average net return on the validation set.

    We only try the top 10% down to the top 0.1% of scores and require at least
    `min_trades` trades, so we don't pick a threshold that fires once by luck.
    The test set is never used here.
    """
    best_thr, best_val = float(np.quantile(scores, 0.99)), -np.inf
    for q in np.linspace(0.90, 0.999, 25):
        thr = float(np.quantile(scores, q))
        res = backtest(scores, net_return, thr, horizon)
        if res["n_trades"] >= min_trades and res["mean_net_return"] > best_val:
            best_thr, best_val = thr, res["mean_net_return"]
    return best_thr


def _safe_auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def _safe_ap(y, s):
    return float(average_precision_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def evaluate(
    model, ds: Dataset, horizon: int, train_frac: float = 0.6, val_frac: float = 0.2
) -> dict:
    """Score the already-trained model on the validation and test segments.

    The threshold is chosen on validation; the final numbers come from test.
    `baseline_always_trade` is the comparison: what happens if we just trade on every row.
    """
    tr, va, te = chronological_split(len(ds.y), train_frac, val_frac, gap=horizon)
    p_va = model.predict_proba(ds.X.iloc[va])[:, 1]
    p_te = model.predict_proba(ds.X.iloc[te])[:, 1]
    r_va = ds.net_return.iloc[va].to_numpy()
    r_te = ds.net_return.iloc[te].to_numpy()
    y_te = ds.y.iloc[te].to_numpy()

    thr = pick_threshold(p_va, r_va, horizon)
    return {
        "test_rows": int(len(y_te)),
        "test_base_rate": float(y_te.mean()),
        "test_roc_auc": _safe_auc(y_te, p_te),
        "test_average_precision": _safe_ap(y_te, p_te),
        "threshold": thr,
        "strategy": backtest(p_te, r_te, thr, horizon),
        "baseline_always_trade": backtest(np.ones(len(r_te)), r_te, 0.5, horizon),
    }
