"""lobtrade: profitability labeling and prediction on crypto limit order book snapshots."""

__version__ = "0.1.0"

from .data import load_book, clean_book, book_columns, N_LEVELS
from .labeling import vwap_fill, compute_labels
from .features import build_features
from .model import make_dataset, chronological_split, train_model, evaluate, backtest
from .pipeline import run_pipeline

__all__ = [
    "load_book",
    "clean_book",
    "book_columns",
    "N_LEVELS",
    "vwap_fill",
    "compute_labels",
    "build_features",
    "make_dataset",
    "chronological_split",
    "train_model",
    "evaluate",
    "backtest",
    "run_pipeline",
]
