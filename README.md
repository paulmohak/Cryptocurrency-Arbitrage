
# Predicting Profitable Short-Term Crypto Trades from Limit Order Book Data


**Question:** given a snapshot of the order book, can we tell whether a quick buy-then-sell round trip will make money *after trading fees*?

I built this to practice a full ML workflow on time-series data: realistic label construction, leakage-free features, time-based validation, and an honest backtest.

## Project at a glance

| | |
|---|---|
| **Data** | [High-Frequency Crypto Limit Order Book Data](https://www.kaggle.com/datasets/martinsn/high-frequency-crypto-limit-order-book-data) (BTC, ETH, ADA; 15 price levels per side; 1s / 1min / 5min snapshots) |
| **Task** | Binary classification: is the round trip profitable after fees? |
| **Model** | Gradient boosting (`HistGradientBoostingClassifier`) with class weights |
| **Validation** | Chronological 60 / 20 / 20 split with a gap between segments |
| **Tools** | Python, pandas, NumPy, scikit-learn, matplotlib, pytest |

## How it works

1. **Clean** (`data.py`): remove the stray index column, parse timestamps, sort, drop duplicates, drop the first ~16 rows where the feed is frozen.
2. **Label** (`labeling.py`): at each row, simulate a market **buy** by walking up the ask side of the book, then a market **sell** `horizon` rows later by walking down the bid side. Fill prices are volume-weighted, and a fee is charged on both trades.

   `net_return = (sell_price - buy_price) / buy_price - 2 * fee`

   A row is labelled profitable if `net_return > 0`. Rows are dropped if the order can't be fully filled from the 15 visible levels or if the data has a time gap.
3. **Features** (`features.py`): spread, past returns, rolling volatility, order book imbalance and depth (top 1 / 5 / 15 levels), and order flow (limit orders, cancellations, market orders, executed buys vs sells). Every feature only uses current and past rows.
4. **Model and evaluation** (`model.py`): train on the first 60% of time, choose a trade threshold on the next 20%, and report results on the final 20%. The backtest holds one position at a time and is compared with an "always trade" baseline.

## Results

### What I found on real data (BTC 1-second, exploration step)

Setup: 1,030,728 snapshots (7 to 19 April 2021), $1,000 order, 5-second hold, 0.1% fee per side.

| Metric | Value |
|---|---|
| Profitable round trips | **0.035%** |
| Mean net return | **-0.203%** (almost exactly the 0.2% fee cost) |
| Best single round trip | +0.61% |

At this speed, fees dominate. Profitable trades are extremely rare, so the model has to find rare large moves rather than small directional edges.

### Model results on real data

<!-- RESULTS:START -->
Setup: 1,030,610 labelled rows, $1,000 order, 5-step hold, 0.10% fee per side. Test set: 206,117 rows, 0.14% profitable.

| Metric | Value |
|---|---|
| Test ROC AUC | 0.877 |
| Test average precision | 0.013 |

| Backtest (test set) | Trades | Mean net return | Hit rate | Total net return |
|---|---|---|---|---|
| Model strategy | 8413 | -0.214% | 0.6% | -1801.48% |
| Always trade | 41224 | -0.206% | 0.1% | -8499.36% |
<!-- RESULTS:END -->

### Sanity check on synthetic data

`python scripts/run_pipeline.py --synthetic` runs the pipeline on generated data with a deliberately planted signal. The model beats the baseline there, which only confirms that the code works end to end. It says nothing about real markets.

## Quick start

```bash
git clone https://github.com/paulmohak/Cryptocurrency-Arbitrage.git

cd "Cryptocurrency project"

python scripts/run_all.py --data high-frequency-crypto-limit-order-book-data.zip
```

That one command installs the package, runs the tests, trains and evaluates the model, and writes the results into `outputs/` and this readme file. Use `--synthetic` instead of `--data` to try it without downloading anything.

To run the steps separately:

```bash
pip install -e ".[dev]"
pytest
python scripts/run_pipeline.py --data high-frequency-crypto-limit-order-book-data.zip --member BTC_1sec.csv
```

Outputs go to `outputs/`: `metrics.json`, `model.joblib`, `net_return_hist.png`.
Useful options: `--horizon` (rows to hold), `--fee-rate`, `--trade-notional`, `--nrows`.

For a step-by-step explanation with plots, open `notebooks/work.ipynb`.

## Design choices worth knowing about

- **Time-based split, not random.** Shuffling would let the model peek at the future.
- **Gap between train / validation / test.** Labels look `horizon` rows ahead, so neighbouring rows overlap; the gap stops that leaking across segments.
- **Causal features.** A test changes future rows and checks that past features don't change.
- **Realistic fills.** Orders walk the book, so a bigger order gets a worse average price.
- **Threshold chosen on validation only.** The test set is used once for the final numbers.

## Limitations

- Snapshots are 1 second apart; real trading has latency, partial fills and queue position, which this ignores.
- Only 15 levels of depth are visible, so very large orders can't be simulated.
- The real-data results come from one pair, one time window and one fee level.
- This is a learning project, not trading advice.

## Next steps

- Try lower fees (maker orders), longer horizons (`--horizon 60`), the 1min / 5min files, and ETH / ADA.
- Predict `net_return` directly (regression) instead of a yes/no label.
- Walk-forward validation over several time windows.
- Compare against a simple logistic regression baseline.

## Project structure

```
src/lobtrade/    data.py, labeling.py, features.py, model.py, pipeline.py, synthetic.py
scripts/         run_pipeline.py (command line), price_logger.py (optional Binance price logger)
tests/           labeling math, no-lookahead check, splits, end-to-end smoke test
notebooks/       work.ipynb
```


=======
# Cryptocurrency-Arbitrage
An arbitrage prediction model working to find the exploits in the market to make riskfree profit in the cryptocurrency market with the help of python
>>>>>>> 95176aa0b754f2b7d3c4819c5d20b0b734291e4e
