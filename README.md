# Monthly Price Direction Predictor

A RandomForest classifier predicting whether a stock or cryptocurrency's price
will be higher ~1 month from now, evaluated with walk-forward cross-validation
across 5 stocks and 5 cryptocurrencies.

## What it does

```
ingest.py / ingest_stocks.py -> data/prices.db -> features.py -> model.py -> diagnose.py / significance.py
```

- **ingest.py / ingest_stocks.py** — pull daily price history (via yfinance)
for 5 cryptocurrencies and 5 stocks into a local SQLite database
- **features.py** — engineers RSI, rolling volatility, and price-relative
moving averages from raw price history
- **model.py** — builds the up/down target `horizon` days ahead, trains a
RandomForest, evaluates with walk-forward cross-validation
- **diagnose.py** — per-symbol breakdown: accuracy, baseline, feature
importances, most-recent-fold prediction behavior
- **significance.py** — block-bootstraps the accuracy-vs-baseline edge to
test whether it's statistically distinguishable from zero, accounting for
the autocorrelation that overlapping target windows introduce (see
[Is that edge statistically real?](#is-that-edge-statistically-real))
- **tune_class_weight.py** — one-off diagnostic comparing `class_weight=None` vs `"balanced"` per asset class

## Results

Walk-forward accuracy vs. a fair baseline (majority class of the *training* fold, not the test fold — see bug #3 below for why that distinction matters).

**Crypto** (horizon = 30 days, class_weight=None)

| Symbol       | Accuracy  | Baseline  | Edge       |
| ------------ | --------- | --------- | ---------- |
| bitcoin      | 0.481     | 0.424     | +5.7pp     |
| ethereum     | 0.467     | 0.512     | -4.5pp     |
| solana       | 0.461     | 0.501     | -4.0pp     |
| cardano      | 0.607     | 0.603     | +0.4pp     |
| dogecoin     | 0.554     | 0.549     | +0.5pp     |
| **Combined** | **0.514** | **0.518** | **-0.4pp** |

**Stocks** (horizon = 21 trading days, class_weight=None)

| Symbol       | Accuracy  | Baseline  | Edge       |
| ------------ | --------- | --------- | ---------- |
| AAPL         | 0.583     | 0.577     | +0.6pp     |
| MSFT         | 0.442     | 0.426     | +1.6pp     |
| GOOGL        | 0.490     | 0.556     | -6.6pp     |
| AMZN         | 0.477     | 0.519     | -4.2pp     |
| TSLA         | 0.461     | 0.507     | -4.6pp     |
| **Combined** | **0.490** | **0.517** | **-2.7pp** |

**Overall**: accuracy 0.504 vs baseline 0.518 (-1.4pp)

The honest takeaway: results are mixed and close to baseline either way. Four
of ten symbols (bitcoin, cardano, dogecoin, AAPL, MSFT) show a small positive
edge; the rest don't. Combined, the model sits within ~3 points of a fair
baseline in both directions — consistent with weak-form market efficiency:
price-derived technical features alone don't reliably forecast monthly-ahead
direction. `class_weight=None` was chosen because `tune_class_weight.py` showed it slightly outperforming `"balanced"` on average across crypto
symbols (0.514 vs 0.504) — though notably bitcoin specifically does better
with `"balanced"` (0.491), a reminder that a single class_weight setting per
asset class is a simplification, not a universally optimal choice per symbol.

### Is that edge statistically real?

Point estimates from 5 symbols and 5 walk-forward folds don't say whether an
edge is real or noise — and naive resampling would overstate confidence
anyway, since rows sharing an overlapping `horizon`-day target window aren't
independent trials. `src/significance.py` block-bootstraps (accuracy -
baseline), resampling in contiguous chunks sized to the horizon so the
autocorrelation from that overlap doesn't get treated as independent
evidence. It reports a 95% CI and two-sided p-value per asset class and
pooled overall:

| Group    | Edge   | 95% CI            | p-value |
| -------- | ------ | ------------------ | ------- |
| Crypto   | -0.1pp | [-4.3pp, +1.8pp]   | 0.422   |
| Stock    | -2.1pp | [-5.7pp, +0.2pp]   | 0.068   |
| Overall  | -0.9pp | [-4.2pp, +0.2pp]   | 0.089   |

None of the three intervals excludes zero. Stock comes closest (p=0.068)
without crossing the conventional 0.05 threshold. Five symbols per asset
class simply isn't enough independent evidence to distinguish a small
negative edge from no edge at all — which reinforces the "mixed and close to
baseline" reading above rather than contradicting it. Run
`python src/significance.py` to reproduce.

---

## Bugs found and fixed

This is the part worth reading if you're evaluating the project rather than
the stock picks.

### 1. Non-stationary features

`ma_7` / `ma_30` were stored as raw price levels. Over a 5-year window where
prices move several-fold, a split like "ma_30 > 180" learned early in training
becomes meaningless once price has drifted to a different range — the model
made confident, wrong-direction bets as a result (up to 75% of a symbol's
predictive weight sat on these two non-stationary features). Fixed by
switching to the price's *relative* distance from its moving averages
(`price / ma_30 - 1`) instead of the raw level.

### 2. Overlapping labels + a single train/test split

Once the prediction target moved from next-day to next-month, consecutive
rows' labels started overlapping by up to 29 days, and a single 80/20 split
meant the entire evaluation rested on one arbitrary few-month window. Fixed
with walk-forward (expanding-window) cross-validation across 5 folds, with
each fold's training data purged of its final `horizon` rows to prevent a
row's label from leaking information about the following test window.

Note: this fixed the leakage *between* train and test, but rows *within* a
single test fold still overlap each other — see
[Is that edge statistically real?](#is-that-edge-statistically-real) for how
`significance.py` accounts for that separately.

### 3. The baseline metric was leaking the future

The original "naive baseline" was computed as `max(test_target.mean(), 1 - test_target.mean())` — the best score a
predictor could get *if it already knew the test set's actual outcomes*.
At a monthly horizon during a mostly-bullish 5-year window, this made the
benchmark nearly impossible to beat and made a genuinely reasonable model
look like it was failing badly. Fixed by computing the baseline from the **training** fold's majority class instead — what a real naive predictor
could actually know at the time.

### 4. Crypto data source hit a hard cap

CoinGecko's free tier caps historical data at 365 days regardless of the `days` parameter requested — not enough history for reliable monthly-horizon
training, and also inconsistent granularity with the 5 years of daily stock
data. Switched crypto ingestion to yfinance (same source as stocks), which
also eliminated a recurring 429 rate-limit failure on the free CoinGecko tier.

---

## Setup

Requires Python 3.10+.

```
git clone https://github.com/areebislam1234/price-predictor
cd price-predictor
pip install -r requirements.txt

python src/ingest.py          # crypto prices (yfinance)
python src/ingest_stocks.py   # stock prices (yfinance)
python src/tune_class_weight.py   # optional: compare class_weight options
python src/model.py           # walk-forward evaluation summary
python src/diagnose.py        # per-symbol detail + feature importances
python src/significance.py    # is the accuracy-vs-baseline edge statistically real?
```

## Layout

```
├── src/
│   ├── ingest.py             crypto price ingestion (yfinance)
│   ├── ingest_stocks.py      stock price ingestion (yfinance)
│   ├── features.py           RSI, volatility, relative moving averages
│   ├── model.py               dataset building, training, walk-forward eval
│   ├── diagnose.py           per-symbol diagnostics
│   ├── significance.py       block-bootstrap significance test on the edge
│   └── tune_class_weight.py  class_weight comparison
└── data/
    └── prices.db             generated locally, gitignored
```

## Known limitations

- Each symbol trains its own model on a few thousand rows — likely
undertrained; pooling data across symbols within an asset class would
give the model more to learn from.
- Features are derived purely from each symbol's own price series — no
volume, no market-wide/sector context.
- The binary up/down target treats a +0.1% move identically to a +20% move.
- `class_weight` is set from a one-time comparison run rather than
re-validated automatically whenever the horizon changes.
- Only 5 symbols per asset class — as `significance.py` shows, that's not
enough independent evidence to distinguish a small edge from no edge at all
with real statistical confidence.

## License

MIT — see [LICENSE](https://github.com/areebislam1234/price-predictor/blob/main/LICENSE).
