# src/model.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from features import add_features, load_prices
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

FEATURE_COLS = ["price_vs_ma7", "price_vs_ma30", "daily_return", "volatility", "rsi"]
ASSETS = {
    "crypto": ["bitcoin", "ethereum", "solana", "cardano", "dogecoin"],
    "stock": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
}

# NOTE: validated against the next-day (horizon=1) target. Not guaranteed
# to still be right at the new horizon -- run tune_class_weight.py before
# trusting these.
ASSET_CLASS_WEIGHT = {
    "crypto": None,
    "stock": None,
}

ASSET_CLASS_HORIZON = {
    "crypto": 30,  # crypto trades every day of the week -> ~30 rows = ~1 month
    "stock": 21,   # stocks trade ~5 days/week -> ~21 trading days = ~1 month
}

N_SPLITS = 5            # walk-forward folds used for evaluation
MIN_ROWS_PER_FOLD = 10  # sanity floor: skip symbols too short to fold reliably


def build_dataset(symbol, horizon=1):
    df = add_features(load_prices(symbol))
    df["target"] = (df["price"].shift(-horizon) > df["price"]).astype(int)
    return df.iloc[:-horizon] if horizon > 0 else df


def fit_model(df, class_weight=None):
    """Fit a RandomForest on the entire given dataframe (no split).
    Used to produce the final 'deployable' model, e.g. for feature importances."""
    model = RandomForestClassifier(
        n_estimators=200, max_depth=5, random_state=42, class_weight=class_weight
    )
    model.fit(df[FEATURE_COLS], df["target"])
    return model


def fit_and_eval(train_df, test_df, class_weight=None):
    model = fit_model(train_df, class_weight=class_weight)
    preds = model.predict(test_df[FEATURE_COLS])
    acc = accuracy_score(test_df["target"], preds)
    # Baseline = accuracy of a naive predictor that always guesses whichever
    # class was more common in the TRAINING data -- what a real naive
    # predictor could actually achieve without seeing the future.
    train_majority_class = int(train_df["target"].mean() >= 0.5)
    baseline = (test_df["target"] == train_majority_class).mean()
    return model, preds, {"accuracy": acc, "baseline": baseline, "n_test": len(test_df)}


def weighted_average(metric_list, key):
    total_n = sum(m["n_test"] for m in metric_list)
    return sum(m[key] * m["n_test"] for m in metric_list) / total_n


def walk_forward_folds(n_rows, n_splits=N_SPLITS, horizon=1):
    """
    Expanding-window walk-forward folds, oldest-to-newest.

    Each fold's train set drops its final `horizon` rows before the test
    window starts. Those rows' targets are computed from prices that land
    inside (or after) the following test window, so keeping them would
    leak test-period information into training.
    """
    fold_size = n_rows // (n_splits + 1)
    if fold_size < 1:
        return
    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        test_end = fold_size * (i + 1) if i < n_splits else n_rows
        train_end_purged = max(train_end - horizon, 0)
        train_idx = list(range(0, train_end_purged))
        test_idx = list(range(train_end, test_end))
        if train_idx and test_idx:
            yield train_idx, test_idx


def walk_forward_evaluate(df, class_weight=None, horizon=1, n_splits=N_SPLITS):
    fold_metrics = []
    for train_idx, test_idx in walk_forward_folds(len(df), n_splits=n_splits, horizon=horizon):
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
        _, _, m = fit_and_eval(train_df, test_df, class_weight=class_weight)
        fold_metrics.append(m)
    if not fold_metrics:
        raise ValueError("not enough rows for any walk-forward fold")
    return {
        "accuracy": weighted_average(fold_metrics, "accuracy"),
        "baseline": weighted_average(fold_metrics, "baseline"),
        "n_test": sum(m["n_test"] for m in fold_metrics),
        "n_folds": len(fold_metrics),
    }


def evaluate_symbol(symbol, class_weight=None, horizon=1, n_splits=N_SPLITS):
    df = build_dataset(symbol, horizon=horizon)
    if len(df) < (n_splits + 1) * MIN_ROWS_PER_FOLD:
        raise ValueError(f"not enough rows ({len(df)}) for {n_splits}-fold walk-forward eval")
    return walk_forward_evaluate(df, class_weight=class_weight, horizon=horizon, n_splits=n_splits)


def evaluate_group(asset_type):
    class_weight = ASSET_CLASS_WEIGHT.get(asset_type)
    horizon = ASSET_CLASS_HORIZON.get(asset_type, 1)
    results = {}
    for symbol in ASSETS[asset_type]:
        try:
            results[symbol] = evaluate_symbol(symbol, class_weight=class_weight, horizon=horizon)
            m = results[symbol]
            print(f"  {symbol:10s} accuracy={m['accuracy']:.3f}  baseline={m['baseline']:.3f}  n_test={m['n_test']}  folds={m['n_folds']}")
        except Exception as e:
            print(f"  {symbol:10s} skipped ({e})")
    return list(results.values())


def main():
    all_metrics = []
    for asset_type in ASSETS:
        print(f"\n{asset_type.upper()}")
        metrics = evaluate_group(asset_type)
        all_metrics.extend(metrics)
        if metrics:
            acc = weighted_average(metrics, "accuracy")
            base = weighted_average(metrics, "baseline")
            n = sum(m["n_test"] for m in metrics)
            print(f"  -> {asset_type} combined: accuracy={acc:.3f}  baseline={base:.3f}  n_test={n}")

    print("\nOVERALL (crypto + stock combined)")
    if all_metrics:
        acc = weighted_average(all_metrics, "accuracy")
        base = weighted_average(all_metrics, "baseline")
        n = sum(m["n_test"] for m in all_metrics)
        print(f"  -> overall: accuracy={acc:.3f}  baseline={base:.3f}  n_test={n}")


if __name__ == "__main__":
    main()