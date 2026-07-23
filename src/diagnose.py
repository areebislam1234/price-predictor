# src/diagnose.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from model import (
    ASSET_CLASS_WEIGHT,
    ASSET_CLASS_HORIZON,
    ASSETS,
    FEATURE_COLS,
    MIN_ROWS_PER_FOLD,
    N_SPLITS,
    build_dataset,
    fit_and_eval,
    fit_model,
    walk_forward_evaluate,
    walk_forward_folds,
)


def diagnose_symbol(symbol, class_weight=None, horizon=1, n_splits=N_SPLITS):
    df = build_dataset(symbol, horizon=horizon)
    if len(df) < (n_splits + 1) * MIN_ROWS_PER_FOLD:
        print(f"  {symbol}: skipped, not enough data for {n_splits}-fold walk-forward eval")
        return None

    # Robust accuracy/baseline, averaged across walk-forward folds
    wf_metrics = walk_forward_evaluate(df, class_weight=class_weight, horizon=horizon, n_splits=n_splits)

    # A concrete look at the most recent out-of-sample fold
    folds = list(walk_forward_folds(len(df), n_splits=n_splits, horizon=horizon))
    last_train_idx, last_test_idx = folds[-1]
    last_train_df, last_test_df = df.iloc[last_train_idx], df.iloc[last_test_idx]
    _, preds, _ = fit_and_eval(last_train_df, last_test_df, class_weight=class_weight)
    pred_up_rate = preds.mean()
    actual_up_rate = last_test_df["target"].mean()

    # Final "deployable" model, fit on all available history
    final_model = fit_model(df, class_weight=class_weight)
    importances = dict(zip(FEATURE_COLS, final_model.feature_importances_))

    print(f"\n{symbol}")
    print(f"  accuracy={wf_metrics['accuracy']:.3f}  baseline={wf_metrics['baseline']:.3f}"
          f"  ({wf_metrics['n_folds']} walk-forward folds, n_test={wf_metrics['n_test']})")
    print(f"  most recent fold -- actual 'up' rate: {actual_up_rate:.3f}")
    print(f"  most recent fold -- pred 'up' rate:   {pred_up_rate:.3f}")
    print("  feature importances (model fit on full history):")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"    {feat:15s} {imp:.3f}")

    return importances


def main():
    for asset_type, symbols in ASSETS.items():
        class_weight = ASSET_CLASS_WEIGHT.get(asset_type)
        horizon = ASSET_CLASS_HORIZON.get(asset_type, 1)
        print(f"\n=== {asset_type.upper()} (class_weight={class_weight}, horizon={horizon}) ===")
        for symbol in symbols:
            diagnose_symbol(symbol, class_weight=class_weight, horizon=horizon)


if __name__ == "__main__":
    main()