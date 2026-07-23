# src/tune_class_weight.py
"""
One-off diagnostic: compares class_weight=None vs class_weight="balanced"
per asset class using walk-forward evaluation.

ASSET_CLASS_WEIGHT in model.py was validated against the next-day
(horizon=1) target. Whenever the horizon changes, re-run this and update
ASSET_CLASS_WEIGHT by hand based on the result -- don't assume the old
setting still holds.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from model import (
    ASSETS,
    ASSET_CLASS_HORIZON,
    MIN_ROWS_PER_FOLD,
    N_SPLITS,
    build_dataset,
    walk_forward_evaluate,
    weighted_average,
)


def compare_class_weight(asset_type, horizon, n_splits=N_SPLITS):
    print(f"\n{asset_type.upper()} (horizon={horizon})")
    per_option_metrics = {None: [], "balanced": []}

    for symbol in ASSETS[asset_type]:
        df = build_dataset(symbol, horizon=horizon)
        if len(df) < (n_splits + 1) * MIN_ROWS_PER_FOLD:
            print(f"  {symbol:10s} skipped (not enough rows)")
            continue

        line = f"  {symbol:10s}"
        for cw in (None, "balanced"):
            m = walk_forward_evaluate(df, class_weight=cw, horizon=horizon, n_splits=n_splits)
            per_option_metrics[cw].append(m)
            line += f"   class_weight={str(cw):9s} acc={m['accuracy']:.3f}"
        print(line)

    print("  --------")
    for cw in (None, "balanced"):
        metrics = per_option_metrics[cw]
        if metrics:
            acc = weighted_average(metrics, "accuracy")
            base = weighted_average(metrics, "baseline")
            print(f"  {asset_type} avg  class_weight={str(cw):9s} accuracy={acc:.3f}  baseline={base:.3f}")


def main():
    for asset_type in ASSETS:
        horizon = ASSET_CLASS_HORIZON.get(asset_type, 1)
        compare_class_weight(asset_type, horizon)


if __name__ == "__main__":
    main()