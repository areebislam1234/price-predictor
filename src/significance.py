# src/significance.py
"""
Block-bootstrap significance test on the walk-forward (accuracy - baseline)
edge reported by model.py / diagnose.py.

Why this exists
----------------
model.py reports a single point estimate -- e.g. "combined accuracy 0.504 vs
baseline 0.518" -- with no sense of whether -1.4pp is a real effect or just
noise from evaluating on a few thousand rows across 10 symbols.

Naive iid bootstrapping of individual rows would understate that noise,
because rows within a single walk-forward fold are NOT independent: at
horizon=21 (stocks) or horizon=30 (crypto), each row's label is "is price
higher `horizon` days from now", so adjacent rows' labels are computed from
almost entirely the same future price path. A model that's right on Tuesday
is likely right on Wednesday for the same underlying reason -- these are
correlated trials, not independent coin flips. Treating them as independent
overstates the effective sample size and produces falsely tight confidence
intervals.

This module instead resamples in contiguous BLOCKS of length >= horizon, so
each block is treated as (approximately) one unit of independent evidence
rather than `horizon` separate ones. Folds themselves are never mixed across
each other or resampled together -- walk_forward_folds() already constructs
them as distinct, non-overlapping time windows, so they are pooled as-is.

Run: python src/significance.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np

from model import (
    ASSETS,
    ASSET_CLASS_WEIGHT,
    ASSET_CLASS_HORIZON,
    N_SPLITS,
    evaluate_symbol,
)

N_BOOT = 2000
SEED = 42


def collect_fold_metrics(asset_type, class_weight, horizon, n_splits=N_SPLITS):
    """Run walk-forward eval for every symbol in an asset class and pool the
    raw per-fold detail (not just the aggregated accuracy/baseline) for use
    in the block bootstrap below."""
    fold_metrics = []
    for symbol in ASSETS[asset_type]:
        try:
            result = evaluate_symbol(symbol, class_weight=class_weight, horizon=horizon, n_splits=n_splits)
        except ValueError as e:
            print(f"  {symbol:10s} skipped ({e})")
            continue
        fold_metrics.extend(result["fold_metrics"])
    return fold_metrics


def block_bootstrap_edge(fold_metrics_list, block_len, n_boot=N_BOOT, seed=SEED):
    """
    Paired block-bootstrap CI + p-value for (model accuracy - baseline
    accuracy), pooled across every fold passed in.

    Paired, because model and baseline are evaluated on the exact same test
    rows -- resampling them together (not as two separate accuracy numbers)
    cancels out shared row-level noise and gives a tighter, more honest
    interval on the actual edge.

    block_len should be >= the horizon used to build the targets, so each
    resampled block is no more fragmented than the label overlap that
    created the autocorrelation in the first place.
    """
    if not fold_metrics_list:
        raise ValueError("no fold metrics to bootstrap")

    rng = np.random.default_rng(seed)

    all_model = np.concatenate([fm["correct_model"] for fm in fold_metrics_list])
    all_baseline = np.concatenate([fm["correct_baseline"] for fm in fold_metrics_list])
    observed_edge = all_model.mean() - all_baseline.mean()

    boot_edges = np.empty(n_boot)
    for b in range(n_boot):
        resampled_model = []
        resampled_baseline = []
        for fm in fold_metrics_list:
            n = len(fm["correct_model"])
            if n == 0:
                continue
            bl = min(block_len, n)
            n_blocks = -(-n // bl)  # ceil division
            starts = rng.integers(0, n - bl + 1, size=n_blocks)
            idx = np.concatenate([np.arange(s, s + bl) for s in starts])[:n]
            resampled_model.append(fm["correct_model"][idx])
            resampled_baseline.append(fm["correct_baseline"][idx])
        boot_edges[b] = (
            np.concatenate(resampled_model).mean() - np.concatenate(resampled_baseline).mean()
        )

    ci_low, ci_high = np.percentile(boot_edges, [2.5, 97.5])
    # Two-sided p-value via the percentile method: how often the bootstrap
    # distribution crosses zero, doubled, capped at 1.
    p_value = min(2 * min((boot_edges <= 0).mean(), (boot_edges >= 0).mean()), 1.0)

    return {
        "n_folds": len(fold_metrics_list),
        "n_rows": len(all_model),
        "edge": observed_edge,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
    }


def _print_result(label, r):
    sig = "not distinguishable from 0" if r["ci_low"] <= 0 <= r["ci_high"] else "distinguishable from 0"
    print(
        f"  {label:9s} edge={r['edge']:+.3f}  95% CI=[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]  "
        f"p={r['p_value']:.3f}  n_rows={r['n_rows']}  n_folds={r['n_folds']}  -> {sig}"
    )


def main():
    print(f"Block-bootstrap significance test on (accuracy - baseline), n_boot={N_BOOT}\n")

    all_fold_metrics = []
    for asset_type in ASSETS:
        class_weight = ASSET_CLASS_WEIGHT.get(asset_type)
        horizon = ASSET_CLASS_HORIZON.get(asset_type, 1)
        print(f"{asset_type.upper()} (block_len={horizon}, from horizon)")
        fold_metrics = collect_fold_metrics(asset_type, class_weight, horizon)
        if not fold_metrics:
            print("  no folds available, skipping\n")
            continue
        all_fold_metrics.extend(fold_metrics)
        r = block_bootstrap_edge(fold_metrics, block_len=horizon)
        _print_result(asset_type, r)
        print()

    if all_fold_metrics:
        # Crypto and stocks use different horizons (30 vs 21). Pooling them
        # under one block length uses the larger of the two -- always safe,
        # since a bigger block only makes the resampling more conservative
        # (wider CI), never less.
        block_len = max(ASSET_CLASS_HORIZON.values())
        print(f"OVERALL (crypto + stock, block_len={block_len} = max horizon, conservative)")
        r = block_bootstrap_edge(all_fold_metrics, block_len=block_len)
        _print_result("overall", r)


if __name__ == "__main__":
    main()