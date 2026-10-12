"""
Paired difference summary + bootstrap confidence interval, in one place.

Two entry points, because the two kinds of "paired difference" in this
project have different dependence structure:

- paired_seed_summary(rmse_a, rmse_b, ...)
    For per-seed test RMSE (one value per seed, no temporal order between
    seeds). Reports mean +/- std of the paired difference, a paired t-test,
    and a plain (iid) bootstrap CI on the mean difference.

- paired_timeseries_summary(e_a, e_b, ...)
    For a date-level paired error series (e.g. e_quantum,t vs e_classical,t
    across T chronological test dates). Computes the loss differential
    d_t = L(e_a,t) - L(e_b,t), reports its mean +/- std, and a CIRCULAR
    MOVING-BLOCK bootstrap CI that preserves serial dependence (matches the
    paper's Table 2 methodology: block length 10, 4000 reps).
"""
from typing import Optional

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------
# Seed-level (independent observations)
# ---------------------------------------------------------------------

def paired_seed_summary(
    rmse_a,
    rmse_b,
    name_a: str = "Model A",
    name_b: str = "Model B",
    n_boot: int = 4000,
    ci: float = 0.95,
    random_state: Optional[int] = None,
):
    """
    rmse_a, rmse_b : per-seed test RMSE, one entry per seed, matching seed
                     order (rmse_a[i] and rmse_b[i] must be the same seed).
    """
    rmse_a = np.asarray(rmse_a, dtype=float)
    rmse_b = np.asarray(rmse_b, dtype=float)
    if len(rmse_a) != len(rmse_b):
        raise ValueError("rmse_a and rmse_b must have the same number of seeds.")

    diff = rmse_a - rmse_b  # negative => model A better on that seed
    n = len(diff)
    mean_diff = diff.mean()
    std_diff = diff.std(ddof=1)
    se_diff = std_diff / np.sqrt(n)

    t_stat = mean_diff / se_diff if se_diff > 0 else np.nan
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - 1)) if se_diff > 0 else np.nan

    # plain iid bootstrap on the mean difference
    rng = np.random.default_rng(random_state)
    boot_means = np.array([rng.choice(diff, size=n, replace=True).mean() for _ in range(n_boot)])
    alpha = 1 - ci
    ci_low, ci_high = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    print("\n" + "=" * 65)
    print(f"PAIRED SEED-LEVEL RMSE: {name_a} vs {name_b}  (n={n} seeds)")
    print("=" * 65)
    print(f"{name_a} RMSE: {rmse_a.mean():.4f} +/- {rmse_a.std(ddof=1):.4f}")
    print(f"{name_b} RMSE: {rmse_b.mean():.4f} +/- {rmse_b.std(ddof=1):.4f}")
    print(f"Paired diff ({name_a} - {name_b}): {mean_diff:.4f} +/- {std_diff:.4f}")
    print(f"Paired t-stat: {t_stat:.4f}   p-value: {p_value:.4f}")
    print(f"{int(ci*100)}% iid bootstrap CI on mean diff: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"(negative diff favors {name_a})")

    return {
        "diff": diff,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": ci,
    }


# ---------------------------------------------------------------------
# Date-level (autocorrelated time series)
# ---------------------------------------------------------------------

def paired_timeseries_summary(
    e_a,
    e_b,
    name_a: str = "Model A",
    name_b: str = "Model B",
    loss: str = "squared",
    block_length: int = 10,
    n_boot: int = 4000,
    ci: float = 0.95,
    random_state: Optional[int] = None,
):
    """
    e_a, e_b : forecast error arrays (y_true - y_pred), same length, in
               chronological order across the test dates.
    loss     : 'squared' for L(e) = e^2, 'absolute' for L(e) = |e|
    """
    e_a = np.asarray(e_a, dtype=float)
    e_b = np.asarray(e_b, dtype=float)
    if len(e_a) != len(e_b):
        raise ValueError("e_a and e_b must have the same length.")

    if loss == "squared":
        d = e_a ** 2 - e_b ** 2
    elif loss == "absolute":
        d = np.abs(e_a) - np.abs(e_b)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")

    T = len(d)
    mean_diff = d.mean()
    std_diff = d.std(ddof=1)

    # circular moving-block bootstrap
    rng = np.random.default_rng(random_state)
    n_blocks_needed = int(np.ceil(T / block_length))
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, T, size=n_blocks_needed)
        resampled = np.concatenate([d[np.arange(s, s + block_length) % T] for s in starts])[:T]
        boot_means[b] = resampled.mean()

    alpha = 1 - ci
    ci_low, ci_high = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    print("\n" + "=" * 65)
    print(f"PAIRED DATE-LEVEL LOSS DIFFERENTIAL: {name_a} vs {name_b}  "
          f"(T={T}, loss={loss})")
    print("=" * 65)
    print(f"Mean d_t +/- std: {mean_diff:.4f} +/- {std_diff:.4f}")
    print(f"{int(ci*100)}% circular block-bootstrap CI (block={block_length}, "
          f"n_boot={n_boot}): [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"(negative values favor {name_a})")

    return {
        "d": d,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": ci,
        "block_length": block_length,
        "n_boot": n_boot,
    }


# ---------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------

def _load_test_split(csv_path: str, split_col: str = "split", split_value: str = "test"):
    import pandas as pd
    df = pd.read_csv(csv_path, index_col=0)
    return df[df[split_col] == split_value].copy()


def run_seed_from_csv(
    csv_path_a: str,
    csv_path_b: str,
    name_a: str = "Model A",
    name_b: str = "Model B",
    true_col: str = "true_vol_proxy",
    pred_col: str = "predicted_vol",
    seed_col: str = "seed",
    split_col: str = "split",
    **kwargs,
):
    """Load two per-seed prediction CSVs, compute per-seed test RMSE for
    each, and run paired_seed_summary on the result.

    Expects each CSV to have all seeds stacked in one file, with columns
    for seed, split, true value, and predicted value (adjust the *_col
    arguments if your column names differ)."""
    test_a = _load_test_split(csv_path_a, split_col)
    test_b = _load_test_split(csv_path_b, split_col)

    rmse_a = test_a.groupby(seed_col).apply(
        lambda g: np.sqrt(np.mean((g[true_col] - g[pred_col]) ** 2))
    ).sort_index()
    rmse_b = test_b.groupby(seed_col).apply(
        lambda g: np.sqrt(np.mean((g[true_col] - g[pred_col]) ** 2))
    ).sort_index()

    if not rmse_a.index.equals(rmse_b.index):
        raise ValueError(f"Seed mismatch between files: {list(rmse_a.index)} vs {list(rmse_b.index)}")

    return paired_seed_summary(rmse_a.values, rmse_b.values, name_a=name_a, name_b=name_b, **kwargs)


def run_timeseries_from_csv(
    csv_path_a: str,
    csv_path_b: str,
    name_a: str = "Model A",
    name_b: str = "Model B",
    true_col: str = "true_vol_proxy",
    pred_col: str = "predicted_vol",
    seed_col: str = "seed",
    split_col: str = "split",
    ensemble_across_seeds: bool = True,
    **kwargs,
):
    """Load two per-seed prediction CSVs and run paired_timeseries_summary
    on the date-level errors.

    If ensemble_across_seeds=True (default), predictions are averaged
    across all seeds for each date first, giving one error value per date
    (recommended -- this is what a date-level test needs). If False, only
    seed 0's predictions are used instead.
    """
    test_a = _load_test_split(csv_path_a, split_col)
    test_b = _load_test_split(csv_path_b, split_col)

    def get_series(test_df):
        seeds = sorted(test_df[seed_col].unique())
        true_vals = test_df[test_df[seed_col] == seeds[0]][true_col].reset_index(drop=True).values
        if ensemble_across_seeds:
            pred_matrix = np.vstack([
                test_df[test_df[seed_col] == s][pred_col].reset_index(drop=True).values
                for s in seeds
            ])
            pred_vals = pred_matrix.mean(axis=0)
        else:
            pred_vals = test_df[test_df[seed_col] == seeds[0]][pred_col].reset_index(drop=True).values
        return true_vals - pred_vals

    e_a = get_series(test_a)
    e_b = get_series(test_b)

    if not np.allclose(
        (test_a[test_a[seed_col] == sorted(test_a[seed_col].unique())[0]][true_col].reset_index(drop=True).values),
        (test_b[test_b[seed_col] == sorted(test_b[seed_col].unique())[0]][true_col].reset_index(drop=True).values),
    ):
        raise ValueError("True values differ between the two files -- these don't look like the same test set.")

    return paired_timeseries_summary(e_a, e_b, name_a=name_a, name_b=name_b, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Paired difference + bootstrap CI from your own CSV files.")
    parser.add_argument("mode", choices=["seed", "timeseries"], help="Which comparison to run.")
    parser.add_argument("csv_a", help="Path to model A's predictions CSV.")
    parser.add_argument("csv_b", help="Path to model B's predictions CSV.")
    parser.add_argument("--name_a", default="Model A")
    parser.add_argument("--name_b", default="Model B")
    parser.add_argument("--true_col", default="true_vol_proxy")
    parser.add_argument("--pred_col", default="predicted_vol")
    parser.add_argument("--seed_col", default="seed")
    parser.add_argument("--split_col", default="split")
    parser.add_argument("--block_length", type=int, default=10, help="Block bootstrap block length (timeseries mode only).")
    parser.add_argument("--n_boot", type=int, default=4000)
    args = parser.parse_args()

    if args.mode == "seed":
        run_seed_from_csv(
            args.csv_a, args.csv_b,
            name_a=args.name_a, name_b=args.name_b,
            true_col=args.true_col, pred_col=args.pred_col,
            seed_col=args.seed_col, split_col=args.split_col,
            n_boot=args.n_boot,
        )
    else:
        run_timeseries_from_csv(
            args.csv_a, args.csv_b,
            name_a=args.name_a, name_b=args.name_b,
            true_col=args.true_col, pred_col=args.pred_col,
            seed_col=args.seed_col, split_col=args.split_col,
            block_length=args.block_length, n_boot=args.n_boot,
        )