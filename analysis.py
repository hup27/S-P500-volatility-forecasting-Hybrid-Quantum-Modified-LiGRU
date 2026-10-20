"""Statistical comparison of models: confidence intervals, bootstrap, Diebold-Mariano
tests, paired differences.

Consolidated from common/metrics.py, analysis/{compare_models_dm_test,
paired_difference, validation_rmse_check}.py, experiments/compare_dm.py, and the
two statistics helpers (``block_bootstrap_ci``, ``sqrt_asymmetric_mse``) that
used to sit inside common/plotting.py.

Command-line use (see ``python analysis.py --help``)::

    python analysis.py paired seed       A.csv B.csv --name_a Quantum --name_b Classical
    python analysis.py paired timeseries A.csv B.csv
    python analysis.py dm-ensemble       # quantum vs classical, seed-ensemble DM test
    python analysis.py dm-per-seed       # per-seed DM test, mean DM stat with CI
    python analysis.py val-rmse          # validation RMSE of a predictions CSV
"""
import math
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy import stats

from src import asymmetric_mse_loss


# ===========================================================================
# Basic metrics  (was common/metrics.py)
# ===========================================================================


def mean_ci(data, confidence: float = 0.95):
    """Return (mean, (lower, upper)) confidence interval using the t-distribution."""
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    t_val = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_val * sem
    return mean, (mean - margin, mean + margin)


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> tuple:
    """Diebold-Mariano test comparing squared-error loss of two forecasts,
    with the Harvey-Leybourne-Newbold (1997) small-sample correction.

    Negative dm_stat favors e1 (model 1 is more accurate).
    """
    d = e1 ** 2 - e2 ** 2
    T = len(d)
    d_bar = d.mean()
    nw_var = np.var(d, ddof=1)
    for lag in range(1, h):
        nw_var += 2 * np.cov(d[lag:], d[:-lag], ddof=1)[0, 1]
    nw_var = max(nw_var, 1e-12)
    dm_stat = d_bar / np.sqrt(nw_var / T)

    # HLN small-sample correction
    hln_factor = np.sqrt((T + 1 - 2 * h + (h * (h - 1)) / T) / T)
    dm_stat_corrected = dm_stat * hln_factor

    # Use t-distribution (df = T - 1) instead of normal for the p-value
    p_val = 2 * stats.t.sf(np.abs(dm_stat_corrected), df=T - 1)

    return float(dm_stat_corrected), float(p_val)


# ===========================================================================
# Block bootstrap on test error  (was in common/plotting.py)
# ===========================================================================


def sqrt_asymmetric_mse(y_true, y_pred):
    yt = torch.tensor(y_true, dtype=torch.float32)
    yp = torch.tensor(y_pred, dtype=torch.float32)
    return math.sqrt(asymmetric_mse_loss(yp, yt).item())


def block_bootstrap_ci(y_true, y_pred, loss_fn=None, block_length=20, n_boot=2000, ci=0.95, seed=0):
    """Time-series-aware block bootstrap CI for a scalar error metric computed
    over paired (y_true, y_pred) sequences. Resamples contiguous blocks of
    length `block_length` (with replacement) to preserve autocorrelation,
    instead of resampling individual time points independently.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    n = len(y_true)
    n_blocks = int(np.ceil(n / block_length))

    if loss_fn is None:
        def loss_fn(yt, yp):
            return np.sqrt(np.mean((yt - yp) ** 2))

    boot_stats = np.empty(n_boot)
    for b in range(n_boot):
        start_idxs = rng.integers(0, n - block_length + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_length) for s in start_idxs])[:n]
        boot_stats[b] = loss_fn(y_true[idx], y_pred[idx])

    point_estimate = loss_fn(y_true, y_pred)
    alpha = 1 - ci
    lo, hi = np.percentile(boot_stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return {
        "point_estimate": point_estimate,
        "ci_low": lo,
        "ci_high": hi,
        "boot_stats": boot_stats,
    }


# ===========================================================================
# Validation RMSE check  (was analysis/validation_rmse_check.py)
# ===========================================================================
# Quick sanity check on validation-split RMSE for the Quantum EGARCH-LiGRU
# predictions CSV. Was a standalone notebook cell (cell 2); needs a fitted
# `scaler_target` (e.g. from `common.data.build_dataset`) to rescale properly.


def validation_rmse(csv_path: str, scaler_target) -> float:
    df = pd.read_csv(csv_path, index_col=0)
    val_df = df[df["split"] == "val"]

    true_scaled = scaler_target.transform(val_df["true_vol_proxy"].values.reshape(-1, 1))
    pred_scaled = scaler_target.transform(val_df["predicted_vol"].values.reshape(-1, 1))

    rmse = np.sqrt(np.mean((true_scaled - pred_scaled) ** 2))
    print(f"Validation RMSE (scaled): {rmse:.6f}")
    return rmse


# ===========================================================================
# Diebold-Mariano on seed-ensembled predictions  (was analysis/compare_models_dm_test.py)
# ===========================================================================
# Diebold-Mariano test (with the Harvey-Leybourne-Newbold small-sample
# correction) comparing Quantum LiGRU vs Classical LiGRU, using
# SEED-ENSEMBLE-AVERAGED predictions rather than pooled per-seed errors.
#
# Why ensemble-average instead of pooling errors:
# `true_vol_proxy` is identical across all seeds and across both models (same
# fixed test set) -- only `predicted_vol` differs by seed. So instead of
# stacking each seed's error series end-to-end (which introduces artificial
# breaks in the series at seed boundaries), we average the seeds' predictions
# date-by-date into one "ensemble" prediction per model, then run a single
# DM test on that one properly-ordered time series of length T.


def load_ensemble_test_series(
    csv_path: str,
    true_col: str = "true_vol_proxy",
    pred_col: str = "predicted_vol",
    seed_col: str = "seed",
    split_col: str = "split",
):
    """Return (y_true, ensemble_pred) for the test split, where
    ensemble_pred[t] is the mean prediction across all seeds for date t."""
    df = pd.read_csv(csv_path, index_col=0)
    test_df = df[df[split_col] == "test"].copy()

    seeds = sorted(test_df[seed_col].unique())
    lengths = test_df.groupby(seed_col).size()
    if lengths.nunique() != 1:
        raise ValueError(f"Seeds have mismatched test lengths: {lengths.to_dict()}")

    # true values should be identical across seeds -> just take the first seed's
    first_seed_true = test_df[test_df[seed_col] == seeds[0]][true_col].reset_index(drop=True)

    # stack predictions: rows = seeds, columns = time index within test split
    pred_matrix = np.vstack([
        test_df[test_df[seed_col] == s][pred_col].reset_index(drop=True).values
        for s in seeds
    ])
    ensemble_pred = pred_matrix.mean(axis=0)

    return first_seed_true.values, ensemble_pred


def diebold_mariano_hln(e_a, e_b, h: int = 1, loss: str = "squared"):
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold (1997)
    small-sample correction.

    e_a, e_b : forecast error arrays (y_true - y_pred), equal length
    h        : forecast horizon (1 for one-step-ahead)
    loss     : 'squared' for L(e) = e^2, 'absolute' for L(e) = |e|

    Returns (dm_stat_hln, p_value). Negative dm_stat => model A (e_a) has
    lower average loss than model B (e_b).
    """
    e_a = np.asarray(e_a, dtype=float)
    e_b = np.asarray(e_b, dtype=float)
    if len(e_a) != len(e_b):
        raise ValueError("Error series must be the same length.")

    if loss == "squared":
        d = e_a ** 2 - e_b ** 2
    elif loss == "absolute":
        d = np.abs(e_a) - np.abs(e_b)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")

    T = len(d)
    d_bar = d.mean()

    # long-run variance of d_bar, with (h-1) lags of autocovariance (Newey-West style)
    gamma0 = np.sum((d - d_bar) ** 2) / T
    var_d = gamma0
    for k in range(1, h):
        gamma_k = np.sum((d[k:] - d_bar) * (d[:-k] - d_bar)) / T
        var_d += 2 * gamma_k
    var_d /= T  # variance of the mean, not of d itself

    if var_d <= 0:
        raise ValueError("Estimated variance of the loss differential is non-positive; "
                          "check for a degenerate (near-constant) series.")

    dm_stat = d_bar / np.sqrt(var_d)

    # Harvey-Leybourne-Newbold (1997) small-sample correction factor
    hln_factor = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_stat_hln = dm_stat * hln_factor

    # HLN recommend comparing against a Student-t distribution with T-1 df
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat_hln), df=T - 1))

    return dm_stat_hln, p_value


def compare_quantum_vs_classical_ligru_ensemble(
    quantum_csv: str = "Quantum_LiGRU_Predictions_event2.csv",
    classical_csv: str = "Classical_LiGRU_Predictions_event2.csv",
    loss: str = "squared",
    h: int = 1,
):
    y_true_q, pred_q = load_ensemble_test_series(quantum_csv)
    y_true_c, pred_c = load_ensemble_test_series(classical_csv)

    if not np.allclose(y_true_q, y_true_c):
        raise ValueError("Quantum and Classical test sets don't match -- true values differ.")

    e_quantum = y_true_q - pred_q
    e_classical = y_true_c - pred_c

    dm_stat, p_value = diebold_mariano_hln(e_quantum, e_classical, h=h, loss=loss)

    if p_value < 0.05:
        winner = "Quantum LiGRU" if dm_stat < 0 else "Classical LiGRU"
        result = f"{winner} wins"
    else:
        result = "No significant difference"

    print("\n" + "=" * 70)
    print("DIEBOLD-MARIANO (HLN-corrected), ENSEMBLE-AVERAGED ACROSS SEEDS")
    print(f"Loss: {loss} | horizon h={h} | T={len(e_quantum)}")
    print("=" * 70)
    print(f"DM statistic: {dm_stat:.4f}")
    print(f"p-value:      {p_value:.4f}")
    print(f"Conclusion:   {result}")
    print(f"(negative DM stat => Quantum LiGRU has lower average loss)")

    return {"dm_stat": dm_stat, "p_value": p_value, "result": result}


# ===========================================================================
# Paired difference + bootstrap CI  (was analysis/paired_difference.py)
# ===========================================================================
# Paired difference summary + bootstrap confidence interval, in one place.
#
# Two entry points, because the two kinds of "paired difference" in this
# project have different dependence structure:
#
# - paired_seed_summary(rmse_a, rmse_b, ...)
#     For per-seed test RMSE (one value per seed, no temporal order between
#     seeds). Reports mean +/- std of the paired difference, a paired t-test,
#     and a plain (iid) bootstrap CI on the mean difference.
#
# - paired_timeseries_summary(e_a, e_b, ...)
#     For a date-level paired error series (e.g. e_quantum,t vs e_classical,t
#     across T chronological test dates). Computes the loss differential
#     d_t = L(e_a,t) - L(e_b,t), reports its mean +/- std, and a CIRCULAR
#     MOVING-BLOCK bootstrap CI that preserves serial dependence (matches the
#     paper's Table 2 methodology: block length 10, 4000 reps).


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


# ===========================================================================
# Per-seed Diebold-Mariano, classical vs quantum  (was experiments/compare_dm.py)
# ===========================================================================


def get_test_errors(df, seed):
    seed_df = df[(df["seed"] == seed) & (df["split"] == "test")]
    y_true = seed_df["true_vol_proxy"].values
    y_pred = seed_df["predicted_vol"].values
    return y_true - y_pred


def compare_dm_per_seed(
    classical_csv: str = "Classical_LiGRU_Predictions.csv",
    quantum_csv: str = "Quantum_LiGRU_Predictions.csv",
    num_seeds: int = 12,
):
    print("STARTING compare_dm")

    classical_df = pd.read_csv(classical_csv)
    quantum_df = pd.read_csv(quantum_csv)

    dm_stats, p_vals = [], []

    for seed in range(num_seeds):
        e1 = get_test_errors(classical_df, seed)
        e2 = get_test_errors(quantum_df, seed)
        dm_stat, p_val = diebold_mariano(e1, e2, h=1)
        dm_stats.append(dm_stat)
        p_vals.append(p_val)
        print(f"Seed {seed}: DM stat = {dm_stat:.4f}, p = {p_val:.4f}")

    dm_stats = np.array(dm_stats)
    mean_dm, (ci_low, ci_high) = mean_ci(dm_stats)

    print(f"\nMean DM stat across seeds: {mean_dm:.4f}  95% CI [{ci_low:.4f}, {ci_high:.4f}]")

    n_favor_classical = np.sum(dm_stats < 0)
    n_favor_quantum = np.sum(dm_stats > 0)
    n_significant = np.sum(np.array(p_vals) < 0.05)

    print(f"Seeds favoring classical: {n_favor_classical}/{num_seeds}")
    print(f"Seeds favoring quantum:   {n_favor_quantum}/{num_seeds}")
    print(f"Seeds with p < 0.05:      {n_significant}/{num_seeds}")


# ===========================================================================
# Command line
# ===========================================================================


def _build_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Statistical comparisons between model prediction CSVs.")
    sub = parser.add_subparsers(dest="command", required=True)

    # paired seed|timeseries  (was: python paired_difference.py ...)
    pp = sub.add_parser("paired", help="Paired difference + bootstrap CI from your own CSV files.")
    pp.add_argument("mode", choices=["seed", "timeseries"], help="Which comparison to run.")
    pp.add_argument("csv_a", help="Path to model A's predictions CSV.")
    pp.add_argument("csv_b", help="Path to model B's predictions CSV.")
    pp.add_argument("--name_a", default="Model A")
    pp.add_argument("--name_b", default="Model B")
    pp.add_argument("--true_col", default="true_vol_proxy")
    pp.add_argument("--pred_col", default="predicted_vol")
    pp.add_argument("--seed_col", default="seed")
    pp.add_argument("--split_col", default="split")
    pp.add_argument("--block_length", type=int, default=10, help="Block bootstrap block length (timeseries mode only).")
    pp.add_argument("--n_boot", type=int, default=4000)

    # dm-ensemble  (was: python compare_models_dm_test.py)
    de = sub.add_parser("dm-ensemble", help="DM test (HLN) on seed-ensemble-averaged predictions.")
    de.add_argument("--quantum_csv", default="Quantum_LiGRU_Predictions_event2.csv")
    de.add_argument("--classical_csv", default="Classical_LiGRU_Predictions_event2.csv")
    de.add_argument("--loss", choices=["squared", "absolute"], default="squared")
    de.add_argument("--h", type=int, default=1)

    # dm-per-seed  (was: python compare_dm.py)
    dp = sub.add_parser("dm-per-seed", help="Per-seed DM test, then mean DM stat with a t-CI.")
    dp.add_argument("--classical_csv", default="Classical_LiGRU_Predictions.csv")
    dp.add_argument("--quantum_csv", default="Quantum_LiGRU_Predictions.csv")
    dp.add_argument("--num_seeds", type=int, default=12)

    # val-rmse  (was: python validation_rmse_check.py)
    vr = sub.add_parser("val-rmse", help="Validation RMSE (scaled) of a predictions CSV.")
    vr.add_argument("csv", nargs="?", default="Quantum_EGARCH-LiGRU_Predictions.csv")

    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.command == "paired":
        common = dict(name_a=args.name_a, name_b=args.name_b, true_col=args.true_col,
                      pred_col=args.pred_col, seed_col=args.seed_col, split_col=args.split_col,
                      n_boot=args.n_boot)
        if args.mode == "seed":
            run_seed_from_csv(args.csv_a, args.csv_b, **common)
        else:
            run_timeseries_from_csv(args.csv_a, args.csv_b, block_length=args.block_length, **common)

    elif args.command == "dm-ensemble":
        compare_quantum_vs_classical_ligru_ensemble(
            quantum_csv=args.quantum_csv, classical_csv=args.classical_csv, loss=args.loss, h=args.h)

    elif args.command == "dm-per-seed":
        compare_dm_per_seed(args.classical_csv, args.quantum_csv, args.num_seeds)

    elif args.command == "val-rmse":
        from data import build_dataset

        data = build_dataset(print_egarch_summary=False)
        validation_rmse(args.csv, data["scaler_target"])
