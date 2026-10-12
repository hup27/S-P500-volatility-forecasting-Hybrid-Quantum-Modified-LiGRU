"""
Diebold-Mariano test (with the Harvey-Leybourne-Newbold small-sample
correction) comparing Quantum LiGRU vs Classical LiGRU, using
SEED-ENSEMBLE-AVERAGED predictions rather than pooled per-seed errors.

Why ensemble-average instead of pooling errors:
`true_vol_proxy` is identical across all seeds and across both models (same
fixed test set) -- only `predicted_vol` differs by seed. So instead of
stacking each seed's error series end-to-end (which introduces artificial
breaks in the series at seed boundaries), we average the seeds' predictions
date-by-date into one "ensemble" prediction per model, then run a single
DM test on that one properly-ordered time series of length T.
"""
import numpy as np
import pandas as pd
from scipy import stats


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


if __name__ == "__main__":
    compare_quantum_vs_classical_ligru_ensemble(loss="squared", h=1)