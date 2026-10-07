import pandas as pd
import numpy as np
from common.metrics import diebold_mariano, mean_ci


def get_test_errors(df, seed):
    seed_df = df[(df["seed"] == seed) & (df["split"] == "test")]
    y_true = seed_df["true_vol_proxy"].values
    y_pred = seed_df["predicted_vol"].values
    return y_true - y_pred


def main():
    print("STARTING compare_dm")
    
    classical_df = pd.read_csv("Classical_LiGRU_Predictions.csv")
    quantum_df = pd.read_csv("Quantum_LiGRU_Predictions.csv")  # adjust filename

    num_seeds = 12  # adjust to match your actual num_seeds

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


if __name__ == "__main__":
    main()