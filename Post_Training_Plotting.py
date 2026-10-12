import pandas as pd
from common.plotting import plot_prediction_fit
import numpy as np

from common.data import CRISIS_NAMES, CRISIS_SEGMENTS, build_dataset

# ============================================================
# Recreate dataset metadata
# ============================================================
# This is only needed to recover the dates.
# No LiGRU training is performed.
data = build_dataset(
    seq_length=20,
    use_egarch_vol=True,
    print_egarch_summary=False
)

# ============================================================
# Load saved predictions
# ============================================================
df = pd.read_csv("Quantum_LiGRU_Predictions_event2.csv")

# Only seed 0
df = df[df["seed"] == 0].reset_index(drop=True)

# ============================================================
# Get split boundaries
# ============================================================
train_end = (df["split"] == "train").sum()

val_end = train_end + (df["split"] == "val").sum()

# ============================================================
# Get predictions
# ============================================================
all_true = df["true_vol_proxy"].to_numpy()
all_pred = df["predicted_vol"].to_numpy()

# ============================================================
# Get the actual dates
# ============================================================
all_dates = np.concatenate([
    data["train_dates"],
    data["val_dates"],
    data["test_dates"]
])

# Check that everything lines up
assert len(all_dates) == len(all_true), (
    f"Dates: {len(all_dates)}, Predictions: {len(all_true)}"
)

# ============================================================
# Reconstruct eval_result
# ============================================================
eval_result = {
    "all_true": all_true,
    "all_pred": all_pred,
    "train_end": train_end,
    "val_end": val_end,
    "all_dates": all_dates,
}

# ============================================================
# Plot using your existing function
# ============================================================
plot_prediction_fit(
    eval_result,
    title="Modified LiGRU - True vs Predicted First Difference of the Logarithm of the Squared Daily Returns",
    save_path="Classical_LiGRU_seed0_prediction_fit.png"
)