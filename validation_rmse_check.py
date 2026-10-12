"""Quick sanity check on validation-split RMSE for the Quantum EGARCH-LiGRU
predictions CSV. Was a standalone notebook cell (cell 2); needs a fitted
`scaler_target` (e.g. from `common.data.build_dataset`) to rescale properly.
"""
import numpy as np
import pandas as pd


def validation_rmse(csv_path: str, scaler_target) -> float:
    df = pd.read_csv(csv_path, index_col=0)
    val_df = df[df["split"] == "val"]

    true_scaled = scaler_target.transform(val_df["true_vol_proxy"].values.reshape(-1, 1))
    pred_scaled = scaler_target.transform(val_df["predicted_vol"].values.reshape(-1, 1))

    rmse = np.sqrt(np.mean((true_scaled - pred_scaled) ** 2))
    print(f"Validation RMSE (scaled): {rmse:.6f}")
    return rmse


if __name__ == "__main__":
    from common.data import build_dataset

    data = build_dataset(print_egarch_summary=False)
    validation_rmse("Quantum_EGARCH-LiGRU_Predictions.csv", data["scaler_target"])
