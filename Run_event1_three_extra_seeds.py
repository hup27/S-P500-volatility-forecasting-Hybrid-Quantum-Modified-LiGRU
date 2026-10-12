"""Run 3 extra seeds (12, 13, 14) on top of the original 12-seed run and save
results to separately-named files (Quantum_LiGRU_Predictions_event1_three*).

This does NOT retrain seeds 0-11 -- it only trains the 3 seeds listed below,
using the same data / hyperparameters as the original run.
"""
from common import data
import torch
from common.training import train_multi_seed
from common.plotting import evaluate_full_timeline, save_predictions_csv, plot_loss_curve, plot_prediction_fit
from models.quantum_ligru import MultiLayerGRU as QuantumLiGRU
from common.data import build_dataset  
# --- CONFIG: filled in with your reported hyperparameters ---
# model_cls = QuantumLiGRU          # <-- import/define your actual quantum model class here
# data = build_dataset(...)         # <-- reuse the same `data` dict from the original 12-seed run

hidden_dim = 4
num_layers = 2
dropout = 0.3
batch_size = 32
optimizer_cls = torch.optim.Adam
optimizer_kwargs = {"lr": 1e-3}     # <-- set this to whatever learning rate your original run used
num_epochs = 300
model_cls = QuantumLiGRU
data = build_dataset(use_egarch_vol=False) 
 # <-- set this to True if your original run used EGARCH vol
EXTRA_SEEDS = [12, 13, 14]
OUT_PREFIX = "Quantum_LiGRU"
SUFFIX = "_event1_three"


def main():
    extra_result = train_multi_seed(
        model_cls=model_cls,
        data=data,
        num_seeds=len(EXTRA_SEEDS),          # only these 3 train -- seeds 0-11 are NOT rerun
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        batch_size=batch_size,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs,
        num_epochs=num_epochs,
    )

    extra_eval_results = [
        evaluate_full_timeline(m, data, hidden_dim, num_layers)
        for m in extra_result["all_models"]
    ]

    save_predictions_csv(
        extra_eval_results,
        extra_result,
        out_prefix=OUT_PREFIX,
        suffix=SUFFIX,
    )

    best_eval = extra_eval_results[extra_result["best_seed_idx"]]

    plot_loss_curve(
        extra_result["train_losses"],
        extra_result["val_losses"],
        title="Quantum LiGRU — Event 1 (3 extra seeds)",
        save_path=f"{OUT_PREFIX}_Predictions{SUFFIX}_loss_curve.png",
    )

    plot_prediction_fit(
        best_eval,
        title="Quantum LiGRU — Prediction Fit, Event 1 (3 extra seeds)",
        save_path=f"{OUT_PREFIX}_Predictions{SUFFIX}_prediction_fit.png",
    )

    print(f"\nDone. Seeds used: {extra_result['seeds_used']}")
    print(f"Best seed (of the 3): {extra_result['seeds_used'][extra_result['best_seed_idx']]}")


if __name__ == "__main__":
    main()