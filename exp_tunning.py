import optuna
import numpy as np
import torch
from common.data import build_dataset
from common.training import train_multi_seed
from models.quantum_ligru import MultiLayerGRU as QuantumMultiLayerGRU  # adjust to your actual filename


def objective(trial, data: dict):
    hidden_dim = trial.suggest_categorical("hidden_dim", [4, 6, 8])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    lr = trial.suggest_categorical("lr", [0.001, 0.0005])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)

    num_qubits = trial.suggest_categorical("num_qubits", [4, 6, 8])
    num_qlayers = trial.suggest_int("num_qlayers", 1, 3)

    result = train_multi_seed(
        QuantumMultiLayerGRU,
        data,
        num_seeds=1,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout_rate,
        batch_size=batch_size,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs={"lr": lr},
        num_epochs=100,
        log_every=1000,
        trial=trial,   # enables pruning inside train_multi_seed
    )

    # --- CHANGED: score trials on validation loss, not test loss. ---
    # train_multi_seed already selects each seed's checkpoint by best
    # validation loss (see all_best_epochs / best_val_loss inside it), so the
    # minimum of that seed's per-epoch val history *is* the best val loss
    # reached. With num_seeds=1 here, all_val_hist has exactly one list.
    best_val_loss = min(result["all_val_hist"][0])
    return best_val_loss


data = build_dataset(seq_length=20, use_egarch_vol=True)

study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20),
)
study.optimize(lambda trial: objective(trial, data), n_trials=50)

print("Best params:", study.best_params)
print("Best mean val loss:", study.best_value)  # renamed for clarity -- this is val loss now

best = study.best_params

# --- Automatically runs right after tuning completes, using best params, 12 seeds ---
# Unchanged: this block only reports TEST loss, which is correct here since
# tuning (above) never touched the test set.
final_result = train_multi_seed(
    QuantumMultiLayerGRU,
    data,
    num_seeds=12,
    hidden_dim=best["hidden_dim"],
    num_layers=best["num_layers"],
    dropout=best["dropout_rate"],
    batch_size=best["batch_size"],
    optimizer_cls=torch.optim.Adam,
    optimizer_kwargs={"lr": best["lr"]},
    num_epochs=300,
    # trial not passed -> no pruning, full training
)

print("Final 12-seed mean test loss:", np.mean(final_result["all_test_losses"]))