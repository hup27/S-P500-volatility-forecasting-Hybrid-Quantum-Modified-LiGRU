"""Experiment configs, the generic experiment driver, and a command-line launcher.

Consolidated from experiments/{configs, run_experiment, exp_*, exp_tunning}.py and
common/Run_event1_three_extra_seeds.py. (The nine tiny ``exp_*.py`` launchers
are replaced by the command line below.)

Usage::

    python experiments.py --list                 # show every experiment name
    python experiments.py quantum_ligru          # run one experiment
    python experiments.py standalone_ligru
    python experiments.py tune                   # Optuna search, then 12-seed final run
    python experiments.py extra-seeds            # 3 extra seeds for quantum_ligru
"""
import argparse
import time

import numpy as np
import torch
import torch.optim as optim

from analysis import block_bootstrap_ci, sqrt_asymmetric_mse
from data import build_dataset
from models import (
    ClassicalFeatureMultiLayerGRU,
    ClassicalMultiLayerGRU,
    FrozenQuantumMultiLayerGRU,
    GateMapMultiLayerGRU,
    QuantumMultiLayerGRU,
    SharedQuantumMultiLayerGRU,
)
from plotting import plot_loss_curve, plot_prediction_fit, save_predictions_csv
from src import evaluate_full_timeline, train_multi_seed


# ===========================================================================
# Configs  (was experiments/configs.py)
# ===========================================================================
# Every experiment in the original notebook = the same pipeline with different
# hyperparameters. This table is the single source of truth for those
# differences (previously scattered as hard-coded values inside 4 giant cells).


CONFIGS = {
    "egarch_quantum_ligru": {
        "title_prefix": "EGARCH-quantum LiGRU",
        "out_prefix": "Quantum_EGARCH-LiGRU",
        "model_cls": QuantumMultiLayerGRU,
        "use_egarch_vol": True,
        "num_seeds": 7,
        "hidden_dim": 12,
        "num_layers": 2,
        "dropout": 0.4,
        "batch_size": 128,
        "optimizer_cls": optim.RMSprop,
        "optimizer_kwargs": {"lr": 0.001},
    },
    "egarch_classical_ligru": {
        "title_prefix": "EGARCH-classical LiGRU",
        "out_prefix": "Classical_EGARCH-LiGRU",
        "model_cls": ClassicalMultiLayerGRU,
        "use_egarch_vol": True,
        "num_seeds": 7,
        "hidden_dim": 4,
        "num_layers": 2,
        "dropout": 0.3,
        "batch_size": 32,
        "optimizer_cls": optim.Adam,
        "optimizer_kwargs": {"lr": 0.001},
    },
    "standalone_ligru": {
        "title_prefix": "Classical LiGRU",
        "out_prefix": "Classical_LiGRU",
        "model_cls": ClassicalMultiLayerGRU,
        "use_egarch_vol": False,  # raw returns only, input_dim=1
        "num_seeds": 12,
        "hidden_dim": 4,
        "num_layers": 2,
        "dropout": 0.3,
        "batch_size": 32,
        "optimizer_cls": optim.Adam,
        "optimizer_kwargs": {"lr": 0.001},
    },
    "quantum_ligru": {
        "title_prefix": "Quantum LiGRU",
        "out_prefix": "Quantum_LiGRU",
        "model_cls": QuantumMultiLayerGRU,
        "use_egarch_vol": False,  # raw returns only, input_dim=1
        "num_seeds": 12,
        "hidden_dim": 4,
        "num_layers": 2,
        "dropout": 0.3,
        "batch_size": 32,
        "optimizer_cls": optim.Adam,
        "optimizer_kwargs": {"lr": 0.001},
    },
    # NOTE: this entry trains QuantumMultiLayerGRU (the normal entangled model), exactly as
    # the original configs.py did. The no-entanglement architecture lives in models.py as
    # NoEntanglementQuantumMultiLayerGRU but no config uses it. If this experiment is meant
    # to be the no-entanglement ablation, change model_cls below.
    "quantum_ligru_no_entanglement": {
        "title_prefix": "Quantum LiGRU (no entanglement)",
        "out_prefix": "Quantum_LiGRU_No_Entanglement",
        "model_cls": QuantumMultiLayerGRU,
        "use_egarch_vol": False,  # raw returns only, input_dim=1
        "num_seeds": 12,
        "hidden_dim": 4,
        "num_layers": 2,
        "dropout": 0.4,
        "batch_size": 32,
        "optimizer_cls": optim.Adam,
        "optimizer_kwargs": {"lr": 0.001},
    },
    "quantum_ligru_frozen_circuit": {
        "title_prefix": "Quantum LiGRU (frozen circuit)",
        "out_prefix": "Quantum_LiGRU_Frozen_Circuit",
        "model_cls": FrozenQuantumMultiLayerGRU,
        "use_egarch_vol": False,  # raw returns only, input_dim=1
        "num_seeds": 12,
        "hidden_dim": 4,
        "num_layers": 2,
        "dropout": 0.4,
        "batch_size": 32,
        "optimizer_cls": optim.Adam,
        "optimizer_kwargs": {"lr": 0.001},
    },
    "quantum_ligru_shared": {
        "title_prefix": "Quantum LiGRU (shared circuit)",
        "out_prefix": "Quantum_LiGRU_shared_circuit",
        "model_cls": SharedQuantumMultiLayerGRU,
        "use_egarch_vol": False,  # raw returns only, input_dim=1
        "num_seeds": 12,
        "hidden_dim": 4,
        "num_layers": 2,
        "dropout": 0.3,
        "batch_size": 32,
        "optimizer_cls": optim.Adam,
        "optimizer_kwargs": {"lr": 0.001},
    },
    "quantum_ligru_layer": {
            "title_prefix": "Quantum LiGRU Layer",
            "out_prefix": "Quantum_LiGRU_Layer",
            "model_cls": GateMapMultiLayerGRU,
            "use_egarch_vol": False,  
            "num_seeds": 12,
            "hidden_dim": 12,
            "num_layers": 3,
            "dropout": 0.3,
            "batch_size": 64,
            "optimizer_cls": optim.Adam,
            "optimizer_kwargs": {"lr": 0.002},
    },
}

CONFIGS["quantum_ligru_baseline_mlp"] = {
    "title_prefix": "Classical Baseline (param-matched MLP)",
    "out_prefix": "Baseline_MLP_LiGRU",
    "model_cls": ClassicalFeatureMultiLayerGRU,
    "use_egarch_vol": False,
    "num_seeds": 12,
    "hidden_dim": 4,
    "num_layers": 2,
    "dropout": 0.4,
    "batch_size": 32,
    "optimizer_cls": optim.Adam,
    "optimizer_kwargs": {"lr": 0.001},
}


# ===========================================================================
# Generic driver  (was experiments/run_experiment.py)
# ===========================================================================
# Generic driver: build data -> train multi-seed -> evaluate -> plot -> save CSVs.
#
# This single function replaces the ~250 lines of "Training Loop / Plotting /
# Evaluation / Save CSV" that used to be re-typed into every experiment cell.
#
# CHANGE (cost reporting): computational cost is now reported per-seed
# (mean +/- std over all num_seeds runs) instead of a single end-of-pipeline
# snapshot. The old snapshot conflated: 1x data pipeline + 12x training +
# 12x evaluation + 1x bootstrap + 1x plotting + 1x CSV save, all summed into
# one number -- and peak RAM wasn't a true peak at all, just whatever was
# resident at the very end. See src.py for where the per-seed
# numbers are actually measured.


def run(config: dict, seq_length: int = 20):

    end_to_end_start = time.perf_counter()

    data_start = time.perf_counter()
    data = build_dataset(seq_length=seq_length, use_egarch_vol=config["use_egarch_vol"])
    data_pipeline_time = time.perf_counter() - data_start

    train_start = time.perf_counter()

    train_result = train_multi_seed(
        config["model_cls"],
        data,
        num_seeds=config["num_seeds"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        batch_size=config["batch_size"],
        optimizer_cls=config["optimizer_cls"],
        optimizer_kwargs=config["optimizer_kwargs"],
    )

    # Sum of all num_seeds seeds' training time -- kept for reference / sanity
    # checking against sum(train_result["all_train_time_s"]), but the
    # per-seed mean +/- std below is what should actually be reported.
    training_time_total = time.perf_counter() - train_start

    model = train_result["model"]  # best-seed model, used only for the figures/CSVs below

    title_prefix = config["title_prefix"]
    out_prefix = config["out_prefix"]

    plot_loss_curve(
        train_result["train_losses"],
        train_result["val_losses"],
        title=f"{title_prefix} \u2014 Train and Validation Loss",
        save_path=f"{title_prefix}_fig_loss_curve.png",
    )

    eval_start = time.perf_counter()
    all_eval_results = [
        evaluate_full_timeline(m, data, train_result["hidden_dim"], train_result["num_layers"])
        for m in train_result["all_models"]
    ]
    evaluation_time_total = time.perf_counter() - eval_start

    eval_result = all_eval_results[train_result["best_seed_idx"]]

    print(f"\nFinal Scaled Test Loss (Asymmetric MSE): {eval_result['test_loss_value']:.6f}")

    bootstrap_result = block_bootstrap_ci(
    eval_result["y_test"],
    eval_result["test_pred"],
    loss_fn=sqrt_asymmetric_mse,
    block_length=20,
    n_boot=2000,
    )
    print(
        f"Test RMSE (best seed) 95% CI [block bootstrap]: "
        f"{bootstrap_result['point_estimate']:.6f} "
        f"[{bootstrap_result['ci_low']:.6f}, {bootstrap_result['ci_high']:.6f}]"
    )

    plot_prediction_fit(
        eval_result,
        title=f"{title_prefix} \u2014 True vs Predicted First Difference of Proxy Realized Volatility",
        save_path=f"{title_prefix} Prediction Fit Graph.png",
    )

    save_predictions_csv(all_eval_results, train_result, out_prefix)

    end_to_end_time = time.perf_counter() - end_to_end_start

    # --- per-seed cost numbers, straight from train_multi_seed ---
    train_time_arr = np.array(train_result["all_train_time_s"])
    peak_ram_arr = np.array(train_result["all_peak_ram_mb"])
    peak_gpu_arr = np.array(train_result["all_peak_gpu_mb"])
    latency_arr = np.array(train_result["all_latency_ms"])
    params_arr = np.array(train_result["all_trainable_params"])

    print("\n" + "=" * 65)
    print(f"COMPUTATIONAL COST — {title_prefix}  (mean \u00b1 std over {config['num_seeds']} seeds)")
    print("=" * 65)
    print(f"Trainable parameters:       {int(params_arr.mean()):,}  (identical across seeds: {params_arr.min()==params_arr.max()})")
    print(f"Per-seed training time:     {train_time_arr.mean():.3f} \u00b1 {train_time_arr.std():.3f} s")
    print(f"Per-seed peak RAM:          {peak_ram_arr.mean():.2f} \u00b1 {peak_ram_arr.std():.2f} MB")
    print(f"Per-seed peak GPU memory:   {peak_gpu_arr.mean():.2f} \u00b1 {peak_gpu_arr.std():.2f} MB")
    print(f"One-step forecast latency:  {latency_arr.mean():.4f} \u00b1 {latency_arr.std():.4f} ms")
    print("-" * 65)
    print("Pipeline-level timing (NOT per-model -- for reference only):")
    print(f"  Data pipeline (1x):        {data_pipeline_time:.3f} s")
    print(f"  Training, all seeds summed:{training_time_total:.3f} s")
    print(f"  Evaluation, all seeds summed:{evaluation_time_total:.3f} s")
    print(f"  End-to-end (whole run):    {end_to_end_time:.3f} s")

    return {
        "data": data,
        "train_result": train_result,
        "eval_result": eval_result,
        "cost_summary": {
            "trainable_params": int(params_arr.mean()),
            "train_time_s_mean": float(train_time_arr.mean()),
            "train_time_s_std": float(train_time_arr.std()),
            "peak_ram_mb_mean": float(peak_ram_arr.mean()),
            "peak_ram_mb_std": float(peak_ram_arr.std()),
            "peak_gpu_mb_mean": float(peak_gpu_arr.mean()),
            "peak_gpu_mb_std": float(peak_gpu_arr.std()),
            "latency_ms_mean": float(latency_arr.mean()),
            "latency_ms_std": float(latency_arr.std()),
            "data_pipeline_time_s": data_pipeline_time,
            "end_to_end_time_s": end_to_end_time,
        },
    }


# ===========================================================================
# Hyperparameter search  (was experiments/exp_tunning.py)
# ===========================================================================


def objective(trial, data: dict):
    hidden_dim = trial.suggest_categorical("hidden_dim", [4, 6, 8])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    lr = trial.suggest_categorical("lr", [0.001, 0.0005])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)

    # NOTE: num_qubits and num_qlayers are sampled but never passed to the model (the qubit
    # count is fixed at NUM_QUBITS = 2 in models.py), so these two have no effect on the search.
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


def tune(n_trials: int = 50):
    import optuna

    data = build_dataset(seq_length=20, use_egarch_vol=True)

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20),
    )
    study.optimize(lambda trial: objective(trial, data), n_trials=n_trials)

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


# ===========================================================================
# Three extra seeds  (was common/Run_event1_three_extra_seeds.py)
# ===========================================================================
# Run 3 extra seeds (12, 13, 14) on top of the original 12-seed run and save
# results to separately-named files (Quantum_LiGRU_Predictions_event1_three*).
#
# This does NOT retrain seeds 0-11 -- it only trains the 3 seeds listed below,
# using the same data / hyperparameters as the original run.


def run_extra_seeds():
    # NOTE (kept exactly as in the original script -- these look like bugs):
    #   1. train_multi_seed() has no `seeds=` argument; it always trains seeds 0..num_seeds-1, so
    #      this retrains seeds 0, 1, 2 -- NOT 12, 13, 14 as the docstring above says.
    #   2. save_predictions_csv() does not accept `suffix=`, so the call below raises a TypeError.
    #   3. extra_result has no "seeds_used" key, so the final two prints would raise a KeyError.
    # --- hyperparameters of the original quantum_ligru run ---
    hidden_dim = 4
    num_layers = 2
    dropout = 0.3
    batch_size = 32
    optimizer_cls = torch.optim.Adam
    optimizer_kwargs = {"lr": 1e-3}  # set this to whatever learning rate your original run used
    num_epochs = 300
    model_cls = QuantumMultiLayerGRU
    data = build_dataset(use_egarch_vol=False)  # set to True if your original run used EGARCH vol
    EXTRA_SEEDS = [12, 13, 14]
    OUT_PREFIX = "Quantum_LiGRU"
    SUFFIX = "_event1_three"

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


# ===========================================================================
# Command line
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(description="Run LiGRU volatility experiments.")
    parser.add_argument(
        "name", nargs="?",
        help="experiment name (see --list), or 'tune' / 'extra-seeds'",
    )
    parser.add_argument("--list", action="store_true", help="list available experiments and exit")
    parser.add_argument("--seq_length", type=int, default=20, help="input window length (default 20)")
    args = parser.parse_args()

    if args.list or args.name is None:
        print("Experiments:")
        for key, cfg in CONFIGS.items():
            print(f"  {key:32s} {cfg['title_prefix']}")
        print("Other:")
        print(f"  {'tune':32s} Optuna search for the quantum LiGRU, then a 12-seed final run")
        print(f"  {'extra-seeds':32s} three extra seeds for the quantum LiGRU")
        return

    if args.name == "tune":
        tune()
    elif args.name == "extra-seeds":
        run_extra_seeds()
    elif args.name in CONFIGS:
        run(CONFIGS[args.name], seq_length=args.seq_length)
    else:
        parser.error(f"unknown experiment {args.name!r}; use --list")


if __name__ == "__main__":
    main()
