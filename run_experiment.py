"""Generic driver: build data -> train multi-seed -> evaluate -> plot -> save CSVs.

This single function replaces the ~250 lines of "Training Loop / Plotting /
Evaluation / Save CSV" that used to be re-typed into every experiment cell.

CHANGE (cost reporting): computational cost is now reported per-seed
(mean +/- std over all num_seeds runs) instead of a single end-of-pipeline
snapshot. The old snapshot conflated: 1x data pipeline + 12x training +
12x evaluation + 1x bootstrap + 1x plotting + 1x CSV save, all summed into
one number -- and peak RAM wasn't a true peak at all, just whatever was
resident at the very end. See common/training.py for where the per-seed
numbers are actually measured.
"""
import time

import numpy as np

from common.data import build_dataset
from common.plotting import evaluate_full_timeline, plot_loss_curve, plot_prediction_fit, save_predictions_csv
from common.training import train_multi_seed
from common.benchmark import benchmark_latency, get_memory_mb, count_parameters
from common.plotting import evaluate_full_timeline, plot_loss_curve, plot_prediction_fit, save_predictions_csv, block_bootstrap_ci
from common.plotting import sqrt_asymmetric_mse

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