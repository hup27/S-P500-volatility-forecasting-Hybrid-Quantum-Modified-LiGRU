"""Evaluation + plotting helpers shared by every experiment.

Every experiment cell in the notebook re-defined the same matplotlib style,
the same "loss curve" figure, and the same "true vs predicted with crisis
shading" figure, changing only titles/filenames. Consolidated here.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import math
from common.data import CRISIS_NAMES, CRISIS_SEGMENTS
from common.losses import asymmetric_mse_loss

MPL_STYLE = {
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "lines.linewidth": 1.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}


def apply_style():
    mpl.rcParams.update(MPL_STYLE)


def plot_loss_curve(train_losses, val_losses, title: str, save_path: str):
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Train Loss", color="steelblue")
    ax.plot(val_losses, label="Validation Loss", color="#FF5500")
    ax.set_ylim(0, 0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, color="black", linestyle="-", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()


def evaluate_full_timeline(model, data: dict, hidden_dim: int, num_layers: int):
    """Run the (already-trained) model over train/val/test and inverse-transform
    predictions back to the original log(r^2) scale.
    """
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    scaler_target = data["scaler_target"]

    model.eval()
    with torch.no_grad():
        h_train = [torch.zeros(X_train.size(0), hidden_dim) for _ in range(num_layers)]
        train_pred = model(X_train, h_train)[0][:, -1, :].numpy()

        h_val = [torch.zeros(X_val.size(0), hidden_dim) for _ in range(num_layers)]
        val_pred = model(X_val, h_val)[0][:, -1, :].numpy()

        h_test = [torch.zeros(X_test.size(0), hidden_dim) for _ in range(num_layers)]
        test_pred = model(X_test, h_test)[0][:, -1, :].numpy()

    train_pred_inv = scaler_target.inverse_transform(train_pred)
    val_pred_inv = scaler_target.inverse_transform(val_pred)
    test_pred_inv = scaler_target.inverse_transform(test_pred)

    y_train_inv = scaler_target.inverse_transform(y_train.numpy())
    y_val_inv = scaler_target.inverse_transform(y_val.numpy())
    y_test_inv = scaler_target.inverse_transform(y_test.numpy())

    all_true = np.concatenate([y_train_inv, y_val_inv, y_test_inv], axis=0).flatten()
    all_pred = np.concatenate([train_pred_inv, val_pred_inv, test_pred_inv], axis=0).flatten()

    train_end = len(y_train_inv)
    val_end = train_end + len(y_val_inv)

    all_dates = np.concatenate([data["train_dates"], data["val_dates"], data["test_dates"]])

    test_loss_value = asymmetric_mse_loss(torch.tensor(test_pred), y_test).item()

    return {
    "all_true": all_true,
    "all_pred": all_pred,
    "all_dates": all_dates,
    "train_end": train_end,
    "val_end": val_end,
    "test_loss_value": test_loss_value,
    "y_test": y_test.numpy(),
    "test_pred": test_pred
    }

from common.losses import asymmetric_mse_loss
import torch

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
def plot_prediction_fit(eval_result: dict, title: str, save_path: str):
    apply_style()
    all_true = eval_result["all_true"]
    all_pred = eval_result["all_pred"]
    train_end = eval_result["train_end"]
    val_end = eval_result["val_end"]
    all_dates = eval_result["all_dates"]

    # Full dataset is used now (60/30/10 positional split, nothing dropped),
    # so train_end/val_end from eval_result already line up with this array
    # -- no filtering/remapping needed like the old crisis-drop version.
    #
    # Segment boundaries are still derived from real per-sample dates rather
    # than the hardcoded CRISIS_SEGMENTS row numbers, since those were
    # computed against the raw unfiltered return series and won't match this
    # array's index space after windowing/gap-filtering.
    CRISIS_YEAR_GROUPS = [
    (["1987"], "Black Monday\n1987"),
    (["2000", "2001", "2002"], "Dot-Com Bubble\n2000–2002"),
    (["2008"], "Financial Crisis\n2008"),
    (["2020"], "COVID-19\n2020")
    ]
    years = pd.to_datetime(all_dates).year.astype(str)
    segments = []
    segment_names = []
    for labels, name in CRISIS_YEAR_GROUPS:
        idxs = np.where(np.isin(years, labels))[0]
        if len(idxs) == 0:
            continue
        segments.append((idxs[0], idxs[-1] + 1, labels))
        segment_names.append(name)

    tick_positions, tick_labels = [], []
    for start, end, labels in segments:
        positions = np.linspace(start, end, len(labels) + 2)[1:-1]
        tick_positions.extend(positions)
        tick_labels.extend(labels)

    fig, ax = plt.subplots(figsize=(20, 6))

    ax.plot(all_true, label="True First Difference of the Logarithm of the Squared Daily Returns", color="steelblue", linewidth=2.0)
    ax.plot(all_pred, label="Predicted First Difference of the Logarithm of the Squared Daily Returns", color="#FF5E00", linewidth=2.0, alpha=0.8)

    ax.axvspan(0, train_end, alpha=0.07, color="green", label="Train")
    ax.axvspan(train_end, val_end, alpha=0.07, color="orange", label="Validation")
    ax.axvspan(val_end, len(all_true), alpha=0.07, color="red", label="Test")

    ax.axvline(train_end, color="green", linestyle="--", linewidth=1.5)
    ax.axvline(val_end, color="orange", linestyle="--", linewidth=1.5)

    ymin, ymax = ax.get_ylim()
    for i, (start, end, labels) in enumerate(segments):
        ax.axvline(start, color="#555555", linestyle="-", linewidth=1.2, alpha=0.7, zorder=3)
        ax.axvline(end, color="#555555", linestyle="-", linewidth=1.2, alpha=0.7, zorder=3)
        ax.text(
            (start + end) / 2, ymax, segment_names[i],
            ha="center", va="top", fontsize=14, color="#333333", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"),
            zorder=4,
        )

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Period")
    ax.set_ylabel("First Difference of the\nLogarithm of the Squared Daily Returns")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, color="gray", linestyle="-", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()


def save_predictions_csv(all_eval_results: list, train_result: dict, out_prefix: str):
    frames = []
    for seed_idx, eval_result in enumerate(all_eval_results):
        all_true = eval_result["all_true"]
        all_pred = eval_result["all_pred"]
        train_end = eval_result["train_end"]
        val_end = eval_result["val_end"]

        frames.append(pd.DataFrame({
            "seed": seed_idx,
            "true_vol_proxy": all_true,
            "predicted_vol": all_pred,
            "split": (["train"] * train_end
                      + ["val"] * (val_end - train_end)
                      + ["test"] * (len(all_true) - val_end)),
        }))

    predictions_df = pd.concat(frames, ignore_index=True)
    predictions_df.to_csv(f"{out_prefix}_Predictions.csv", index=True)
    print(f"Saved {len(predictions_df)} rows across {len(all_eval_results)} seeds to {out_prefix}_Predictions.csv")

    # unchanged — already per-seed, equal-length, no padding needed
    pd.DataFrame(train_result["all_train_hist"]).to_csv(f"{out_prefix}_Predictions_train_losses.csv", index=False)
    pd.DataFrame(train_result["all_val_hist"]).to_csv(f"{out_prefix}_Predictions_val_losses.csv", index=False)

    pd.DataFrame({
        "seed": np.arange(len(train_result["all_test_losses"])),
        "test_loss": train_result["all_test_losses"],
        "test_rmse": np.sqrt(train_result["all_test_losses"]),
    }).to_csv(f"{out_prefix}_Predictions_test_losses.csv", index=False)


