"""Plots and result-file output shared by every experiment.

Consolidated from common/plotting.py and common/Post_Training_Plotting.py.
(The bootstrap-CI helpers that used to sit in the middle of plotting.py now live
in analysis.py, and ``evaluate_full_timeline`` lives in src.py.)

Every experiment used to re-define the same matplotlib style, the same "loss
curve" figure, and the same "true vs predicted with crisis shading" figure,
changing only titles/filenames.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


# ===========================================================================
# Re-plot from a saved predictions CSV  (was common/Post_Training_Plotting.py)
# ===========================================================================
# No training happens here: build_dataset() is only called to recover the dates,
# then a saved *_Predictions*.csv is re-plotted with plot_prediction_fit().


def replot_from_csv(
    predictions_csv: str = "Quantum_LiGRU_Predictions_event2.csv",
    seed: int = 0,
    title: str = "Modified LiGRU - True vs Predicted First Difference of the Logarithm of the Squared Daily Returns",
    save_path: str = "Classical_LiGRU_seed0_prediction_fit.png",
):
    from data import build_dataset  # imported lazily: pulls in torch, sklearn, EGARCH

    # --- Recreate dataset metadata (only needed to recover the dates) ---
    data = build_dataset(
        seq_length=20,
        use_egarch_vol=True,
        print_egarch_summary=False
    )

    # --- Load saved predictions (one seed) ---
    df = pd.read_csv(predictions_csv)
    df = df[df["seed"] == seed].reset_index(drop=True)

    # --- Get split boundaries ---
    train_end = (df["split"] == "train").sum()
    val_end = train_end + (df["split"] == "val").sum()

    # --- Get predictions ---
    all_true = df["true_vol_proxy"].to_numpy()
    all_pred = df["predicted_vol"].to_numpy()

    # --- Get the actual dates ---
    all_dates = np.concatenate([
        data["train_dates"],
        data["val_dates"],
        data["test_dates"]
    ])

    # Check that everything lines up
    assert len(all_dates) == len(all_true), (
        f"Dates: {len(all_dates)}, Predictions: {len(all_true)}"
    )

    # --- Reconstruct eval_result ---
    eval_result = {
        "all_true": all_true,
        "all_pred": all_pred,
        "train_end": train_end,
        "val_end": val_end,
        "all_dates": all_dates,
    }

    # --- Plot using the existing function ---
    plot_prediction_fit(eval_result, title=title, save_path=save_path)


if __name__ == "__main__":
    replot_from_csv()
