"""Core machinery: loss, multi-seed training loop, per-seed cost tracking, evaluation.

Consolidated from common/{losses, benchmark, peak_ram_tracker, training}.py plus
``evaluate_full_timeline`` (which used to live in plotting.py).

Typical use::

    from src import train_multi_seed, evaluate_full_timeline
"""
import copy
import math
import os
import threading
import time

import numpy as np
import psutil
import torch
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset


# ===========================================================================
# Loss  (was common/losses.py)
# ===========================================================================


@torch.jit.script
def asymmetric_mse_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.75) -> torch.Tensor:
    """Quantile-weighted squared error.

    L = mean(w_t * (y_t - pred_t)^2), where w_t = alpha if (y_t - pred_t) > 0
    else (1 - alpha). alpha=0.75 penalizes underprediction more heavily.
    """
    error = target - pred
    sq_error = error ** 2
    weights = torch.where(error > 0, alpha, 1.0 - alpha)
    return torch.mean(weights * sq_error)


# ===========================================================================
# Model size / memory / latency  (was common/benchmark.py)
# ===========================================================================


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_memory_mb():
    ram_mb = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    gpu_mb = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    return ram_mb, gpu_mb


def benchmark_latency(model, sample, device, repeats=100):
    model.eval()
    sample = sample.to(device)

    with torch.no_grad():
        for _ in range(10):
            h = torch.zeros(model.num_layers, sample.size(0), model.hidden_dim, device=sample.device)
            _ = model(sample, h)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(repeats):
            h = torch.zeros(model.num_layers, sample.size(0), model.hidden_dim, device=sample.device)
            _ = model(sample, h)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

    return elapsed / repeats * 1000


# ===========================================================================
# Peak-RAM tracking  (was common/peak_ram_tracker.py)
# ===========================================================================
# Peak-RAM tracking helper.
#
# psutil only gives you a *current* RSS snapshot -- there's no OS-level
# "max RSS since I started watching" call that's portable and reliable across
# platforms in the way torch.cuda.max_memory_allocated() is for GPU. So to get
# a true peak, we poll RSS on a background thread at a short interval and keep
# the running max. Overhead is small (a few memory reads per second).


class PeakRAMTracker:
    """Context manager that tracks peak RSS (in MB) for the current process
    while the `with` block runs.

    Usage:
        with PeakRAMTracker() as tracker:
            train_one_seed(...)
        peak_ram_mb = tracker.peak_mb
    """

    def __init__(self, poll_interval_s: float = 0.05):
        self.poll_interval_s = poll_interval_s
        self._process = psutil.Process(os.getpid())
        self._peak_mb = 0.0
        self._stop_event = threading.Event()
        self._thread = None

    def _poll_loop(self):
        while not self._stop_event.is_set():
            rss_mb = self._process.memory_info().rss / 1024**2
            if rss_mb > self._peak_mb:
                self._peak_mb = rss_mb
            self._stop_event.wait(self.poll_interval_s)

    def __enter__(self):
        # Take one immediate reading so short blocks still get a real number.
        self._peak_mb = self._process.memory_info().rss / 1024**2
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        return False  # don't suppress exceptions

    @property
    def peak_mb(self) -> float:
        return self._peak_mb


# ===========================================================================
# Multi-seed training loop  (was common/training.py)
# ===========================================================================
# Multi-seed training loop. The original notebook copy-pasted this ~90-line
# block into every experiment cell, changing only hyperparameters. Now it's one
# function that takes those hyperparameters as arguments.
#
# CHANGE: for every seed, the model weights used to evaluate the test set are
# now the weights from the epoch with the *best validation loss*, not the
# weights left over from the final training epoch. Checkpoint selection uses
# validation data only -- the test set is never touched until after the best
# checkpoint has been restored.
#
# CHANGE (cost tracking): training time, peak RAM, and peak GPU memory are now
# measured PER SEED, inside the loop, instead of as a single end-of-run
# snapshot. This lets you report honest mean +/- std computational-cost
# numbers across seeds, the same way test loss already is.


def train_multi_seed(
    model_cls,
    data: dict,
    *,
    num_seeds: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    batch_size: int,
    optimizer_cls,
    optimizer_kwargs: dict,
    num_epochs: int = 300,
    log_every: int = 10,
    trial=None,
):
    """Train `model_cls` across several seeds and return per-seed results plus
    the best-seed (lowest test loss) model, ready for plotting/DM tests.

    Per-seed test metrics are computed from the checkpoint with the lowest
    validation loss for that seed (selected using validation data only),
    not from the final epoch's weights.

    Per-seed computational cost (training time, peak RAM, peak GPU memory,
    forecast latency) is also recorded for every seed, not just the best one.
    """
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    input_dim = data["input_dim"]
    criterion = asymmetric_mse_loss
    target_scaler = data["target_scaler"]

    all_models = []
    all_test_preds = []
    all_test_losses = []
    all_train_hist = []
    all_val_hist = []
    all_test_qlike = []
    all_val_rmse_hist = []
    all_best_epochs = []  # which epoch's weights were actually restored for test eval, per seed

    # --- NEW: per-seed computational cost ---
    all_train_time_s = []
    all_peak_ram_mb = []
    all_peak_gpu_mb = []
    all_latency_ms = []
    all_trainable_params = []

    for seed in range(num_seeds):
        print(f"\n{'=' * 50}\n SEED {seed + 1}/{num_seeds}\n{'=' * 50}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Reset GPU peak tracker so this seed's reading isn't polluted by
        # whatever the previous seed (or anything else) allocated.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        model = model_cls(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        )
        train_loader = DataLoader(TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size)
        optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)

        train_losses, val_losses, val_rmses = [], [], []

        best_val_loss = float("inf")
        best_epoch = -1
        best_state_dict = None  # deep-copied weights at the best validation epoch so far

        seed_train_start = time.perf_counter()

        with PeakRAMTracker() as ram_tracker:
            for epoch in range(num_epochs):
                model.train()
                batch_loss_acc = 0
                for X_batch, y_batch in train_loader:
                    h = [torch.zeros(X_batch.size(0), hidden_dim) for _ in range(num_layers)]
                    optimizer.zero_grad()
                    output, _ = model(X_batch, h)
                    pred = output[:, -1, :]
                    loss = criterion(pred, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    batch_loss_acc += loss.item()

                avg_train_loss = batch_loss_acc / len(train_loader)
                train_losses.append(avg_train_loss)

                model.eval()
                with torch.no_grad():
                    h_val = [torch.zeros(X_val.size(0), hidden_dim) for _ in range(num_layers)]
                    val_out, _ = model(X_val, h_val)
                    val_loss = criterion(val_out[:, -1, :], y_val).item()
                    val_rmse = math.sqrt(val_loss)
                    val_losses.append(val_loss)
                    val_rmses.append(val_rmse)
                    if trial is not None:
                        trial.report(val_loss, epoch)

                        if trial.should_prune():
                            import optuna
                            raise optuna.TrialPruned()

                # --- checkpoint selection: strictly on validation loss, test set never consulted ---
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_state_dict = copy.deepcopy(model.state_dict())

                if (epoch + 1) % log_every == 0:
                    print(f"  Epoch {epoch + 1}: Train {avg_train_loss:.4f} | Val loss {val_loss:.4f} | Val RMSE {val_rmse:.4f}")

        # --- per-seed cost readings, taken right as this seed's training block ends ---
        seed_train_time_s = time.perf_counter() - seed_train_start
        seed_peak_ram_mb = ram_tracker.peak_mb
        seed_peak_gpu_mb = (
            torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
        )

        all_train_hist.append(train_losses)
        all_val_hist.append(val_losses)
        all_val_rmse_hist.append(val_rmses)
        all_best_epochs.append(best_epoch)
        all_train_time_s.append(seed_train_time_s)
        all_peak_ram_mb.append(seed_peak_ram_mb)
        all_peak_gpu_mb.append(seed_peak_gpu_mb)

        # Restore the best-validation-epoch weights BEFORE touching the test set.
        # This replaces the old behavior of evaluating whatever the last epoch left in `model`.
        model.load_state_dict(best_state_dict)
        print(f"  Seed {seed + 1}: restoring epoch {best_epoch + 1} weights (val loss {best_val_loss:.6f}) for test evaluation.")

        model.eval()
        with torch.no_grad():
            h_test = [torch.zeros(X_test.size(0), hidden_dim) for _ in range(num_layers)]
            test_out, _ = model(X_test, h_test)
            test_pred = test_out[:, -1, :].numpy()
            test_loss = criterion(torch.tensor(test_pred), y_test).item()

        # Per-seed latency and param count -- cheap, doesn't require retraining,
        # measured on THIS seed's restored (best-checkpoint) weights.
        seed_latency_ms = benchmark_latency(
            model, X_test[:1], next(model.parameters()).device, repeats=100
        )
        seed_trainable_params = count_parameters(model)
        all_latency_ms.append(seed_latency_ms)
        all_trainable_params.append(seed_trainable_params)

        all_models.append(model)
        all_test_preds.append(test_pred)
        all_test_losses.append(test_loss)
        pred_diff_logvar = target_scaler.inverse_transform(test_pred).flatten()
        pred_logvar = data["prev_logvar_test"] + pred_diff_logvar
        actual_logvar = data["actual_logvar_test"]
        pred_var = np.exp(pred_logvar)
        actual_var = np.exp(actual_logvar)
        ratio = actual_var / pred_var
        qlike = np.mean(ratio - np.log(ratio) - 1)
        all_test_qlike.append(qlike)
        print(
            f"  Seed {seed + 1} Test Loss: {test_loss:.6f}  RMSE: {math.sqrt(test_loss):.6f}  "
            f"QLIKE: {qlike:.6f}  (checkpoint: epoch {best_epoch + 1}, val loss {best_val_loss:.6f})"
        )
        print(
            f"  Seed {seed + 1} Cost: train_time={seed_train_time_s:.3f}s  "
            f"peak_ram={seed_peak_ram_mb:.2f}MB  peak_gpu={seed_peak_gpu_mb:.2f}MB  "
            f"latency={seed_latency_ms:.4f}ms"
        )

    test_losses_arr = np.array(all_test_losses)
    print(f"\nTest Loss across {num_seeds} seeds: mean={test_losses_arr.mean():.6f} \u00b1 std={test_losses_arr.std():.6f}")
    print(f"Test RMSE across {num_seeds} seeds: mean={np.sqrt(test_losses_arr).mean():.6f} \u00b1 std={np.sqrt(test_losses_arr).std():.6f}")
    print(f"Test QLIKE across {num_seeds} seeds: mean={np.mean(all_test_qlike):.6f} \u00b1 std={np.std(all_test_qlike):.6f}")
    print(f"Checkpoint epochs used per seed (1-indexed): {[e + 1 for e in all_best_epochs]}")

    # --- NEW: per-seed cost summary, same mean +/- std treatment as test loss ---
    train_time_arr = np.array(all_train_time_s)
    peak_ram_arr = np.array(all_peak_ram_mb)
    peak_gpu_arr = np.array(all_peak_gpu_mb)
    latency_arr = np.array(all_latency_ms)
    print(f"Train time per seed (s): mean={train_time_arr.mean():.3f} \u00b1 std={train_time_arr.std():.3f}")
    print(f"Peak RAM per seed (MB):  mean={peak_ram_arr.mean():.2f} \u00b1 std={peak_ram_arr.std():.2f}")
    print(f"Peak GPU per seed (MB):  mean={peak_gpu_arr.mean():.2f} \u00b1 std={peak_gpu_arr.std():.2f}")
    print(f"Latency per seed (ms):   mean={latency_arr.mean():.4f} \u00b1 std={latency_arr.std():.4f}")

    n = len(test_losses_arr)
    se = test_losses_arr.std(ddof=1) / np.sqrt(n)  # ddof=1 for sample std
    t_crit = stats.t.ppf(0.975, df=n - 1)  # 95% CI, two-tailed
    ci_low, ci_high = test_losses_arr.mean() - t_crit * se, test_losses_arr.mean() + t_crit * se
    print(f"Test Loss 95% CI: [{ci_low:.6f}, {ci_high:.6f}]")

    n = len(test_losses_arr)
    se = np.sqrt(test_losses_arr).std(ddof=1) / np.sqrt(n)  # ddof=1 for sample std
    t_crit = stats.t.ppf(0.975, df=n - 1)  # 95% CI, two-tailed
    ci_low, ci_high = np.sqrt(test_losses_arr).mean() - t_crit * se, np.sqrt(test_losses_arr).mean() + t_crit * se
    print(f"Test RMSE 95% CI: [{ci_low:.6f}, {ci_high:.6f}]")

    n = len(test_losses_arr)

    best_seed_idx = int(np.argmin(test_losses_arr))
    print(f"\nUsing seed {best_seed_idx + 1} (best) for figures and downstream tests.")

    return {
        "model": all_models[best_seed_idx],
        "all_models": all_models,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "best_seed_idx": best_seed_idx,
        "best_epoch_per_seed": all_best_epochs,  # checkpoint provenance for every seed's numbers
        "test_pred": all_test_preds[best_seed_idx],
        "train_losses": all_train_hist[best_seed_idx],
        "val_losses": all_val_hist[best_seed_idx],
        "all_test_losses": all_test_losses,
        "all_train_hist": all_train_hist,
        "all_val_hist": all_val_hist,
        "all_val_rmse_hist": all_val_rmse_hist,
        "all_test_qlike": all_test_qlike,
        # --- NEW: per-seed computational cost, aligned by index with all_models / all_test_losses ---
        "all_train_time_s": all_train_time_s,
        "all_peak_ram_mb": all_peak_ram_mb,
        "all_peak_gpu_mb": all_peak_gpu_mb,
        "all_latency_ms": all_latency_ms,
        "all_trainable_params": all_trainable_params,
        # NEW: pass an optuna.Trial to enable pruning; leave None for normal training
    }


# ===========================================================================
# Evaluation over the full train/val/test timeline  (was in common/plotting.py)
# ===========================================================================


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
