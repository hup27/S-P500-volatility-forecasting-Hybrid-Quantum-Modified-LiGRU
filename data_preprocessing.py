"""Data loading, scaling, and sequence construction.

Every experiment cell in the original notebook re-loaded the same four
S&P 500 CSVs and re-implemented the same two sequence-builder functions.
Both are consolidated here.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler, StandardScaler

from egarch import fit_egarch, get_conditional_volatility, noise_from_path

# CSVs are looked up in ./datasets next to this file first, then in the current
# working directory (the original behaviour), so either layout works.
DATA_DIR = Path(__file__).resolve().parent / "datasets"

SP500_FILES = [
    "S&P 500 Historical Data 1987.csv",
    "S&P 500 Historical Data 2000-2002.csv",
    "S&P 500 Historical Data 2008.csv",
    "S&P 500 Historical Data 2020-2022.csv",
]

# Segment boundaries / crisis labels shared by every "prediction fit" plot.
CRISIS_SEGMENTS = [
    (0, 253, ["1987"]),
    (253, 253 + 752, ["2000", "2001", "2002"]),
    (1005, 1005 + 253, ["2008"])  # end filled in at plot time
]
CRISIS_NAMES = [
    "Black Monday\n1987",
    "Dot-Com Bubble\n2000–2002",
    "Financial Crisis\n2008"
]


def _resolve(filename: str) -> Path:
    candidate = DATA_DIR / filename
    return candidate if candidate.exists() else Path(filename)


def load_sp500_returns():
    charts = [pd.read_csv(_resolve(f)) for f in SP500_FILES]

    # Investing.com exports are newest-first; sort each ascending by date
    # BEFORE concatenating, so the combined series is strictly chronological.
    for c in charts:
        c["Date"] = pd.to_datetime(c["Date"])
        c.sort_values("Date", inplace=True)

    returns = np.concatenate([
        c["Change %"].dropna().str.rstrip("%").astype(float).values
        for c in charts
    ]) / 100.0

    dates = np.concatenate([
        pd.to_datetime(c.loc[c["Change %"].notna(), "Date"]).values
        for c in charts
    ])

    returns = returns - returns.mean()
    return returns, pd.to_datetime(dates)


def create_multivariate_sequences(ret_data, vol_data, target_data, dates, seq_length):
    xs, ys, target_dates = [], [], []
    dates = pd.to_datetime(dates).to_numpy()

    for i in range(len(ret_data) - seq_length):
        window_dates = dates[i:i + seq_length + 1]
        if np.any(np.diff(window_dates).astype("timedelta64[D]").astype(int) > 10):
            continue

        xs.append(np.column_stack((ret_data[i:i + seq_length], vol_data[i:i + seq_length])))
        ys.append(target_data[i + seq_length])
        target_dates.append(dates[i + seq_length])

    return np.array(xs), np.array(ys), np.array(target_dates)


def create_sequences(data, target_data, logvar_data, dates, seq_length):
    xs, ys, prev_logvars, actual_logvars, target_dates = [], [], [], [], []
    dates = pd.to_datetime(dates).to_numpy()

    for i in range(len(data) - seq_length):
        window_dates = dates[i:i + seq_length + 1]
        if np.any(np.diff(window_dates).astype("timedelta64[D]").astype(int) > 10):
            continue
        xs.append(data[i:i + seq_length].reshape(-1, 1))
        ys.append(target_data[i + seq_length])
        prev_logvars.append(logvar_data[i + seq_length - 1])
        actual_logvars.append(logvar_data[i + seq_length])
        target_dates.append(dates[i + seq_length])

    return np.array(xs), np.array(ys), np.array(prev_logvars), np.array(actual_logvars), np.array(target_dates)


def build_dataset(
    seq_length: int = 20,
    use_egarch_vol: bool = True,
    print_egarch_summary: bool = True,
    test_end_date: str = "2008-12-31",
):
    """Full pipeline: load returns -> fit EGARCH -> scale -> build sequences -> tensors.

    Split:
      pool  = all sequences dated on or before test_end_date, i.e. 1987
              Black Monday, Dot-Com 2000-2002, and 2008 (through the end
              of the Financial Crisis). Everything after test_end_date
              (2020-2022) is dropped entirely -- it isn't assigned to
              any split.
      train/val/test = a straight positional 70:15:15 split of the pool,
              taken in chronological order, so test is the last 15% of
              the pool by date (train the first 70%, val the next 15%,
              test the final 15%).
    """
    returns, dates = load_sp500_returns()
    params = fit_egarch(returns, print_summary=print_egarch_summary)

    residuals = noise_from_path(returns, params).flatten()  # noqa: F841 (kept for parity/inspection)
    sigma_egarch = get_conditional_volatility(returns, params)

    vol_proxy = np.log((returns ** 2) + 1e-6)
    vol_proxy_diff = np.diff(vol_proxy)

    returns_trim = returns[1:]
    sigma_egarch_trim = sigma_egarch[1:]
    dates_trim = dates[1:]

    train_split = int(len(returns) * 0.70)

    scaler_ret = RobustScaler()
    scaler_vol = RobustScaler()
    scaler_target = StandardScaler()

    scaler_ret.fit(returns_trim[:train_split].reshape(-1, 1))
    scaler_vol.fit(sigma_egarch_trim[:train_split].reshape(-1, 1))
    scaler_target.fit(vol_proxy_diff[:train_split].reshape(-1, 1))

    ret_scaled = scaler_ret.transform(returns_trim.reshape(-1, 1)).flatten()
    vol_scaled = scaler_vol.transform(sigma_egarch_trim.reshape(-1, 1)).flatten()
    target_scaled = scaler_target.transform(vol_proxy_diff.reshape(-1, 1)).flatten()

    # ALWAYS compute prev_logvar/actual_logvar + target dates, regardless of use_egarch_vol.
    _, _, prev_logvar, actual_logvar, seq_dates = create_sequences(
        ret_scaled, target_scaled, vol_proxy[1:], dates_trim, seq_length
    )

    if use_egarch_vol:
        X, y, _ = create_multivariate_sequences(ret_scaled, vol_scaled, target_scaled, dates_trim, seq_length)
        input_dim = 2
    else:
        X, y, _, _, _ = create_sequences(ret_scaled, target_scaled, vol_proxy[1:], dates_trim, seq_length)
        input_dim = 1

    # --- pool = everything through test_end_date (drops 2020-2022 entirely) ---
    seq_dates_pd = pd.to_datetime(seq_dates)
    test_end = pd.Timestamp(test_end_date)

    pool_mask = seq_dates_pd <= test_end
    pool_idx = np.where(pool_mask)[0]  # already chronological

    n_pool = len(pool_idx)
    train_end = int(n_pool * 0.70)
    val_end = int(n_pool * 0.85)  # 0.70 + 0.15

    train_idx = pool_idx[:train_end]
    val_idx = pool_idx[train_end:val_end]
    test_idx = pool_idx[val_end:]

    prev_logvar_test = prev_logvar[test_idx]
    actual_logvar_test = actual_logvar[test_idx]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(-1)

    X_val = torch.tensor(X[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y[val_idx], dtype=torch.float32).unsqueeze(-1)

    X_test = torch.tensor(X[test_idx], dtype=torch.float32)
    y_test = torch.tensor(y[test_idx], dtype=torch.float32).unsqueeze(-1)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "input_dim": input_dim,
        "scaler_target": scaler_target,
        "egarch_params": params,
        "target_scaler": scaler_target,
        "prev_logvar_test": prev_logvar_test,
        "actual_logvar_test": actual_logvar_test,
        "test_dates": seq_dates[test_idx],
        "train_dates": seq_dates[train_idx],
        "val_dates": seq_dates[val_idx],
    }

def summarize_splits(data: dict, seq_length: int = 20) -> None:
    _, all_dates = load_sp500_returns()
    dates_trim = pd.DatetimeIndex(all_dates[1:])

    for name in ("train", "val", "test"):
        tgt = pd.DatetimeIndex(data[f"{name}_dates"])
        if len(tgt) == 0:
            print(f"\n{name.upper()}: empty")
            continue
        idx = dates_trim.searchsorted(tgt)
        print(f"\n{name.upper()}: {len(idx):,} sequences")
        print(f"  first input window starts : {dates_trim[idx[0] - seq_length].date()}")
        print(f"  first target date         : {tgt[0].date()}")
        print(f"  last target date          : {tgt[-1].date()}")

        blocks = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        print(f"  contiguous blocks         : {len(blocks)}")
        for b in blocks:
            print(f"    {dates_trim[b[0]-seq_length].date()} -> {dates_trim[b[-1]].date()} ({len(b):,} seqs)")

        # --- unique raw data points (trading days) actually touched by this split ---
        raw_day_positions = set()
        for target_pos in idx:
            raw_day_positions.update(range(target_pos - seq_length, target_pos + 1))
        print(f"  unique raw data points    : {len(raw_day_positions):,}"
              f"  (vs {len(idx):,} sequences -> {len(idx) * (seq_length + 1):,} "
              f"raw slots before dedup)")

    # cross-split overlap check, since raw days can repeat across train/val/test
    raw_sets = {}
    for name in ("train", "val", "test"):
        tgt = pd.DatetimeIndex(data[f"{name}_dates"])
        if len(tgt) == 0:
            continue
        idx = dates_trim.searchsorted(tgt)
        s = set()
        for target_pos in idx:
            s.update(range(target_pos - seq_length, target_pos + 1))
        raw_sets[name] = s

    print("\nRaw-day overlap between splits (should be 0 for a clean split):")
    names = list(raw_sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = raw_sets[a] & raw_sets[b]
            print(f"  {a} <-> {b}: {len(overlap)} shared raw days")


if __name__ == "__main__":
    data = build_dataset(seq_length=20, use_egarch_vol=True, print_egarch_summary=False)
    summarize_splits(data, seq_length=20)
