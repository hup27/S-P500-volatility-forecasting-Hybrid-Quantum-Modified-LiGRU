import numpy as np
import scipy.stats as stats
from scipy import stats as dm_stats


def mean_ci(data, confidence: float = 0.95):
    """Return (mean, (lower, upper)) confidence interval using the t-distribution."""
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    t_val = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_val * sem
    return mean, (mean - margin, mean + margin)


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> tuple:
    """Diebold-Mariano test comparing squared-error loss of two forecasts,
    with the Harvey-Leybourne-Newbold (1997) small-sample correction.

    Negative dm_stat favors e1 (model 1 is more accurate).
    """
    d = e1 ** 2 - e2 ** 2
    T = len(d)
    d_bar = d.mean()
    nw_var = np.var(d, ddof=1)
    for lag in range(1, h):
        nw_var += 2 * np.cov(d[lag:], d[:-lag], ddof=1)[0, 1]
    nw_var = max(nw_var, 1e-12)
    dm_stat = d_bar / np.sqrt(nw_var / T)

    # HLN small-sample correction
    hln_factor = np.sqrt((T + 1 - 2 * h + (h * (h - 1)) / T) / T)
    dm_stat_corrected = dm_stat * hln_factor

    # Use t-distribution (df = T - 1) instead of normal for the p-value
    p_val = 2 * dm_stats.t.sf(np.abs(dm_stat_corrected), df=T - 1)

    return float(dm_stat_corrected), float(p_val)
