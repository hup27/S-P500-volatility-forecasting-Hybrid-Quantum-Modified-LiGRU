"""EGARCH(1,1) fitting and conditional-volatility helpers.

This module was duplicated verbatim across four cells in the original
notebook (b1_square, b1, loglik, mle, noise_from_path,
get_conditional_volatility). It now lives in exactly one place.
"""
from typing import Sequence

import numpy as np
import scipy.optimize
import scipy.stats

E_ABS_Z = np.sqrt(2 / np.pi)


def b1_square(x_view: np.ndarray) -> np.ndarray:
    return np.square(x_view[-1, :])


def b1(sigma_view: np.ndarray) -> np.ndarray:
    return sigma_view[-1, :]


def loglik(ret: np.ndarray, params: Sequence, vola_ret_features=b1_square, vola_sigma_features=b1):
    """Calculate the (negative) log-likelihood of a process path, with safety rails."""
    ret = ret.reshape((-1, 1))
    omega, alpha, gamma, beta = params

    # --- SAFETY 1: Constraints ---
    if alpha < 0 or beta < 0 or beta >= 1.0:
        return 1e10

    log_sigma_squared = np.zeros_like(ret)
    log_sigma_squared[0, 0] = omega

    for s in range(1, len(ret)):
        # --- SAFETY 2: Prevent overflow in exp() ---
        prev_log = log_sigma_squared[s - 1, 0]
        if prev_log > 700:
            prev_log = 700

        sigma_prev = np.sqrt(np.exp(prev_log))

        # --- SAFETY 3: Prevent division by zero ---
        if sigma_prev < 1e-6:
            sigma_prev = 1e-6

        z_prev = ret[s - 1, 0] / sigma_prev

        log_sigma_squared[s, 0] = (
            omega
            + beta * np.log(sigma_prev**2)
            + alpha * (np.abs(z_prev) - E_ABS_Z)
            + gamma * z_prev
        )

    # --- SAFETY 4: Handle NaNs in final output ---
    sigma_squared = np.exp(np.clip(log_sigma_squared, -700, 700))

    ll_values = scipy.stats.norm.logpdf(ret[1:, 0], loc=0.0, scale=np.sqrt(sigma_squared[1:, 0]))
    ll = np.sum(ll_values)

    if not np.isfinite(ll):
        return 1e10

    return -ll


def mle(ret: np.ndarray, start_params: Sequence):
    """Maximum-likelihood estimator for the EGARCH(1,1) parameters."""

    def error_fuc(theta):
        return loglik(ret, theta)

    start_params = np.array(start_params)
    result = scipy.optimize.minimize(
        error_fuc,
        start_params,
        method="L-BFGS-B",
        bounds=[
            (-20.0, 0.0),  # omega
            (0.001, 1.0),  # alpha (bound > 0)
            (-1.0, 1.0),  # gamma
            (0.001, 0.99),  # beta (bound < 1.0)
        ],
        options={"maxiter": 250},
    )
    return result


def noise_from_path(ret: np.ndarray, params: Sequence):
    """Extract standardized residuals z_t from an EGARCH path."""
    ret = ret.reshape((-1, 1))
    omega, alpha, gamma, beta = params

    log_sigma_sq = np.zeros_like(ret)
    log_sigma_sq[0, 0] = omega / (1 - beta)
    z = np.zeros_like(ret)

    for t in range(1, len(ret)):
        sigma_prev = np.sqrt(np.exp(log_sigma_sq[t - 1, 0]))
        z_prev = ret[t - 1, 0] / sigma_prev

        log_sigma_sq[t, 0] = (
            omega
            + beta * log_sigma_sq[t - 1, 0]
            + alpha * (np.abs(z_prev) - E_ABS_Z)
            + gamma * z_prev
        )

        z[t, 0] = ret[t, 0] / np.sqrt(np.exp(log_sigma_sq[t, 0]))

    return z


def get_conditional_volatility(ret, params):
    """Re-run the EGARCH recursion with fitted params to get sigma_t."""
    ret = ret.reshape((-1, 1))
    omega, alpha, gamma, beta = params

    log_sigma_sq = np.zeros_like(ret)
    log_sigma_sq[0, 0] = omega

    for t in range(1, len(ret)):
        sigma_prev = np.sqrt(np.exp(log_sigma_sq[t - 1, 0]))
        z_prev = ret[t - 1, 0] / sigma_prev

        log_sigma_sq[t, 0] = (
            omega
            + beta * log_sigma_sq[t - 1, 0]
            + alpha * (np.abs(z_prev) - E_ABS_Z)
            + gamma * z_prev
        )

    return np.sqrt(np.exp(log_sigma_sq)).flatten()


def fit_egarch(returns: np.ndarray, print_summary: bool = True):
    """Fit EGARCH(1,1) via MLE and (optionally) print a coefficient table.

    Returns the fitted `params` array (omega, alpha, gamma, beta).
    """
    log_var_init = np.log(np.var(returns))
    start_params = [
        log_var_init * (1 - 0.85),  # omega
        0.05,  # alpha
        -0.05,  # gamma
        0.85,  # beta
    ]

    result = mle(returns, start_params)
    params = result.x

    if print_summary:
        param_names = ["ω (omega)", "α (alpha)", "γ (gamma)", "β (beta)"]
        cov_matrix = result.hess_inv.todense()
        std_errors = np.sqrt(np.diag(cov_matrix))
        z_stats = params / std_errors
        p_values = 2 * (1 - scipy.stats.norm.cdf(np.abs(z_stats)))

        print("\n" + "=" * 70)
        print("EGARCH(1,1) Maximum Likelihood Estimates")
        print("=" * 70)
        print(f"{'Parameter':<15}{'Estimate':>15}{'Std.Error':>15}{'z':>12}{'p-value':>13}")
        for i in range(len(param_names)):
            print(
                f"{param_names[i]:<15}"
                f"{params[i]:>15.6f}"
                f"{std_errors[i]:>15.6f}"
                f"{z_stats[i]:>12.3f}"
                f"{p_values[i]:>13.5f}"
            )
        print("=" * 70)
        print(f"\nEstimated leverage coefficient (gamma): {params[2]:.6f}")
        print(
            "✓ gamma < 0 : Leverage effect is present."
            if params[2] < 0
            else "✗ gamma >= 0 : Leverage effect is NOT present."
        )

    return params
