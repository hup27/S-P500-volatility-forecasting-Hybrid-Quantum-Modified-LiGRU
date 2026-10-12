import torch


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
