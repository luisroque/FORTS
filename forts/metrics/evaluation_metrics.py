import numpy as np


def smape(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    epsilon = 1e-8  # Use a smaller epsilon for floating point comparisons

    # Ensure the denominator is not zero to avoid division by zero errors
    safe_denominator = np.where(denominator < epsilon, epsilon, denominator)

    smape_value = 100 * np.mean(
        np.where(
            denominator < epsilon, 0, 2 * np.abs(y_true - y_pred) / safe_denominator
        )
    )
    return smape_value


def mase(y_true, y_pred, h, m: int = 1):
    """
    Mean Absolute Scaled Error (MASE).
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)

    if y_true.size <= max(m, h):
        return np.nan

    y_true_insample = y_true[:-h]
    y_true_h = y_true[-h:]

    y_pred_h = y_pred[-h:]

    mask = ~np.isnan(y_pred_h)
    if mask.sum() == 0:
        return np.nan

    # Ensure there are enough points for seasonal differencing
    if len(y_true_insample) < m + 1:
        return np.nan  # Not enough data to calculate seasonal difference

    scale = np.mean(np.abs(y_true_insample[m:] - y_true_insample[:-m]))
    if np.isclose(scale, 0) or np.isnan(scale):
        # Fallback to non-seasonal differencing
        if len(y_true_insample) < 2:
            return np.nan  # Not enough data for non-seasonal difference
        scale = np.mean(np.abs(y_true_insample[1:] - y_true_insample[:-1]))
        if np.isclose(scale, 0) or np.isnan(scale):
            return np.nan

    mase_value = np.nanmean(np.abs(y_true_h - y_pred_h)) / scale
    return float(mase_value)


def mae(y_true, y_pred):
    """Mean Absolute Error (MAE)"""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.nanmean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    """Root-Mean-Squared Error (RMSE)"""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.nanmean((y_true - y_pred) ** 2)))


def rmsse(y_true, y_pred, h: int, m: int = 1):
    """
    Root Mean Squared Scaled Error (RMSSE).
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    n = y_true.size

    if n <= h or n - h <= m:
        return np.nan

    y_true_insample = y_true[:-h]
    y_true_h = y_true[-h:]
    y_pred_h = y_pred[-h:]

    mask = ~np.isnan(y_pred_h)
    if mask.sum() == 0:
        return np.nan

    mse_forecast = np.mean((y_true_h[mask] - y_pred_h[mask]) ** 2)

    m = max(1, m)
    # Ensure there are enough points for differencing
    if len(y_true_insample) < m + 1:
        return np.nan

    denom = np.mean((y_true_insample[m:] - y_true_insample[:-m]) ** 2)
    if np.isclose(denom, 0) or np.isnan(denom):
        return np.nan

    return float(np.sqrt(mse_forecast / denom))
