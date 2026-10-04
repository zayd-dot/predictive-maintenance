"""Evaluation metrics and visualization for RUL prediction."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Optional


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics for RUL prediction.

    Args:
        y_true: Ground truth RUL values.
        y_pred: Predicted RUL values.

    Returns:
        Dictionary with RMSE, MAE, R2, and NASA scoring function.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    score = nasa_score(y_true, y_pred)

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "NASA_Score": score}


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute NASA's asymmetric scoring function for RUL prediction.

    Penalizes late predictions (under-estimation) more heavily than
    early predictions (over-estimation).

    Args:
        y_true: Ground truth RUL.
        y_pred: Predicted RUL.

    Returns:
        Total NASA score (lower is better).
    """
    diff = y_pred - y_true
    score = 0.0
    for d in diff:
        if d < 0:
            score += np.exp(-d / 13.0) - 1  # Late prediction penalty
        else:
            score += np.exp(d / 10.0) - 1  # Early prediction penalty
    return score


def plot_training_history(history, save_path: Optional[str] = None) -> None:
    """Plot training and validation loss curves.

    Args:
        history: Keras History object.
        save_path: If provided, save the figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history["loss"], label="Train")
    ax1.plot(history.history["val_loss"], label="Validation")
    ax1.set_title("Loss (MSE)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history["mae"], label="Train")
    ax2.plot(history.history["val_mae"], label="Validation")
    ax2.set_title("Mean Absolute Error")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MAE")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_rul_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "RUL Predictions",
    save_path: Optional[str] = None,
) -> None:
    """Plot predicted vs actual RUL with error distribution.

    Args:
        y_true: Ground truth RUL.
        y_pred: Predicted RUL.
        title: Plot title prefix.
        save_path: If provided, save the figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=15, edgecolors="none")
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Perfect")
    axes[0].set_xlabel("Actual RUL")
    axes[0].set_ylabel("Predicted RUL")
    axes[0].set_title(f"{title}: Predicted vs Actual")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Error distribution
    errors = y_pred - y_true
    axes[1].hist(errors, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Prediction Error (cycles)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"{title}: Error Distribution")
    axes[1].grid(True, alpha=0.3)

    # Per-engine prediction
    n_show = min(30, len(y_true))
    idx = np.arange(n_show)
    axes[2].bar(idx - 0.2, y_true[:n_show], 0.4, label="Actual", alpha=0.8)
    axes[2].bar(idx + 0.2, y_pred[:n_show], 0.4, label="Predicted", alpha=0.8)
    axes[2].set_xlabel("Engine Index")
    axes[2].set_ylabel("RUL (cycles)")
    axes[2].set_title(f"{title}: Per-Engine (first {n_show})")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_sensor_degradation(df, sensors: list, engine_id: int = 1, save_path: Optional[str] = None) -> None:
    """Plot sensor readings over time for a single engine.

    Args:
        df: DataFrame with unit_id, cycle, and sensor columns.
        sensors: List of sensor column names to plot.
        engine_id: Engine unit ID to visualize.
        save_path: If provided, save the figure.
    """
    engine = df[df["unit_id"] == engine_id]
    n = len(sensors)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 3 * rows))
    axes = axes.flatten()

    for i, sensor in enumerate(sensors):
        axes[i].plot(engine["cycle"], engine[sensor], linewidth=0.8)
        axes[i].set_title(sensor, fontsize=10)
        axes[i].set_xlabel("Cycle")
        axes[i].grid(True, alpha=0.3)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f"Sensor Degradation — Engine {engine_id}", fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
