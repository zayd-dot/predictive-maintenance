"""Model architectures for RUL prediction.

Provides an LSTM-based deep learning model and a Random Forest baseline,
both targeting Remaining Useful Life regression.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from configs.config import ModelConfig


def build_lstm_model(input_shape: Tuple[int, int], cfg: ModelConfig = ModelConfig()) -> Sequential:
    """Build a stacked LSTM model for RUL regression.

    Args:
        input_shape: (window_size, n_features).
        cfg: Model configuration.

    Returns:
        Compiled Keras Sequential model.
    """
    model = Sequential([
        LSTM(cfg.lstm_units[0], return_sequences=True, input_shape=input_shape),
        Dropout(cfg.dropout),
        BatchNormalization(),
        LSTM(cfg.lstm_units[1], return_sequences=False),
        Dropout(cfg.dropout),
        BatchNormalization(),
        Dense(32, activation="relu"),
        Dropout(cfg.dropout / 2),
        Dense(1),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def get_callbacks(cfg: ModelConfig = ModelConfig()):
    """Create training callbacks for early stopping and LR reduction.

    Args:
        cfg: Model configuration.

    Returns:
        List of Keras callbacks.
    """
    return [
        EarlyStopping(monitor="val_loss", patience=cfg.patience, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=cfg.patience // 2, min_lr=1e-6, verbose=1),
    ]


def build_rf_baseline(cfg: ModelConfig = ModelConfig()) -> RandomForestRegressor:
    """Build a Random Forest baseline model.

    Args:
        cfg: Model configuration.

    Returns:
        Configured RandomForestRegressor.
    """
    return RandomForestRegressor(
        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth,
        random_state=42,
        n_jobs=-1,
    )


def flatten_sequences(X: np.ndarray) -> np.ndarray:
    """Flatten 3D sequences to 2D for tree-based models.

    Uses statistical aggregation (mean, std, min, max, last) over the
    time dimension to create fixed-length feature vectors.

    Args:
        X: Array of shape (N, window_size, n_features).

    Returns:
        Array of shape (N, n_features * 5).
    """
    return np.concatenate([
        X.mean(axis=1),
        X.std(axis=1),
        X.min(axis=1),
        X.max(axis=1),
        X[:, -1, :],  # last time step
    ], axis=1)
