"""Data loading and preprocessing for NASA CMAPSS turbofan dataset.

Handles loading raw text files, computing RUL labels with piecewise
linear clipping, normalization, and sliding window creation for LSTM input.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional

from configs.config import ALL_COLS, DROP_SENSORS, KEEP_SENSORS, SETTING_COLS, DataConfig


def load_dataset(cfg: DataConfig = DataConfig()) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load train, test, and RUL files for a given CMAPSS sub-dataset.

    Args:
        cfg: Data configuration with data_dir and dataset name.

    Returns:
        Tuple of (train_df, test_df, rul_true) where rul_true is a 1D array.
    """
    train_path = os.path.join(cfg.data_dir, f"train_{cfg.dataset}.txt")
    test_path = os.path.join(cfg.data_dir, f"test_{cfg.dataset}.txt")
    rul_path = os.path.join(cfg.data_dir, f"RUL_{cfg.dataset}.txt")

    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=ALL_COLS)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=ALL_COLS)
    rul_true = pd.read_csv(rul_path, sep=r"\s+", header=None).values.flatten()

    return train_df, test_df, rul_true


def add_rul_labels(df: pd.DataFrame, max_rul: int = 125) -> pd.DataFrame:
    """Compute Remaining Useful Life with piecewise linear clipping.

    For each engine unit, RUL decreases linearly from max_cycle to 0.
    Values above max_rul are clipped to max_rul (flat early life assumption).

    Args:
        df: DataFrame with 'unit_id' and 'cycle' columns.
        max_rul: Maximum RUL cap for piecewise linear model.

    Returns:
        DataFrame with added 'rul' column.
    """
    max_cycles = df.groupby("unit_id")["cycle"].max().reset_index()
    max_cycles.columns = ["unit_id", "max_cycle"]
    df = df.merge(max_cycles, on="unit_id")
    df["rul"] = df["max_cycle"] - df["cycle"]
    df["rul"] = df["rul"].clip(upper=max_rul)
    df.drop("max_cycle", axis=1, inplace=True)
    return df


def preprocess_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """Normalize sensor features and drop low-variance sensors.

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.

    Returns:
        Tuple of (processed_train, processed_test, fitted_scaler).
    """
    feature_cols = KEEP_SENSORS

    scaler = MinMaxScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    return train_df, test_df, scaler


def create_sequences(
    df: pd.DataFrame, window_size: int = 30, feature_cols: Optional[list] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding window sequences for LSTM input.

    For each engine, extracts overlapping windows of sensor readings
    paired with the RUL label at the end of each window.

    Args:
        df: Preprocessed DataFrame with 'unit_id', 'rul', and sensor columns.
        window_size: Number of time steps per sequence.
        feature_cols: Sensor columns to include. Defaults to KEEP_SENSORS.

    Returns:
        Tuple of (X, y) where X has shape (N, window_size, n_features)
        and y has shape (N,).
    """
    if feature_cols is None:
        feature_cols = KEEP_SENSORS

    sequences, labels = [], []
    for uid in df["unit_id"].unique():
        engine = df[df["unit_id"] == uid]
        data = engine[feature_cols].values
        rul = engine["rul"].values

        for i in range(len(data) - window_size + 1):
            sequences.append(data[i : i + window_size])
            labels.append(rul[i + window_size - 1])

    return np.array(sequences), np.array(labels)


def create_test_sequences(
    test_df: pd.DataFrame, window_size: int = 30, feature_cols: Optional[list] = None
) -> np.ndarray:
    """Extract the last window from each test engine for RUL prediction.

    Args:
        test_df: Test DataFrame.
        window_size: Sequence length.
        feature_cols: Sensor columns.

    Returns:
        Array of shape (n_engines, window_size, n_features).
    """
    if feature_cols is None:
        feature_cols = KEEP_SENSORS

    sequences = []
    for uid in test_df["unit_id"].unique():
        engine = test_df[test_df["unit_id"] == uid]
        data = engine[feature_cols].values
        if len(data) >= window_size:
            sequences.append(data[-window_size:])
        else:
            # Pad short sequences with zeros on the left
            padded = np.zeros((window_size, len(feature_cols)))
            padded[-len(data):] = data
            sequences.append(padded)

    return np.array(sequences)
