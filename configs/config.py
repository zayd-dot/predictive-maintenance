"""Configuration for NASA CMAPSS Predictive Maintenance pipeline."""

from dataclasses import dataclass, field
from typing import List

SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]
ALL_COLS = ["unit_id", "cycle"] + SETTING_COLS + SENSOR_COLS

# Sensors with near-zero variance (uninformative) — drop these
DROP_SENSORS = ["s_1", "s_5", "s_6", "s_10", "s_16", "s_18", "s_19"]

# Sensors to keep after variance filtering
KEEP_SENSORS = [s for s in SENSOR_COLS if s not in DROP_SENSORS]


@dataclass
class DataConfig:
    data_dir: str = "data"
    dataset: str = "FD001"  # FD001, FD002, FD003, FD004
    max_rul: int = 125  # Piecewise linear RUL cap
    window_size: int = 30  # Sliding window length for LSTM
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class ModelConfig:
    # LSTM
    lstm_units: List[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.3
    learning_rate: float = 0.001
    epochs: int = 50
    batch_size: int = 64
    patience: int = 10  # Early stopping patience

    # Random Forest baseline
    rf_n_estimators: int = 200
    rf_max_depth: int = 15
