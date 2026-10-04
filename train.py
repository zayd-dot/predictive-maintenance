"""Train and evaluate RUL prediction models on NASA CMAPSS data.

Usage:
    python train.py                       # LSTM on FD001
    python train.py --dataset FD002       # LSTM on FD002
    python train.py --baseline            # Also train RF baseline
"""

import os
import argparse
import numpy as np

from configs.config import DataConfig, ModelConfig, KEEP_SENSORS
from src.data_preprocessing import (
    load_dataset, add_rul_labels, preprocess_features,
    create_sequences, create_test_sequences,
)
from src.models import build_lstm_model, get_callbacks, build_rf_baseline, flatten_sequences
from src.evaluation import (
    compute_metrics, plot_training_history, plot_rul_predictions, plot_sensor_degradation,
)


def main(cfg_data: DataConfig, cfg_model: ModelConfig, run_baseline: bool = False) -> None:
    os.makedirs("results", exist_ok=True)

    # Load data
    print(f"Loading CMAPSS {cfg_data.dataset}...")
    train_df, test_df, rul_true = load_dataset(cfg_data)
    print(f"Train engines: {train_df['unit_id'].nunique()} | Test engines: {test_df['unit_id'].nunique()}")

    # Add RUL labels to training data
    train_df = add_rul_labels(train_df, max_rul=cfg_data.max_rul)

    # Normalize
    train_df, test_df, scaler = preprocess_features(train_df, test_df)

    # Visualize sensor degradation for engine 1
    plot_sensor_degradation(train_df, KEEP_SENSORS, engine_id=1, save_path="results/sensor_degradation.png")

    # Create sequences
    print(f"Creating sequences (window={cfg_data.window_size})...")
    X_train, y_train = create_sequences(train_df, window_size=cfg_data.window_size)
    X_test = create_test_sequences(test_df, window_size=cfg_data.window_size)
    print(f"Train sequences: {X_train.shape} | Test engines: {X_test.shape}")

    # Split train into train/val
    split_idx = int(len(X_train) * (1 - cfg_data.test_size))
    X_val, y_val = X_train[split_idx:], y_train[split_idx:]
    X_train, y_train = X_train[:split_idx], y_train[:split_idx]

    # ---- LSTM ----
    print("\n=== LSTM Model ===")
    model = build_lstm_model(input_shape=(cfg_data.window_size, len(KEEP_SENSORS)), cfg=cfg_model)
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg_model.epochs,
        batch_size=cfg_model.batch_size,
        callbacks=get_callbacks(cfg_model),
        verbose=1,
    )

    plot_training_history(history, save_path="results/lstm_training.png")

    y_pred_lstm = model.predict(X_test).flatten()
    metrics_lstm = compute_metrics(rul_true, y_pred_lstm)
    print("\nLSTM Results:")
    for k, v in metrics_lstm.items():
        print(f"  {k}: {v:.2f}")

    plot_rul_predictions(rul_true, y_pred_lstm, title="LSTM", save_path="results/lstm_predictions.png")

    model.save("results/lstm_model.keras")

    # ---- Random Forest Baseline ----
    if run_baseline:
        print("\n=== Random Forest Baseline ===")
        X_train_flat = flatten_sequences(X_train)
        X_val_flat = flatten_sequences(X_val)
        X_test_flat = flatten_sequences(X_test)

        rf = build_rf_baseline(cfg_model)
        rf.fit(X_train_flat, y_train)

        y_pred_rf = rf.predict(X_test_flat)
        metrics_rf = compute_metrics(rul_true, y_pred_rf)
        print("\nRandom Forest Results:")
        for k, v in metrics_rf.items():
            print(f"  {k}: {v:.2f}")

        plot_rul_predictions(rul_true, y_pred_rf, title="Random Forest", save_path="results/rf_predictions.png")

        # Comparison
        print("\n=== Comparison ===")
        print(f"{'Metric':<15} {'LSTM':>10} {'RF':>10}")
        print("-" * 35)
        for k in metrics_lstm:
            print(f"{k:<15} {metrics_lstm[k]:>10.2f} {metrics_rf[k]:>10.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RUL prediction models")
    parser.add_argument("--dataset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--baseline", action="store_true", help="Also train RF baseline")
    args = parser.parse_args()

    cfg_data = DataConfig(dataset=args.dataset, window_size=args.window)
    cfg_model = ModelConfig(epochs=args.epochs, batch_size=args.batch_size)
    main(cfg_data, cfg_model, run_baseline=args.baseline)
