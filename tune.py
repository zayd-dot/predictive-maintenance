"""Hyperparameter tuning for RUL prediction models.

Runs a grid of configurations across 3 model architectures:
1. Stacked LSTM (baseline)
2. BiLSTM + Attention
3. CNN-BiLSTM + Attention

Usage:
    python tune.py --dataset FD001
    python tune.py --dataset FD001 --quick   # Fast run with fewer configs
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime

from configs.config import DataConfig, KEEP_SENSORS
from src.data_preprocessing import (
    load_dataset, add_rul_labels, preprocess_features,
    create_sequences, create_test_sequences,
)
from src.models import build_lstm_model
from src.models_enhanced import (
    build_bilstm_attention, build_cnn_bilstm_attention, get_tuning_callbacks,
)
from src.evaluation import compute_metrics, plot_rul_predictions


# ---- Tuning Configurations ----

FULL_CONFIGS = [
    # (name, model_builder, kwargs, window_size, epochs, batch_size)
    ("LSTM_w30_u64",         "lstm",     {"lstm_units": [64, 32], "dropout": 0.3, "lr": 0.001},  30, 80, 64),
    ("LSTM_w50_u64",         "lstm",     {"lstm_units": [64, 32], "dropout": 0.3, "lr": 0.001},  50, 80, 64),
    ("LSTM_w30_u128",        "lstm",     {"lstm_units": [128, 64], "dropout": 0.3, "lr": 0.001}, 30, 80, 64),
    ("BiLSTM_Attn_w30",      "bilstm",   {"lstm_units": (128, 64), "dropout": 0.3, "lr": 0.001}, 30, 80, 64),
    ("BiLSTM_Attn_w50",      "bilstm",   {"lstm_units": (128, 64), "dropout": 0.3, "lr": 0.001}, 50, 80, 64),
    ("BiLSTM_Attn_w50_d04",  "bilstm",   {"lstm_units": (128, 64), "dropout": 0.4, "lr": 0.001}, 50, 80, 64),
    ("BiLSTM_Attn_w50_lr5e4","bilstm",   {"lstm_units": (128, 64), "dropout": 0.3, "lr": 0.0005},50, 80, 64),
    ("CNN_BiLSTM_w30",       "cnn_bilstm",{"cnn_filters": (64, 64), "lstm_units": (128, 64), "dropout": 0.3, "lr": 0.001}, 30, 80, 64),
    ("CNN_BiLSTM_w50",       "cnn_bilstm",{"cnn_filters": (64, 64), "lstm_units": (128, 64), "dropout": 0.3, "lr": 0.001}, 50, 80, 64),
    ("CNN_BiLSTM_w50_f128",  "cnn_bilstm",{"cnn_filters": (128, 64), "lstm_units": (128, 64), "dropout": 0.3, "lr": 0.001}, 50, 100, 64),
    ("CNN_BiLSTM_w50_b32",   "cnn_bilstm",{"cnn_filters": (64, 64), "lstm_units": (128, 64), "dropout": 0.3, "lr": 0.001}, 50, 80, 32),
]

QUICK_CONFIGS = [
    ("LSTM_baseline",        "lstm",     {"lstm_units": [64, 32], "dropout": 0.3, "lr": 0.001},  30, 50, 64),
    ("BiLSTM_Attn_w50",      "bilstm",   {"lstm_units": (128, 64), "dropout": 0.3, "lr": 0.001}, 50, 50, 64),
    ("CNN_BiLSTM_w50",       "cnn_bilstm",{"cnn_filters": (64, 64), "lstm_units": (128, 64), "dropout": 0.3, "lr": 0.001}, 50, 50, 64),
]


def build_model(arch, input_shape, kwargs):
    """Build model by architecture name."""
    if arch == "lstm":
        from src.models import build_lstm_model as _build
        from configs.config import ModelConfig
        cfg = ModelConfig()
        cfg.lstm_units = kwargs.get("lstm_units", [64, 32])
        cfg.dropout = kwargs.get("dropout", 0.3)
        cfg.learning_rate = kwargs.get("lr", 0.001)
        return _build(input_shape, cfg)
    elif arch == "bilstm":
        return build_bilstm_attention(input_shape, **kwargs)
    elif arch == "cnn_bilstm":
        return build_cnn_bilstm_attention(input_shape, **kwargs)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def run_tuning(cfg_data: DataConfig, configs: list) -> pd.DataFrame:
    """Run all tuning configurations and return results.

    Args:
        cfg_data: Data configuration.
        configs: List of tuning configurations.

    Returns:
        DataFrame with results for each configuration.
    """
    os.makedirs("results/tuning", exist_ok=True)

    # Load data once
    print(f"Loading CMAPSS {cfg_data.dataset}...")
    train_df, test_df, rul_true = load_dataset(cfg_data)
    train_df = add_rul_labels(train_df, max_rul=cfg_data.max_rul)
    train_df, test_df, scaler = preprocess_features(train_df, test_df)

    n_features = len(KEEP_SENSORS)
    results = []

    for name, arch, kwargs, window, epochs, batch_size in configs:
        print(f"\n{'='*60}")
        print(f"Config: {name} | arch={arch} | window={window} | epochs={epochs}")
        print(f"{'='*60}")

        # Create sequences for this window size
        X_train, y_train = create_sequences(train_df, window_size=window)
        X_test = create_test_sequences(test_df, window_size=window)

        # Train/val split
        split_idx = int(len(X_train) * 0.8)
        X_val, y_val = X_train[split_idx:], y_train[split_idx:]
        X_tr, y_tr = X_train[:split_idx], y_train[:split_idx]

        input_shape = (window, n_features)

        try:
            model = build_model(arch, input_shape, kwargs)
            params = model.count_params()
            print(f"Parameters: {params:,}")

            history = model.fit(
                X_tr, y_tr,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=get_tuning_callbacks(patience=15),
                verbose=0,
            )

            # Evaluate
            y_pred = model.predict(X_test, verbose=0).flatten()
            metrics = compute_metrics(rul_true, y_pred)

            actual_epochs = len(history.history["loss"])
            best_val_loss = min(history.history["val_loss"])

            result = {
                "name": name,
                "arch": arch,
                "window": window,
                "epochs_run": actual_epochs,
                "batch_size": batch_size,
                "params": params,
                "best_val_loss": round(best_val_loss, 2),
                **{k: round(v, 2) for k, v in metrics.items()},
            }
            results.append(result)

            print(f"  RMSE: {metrics['RMSE']:.2f} | MAE: {metrics['MAE']:.2f} | "
                  f"R2: {metrics['R2']:.3f} | NASA: {metrics['NASA_Score']:.0f} | "
                  f"Epochs: {actual_epochs} | Params: {params:,}")

            # Save predictions for the best model later
            np.save(f"results/tuning/{name}_predictions.npy", y_pred)

            # Clear memory
            del model
            tf.keras.backend.clear_session()

        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"name": name, "arch": arch, "RMSE": None, "error": str(e)})

    # Results table
    df = pd.DataFrame(results)
    df = df.sort_values("RMSE", na_position="last")

    print(f"\n{'='*80}")
    print("TUNING RESULTS (sorted by RMSE)")
    print(f"{'='*80}")
    print(df.to_string(index=False))

    # Save results
    df.to_csv("results/tuning/tuning_results.csv", index=False)

    # Plot best model predictions
    if len(df.dropna(subset=["RMSE"])) > 0:
        best_name = df.iloc[0]["name"]
        best_preds = np.load(f"results/tuning/{best_name}_predictions.npy")
        plot_rul_predictions(
            rul_true, best_preds,
            title=f"Best: {best_name}",
            save_path="results/tuning/best_predictions.png",
        )

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune RUL prediction models")
    parser.add_argument("--dataset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--quick", action="store_true", help="Run only 3 key configs for speed")
    args = parser.parse_args()

    cfg_data = DataConfig(dataset=args.dataset)
    configs = QUICK_CONFIGS if args.quick else FULL_CONFIGS

    print(f"Running {len(configs)} configurations on {args.dataset}")
    print(f"Mode: {'QUICK' if args.quick else 'FULL'}\n")

    results = run_tuning(cfg_data, configs)
