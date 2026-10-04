"""Enhanced models for RUL prediction.

Adds:
1. Bidirectional LSTM with temporal attention
2. CNN-LSTM hybrid (1D conv feature extraction → BiLSTM → attention)
3. Hyperparameter tuning script
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout, BatchNormalization,
    Conv1D, MaxPooling1D, Flatten, Concatenate, Multiply, Permute,
    RepeatVector, Lambda, Attention, GlobalAveragePooling1D, Add,
    LayerNormalization,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2


def temporal_attention_layer(inputs):
    """Attention mechanism that learns to focus on important time steps.

    Args:
        inputs: Tensor of shape (batch, timesteps, features).

    Returns:
        Context vector of shape (batch, features).
    """
    # Score each time step
    score = Dense(1, activation="tanh")(inputs)       # (batch, timesteps, 1)
    weights = tf.keras.layers.Softmax(axis=1)(score)   # (batch, timesteps, 1)
    context = tf.reduce_sum(inputs * weights, axis=1)  # (batch, features)
    return context, weights


def build_bilstm_attention(input_shape, lstm_units=(128, 64), dropout=0.3, lr=0.001):
    """Bidirectional LSTM with temporal attention.

    Architecture:
        Input → BiLSTM(128) → Dropout → BiLSTM(64) → Attention → Dense → RUL

    Args:
        input_shape: (window_size, n_features).
        lstm_units: Units for each BiLSTM layer.
        dropout: Dropout rate.
        lr: Learning rate.

    Returns:
        Compiled Keras Model.
    """
    inputs = Input(shape=input_shape)

    # BiLSTM layers
    x = Bidirectional(LSTM(lstm_units[0], return_sequences=True, kernel_regularizer=l2(1e-4)))(inputs)
    x = Dropout(dropout)(x)
    x = BatchNormalization()(x)

    x = Bidirectional(LSTM(lstm_units[1], return_sequences=True, kernel_regularizer=l2(1e-4)))(x)
    x = Dropout(dropout)(x)
    x = BatchNormalization()(x)

    # Temporal attention
    context, attn_weights = temporal_attention_layer(x)

    # Dense head
    x = Dense(64, activation="relu", kernel_regularizer=l2(1e-4))(context)
    x = Dropout(dropout / 2)(x)
    x = Dense(32, activation="relu")(x)
    output = Dense(1)(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_cnn_bilstm_attention(input_shape, cnn_filters=(64, 64), lstm_units=(128, 64), dropout=0.3, lr=0.001):
    """CNN-BiLSTM hybrid with attention.

    Architecture:
        Input → Conv1D(64) → Conv1D(64) → MaxPool → BiLSTM(128) → BiLSTM(64) → Attention → Dense → RUL

    The CNN front-end extracts local temporal patterns (sensor trends,
    transitions), while BiLSTM captures long-range dependencies.

    Args:
        input_shape: (window_size, n_features).
        cnn_filters: Filters for each Conv1D layer.
        lstm_units: Units for each BiLSTM layer.
        dropout: Dropout rate.
        lr: Learning rate.

    Returns:
        Compiled Keras Model.
    """
    inputs = Input(shape=input_shape)

    # CNN feature extraction
    x = Conv1D(cnn_filters[0], kernel_size=3, activation="relu", padding="same", kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Conv1D(cnn_filters[1], kernel_size=3, activation="relu", padding="same", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(dropout / 2)(x)

    # BiLSTM sequence modeling
    x = Bidirectional(LSTM(lstm_units[0], return_sequences=True, kernel_regularizer=l2(1e-4)))(x)
    x = Dropout(dropout)(x)
    x = BatchNormalization()(x)

    x = Bidirectional(LSTM(lstm_units[1], return_sequences=True, kernel_regularizer=l2(1e-4)))(x)
    x = Dropout(dropout)(x)
    x = BatchNormalization()(x)

    # Temporal attention
    context, attn_weights = temporal_attention_layer(x)

    # Dense head
    x = Dense(64, activation="relu", kernel_regularizer=l2(1e-4))(context)
    x = Dropout(dropout / 2)(x)
    x = Dense(32, activation="relu")(x)
    output = Dense(1)(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )
    return model


def get_tuning_callbacks(patience=15):
    """Callbacks for tuning runs with more patience.

    Args:
        patience: Early stopping patience.

    Returns:
        List of Keras callbacks.
    """
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience // 3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
