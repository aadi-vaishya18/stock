"""
model.py
--------
Deep learning architecture, training, and evaluation for the Stock Price
Prediction System.

Responsibilities:
  * Build a stacked LSTM regression network in TensorFlow/Keras that captures
    sequential dependencies in the price series.
  * Train the network with early stopping to avoid overfitting.
  * Evaluate performance on the held-out test set using RMSE (plus MAE / MAPE
    for extra context), after inverse-transforming predictions back to real
    price scale.
  * Recursively forecast N days beyond the most recent known data point.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential


def build_lstm_model(input_shape: tuple, units: int = 50, dropout_rate: float = 0.2) -> Sequential:
    """
    Build and compile a two-layer stacked LSTM network for next-step price
    regression.

    input_shape: (timesteps, n_features) -> e.g. (window_size, 1)
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(max(units // 2, 1), activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_model(
    model: Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 25,
    batch_size: int = 32,
    validation_split: float = 0.1,
    patience: int = 5,
):
    """
    Train the model with early stopping on validation loss. Returns the Keras
    History object (history.history contains 'loss' / 'val_loss' lists).
    """
    early_stop = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stop],
        verbose=0,
    )
    return history


def evaluate_model(model: Sequential, X_test: np.ndarray, y_test: np.ndarray, scaler: MinMaxScaler) -> dict:
    """
    Run predictions on the test set, inverse-transform both predictions and
    ground truth back to real price units, and compute RMSE / MAE / MAPE.
    """
    predictions_scaled = model.predict(X_test, verbose=0)
    predictions = scaler.inverse_transform(predictions_scaled)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = float(np.sqrt(mean_squared_error(actual, predictions)))
    mae = float(np.mean(np.abs(actual - predictions)))
    mape = float(np.mean(np.abs((actual - predictions) / actual)) * 100)

    return {
        "predictions": predictions.flatten(),
        "actual": actual.flatten(),
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
    }


def predict_future(
    model: Sequential,
    last_window_scaled: np.ndarray,
    scaler: MinMaxScaler,
    n_days: int = 7,
) -> np.ndarray:
    """
    Recursively forecast `n_days` beyond the last known data point.

    `last_window_scaled` must be a 1-D array of the most recent `window_size`
    *scaled* closing prices (i.e. the last training/test sequence). Each step
    the model's own prediction is fed back in as the newest input, sliding the
    window forward by one day.
    """
    window_size = last_window_scaled.shape[0]
    current_seq = last_window_scaled.reshape(1, window_size, 1)
    future_scaled = []

    for _ in range(n_days):
        next_pred = model.predict(current_seq, verbose=0)[0, 0]
        future_scaled.append(next_pred)
        next_pred_reshaped = np.array(next_pred).reshape(1, 1, 1)
        current_seq = np.append(current_seq[:, 1:, :], next_pred_reshaped, axis=1)

    future_scaled = np.array(future_scaled).reshape(-1, 1)
    future_prices = scaler.inverse_transform(future_scaled)
    return future_prices.flatten()