"""
data_loader.py
--------------
Data ingestion and preprocessing utilities for the Stock Price Prediction System.

Responsibilities:
  * Fetch historical OHLCV data for a given ticker from Yahoo Finance (yfinance).
  * Scale the closing price series with MinMaxScaler.
  * Convert the scaled series into supervised-learning sequences (sliding windows)
    suitable for training an LSTM network.
  * Perform a chronological (non-shuffled) train/test split, taking care to avoid
    data leakage by fitting the scaler on the training portion only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download historical daily OHLCV data for `ticker` between `start_date` and
    `end_date` (both 'YYYY-MM-DD' strings) using yfinance.

    Returns a cleaned DataFrame indexed by date with at least a 'Close' column.
    Raises ValueError if no data is returned (e.g. bad ticker or date range).
    """
    ticker = ticker.strip().upper()
    data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

    # Newer versions of yfinance can return MultiIndex columns (Ticker, Field)
    # even for a single symbol -- flatten them to plain field names.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if data is None or data.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}' between {start_date} and {end_date}. "
            "Double-check the ticker symbol and date range."
        )

    data = data.dropna()
    return data


def create_sequences(scaled_data: np.ndarray, window_size: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a scaled 1-D price series (shape: [n, 1]) into supervised sequences:
    X[i] = the `window_size` values preceding day i, y[i] = the value on day i.

    Returns X reshaped to (samples, window_size, 1) as required by Keras LSTM layers.
    """
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y


def prepare_train_test_data(
    prices: np.ndarray,
    window_size: int = 60,
    train_ratio: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Full preprocessing pipeline:
      1. Chronologically split raw close prices into train/test.
      2. Fit MinMaxScaler on the TRAIN portion only (avoids look-ahead bias),
         then transform both train and test portions with it.
      3. Build sliding-window sequences for training.
      4. Build sliding-window sequences for testing, prepending the last
         `window_size` scaled training points so the very first test sample
         has enough history.

    Returns: X_train, y_train, X_test, y_test, fitted_scaler
    """
    prices = np.asarray(prices).reshape(-1, 1)
    split_idx = int(len(prices) * train_ratio)

    train_prices = prices[:split_idx]
    test_prices = prices[split_idx:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_prices)
    test_scaled = scaler.transform(test_prices) if len(test_prices) > 0 else np.empty((0, 1))

    X_train, y_train = create_sequences(train_scaled, window_size)

    # Prepend the tail of the training data so the first test sequence is valid.
    full_test_input = np.concatenate((train_scaled[-window_size:], test_scaled), axis=0)
    X_test, y_test = create_sequences(full_test_input, window_size)

    return X_train, y_train, X_test, y_test, scaler