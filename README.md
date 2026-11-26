# 📈 Stock Price Prediction System (LSTM)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end deep learning system that predicts stock closing prices using an
LSTM (Long Short-Term Memory) neural network, with an interactive Streamlit
dashboard for visualizing results and technical indicators.

**Tech stack:** Python · TensorFlow / Keras (LSTM) · pandas · NumPy ·
scikit-learn · yfinance · Streamlit

---

## ✨ Features

| Feature | Details |
|---|---|
| **LSTM Forecasting** | Stacked 2-layer LSTM with dropout & early stopping |
| **Any Ticker** | Pulls live data from Yahoo Finance via `yfinance` |
| **Technical Indicators** | 20/50-day Moving Averages, RSI (14-day) |
| **Volume Chart** | Daily trading volume bar chart |
| **Metrics + Badge** | RMSE, MAE, MAPE with Excellent / Good / Fair / Poor quality badge |
| **N-Day Forecast** | Recursive forward forecast up to 30 trading days |
| **CSV Download** | One-click export of the forecast table |
| **Configurable** | Lookback window, train split, epochs, batch size — all adjustable |

---

## 🗂 Project structure

```
stock-price-predictor/
├── app.py             # Streamlit dashboard (entry point)
├── data_loader.py     # Data fetching + preprocessing
├── model.py           # LSTM model build / train / evaluate / forecast
├── requirements.txt   # Python dependencies
└── README.md
```

---

## 🚀 Setup

```bash
# 1. Clone the repository
git clone https://github.com/aadi-vaishya18/stock.git
cd stock

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Run the dashboard

```bash
streamlit run app.py
```

Opens in your browser at **http://localhost:8501**.

---

## 🖥 Using the dashboard

1. Enter a **ticker symbol** in the sidebar (e.g. `AAPL`, `MSFT`, `TSLA`, `RELIANCE.NS`).
2. Choose a **date range** (default: last 5 years).
3. Tune the model knobs — lookback window, train/test split, epochs, batch size, forecast horizon.
4. Toggle the **Technical Indicators** checkboxes to show/hide MA, Volume, and RSI overlays.
5. Click **🚀 Fetch Data & Train Model**. The app will:
   - Download historical OHLCV data from Yahoo Finance
   - Scale and window the closing-price series
   - Train the LSTM with early stopping
   - Evaluate on the held-out test set (RMSE / MAE / MAPE + quality badge)
   - Plot actual vs. predicted prices and a forward forecast
6. Use the **⬇️ Download Forecast as CSV** button to export results.

---

## 📊 How it works

```
Yahoo Finance  →  MinMaxScaler  →  Sliding-window sequences
    ↓
LSTM (50 units) → Dropout → LSTM (50 units) → Dropout → Dense → Dense(1)
    ↓
Inverse-transform predictions  →  RMSE / MAE / MAPE  →  N-day recursive forecast
```

---

## 📌 Notes & tips

- **Training time** scales with epochs, data range, and window size. Start with fewer epochs / shorter range to iterate quickly.
- **RMSE / MAE** are in the stock's native currency (e.g. USD); **MAPE** is a percentage.
- The forward forecast compounds uncertainty — treat longer horizons with more skepticism.
- For Indian stocks, append `.NS` (NSE) or `.BO` (BSE) to the ticker, e.g. `INFY.NS`.

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. Predictions are based solely
on historical price patterns and do **not** account for news, fundamentals, or
market sentiment. Nothing here constitutes financial advice.