"""
app.py
------
Interactive Streamlit dashboard for the Stock Price Prediction System.

Run with:
    streamlit run app.py

Pipeline (triggered by the sidebar "Fetch Data & Train Model" button):
  1. Data ingestion  -> data_loader.fetch_stock_data (yfinance)
  2. Preprocessing    -> data_loader.prepare_train_test_data (MinMaxScaler + sequences)
  3. Model training    -> model.build_lstm_model / model.train_model (TensorFlow/Keras LSTM)
  4. Evaluation        -> model.evaluate_model (RMSE / MAE / MAPE)
  5. Visualization      -> Streamlit charts of actual vs. predicted price, plus a
                          forward price forecast
"""

import datetime as dt
import io

import numpy as np
import pandas as pd
import streamlit as st

from data_loader import fetch_stock_data, prepare_train_test_data
from model import build_lstm_model, evaluate_model, predict_future, train_model

st.set_page_config(page_title="Stock Price Predictor (LSTM)", page_icon="📈", layout="wide")


# --------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index for a price series."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def mape_to_badge(mape: float) -> tuple[str, str]:
    """Return (label, colour) based on MAPE thresholds."""
    if mape < 3:
        return "🟢 Excellent (MAPE < 3%)", "green"
    elif mape < 7:
        return "🟡 Good (MAPE < 7%)", "orange"
    elif mape < 15:
        return "🟠 Fair (MAPE < 15%)", "orange"
    else:
        return "🔴 Poor (MAPE ≥ 15%)", "red"


@st.cache_data(show_spinner=False)
def cached_fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Thin caching wrapper around fetch_stock_data so re-running the app with
    the same inputs doesn't hit Yahoo Finance again."""
    return fetch_stock_data(ticker, start, end)


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialise a DataFrame to UTF-8 CSV bytes for st.download_button."""
    buf = io.StringIO()
    df.to_csv(buf)
    return buf.getvalue().encode()


# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuration")

ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper().strip()

default_start = dt.date.today() - dt.timedelta(days=5 * 365)
start_date = st.sidebar.date_input("Start Date", value=default_start)
end_date = st.sidebar.date_input("End Date", value=dt.date.today())

window_size = st.sidebar.slider("Lookback Window (days)", min_value=20, max_value=120, value=60, step=10)
train_ratio = st.sidebar.slider("Train / Test Split", min_value=0.6, max_value=0.9, value=0.8, step=0.05)
epochs = st.sidebar.slider("Training Epochs", min_value=5, max_value=100, value=25, step=5)
batch_size = st.sidebar.selectbox("Batch Size", options=[16, 32, 64], index=1)
forecast_days = st.sidebar.slider("Forecast Horizon (days)", min_value=1, max_value=30, value=7)

st.sidebar.markdown("---")
st.sidebar.subheader("📉 Technical Indicators")
show_ma20 = st.sidebar.checkbox("20-day MA", value=True)
show_ma50 = st.sidebar.checkbox("50-day MA", value=True)
show_volume = st.sidebar.checkbox("Volume Chart", value=True)
show_rsi = st.sidebar.checkbox("RSI (14)", value=True)

run_button = st.sidebar.button("🚀 Fetch Data & Train Model", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Tech stack: Python · TensorFlow/Keras (LSTM) · pandas · NumPy · "
    "scikit-learn · yfinance · Streamlit"
)

# --------------------------------------------------------------------------
# Main content
# --------------------------------------------------------------------------
st.title("📈 Stock Price Prediction System")
st.caption("LSTM-based time-series forecasting trained on real historical Yahoo Finance data.")

if "trained" not in st.session_state:
    st.session_state.trained = False

if run_button:
    if start_date >= end_date:
        st.error("Start date must be before end date.")
    else:
        try:
            with st.spinner(f"Fetching historical data for {ticker}..."):
                raw_data = cached_fetch(ticker, str(start_date), str(end_date))

            close_prices = raw_data["Close"].values

            X_train, y_train, X_test, y_test, scaler = prepare_train_test_data(
                close_prices, window_size=window_size, train_ratio=train_ratio
            )

            if len(X_train) == 0 or len(X_test) == 0:
                st.error(
                    "Not enough historical data for the chosen lookback window and split ratio. "
                    "Try a longer date range, a smaller window, or a lower train/test split."
                )
            else:
                with st.spinner("Building and training the LSTM model... this can take a minute."):
                    model = build_lstm_model(input_shape=(X_train.shape[1], 1))
                    history = train_model(
                        model, X_train, y_train, epochs=epochs, batch_size=batch_size
                    )

                with st.spinner("Evaluating on the held-out test set..."):
                    results = evaluate_model(model, X_test, y_test, scaler)

                with st.spinner(f"Forecasting the next {forecast_days} trading day(s)..."):
                    last_window = X_test[-1].flatten()
                    future_prices = predict_future(model, last_window, scaler, n_days=forecast_days)

                test_dates = raw_data.index[-len(results["actual"]):]

                st.session_state.trained = True
                st.session_state.raw_data = raw_data
                st.session_state.results = results
                st.session_state.test_dates = test_dates
                st.session_state.future_prices = future_prices
                st.session_state.history = history.history
                st.session_state.ticker = ticker
                st.session_state.forecast_days = forecast_days

        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - surface any unexpected error to the user
            st.error(f"Something went wrong while fetching or training: {exc}")

if st.session_state.trained:
    raw_data = st.session_state.raw_data
    results = st.session_state.results
    test_dates = st.session_state.test_dates
    future_prices = st.session_state.future_prices
    history = st.session_state.history
    ticker_used = st.session_state.ticker
    close = raw_data["Close"]

    # ------------------------------------------------------------------
    # 1. Historical price + moving averages
    # ------------------------------------------------------------------
    st.subheader(f"Historical Closing Price — {ticker_used}")
    price_df = pd.DataFrame({"Close": close})
    if show_ma20:
        price_df["MA 20"] = close.rolling(20).mean()
    if show_ma50:
        price_df["MA 50"] = close.rolling(50).mean()
    st.line_chart(price_df, use_container_width=True)

    # ------------------------------------------------------------------
    # 2. Volume chart
    # ------------------------------------------------------------------
    if show_volume and "Volume" in raw_data.columns:
        st.subheader("📊 Trading Volume")
        st.bar_chart(raw_data["Volume"], use_container_width=True)

    # ------------------------------------------------------------------
    # 3. Model performance metrics + confidence badge
    # ------------------------------------------------------------------
    st.subheader("🏆 Model Performance (Test Set)")
    badge_label, badge_colour = mape_to_badge(results["mape"])
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RMSE", f"${results['rmse']:.2f}")
    col2.metric("MAE", f"${results['mae']:.2f}")
    col3.metric("MAPE", f"{results['mape']:.2f}%")
    col4.markdown(f"**Model Quality**  \n:{badge_colour}[{badge_label}]")

    # ------------------------------------------------------------------
    # 4. Predicted vs. actual
    # ------------------------------------------------------------------
    st.subheader("🔍 Predicted vs. Actual Price")
    comparison_df = pd.DataFrame(
        {"Actual": results["actual"], "Predicted": results["predictions"]},
        index=test_dates,
    )
    st.line_chart(comparison_df, use_container_width=True)

    # ------------------------------------------------------------------
    # 5. RSI indicator
    # ------------------------------------------------------------------
    if show_rsi:
        rsi_series = compute_rsi(close)
        st.subheader("📐 RSI (14-day Relative Strength Index)")
        rsi_df = pd.DataFrame({"RSI": rsi_series})
        st.line_chart(rsi_df, use_container_width=True)
        st.caption("RSI > 70 → overbought  |  RSI < 30 → oversold")

    # ------------------------------------------------------------------
    # 6. Training loss curve
    # ------------------------------------------------------------------
    with st.expander("📉 Training Loss Curve"):
        loss_data = {"Training Loss": history["loss"]}
        if "val_loss" in history:
            loss_data["Validation Loss"] = history["val_loss"]
        st.line_chart(pd.DataFrame(loss_data))

    # ------------------------------------------------------------------
    # 7. Forward forecast + download
    # ------------------------------------------------------------------
    st.subheader(f"🔮 {st.session_state.forecast_days}-Day Forward Forecast")
    last_date = raw_data.index[-1]
    forecast_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=len(future_prices))
    forecast_df = pd.DataFrame({"Forecasted Close": future_prices}, index=forecast_dates)
    st.line_chart(forecast_df, use_container_width=True)
    st.dataframe(forecast_df.style.format("{:.2f}"), use_container_width=True)

    st.download_button(
        label="⬇️ Download Forecast as CSV",
        data=df_to_csv_bytes(forecast_df),
        file_name=f"{ticker_used}_forecast_{dt.date.today()}.csv",
        mime="text/csv",
    )

    # ------------------------------------------------------------------
    # 8. Raw data viewer
    # ------------------------------------------------------------------
    with st.expander("View Raw Historical Data (last 200 rows)"):
        st.dataframe(raw_data.tail(200), use_container_width=True)

    st.caption(
        "⚠️ Educational project only. Predictions are based purely on historical price "
        "patterns and should not be used as financial advice."
    )
else:
    st.info("👈 Set your parameters in the sidebar and click **Fetch Data & Train Model** to get started.")