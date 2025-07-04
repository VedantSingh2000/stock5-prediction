# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from pandas import Timestamp
import gdown
import os

# --- Download large model file from Google Drive ---
MODEL_FILE_ID = "1Thh23UJpwKsyluzwipRM37p1F5xit1hg"
MODEL_DEST = "multi_stock_models.pkl"

if not os.path.exists(MODEL_DEST):
    gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_DEST, quiet=False)

# --- Load Pre-trained Models ---
models = joblib.load('multi_stock_models.pkl')
scalers = joblib.load('multi_stock_scalers.pkl')
errors = joblib.load('multi_stock_errors.pkl')

# --- App Config and Style ---
st.set_page_config(page_title="Stock Prediction App", layout="wide")
st.markdown("""
    <style>
        body, .stApp {
            background-color: #121212;
            color: #FFFFFF;
            font-family: 'Segoe UI', sans-serif;
        }
        .metric-label { font-weight: bold; }
        .prediction-highlight {
            background-color: #1e1e1e;
            border-left: 5px solid #00acc1;
            padding: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Inputs ---
st.sidebar.header("🔧 Configuration")
tickers = {
    'TATAMOTORS.NS': 'Tata Motors',
    'UNITDSPR.NS': 'United Spirits',
    'PETRONET.NS': 'Petronet LNG',
    'COLPAL.NS': 'Colgate-Palmolive',
    'BEL.NS': 'Bharat Electronics'
}
selected_ticker = st.sidebar.selectbox("Choose a stock:", list(tickers.keys()), format_func=lambda x: tickers[x])

manual_open = st.sidebar.text_input("(Optional) Enter today's Open price (₹):", value="")
date_range = st.sidebar.selectbox("Select Date Range for Chart:", ["1M", "3M", "6M", "1Y", "5Y"], index=4)
chart_type = st.sidebar.radio("Select Chart Type:", options=["Line Chart", "Candlestick"], index=0)

# --- Fetch and Prepare Data ---
@st.cache_data(ttl=3600)
def get_data(ticker, range_key):
    end_date = datetime.now()
    start_dict = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "5Y": 1825}
    days = start_dict.get(range_key, 1825)
    start_date = end_date - timedelta(days=days)
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    return df

def create_features(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df_feat = df.copy()
    df_feat['SMA_10'] = df_feat['Close'].rolling(window=10).mean()
    df_feat['SMA_30'] = df_feat['Close'].rolling(window=30).mean()
    df_feat['Price_Change'] = df_feat['Close'].diff()
    df_feat['High_Low_Diff'] = df_feat['High'] - df_feat['Low']
    df_feat['Volume_Change'] = df_feat['Volume'].diff()
    df_feat['Close_Shift_1'] = df_feat['Close'].shift(1)
    df_feat.dropna(inplace=True)
    return df_feat

# --- Prediction Logic ---
with st.spinner("🔄 Fetching data and generating predictions..."):
    df = get_data(selected_ticker, date_range)

    if df.empty:
        st.error("❌ Failed to fetch stock data. Try again later.")
    else:
        feat_df = create_features(df)

        # --- Predict only for next trading day ---
        last_data_date = feat_df.index[-1].date()
        future_day = last_data_date + timedelta(days=1)
        while future_day.weekday() >= 5 or future_day in feat_df.index.date:
            future_day += timedelta(days=1)

        latest = feat_df.loc[[feat_df.index[-1]]].copy()

        if manual_open.strip():
            try:
                latest['Open'] = float(manual_open.strip())
            except ValueError:
                st.warning("⚠️ Invalid open price entered. Ignoring manual input.")

        required_features = scalers[selected_ticker].feature_names_in_
        features = latest[required_features]
        X_scaled = scalers[selected_ticker].transform(features)

        model_open = models[selected_ticker]['open']
        model_close = models[selected_ticker]['close']

        pct_open = float(model_open.predict(X_scaled)[0])
        pct_close = float(model_close.predict(X_scaled)[0])

        close_price = float(latest['Close'].iloc[0])
        open_price = float(latest['Open'].iloc[0])

        pred_open_price = open_price * (1 + pct_open / 100)
        pred_close_price = open_price * (1 + pct_close / 100)

        err = errors[selected_ticker]
        acc_open = 100 - err['open']['mae']
        acc_close = 100 - err['close']['mae']

        st.title(f"📈 {tickers[selected_ticker]} Forecast for {future_day.strftime('%Y-%m-%d')}")

        st.markdown("""
            <div class="prediction-highlight">
                <h3>🎯 Predictions:</h3>
                <ul>
                    <li><b>Predicted Open:</b> ₹{:.2f} ({:+.2f}%)</li>
                    <li><b>Predicted Close:</b> ₹{:.2f} ({:+.2f}%)</li>
                </ul>
            </div>
        """.format(pred_open_price, pct_open, pred_close_price, pct_close), unsafe_allow_html=True)

        display_df = df[['Open', 'Close']].copy()
        display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d')
        display_df = display_df.tail(5)

        future_day_str = future_day.strftime('%Y-%m-%d')
        if future_day_str not in display_df.index:
            prediction_row = pd.DataFrame({
                'Open': [pred_open_price],
                'Close': [pred_close_price]
            }, index=[future_day_str])
            display_df = pd.concat([display_df, prediction_row])

        st.subheader("📋 Last 5 Days + Prediction")
        with st.expander("🔎 Show Table"):
            styled = display_df.copy()
            prev_close = None

            def format_arrow(curr, prev):
                if prev is None or pd.isna(curr) or pd.isna(prev):
                    return ""
                return "🔺" if curr > prev else "🔻"

            for idx in styled.index:
                open_val = pd.to_numeric(styled.loc[idx, 'Open'], errors='coerce')
                close_val = pd.to_numeric(styled.loc[idx, 'Close'], errors='coerce')

                open_arrow = format_arrow(open_val, prev_close)
                close_arrow = format_arrow(close_val, open_val)

                styled.loc[idx, 'Open'] = f"₹{open_val:.2f} {open_arrow}" if not pd.isna(open_val) else "N/A"
                styled.loc[idx, 'Close'] = f"₹{close_val:.2f} {close_arrow}" if not pd.isna(close_val) else "N/A"

                prev_close = close_val

            styled_df = styled.style.apply(
                lambda x: ['background-color: #333333' if x.name == future_day_str else '' for _ in x], axis=1
            )
            st.dataframe(styled_df, use_container_width=True)

        st.subheader("✅ Model Accuracy")
        col1, col2 = st.columns(2)
        col1.metric("Open Prediction Accuracy", f"{acc_open:.2f}%")
        col2.metric("Close Prediction Accuracy", f"{acc_close:.2f}%")

        st.subheader("📊 Price Chart")
        chart_df = df[['Open', 'High', 'Low', 'Close']].copy()
        chart_df.loc[Timestamp(future_day)] = [
            pred_open_price,
            max(pred_open_price, pred_close_price),
            min(pred_open_price, pred_close_price),
            pred_close_price
        ]
        chart_df.sort_index(inplace=True)

        fig = go.Figure()
        if chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=chart_df.index,
                open=chart_df['Open'],
                high=chart_df['High'],
                low=chart_df['Low'],
                close=chart_df['Close'],
                name='Candlestick'
            ))
        else:
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['Open'], mode='lines+markers', name='Open'))
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['Close'], mode='lines+markers', name='Close'))

        fig.update_layout(
            title="Stock Price Chart",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            hovermode="x unified",
            xaxis=dict(tickformat="%Y-%m-%d"),
            plot_bgcolor="#1e1e1e",
            paper_bgcolor="#1e1e1e",
            font=dict(color="white")
        )
        st.plotly_chart(fig, use_container_width=True)
