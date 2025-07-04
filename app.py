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

show_chart = st.sidebar.checkbox("Show Price Chart", value=False)
backtest_mode = st.sidebar.checkbox("📅 Backtest Last Month")

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

        last_data_date = feat_df.index[-1].date()
        future_day = last_data_date + timedelta(days=1)
        while future_day.weekday() >= 5 or pd.Timestamp(future_day) in feat_df.index:
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
        model_high = models[selected_ticker].get('high')
        model_low = models[selected_ticker].get('low')

        pct_open = float(model_open.predict(X_scaled)[0])
        pct_close = float(model_close.predict(X_scaled)[0])
        pct_high = float(model_high.predict(X_scaled)[0]) if model_high else 0.0
        pct_low = float(model_low.predict(X_scaled)[0]) if model_low else 0.0

        close_price = float(latest['Close'].iloc[0])
        open_price = float(latest['Open'].iloc[0])

        pred_open_price = open_price * (1 + pct_open / 100)
        pred_close_price = open_price * (1 + pct_close / 100)
        pred_high_price = open_price * (1 + pct_high / 100)
        pred_low_price = open_price * (1 + pct_low / 100)

        err = errors[selected_ticker]
        acc_open = 100 - err['open']['mae']
        acc_close = 100 - err['close']['mae']
        acc_high = 100 - err['high']['mae'] if 'high' in err else 0.0
        acc_low = 100 - err['low']['mae'] if 'low' in err else 0.0

        mae_open = err['open']['mae'] * open_price / 100
        mae_close = err['close']['mae'] * open_price / 100
        mae_high = err['high']['mae'] * open_price / 100 if 'high' in err else 0.0
        mae_low = err['low']['mae'] * open_price / 100 if 'low' in err else 0.0

        # Target/SL Range Calculation
        target = pred_high_price - mae_high
        stop_loss = pred_low_price + mae_low
        margin_range = target - stop_loss

        st.title(f"📈 {tickers[selected_ticker]} Forecast for {future_day.strftime('%Y-%m-%d')}")

        st.markdown(f"""
            <div class="prediction-highlight">
                <h3>🎯 Predictions:</h3>
                <ul>
                    <li><b>Predicted Open:</b> ₹{pred_open_price:.2f} ± ₹{mae_open:.2f}</li>
                    <li><b>Predicted Close:</b> ₹{pred_close_price:.2f} ± ₹{mae_close:.2f}</li>
                    <li><b>Predicted High:</b> ₹{pred_high_price:.2f} ± ₹{mae_high:.2f}</li>
                    <li><b>Predicted Low:</b> ₹{pred_low_price:.2f} ± ₹{mae_low:.2f}</li>
                </ul>
                <h4>📌 Strategy Insights:</h4>
                <ul>
                    <li><b>Target:</b> ₹{target:.2f}</li>
                    <li><b>Stop Loss:</b> ₹{stop_loss:.2f}</li>
                    <li><b>Margin Range:</b> ₹{margin_range:.2f}</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

        display_df = df[['Open', 'High', 'Low', 'Close']].copy()
        display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d')
        display_df = display_df.tail(5)

        future_day_str = future_day.strftime('%Y-%m-%d')
        if future_day_str not in display_df.index:
            prediction_row = pd.DataFrame({
                'Open': [pred_open_price],
                'High': [pred_high_price],
                'Low': [pred_low_price],
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
                styled.loc[idx, 'High'] = f"₹{styled.loc[idx, 'High']:.2f}" if not pd.isna(styled.loc[idx, 'High']) else "N/A"
                styled.loc[idx, 'Low'] = f"₹{styled.loc[idx, 'Low']:.2f}" if not pd.isna(styled.loc[idx, 'Low']) else "N/A"

                prev_close = close_val

            styled_df = styled.style.apply(
                lambda x: ['background-color: #333333' if x.name == future_day_str else '' for _ in x], axis=1
            )
            st.dataframe(styled_df, use_container_width=True)

        st.subheader("✅ Model Accuracy")
        st.markdown(f"""
        <div style="display: flex; justify-content: space-around; padding: 10px; background-color: #1f1f1f; border-radius: 8px;">
            <div><b>Open:</b><br>{acc_open:.2f}% ± ₹{mae_open:.2f}</div>
            <div><b>Close:</b><br>{acc_close:.2f}% ± ₹{mae_close:.2f}</div>
            <div><b>High:</b><br>{acc_high:.2f}% ± ₹{mae_high:.2f}</div>
            <div><b>Low:</b><br>{acc_low:.2f}% ± ₹{mae_low:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

        if show_chart:
            st.subheader("📊 Price Chart")
            chart_df = df[['Open', 'High', 'Low', 'Close']].copy()
            chart_df.loc[Timestamp(future_day)] = [
                pred_open_price,
                pred_high_price,
                pred_low_price,
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

        # --- Backtesting Section ---
        if backtest_mode:
            st.subheader("🧪 Backtest: Last Month Performance")

            backtest_results = []
            trading_days = feat_df.index[-22:]

            for i in range(1, len(trading_days)):
                prev_date = trading_days[i - 1]
                curr_date = trading_days[i]

                try:
                    prev_feat = feat_df.loc[[prev_date]].copy()
                    actual = df.loc[curr_date] if curr_date in df.index else None
                    if actual is None:
                        continue

                    open_val = float(prev_feat['Open'].iloc[0])
                    X_test = scalers[selected_ticker].transform(prev_feat[required_features])

                    pred_open = open_val * (1 + model_open.predict(X_test)[0] / 100)
                    pred_close = open_val * (1 + model_close.predict(X_test)[0] / 100)
                    pred_high = open_val * (1 + model_high.predict(X_test)[0] / 100) if model_high else None
                    pred_low = open_val * (1 + model_low.predict(X_test)[0] / 100) if model_low else None

                    backtest_results.append({
                        "Date": curr_date.strftime("%Y-%m-%d"),
                        "Actual_Open": actual['Open'],
                        "Pred_Open": pred_open,
                        "Actual_Close": actual['Close'],
                        "Pred_Close": pred_close,
                        "Actual_High": actual['High'],
                        "Pred_High": pred_high,
                        "Actual_Low": actual['Low'],
                        "Pred_Low": pred_low
                    })

                except Exception as e:
                    st.warning(f"⚠️ Skipped {curr_date.date()}: {e}")
                    continue

            if backtest_results:
                bt_df = pd.DataFrame(backtest_results)

                for price_type in ["Open", "Close", "High", "Low"]:
                    if f"Pred_{price_type}" in bt_df.columns:
                        bt_df[f"Error_{price_type}"] = abs(bt_df[f"Actual_{price_type}"] - bt_df[f"Pred_{price_type}"])

                if backtest_results:
                    bt_df = pd.DataFrame(backtest_results)

                    # Convert numeric columns to float (prevent formatting errors)
                    for col in bt_df.columns:
                        if col.startswith("Actual") or col.startswith("Pred"):
                            bt_df[col] = pd.to_numeric(bt_df[col], errors='coerce')

                    for price_type in ["Open", "Close", "High", "Low"]:
                        if f"Pred_{price_type}" in bt_df.columns:
                            bt_df[f"Error_{price_type}"] = abs(bt_df[f"Actual_{price_type}"] - bt_df[f"Pred_{price_type}"])

                    st.markdown("### 📋 Predictions vs Actuals")
                    st.dataframe(
                        bt_df[[
                            "Date",
                            "Actual_Open", "Pred_Open", "Error_Open",
                            "Actual_Close", "Pred_Close", "Error_Close",
                            "Actual_High", "Pred_High", "Error_High",
                            "Actual_Low", "Pred_Low", "Error_Low"
                        ]].style.format({
                            "Actual_Open": "₹{:.2f}", "Pred_Open": "₹{:.2f}", "Error_Open": "₹{:.2f}",
                            "Actual_Close": "₹{:.2f}", "Pred_Close": "₹{:.2f}", "Error_Close": "₹{:.2f}",
                            "Actual_High": "₹{:.2f}", "Pred_High": "₹{:.2f}", "Error_High": "₹{:.2f}",
                            "Actual_Low": "₹{:.2f}", "Pred_Low": "₹{:.2f}", "Error_Low": "₹{:.2f}"
                        }),
                        use_container_width=True
                    )

                    st.markdown("### 📊 Backtest MAE (₹)")
                    for price_type in ["Open", "Close", "High", "Low"]:
                        col_name = f"Error_{price_type}"
                        if col_name in bt_df:
                            mae = bt_df[col_name].mean()
                            st.write(f"**{price_type}**: ₹{mae:.2f}")
                else:
                    st.warning("Not enough data to perform backtest.")
