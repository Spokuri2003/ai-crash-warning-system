import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(page_title="Market Dashboard", layout="wide")

st.title("📈 Simple Market Dashboard")
st.caption("A clean, stable project to close the previous app.")

# -----------------------------------------------------
# ASSET SELECTION
# -----------------------------------------------------
ASSETS = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "S&P 500 (^GSPC)": "^GSPC",
    "Nasdaq 100 (^NDX)": "^NDX",
    "Gold (GC=F)": "GC=F"
}

asset_name = st.sidebar.selectbox("Select Asset", list(ASSETS.keys()))
symbol = ASSETS[asset_name]

# -----------------------------------------------------
# LOAD DATA (SAFE)
# -----------------------------------------------------
@st.cache_data
def load_data(symbol):
    df = yf.download(symbol, period="3y", interval="1d")

    # Fallback if API fails
    if df is None or df.empty:
        st.error("No data available.")
        return pd.DataFrame()

    df = df.reset_index()

    # Normalize column names
    df.columns = df.columns.str.lower().str.replace(" ", "")

    # Find any column starting with "close"
    close_columns = [c for c in df.columns if c.startswith("close")]
    if len(close_columns) == 0:
        st.error("No valid Close price column found.")
        return pd.DataFrame()

    df["closeprice"] = df[close_columns[0]].astype(float)

    # Guarantee date column exists
    if "date" not in df.columns:
        df["date"] = df.index

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "closeprice"])

    # Compute features
    df["returns"] = df["closeprice"].pct_change()
    df["volatility30"] = df["returns"].rolling(30).std()

    return df.dropna().reset_index(drop=True)

df = load_data(symbol)

if df.empty:
    st.stop()

# -----------------------------------------------------
# METRICS
# -----------------------------------------------------
latest_price = df["closeprice"].iloc[-1]
latest_vol = df["volatility30"].iloc[-1]
daily_ret = df["returns"].iloc[-1]

col1, col2, col3 = st.columns(3)
col1.metric("📌 Latest Price", f"${latest_price:,.2f}")
col2.metric("🌬️ 30-Day Volatility", f"{latest_vol:.4f}")
col3.metric("📉 Daily Return", f"{daily_ret:.4f}")

# -----------------------------------------------------
# PLOTS
# -----------------------------------------------------
st.subheader(f"Price Chart: {asset_name}")
fig_price = px.line(df, x="date", y="closeprice", title="Price Over Time")
st.plotly_chart(fig_price, use_container_width=True)

st.subheader("30-Day Volatility")
fig_vol = px.line(df, x="date", y="volatility30", title="Rolling Volatility (30D)")
st.plotly_chart(fig_vol, use_container_width=True)

st.subheader("Daily Returns")
fig_ret = px.line(df, x="date", y="returns", title="Daily Returns")
st.plotly_chart(fig_ret, use_container_width=True)

st.success("Project completed successfully.")
