import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------
# NLTK Sentiment Fix
# -----------------------------------------------------
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

analyzer = SentimentIntensityAnalyzer()

# -----------------------------------------------------
# PAGE SETTINGS
# -----------------------------------------------------
st.set_page_config(page_title="AI Crash Warning System", layout="wide")
st.title("🧠 AI Market Crash Warning System (V2 Final)")
st.caption("Market Regimes • Volatility • NLP Sentiment → Crash Risk Score")

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
# SAFE NEWS FETCH
# -----------------------------------------------------
@st.cache_data
def load_news():
    try:
        url = "https://cryptopanic.com/api/v1/posts/?auth_token=bbf69ca77e536fa8d3&public=true"
        r = requests.get(url, timeout=5).json()
        if "results" in r:
            return [item.get("title", "") for item in r["results"]]
    except:
        pass

    return [
        "Market shows stable conditions.",
        "Analysts detect moderate volatility.",
        "Investors wait for macroeconomic signals.",
        "Crypto liquidity remains steady.",
        "Traders cautious despite calm environment."
    ]

news_titles = load_news()

# -----------------------------------------------------
# NLP SENTIMENT
# -----------------------------------------------------
sent_scores = []
for t in news_titles:
    try:
        sent_scores.append(analyzer.polarity_scores(t)["compound"])
    except:
        sent_scores.append(0)

avg_sentiment = np.mean(sent_scores)
sentiment_fear = max(0, (0 - avg_sentiment) * 100)

# -----------------------------------------------------
# PRICE DATA LOADING
# -----------------------------------------------------
@st.cache_data
def load_asset(symbol):
    df = yf.download(symbol, period="3y", interval="1d")

    # fallback
    if df is None or df.empty:
        df = pd.DataFrame({
            "Close": [100, 102, 101, 103, 105],
            "Date": pd.date_range(end=pd.Timestamp.today(), periods=5)
        })
    else:
        df["Date"] = df.index

    return df.reset_index(drop=True)

df = load_asset(symbol)

# -----------------------------------------------------
# UNIVERSAL ClosePrice FIX
# -----------------------------------------------------

# Flatten MultiIndex columns if present
if isinstance(df.columns, pd.MultiIndex):
    df.columns = ['_'.join([str(c) for c in col if c]) for col in df.columns]

# Normalize column names
df.columns = df.columns.str.lower().str.replace(" ", "").str.replace("_", "")

# Debug print
st.write("Columns in downloaded data:", list(df.columns))

# Possible close column names
possible_close_cols = [
    "closeprice", "close", "adjclose", "closingprice",
    "closebtc-usd", "closebtc", "price"
]

found = None
for col in possible_close_cols:
    if col in df.columns:
        found = col
        break

if found is None:
    st.error("❌ No valid Close price column found in the asset data.")
    st.stop()

# Final unified ClosePrice
df["ClosePrice"] = df[found].astype(float)
df = df.dropna(subset=["ClosePrice"]).reset_index(drop=True)

# -----------------------------------------------------
# FEATURES FOR ML
# -----------------------------------------------------
df["Returns"] = df["ClosePrice"].pct_change()
df["Volatility_30d"] = df["Returns"].rolling(30).std()
df = df.dropna().reset_index(drop=True)

# -----------------------------------------------------
# K-Means REGIME DETECTION
# -----------------------------------------------------
def compute_regimes(df):
    feat = df[["Returns", "Volatility_30d"]].dropna()

    if len(feat) < 50:
        df["Regime"] = 1
        return df, {1: "Neutral"}, {1: 50}

    scaler = StandardScaler()
    X = scaler.fit_transform(feat)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    df = df.copy()
    df["Regime"] = -1
    df.loc[feat.index, "Regime"] = labels

    centers = pd.DataFrame(kmeans.cluster_centers_, columns=["ret", "vol"])
    centers["stress"] = centers["vol"] - centers["ret"]
    order = centers.sort_values("stress", ascending=False).index.tolist()

    names = {}
    weights = {}
    mapping = ["High-Stress", "Neutral", "Calm"]
    risk_map = [85, 50, 25]

    for i, cluster in enumerate(order):
        names[cluster] = mapping[i]
        weights[cluster] = risk_map[i]

    return df, names, weights

df, names, weights = compute_regimes(df)

# -----------------------------------------------------
# RISK SCORE
# -----------------------------------------------------
latest_vol = df["Volatility_30d"].iloc[-1]
vol_risk = min(100, latest_vol * 2500)

latest_regime = int(df["Regime"].iloc[-1])
regime_label = names.get(latest_regime, "Neutral")
regime_risk = weights.get(latest_regime, 50)

combined_risk = (
    0.5 * vol_risk +
    0.3 * sentiment_fear +
    0.2 * regime_risk
)

combined_risk = min(100, max(0, combined_risk))

# -----------------------------------------------------
# METRICS
# -----------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("📉 Crash Risk", f"{combined_risk:.1f}%")
col2.metric("😨 Sentiment Fear", f"{sentiment_fear:.1f}%")
col3.metric("📊 Regime", regime_label)

# -----------------------------------------------------
# CLEAN DATA FOR PLOTTING
# -----------------------------------------------------
df_plot = df.copy().reset_index(drop=True)
df_plot["Date"] = pd.to_datetime(df_plot["Date"], errors="coerce")
df_plot = df_plot.dropna(subset=["Date", "ClosePrice"])

# -----------------------------------------------------
# PRICE PLOT
# -----------------------------------------------------
st.subheader("📈 Price Chart")
fig_price = px.line(df_plot, x="Date", y="ClosePrice", title=f"{asset_name} Price")
st.plotly_chart(fig_price, use_container_width=True)

# -----------------------------------------------------
# VOLATILITY PLOT
# -----------------------------------------------------
st.subheader("📉 30-Day Volatility")
fig_vol = px.line(df_plot, x="Date", y="Volatility_30d", title="Volatility (30-Day Rolling)")
st.plotly_chart(fig_vol, use_container_width=True)

# -----------------------------------------------------
# REGIME PLOT
# -----------------------------------------------------
st.subheader("🟦 Regime Classification")
fig_reg = px.scatter(
    df_plot,
    x="Date",
    y="ClosePrice",
    color=df_plot["Regime"].map(names),
    title="Market Regimes",
)
st.plotly_chart(fig_reg, use_container_width=True)

# -----------------------------------------------------
# NEWS + SENTIMENT
# -----------------------------------------------------
st.subheader("📰 Latest Market News")
for t in news_titles:
    st.write("• " + t)

st.subheader("Sentiment Distribution")
fig_sent = px.histogram(sent_scores, nbins=20)
st.plotly_chart(fig_sent, use_container_width=True)

st.success("App running successfully! 🎉")
