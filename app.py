import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ---------------------------------------
# FIX: Download VADER if missing
# ---------------------------------------
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

analyzer = SentimentIntensityAnalyzer()

# ======================================
# 1. Load Price Data
# ======================================
@st.cache_data
def load_price_data(symbol):
    df = yf.download(symbol, period="2y")

    if df.empty:
        return pd.DataFrame()

    df = df.reset_index()

    # Normalize column names
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Expect close price under column 'close'
    if "close" not in df.columns:
        return pd.DataFrame()

    df = df[["date", "close"]].dropna()
    df.rename(columns={"close": "close_price"}, inplace=True)
    return df


# ======================================
# 2. Smart Sentiment System
# ======================================
def smart_sentiment(text_list):
    positive_words = [
        "up","growth","bull","optimistic","surge","rally",
        "strong","support","recover","positive","gain"
    ]
    negative_words = [
        "risk","fear","crash","panic","selloff","bear","volatile",
        "down","collapse","decline","drop","fall","uncertain"
    ]

    scores = []
    for t in text_list:
        t = t.lower()
        score = 0
        hits = 0
        for w in positive_words:
            if w in t:
                score += 1
                hits += 1
        for w in negative_words:
            if w in t:
                score -= 1
                hits += 1

        if hits > 0:
            scores.append(score / hits)

    avg = np.mean(scores) if scores else 0
    fear = int(np.interp(-avg, [-1, 1], [100, 0]))  # map to 0–100 fear index

    return avg, fear


# ======================================
# 3. Get Crypto News Headlines
# ======================================
@st.cache_data
def load_news():
    url = "https://cryptopanic.com/api/free/v1/posts/?kind=news"
    try:
        r = requests.get(url, timeout=5).json()
        return [p["title"] for p in r.get("results", []) if "title" in p]
    except:
        return []


# ======================================
# 4. Regime Classification
# ======================================
def classify_regimes(df, labels, centers):
    df["regime"] = labels

    vol = centers[:, 1]
    sorted_vol = np.argsort(vol)

    calm = sorted_vol[0]
    neutral = sorted_vol[1]
    stress = sorted_vol[2]

    regime_names = {
        calm: "Calm",
        neutral: "Neutral",
        stress: "High-Stress"
    }

    risk_map = {
        calm: 20,
        neutral: 55,
        stress: 85
    }

    return df, regime_names, risk_map


# ======================================
# STREAMLIT UI
# ======================================
st.set_page_config(page_title="AI Crash Warning System", layout="wide")
st.title("⚠️ AI Market Crash Early-Warning System")

asset = st.selectbox(
    "Select Asset",
    ["BTC-USD", "ETH-USD", "AAPL", "SPY"]
)

# --------------------------------------
# Load Data
# --------------------------------------
df = load_price_data(asset)

if df.empty:
    st.error("No price data found for this asset.")
    st.stop()

# --------------------------------------
# Compute Returns + Volatility
# --------------------------------------
df["returns"] = df["close_price"].pct_change()
df["volatility30"] = df["returns"].rolling(30).std()
df = df.dropna()

# --------------------------------------
# KMeans Regime Detection
# --------------------------------------
X = df[["returns", "volatility30"]].values
X = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

df, regime_names, risk_map = classify_regimes(df, labels, centers)

latest_regime_num = int(df["regime"].iloc[-1])
latest_regime = regime_names[latest_regime_num]
regime_risk = risk_map[latest_regime_num]

# --------------------------------------
# Sentiment
# --------------------------------------
news = load_news()
sent_score, sentiment_fear = smart_sentiment(news)

# ======================================
# DISPLAY RESULTS
# ======================================
col1, col2, col3 = st.columns(3)

col1.metric("Market Regime", latest_regime)
col2.metric("Crash Risk Level", f"{regime_risk}/100")
col3.metric("News Fear Index", f"{sentiment_fear}/100")

st.subheader("📈 Price Chart")
fig_price = px.line(
    df,
    x="date",
    y="close_price",
    title=f"{asset} Closing Price"
)
st.plotly_chart(fig_price, use_container_width=True)

st.subheader("📉 30-Day Volatility")
fig_vol = px.line(
    df,
    x="date",
    y="volatility30",
    title=f"{asset} 30-Day Volatility"
)
st.plotly_chart(fig_vol, use_container_width=True)

st.subheader("📰 Latest Headlines Used for Sentiment")
if news:
    for n in news[:10]:
        st.write("•", n)
else:
    st.info("No news available.")


st.success("✔ System Loaded Successfully – Model Working")
