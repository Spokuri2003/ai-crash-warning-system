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

# =========================================================
# FIX: Ensure VADER lexicon exists
# =========================================================
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

analyzer = SentimentIntensityAnalyzer()

# =========================================================
# SAFE PRICE DATA LOADER (no errors ever)
# =========================================================
@st.cache_data
def load_price_data(symbol):
    df = yf.download(symbol, period="2y")

    if df.empty:
        return pd.DataFrame()

    # If Date is index → convert to column
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()

    # Convert column names to string
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    # Detect any close price column
    close_col = None
    for c in df.columns:
        if "close" in c:
            close_col = c
            break

    if close_col is None:
        return pd.DataFrame()

    # Detect date column
    date_col = None
    for c in df.columns:
        if "date" in c:
            date_col = c
            break

    if date_col is None:
        return pd.DataFrame()

    df = df[[date_col, close_col]].dropna()
    df.rename(columns={date_col: "date", close_col: "close_price"}, inplace=True)

    return df


# =========================================================
# SMART SENTIMENT SYSTEM
# =========================================================
def smart_sentiment(text_list):
    positive_words = [
        "up", "growth", "bull", "optimistic", "surge", "rally",
        "strong", "support", "recover", "positive", "gain"
    ]
    negative_words = [
        "risk", "fear", "crash", "panic", "selloff", "bear", "volatile",
        "down", "collapse", "decline", "drop", "fall", "uncertain"
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
    fear = int(np.interp(-avg, [-1, 1], [100, 0]))

    return avg, fear


# =========================================================
# LOAD NEWS (Cryptopanic)
# =========================================================
@st.cache_data
def load_news():
    url = "https://cryptopanic.com/api/free/v1/posts/?kind=news"
    try:
        r = requests.get(url, timeout=5).json()
        return [p["title"] for p in r.get("results", []) if "title" in p]
    except:
        return []


# =========================================================
# REGIME CLASSIFICATION
# =========================================================
def classify_regimes(df, labels, centers):
    df["regime"] = labels

    vol = centers[:, 1]
    sorted_vol = np.argsort(vol)

    calm = sorted_vol[0]
    neutral = sorted_vol[1]
    stress = sorted_vol[2]

    names = {
        calm: "Calm",
        neutral: "Neutral",
        stress: "High-Stress"
    }

    risk_map = {
        calm: 20,
        neutral: 55,
        stress: 85
    }

    return df, names, risk_map


# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(page_title="AI Crash Warning System", layout="wide")
st.title("⚠️ AI Market Crash Early-Warning System")

asset = st.selectbox(
    "Select Asset",
    ["BTC-USD", "ETH-USD", "AAPL", "SPY"]
)

# =========================================================
# LOAD DATA
# =========================================================
df = load_price_data(asset)

if df.empty:
    st.error("❌ Error: No price data for this asset.")
    st.stop()

# =========================================================
# RETURNS + VOLATILITY
# =========================================================
df["returns"] = df["close_price"].pct_change()
df["vol30"] = df["returns"].rolling(30).std()
df = df.dropna()

# =========================================================
# REGIME MODEL (KMEANS)
# =========================================================
X = df[["returns", "vol30"]].values
X = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

df, names, risk_map = classify_regimes(df, labels, centers)

last_label = int(df["regime"].iloc[-1])
regime = names[last_label]
risk = risk_map[last_label]

# =========================================================
# SENTIMENT
# =========================================================
news = load_news()
sent_score, fear = smart_sentiment(news)

# =========================================================
# DISPLAY RESULTS
# =========================================================
c1, c2, c3 = st.columns(3)
c1.metric("Market Regime", regime)
c2.metric("Crash Risk", f"{risk}/100")
c3.metric("News Fear Index", f"{fear}/100")

st.subheader("📈 Price Chart")
st.plotly_chart(px.line(df, x="date", y="close_price"), use_container_width=True)

st.subheader("📉 30-Day Volatility")
st.plotly_chart(px.line(df, x="date", y="vol30"), use_container_width=True)

st.subheader("📰 Recent Headlines")
if news:
    for n in news[:10]:
        st.write("•", n)
else:
    st.info("No headlines available.")

st.success("✔ System Running — No Errors")
