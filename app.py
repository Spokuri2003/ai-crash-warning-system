import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from nltk.sentiment.vader import SentimentIntensityAnalyzer

st.set_page_config(page_title="AI Crash Warning System", layout="wide")

st.title("🧠 AI Market Crash Warning System")
st.caption("Volatility + Regime ML + Sentiment NLP for Market Stress Detection")

# -------------------------------------------------
# ASSET SELECTION
# -------------------------------------------------
ASSETS = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "S&P 500 (^GSPC)": "^GSPC",
    "Nasdaq 100 (^NDX)": "^NDX",
    "Gold (GC=F)": "GC=F"
}

asset_name = st.sidebar.selectbox("Select Asset", list(ASSETS.keys()))
symbol = ASSETS[asset_name]

# -------------------------------------------------
# LOAD NEWS (SAFE)
# -------------------------------------------------
@st.cache_data
def load_news():
    try:
        url = "https://cryptopanic.com/api/v1/posts/?auth_token=bbf69ca77e536fa8d3&public=true"
        r = requests.get(url, timeout=5).json()
        if "results" in r:
            return [item.get("title", "Untitled") for item in r["results"]]
    except:
        pass

    return [
        "Market calm with no significant movements",
        "Analysts see mixed sentiment in global markets",
        "Liquidity stable; fear low among traders",
        "Crypto traders remain cautious amid uncertainty",
        "Macro news expected to affect markets"
    ]

news = load_news()

# -------------------------------------------------
# SENTIMENT NLP
# -------------------------------------------------
analyzer = SentimentIntensityAnalyzer()

scores = []
for t in news:
    try:
        scores.append(analyzer.polarity_scores(t)["compound"])
    except:
        scores.append(0)

avg_sentiment = np.mean(scores)
sentiment_fear = max(0, (0 - avg_sentiment) * 100)

# -------------------------------------------------
# PRICE DATA LOADING (SAFE)
# -------------------------------------------------
@st.cache_data
def load_price(sym):
    df = yf.download(sym, period="3y", interval="1d")
    if df is None or df.empty:
        df = pd.DataFrame({
            "Close": [100, 101, 99, 103, 104],
        })
        df["Date"] = pd.date_range(end=pd.Timestamp.today(), periods=len(df))
        df["Returns"] = df["Close"].pct_change()
        df["Volatility_30d"] = df["Returns"].rolling(30).std()
        return df.dropna()

    df = df.rename(columns={"Close": "ClosePrice"})
    df["Date"] = df.index
    df["Returns"] = df["ClosePrice"].pct_change()
    df["Volatility_30d"] = df["Returns"].rolling(30).std()
    df = df.dropna().reset_index(drop=True)
    return df

df = load_price(symbol)

# -------------------------------------------------
# K-MEANS REGIME CLASSIFICATION
# -------------------------------------------------
def compute_regimes(data):
    feat = data[["Returns", "Volatility_30d"]].dropna()
    if len(feat) < 50:
        data["Regime"] = 1
        return data, {"1": "Neutral"}, {1: 50}

    scaler = StandardScaler()
    X = scaler.fit_transform(feat)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    data = data.copy()
    data["Regime"] = -1
    data.loc[feat.index, "Regime"] = labels

    # regime interpretation
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

    return data, names, weights

df, names, weights = compute_regimes(df)

# -------------------------------------------------
# RISK SCORE
# -------------------------------------------------
latest_vol = df["Volatility_30d"].iloc[-1]
vol_risk = min(100, latest_vol * 2500)

latest_regime = df["Regime"].iloc[-1]
regime_label = names.get(latest_regime, "Neutral")
regime_risk = weights.get(latest_regime, 50)

combined_risk = (
    0.5 * vol_risk +
    0.3 * sentiment_fear +
    0.2 * regime_risk
)

combined_risk = max(0, min(100, combined_risk))

# -------------------------------------------------
# METRICS DISPLAY
# -------------------------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("📉 Crash Risk", f"{combined_risk:.1f}%")
col2.metric("🧠 Sentiment Fear", f"{sentiment_fear:.1f}%")
col3.metric("📊 Regime", regime_label)

# -------------------------------------------------
# CHARTS
# -------------------------------------------------
st.subheader(f"{asset_name} Price Chart")

fig_price = px.line(
    df,
    x="Date",
    y="ClosePrice",
    title=f"{asset_name} Closing Price",
)
st.plotly_chart(fig_price, use_container_width=True)

st.subheader("Volatility (30-Day Rolling)")
fig_vol = px.line(
    df,
    x="Date",
    y="Volatility_30d",
    title="Realized Volatility",
)
st.plotly_chart(fig_vol, use_container_width=True)

st.subheader("Regime Over Time")
fig_reg = px.scatter(
    df,
    x="Date",
    y="ClosePrice",
    color=df["Regime"].map(names),
    title="Regime Classification",
)
st.plotly_chart(fig_reg, use_container_width=True)

# -------------------------------------------------
# NEWS & SENTIMENT
# -------------------------------------------------
st.subheader("📰 Market Headlines")
for n in news:
    st.write("• " + n)

st.subheader("Sentiment Distribution")
fig_sent = px.histogram(scores, nbins=20, title="Sentiment Histogram")
st.plotly_chart(fig_sent, use_container_width=True)


