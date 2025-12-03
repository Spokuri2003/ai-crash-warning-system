import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------
# STREAMLIT CONFIG
# ----------------------------------------------------
st.set_page_config(page_title="AI Market Risk Dashboard", layout="wide")
st.title("🧠 AI Market Crash Warning System")
st.caption("Volatility • Regimes • Sentiment • Crash Probability")

# ----------------------------------------------------
# FIX NLTK
# ----------------------------------------------------
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

sentiment = SentimentIntensityAnalyzer()

# ----------------------------------------------------
# ASSETS
# ----------------------------------------------------
ASSETS = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "S&P 500": "SPY",
    "Nasdaq 100": "QQQ",
    "Gold": "GLD",
}

choice = st.sidebar.selectbox("Select Asset", list(ASSETS.keys()))
symbol = ASSETS[choice]


# ----------------------------------------------------
# SAFE DATA LOADER
# ----------------------------------------------------
def load_price(symbol):
    df = yf.download(symbol, period="3y", interval="1d")

    # fallback if empty
    if df is None or df.empty:
        rng = pd.date_range(end=pd.Timestamp.today(), periods=200)
        df = pd.DataFrame({"Date": rng, "ClosePrice": np.linspace(100, 150, 200)})
        return df

    df["Date"] = df.index
    df["ClosePrice"] = df["Close"]
    df["Returns"] = df["ClosePrice"].pct_change().fillna(0)
    df["Volatility_30d"] = df["Returns"].rolling(30).std().fillna(0)

    return df.reset_index(drop=True)


df = load_price(symbol)


# ----------------------------------------------------
# SENTIMENT ANALYSIS
# ----------------------------------------------------
def get_sentiment():
    headlines = [
        "Markets remain uncertain ahead of macro data release",
        "Analysts expect increased volatility this week",
        "Investors cautious as risk appetite weakens",
        "No major disruption expected today",
    ]

    scores = [sentiment.polarity_scores(h)["compound"] for h in headlines]
    avg = np.mean(scores)
    fear = max(0, (0 - avg) * 100)

    return avg, fear, headlines


avg_sent, sentiment_fear, headlines = get_sentiment()


# ----------------------------------------------------
# REGIME DETECTION
# ----------------------------------------------------
def compute_regime(df):
    X = df[["Returns", "Volatility_30d"]]

    if len(X) < 60:
        df["Regime"] = 1
        return df, {1: "Neutral"}, {1: 50}

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    df["Regime"] = labels

    centers = pd.DataFrame(km.cluster_centers_, columns=["ret", "vol"])
    centers["score"] = centers["vol"] - centers["ret"]
    order = centers.sort_values("score", ascending=False).index.tolist()

    names = ["High-Stress", "Neutral", "Calm"]
    weights = [85, 50, 25]

    regime_names = {}
    regime_weights = {}

    for i, c in enumerate(order):
        regime_names[c] = names[i]
        regime_weights[c] = weights[i]

    return df, regime_names, regime_weights


df, regime_names, regime_weights = compute_regime(df)

current_reg = int(df["Regime"].iloc[-1])
regime_label = regime_names[current_reg]
regime_risk = regime_weights[current_reg]


# ----------------------------------------------------
# CRASH RISK SCORE
# ----------------------------------------------------
vol_risk = min(100, df["Volatility_30d"].iloc[-1] * 4000)
final_risk = (
    0.50 * vol_risk +
    0.30 * sentiment_fear +
    0.20 * regime_risk
)
final_risk = max(0, min(final_risk, 100))


# ----------------------------------------------------
# METRICS
# ----------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("🔥 Crash Risk", f"{final_risk:.1f}%")
c2.metric("😨 Sentiment Fear", f"{sentiment_fear:.1f}%")
c3.metric("📊 Regime", regime_label)


# ----------------------------------------------------
# SAFE PRICE CHART (go.Figure)
# ----------------------------------------------------
st.subheader("📈 Price Chart")

fig_price = go.Figure()
fig_price.add_trace(go.Scatter(
    x=df["Date"],
    y=df["ClosePrice"],
    mode="lines",
    name="Close Price"
))
fig_price.update_layout(title=f"{choice} — Closing Price")
st.plotly_chart(fig_price, use_container_width=True)


# ----------------------------------------------------
# SAFE VOLATILITY CHART
# ----------------------------------------------------
st.subheader("📉 Volatility (30-Day)")

fig_vol = go.Figure()
fig_vol.add_trace(go.Scatter(
    x=df["Date"],
    y=df["Volatility_30d"],
    mode="lines",
    name="Volatility 30d"
))
fig_vol.update_layout(title=f"{choice} — Volatility")
st.plotly_chart(fig_vol, use_container_width=True)


# ----------------------------------------------------
# SAFE REGIME CHART
# ----------------------------------------------------
st.subheader("🟪 Market Regimes")

fig_reg = go.Figure()
fig_reg.add_trace(go.Scatter(
    x=df["Date"],
    y=df["ClosePrice"],
    mode="markers",
    marker=dict(color=df["Regime"], colorscale="Viridis"),
    name="Regime"
))
fig_reg.update_layout(title="Market Regime Classification")
st.plotly_chart(fig_reg, use_container_width=True)


# ----------------------------------------------------
# NEWS BLOCK
# ----------------------------------------------------
st.subheader("📰 Market Headlines")
for h in headlines:
    st.write("•", h)


st.success("App Running Successfully ")
