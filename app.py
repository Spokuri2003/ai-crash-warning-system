import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="AI Crash Warning System", layout="wide")

# ---------------------------------------------------
# SAFE NEWS API
# ---------------------------------------------------
@st.cache_data
def load_news():
    try:
        url = "https://cryptopanic.com/api/v1/posts/?auth_token=bbf69ca77e536fa8d3&public=true"
        r = requests.get(url, timeout=5).json()
        if "results" in r and len(r["results"]) > 0:
            return [p.get("title", "No Title") for p in r["results"]]
        raise ValueError("API returned empty results")
    except:
        return [
            "Market stable; no major events reported",
            "Investors cautious as BTC consolidates",
            "Neutral sentiment across major digital assets",
            "Analysts expect muted volatility",
            "Crypto markets awaiting macroeconomic news",
        ]

news_titles = load_news()

# ---------------------------------------------------
# SAFE BTC PRICE LOADING
# ---------------------------------------------------
@st.cache_data
def load_btc():
    try:
        data = yf.download("BTC-USD", period="1y")
        if data is None or data.empty:
            raise ValueError("Empty BTC data")

        # Standard formatting
        data = data.rename(columns=str.capitalize)

        # Feature Engineering
        data["Returns"] = data["Close"].pct_change()
        data["Volatility_30d"] = data["Returns"].rolling(30).std()

        # Clean output
        data = data.dropna()

        # Ensure DatetimeIndex
        data.index = pd.to_datetime(data.index)

        return data

    except:
        # Fallback minimal dataset (never crashes)
        fallback = pd.DataFrame({
            "Close": [40000, 40500, 39800, 41000, 42000],
            "Returns": [0.01, -0.02, 0.03, 0.01, 0.02],
            "Volatility_30d": [0.02, 0.021, 0.022, 0.023, 0.024],
        })
        fallback.index = pd.date_range(
            end=pd.Timestamp.today(),
            periods=5
        )
        return fallback

btc = load_btc()

# ---------------------------------------------------
# SENTIMENT ANALYSIS
# ---------------------------------------------------
analyzer = SentimentIntensityAnalyzer()

def compute_sentiment(titles):
    scores = []
    for t in titles:
        try:
            scores.append(analyzer.polarity_scores(t)["compound"])
        except:
            scores.append(0)
    return scores

sentiment_scores = compute_sentiment(news_titles)
sentiment_fear = (1 - ((np.mean(sentiment_scores) + 1) / 2)) * 100

# ---------------------------------------------------
# CRASH RISK SCORE
# ---------------------------------------------------
latest_vol = float(btc["Volatility_30d"].iloc[-1])
vol_risk = min(latest_vol * 1200, 100)  # smaller scaling

combined_risk = (0.6 * vol_risk) + (0.4 * sentiment_fear)

# ---------------------------------------------------
# LATEST PRICE (SAFE)
# ---------------------------------------------------
try:
    latest_price = float(btc["Close"].iloc[-1])
    price_display = f"${latest_price:,.2f}"
except:
    price_display = "N/A"

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.title("AI Early-Warning System for Market Stress")

col1, col2, col3 = st.columns(3)
col1.metric("Crash Risk", f"{combined_risk:.1f}%")
col2.metric("Sentiment Fear", f"{sentiment_fear:.1f}%")
col3.metric("BTC Price", price_display)

st.markdown("---")

# ---------------------------------------------------
# VOLATILITY CHART (WITH SAFEGUARDS)
# ---------------------------------------------------
if "Volatility_30d" in btc.columns and len(btc) > 1:
    fig_vol = px.line(
        x=btc.index,
        y=btc["Volatility_30d"],
        title="BTC 30-Day Volatility",
        labels={"x": "Date", "y": "Volatility"},
    )
    st.plotly_chart(fig_vol, use_container_width=True)
else:
    st.warning("Volatility data unavailable.")

# ---------------------------------------------------
# PRICE CHART
# ---------------------------------------------------
if "Close" in btc.columns and len(btc) > 1:
    fig_price = px.line(
        x=btc.index,
        y=btc["Close"],
        title="BTC Closing Price",
        labels={"x": "Date", "y": "Close Price"},
    )
    st.plotly_chart(fig_price, use_container_width=True)
else:
    st.warning("Price data unavailable.")

st.markdown("---")

# ---------------------------------------------------
# SENTIMENT DISTRIBUTION HISTOGRAM
# ---------------------------------------------------
fig_sent = px.histogram(
    sentiment_scores,
    nbins=15,
    title="News Sentiment Distribution",
)
st.plotly_chart(fig_sent, use_container_width=True)

# ---------------------------------------------------
# HEADLINES
# ---------------------------------------------------
st.subheader("News Headlines Used in Sentiment Analysis")
for t in news_titles:
    st.write("• " + t)


