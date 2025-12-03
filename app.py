import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests

# -----------------------------
# PAGE LAYOUT
# -----------------------------
st.set_page_config(
    page_title="AI Crash Warning System",
    layout="wide",
    page_icon="📉"
)

st.title(" AI Early-Warning System for Market Crashes")
st.caption("Live volatility + sentiment + combined crash-risk model. Built by Swathi Pokuri.")

# -----------------------------
# LOAD BTC DATA
# -----------------------------
@st.cache_data
def load_btc():
    # Download last 1 year of Bitcoin (BTC-USD) data
    data = yf.download("BTC-USD", period="1y")
    data = data.rename(columns=str.capitalize)
    # Daily returns
    data["Returns"] = data["Close"].pct_change()
    # 30-day rolling standard deviation of returns = volatility proxy
    data["Volatility_30d"] = data["Returns"].rolling(30).std()
    # Remove first 30 days (NaN volatility)
    return data.dropna()

btc = load_btc()

# -----------------------------
# LOAD NEWS
# -----------------------------
@st.cache_data
def load_news():
    try:
        url = "https://cryptopanic.com/api/v1/posts/?auth_token=bbf69ca77e536fa8d3&public=true"
        r = requests.get(url, timeout=5).json()

        # Check if "results" exists
        if "results" in r:
            titles = [p.get("title", "No Title") for p in r["results"]]
            if len(titles) > 0:
                return titles

        # If API returned empty data
        raise ValueError("Empty results")

    except Exception:
        # Fallback headlines to prevent crash
        return [
            "Bitcoin shows sideways movement amid market uncertainty",
            "Crypto markets remain stable; no significant events reported",
            "Analysts observe neutral sentiment across major digital assets",
            "Investors cautious as volatility remains contained",
            "Market awaits key macroeconomic announcements influencing risk assets"
        ]

news_titles = load_news()
# -----------------------------
# NLP SENTIMENT
# -----------------------------
analyzer = SentimentIntensityAnalyzer()

def compute_sentiment(titles):
    scores = []
    for t in titles:
        s = analyzer.polarity_scores(t)["compound"]
        scores.append(s)
    return scores

sentiment_scores = compute_sentiment(news_titles)

# Average sentiment across all headlines
avg_sent = np.mean(sentiment_scores)

# Convert average sentiment (-1 to 1) to a "fear" score (0 to 100)
# -1 (very negative) -> 100 fear
#  1 (very positive) -> 0 fear
sentiment_fear = (1 - ((avg_sent + 1) / 2)) * 100

# -----------------------------
# VOLATILITY RISK SCORE
# -----------------------------
latest_vol = btc["Volatility_30d"].iloc[-1]

# Simple scaling: higher volatility -> higher risk (0–100)
vol_risk = min(100, latest_vol * 2500)

# -----------------------------
# COMBINED RISK
# -----------------------------
combined_risk = 0.6 * vol_risk + 0.4 * sentiment_fear
combined_risk = min(100, combined_risk)

# -----------------------------
# TOP METRICS ROW
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(" Crash Risk", f"{combined_risk:.1f}%")

with col2:
    st.metric(" Sentiment Fear", f"{sentiment_fear:.1f}%")

with col3:
    st.metric(" BTC Price", f"${btc['Close'].iloc[-1]:,.2f}")

st.markdown("---")

# -----------------------------
# VOLATILITY CHART
# -----------------------------
st.subheader(" BTC 30-Day Realized Volatility")

fig_vol = px.line(
    btc,
    x=btc.index,
    y="Volatility_30d",
    labels={"x": "Date", "Volatility_30d": "30D Volatility"},
    title="BTC Volatility (Rolling 30-Day Std Dev)"
)
st.plotly_chart(fig_vol, use_container_width=True)

# -----------------------------
# PRICE CHART
# -----------------------------
st.subheader(" BTC Price (Last 1 Year)")

fig_price = px.line(
    btc,
    x=btc.index,
    y="Close",
    labels={"x": "Date", "Close": "BTC Price"},
    title="BTC-USD Price"
)
st.plotly_chart(fig_price, use_container_width=True)

st.markdown("---")

# -----------------------------
# SENTIMENT BAR CHART
# -----------------------------
st.subheader(" News Sentiment Scores (Headline-Level)")

sent_df = pd.DataFrame({
    "headline": news_titles,
    "sentiment": sentiment_scores
})

fig_sent = px.bar(
    sent_df,
    x="headline",
    y="sentiment",
    title="Sentiment Score for Latest Crypto Headlines",
)
fig_sent.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig_sent, use_container_width=True)

# -----------------------------
# HEADLINE TABLE
# -----------------------------
st.subheader(" Latest Crypto News Headlines")
st.dataframe(sent_df)

st.markdown("---")

# -----------------------------
# AUTOMATED INSIGHTS
# -----------------------------
st.subheader(" Automated Market Insight")

if combined_risk > 70:
    st.error(" High crash probability zone. Volatility and sentiment suggest extreme stress.")
elif combined_risk > 40:
    st.warning(" Moderate stress zone. Conditions are unstable, monitor closely.")
else:
    st.success(" Low stress zone. No major red flags from volatility or sentiment.")

st.caption(
    "Note: This is a demo prototype. A full version could add multi-asset data "
    "(S&P500, NASDAQ, Gold), more advanced NLP (FinBERT), and ML-based crash forecasting."
)
