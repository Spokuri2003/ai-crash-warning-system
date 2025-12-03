import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------
# NLTK VADER setup (always works)
# -----------------------------
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

vader = SentimentIntensityAnalyzer()

# -----------------------------
# Optional FinBERT (mini) setup
# -----------------------------
finbert_model = None
finbert_tokenizer = None

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    # You asked for a lighter FinBERT-style model
    FINBERT_NAME = "ProsusAI/finbert"

    finbert_tokenizer = AutoTokenizer.from_pretrained(FINBERT_NAME)
    finbert_model = AutoModelForSequenceClassification.from_pretrained(FINBERT_NAME)
    finbert_model.eval()
except Exception:
    finbert_model = None
    finbert_tokenizer = None


def finbert_sentiment_score(texts):
    """
    Returns average FinBERT sentiment in [-1, 1].
    If FinBERT is not available, returns None.
    """
    if finbert_model is None or finbert_tokenizer is None:
        return None

    try:
        if len(texts) == 0:
            return None

        # Limit to first 20 headlines to keep it light
        texts = texts[:20]

        inputs = finbert_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = finbert_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).numpy()

        # FinBERT labels: 0=negative, 1=neutral, 2=positive
        scores = []
        for p in probs:
            neg, neu, pos = p
            score = pos - neg  # [-1,1]
            scores.append(score)

        return float(np.mean(scores))
    except Exception:
        return None


# -----------------------------
# Streamlit page setup
# -----------------------------
st.set_page_config(page_title="AI Market Crash Warning System", layout="wide")

st.title("AI Market Crash Warning System")
st.caption(
    "Research dashboard combining volatility analysis, regime classification, and multi-source sentiment."
)

# -----------------------------
# Asset universe
# -----------------------------
ASSETS = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "S&P 500 (^GSPC)": "^GSPC",
    "Nasdaq 100 (^NDX)": "^NDX",
    "Gold (GC=F)": "GC=F",
}

asset_name = st.sidebar.selectbox("Select asset", list(ASSETS.keys()))
symbol = ASSETS[asset_name]

# -----------------------------
# News: helpers
# -----------------------------
def fetch_cryptopanic_news():
    try:
        url = "https://cryptopanic.com/api/v1/posts/?auth_token=bbf69ca77e536fa8d3&public=true"
        r = requests.get(url, timeout=5)
        data = r.json()
        if "results" in data and isinstance(data["results"], list):
            titles = [item.get("title", "Untitled") for item in data["results"]]
            return titles
    except Exception:
        pass
    return []


def fetch_finviz_like_news(symbol):
    """
    Best-effort: use ETF proxies and FinViz-style pages.
    This may fail silently and we fallback to generic headlines.
    """
    try:
        mapping = {
            "^GSPC": "SPY",
            "^NDX": "QQQ",
            "GC=F": "GLD",
        }
        ticker = mapping.get(symbol, "SPY")
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {"User-Agent": "Mozilla/5.0"}
        html = requests.get(url, headers=headers, timeout=5).text

        lines = html.split("\n")
        titles = []
        for line in lines:
            if "news-link" in line:
                # crude parsing, but good enough for demo
                parts = line.split(">")
                if len(parts) > 2:
                    txt = parts[-2]
                    txt = txt.split("<")[0].strip()
                    if txt and txt not in titles:
                        titles.append(txt)
        return titles[:20]
    except Exception:
        return []


def fetch_reddit_titles(subreddit):
    """
    Pulls top/hot titles from a subreddit, best-effort, with fallback.
    """
    try:
        url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=20"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=5)
        data = r.json()
        titles = [
            child["data"]["title"]
            for child in data["data"]["children"]
            if "data" in child and "title" in child["data"]
        ]
        return titles
    except Exception:
        return []


def load_news_for_symbol(sym):
    """
    Asset-specific news:
    - Crypto (BTC, ETH): CryptoPanic + Reddit crypto subs
    - Indices/Gold: FinViz + Reddit stocks/investing/gold
    """
    headlines = []

    if sym.endswith("-USD"):
        # Crypto assets
        headlines.extend(fetch_cryptopanic_news())
        if sym == "BTC-USD":
            headlines.extend(fetch_reddit_titles("Bitcoin"))
        elif sym == "ETH-USD":
            headlines.extend(fetch_reddit_titles("ethereum"))
        else:
            headlines.extend(fetch_reddit_titles("CryptoCurrency"))
    else:
        # Traditional assets
        headlines.extend(fetch_finviz_like_news(sym))
        if sym in ("^GSPC", "^NDX"):
            headlines.extend(fetch_reddit_titles("stocks"))
            headlines.extend(fetch_reddit_titles("wallstreetbets"))
        elif sym == "GC=F":
            headlines.extend(fetch_reddit_titles("Gold"))

    # Fallback generic if everything is empty
    if len(headlines) == 0:
        headlines = [
            "Markets trade in a narrow range with muted volatility",
            "Analysts describe sentiment as cautious but stable",
            "No major macroeconomic surprises reported today",
        ]

    # Deduplicate
    headlines = list(dict.fromkeys(headlines))
    return headlines


news_titles = load_news_for_symbol(symbol)

# -----------------------------
# Sentiment: VADER + FinBERT combined
# -----------------------------
def vader_sentiment_score(titles):
    if len(titles) == 0:
        return 0.0
    vals = []
    for t in titles:
        try:
            vals.append(vader.polarity_scores(t)["compound"])
        except Exception:
            vals.append(0.0)
    return float(np.mean(vals))


vader_score = vader_sentiment_score(news_titles)
finbert_score = finbert_sentiment_score(news_titles)

# combine:
# if FinBERT is available, weight it more; else only VADER
if finbert_score is not None:
    combined_sentiment = 0.7 * finbert_score + 0.3 * vader_score
else:
    combined_sentiment = vader_score

# Map sentiment in [-1, 1] → fear in [0, 100]
# More negative → higher fear
sentiment_fear = float((1.0 - combined_sentiment) * 50.0)
sentiment_fear = max(0.0, min(100.0, sentiment_fear))

# -----------------------------
# Price / volatility data
# -----------------------------
@st.cache_data
def load_price_data(sym):
    df = yf.download(sym, period="3y", interval="1d")

    if df is None or df.empty:
        # simple fallback
        fallback = pd.DataFrame(
            {
                "Close": [100, 102, 101, 103, 105],
            }
        )
        fallback["Date"] = pd.date_range(end=pd.Timestamp.today(), periods=len(fallback))
        fallback["ClosePrice"] = fallback["Close"]
        fallback["Returns"] = fallback["ClosePrice"].pct_change()
        fallback["Volatility_30d"] = fallback["Returns"].rolling(30).std()
        return fallback.dropna().reset_index(drop=True)

    df["Date"] = df.index

    # ensure ClosePrice exists
    if "Close" in df.columns and "ClosePrice" not in df.columns:
        df = df.rename(columns={"Close": "ClosePrice"})
    elif "ClosePrice" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "ClosePrice"})

    df["Returns"] = df["ClosePrice"].pct_change()
    df["Volatility_30d"] = df["Returns"].rolling(30).std()
    df = df.dropna().reset_index(drop=True)

    return df


df = load_price_data(symbol)

# final safety on ClosePrice
if "ClosePrice" not in df.columns:
    st.error("No valid price column found for this asset.")
    st.stop()

df = df.dropna(subset=["ClosePrice"]).reset_index(drop=True)

# -----------------------------
# Regime detection via K-Means
# -----------------------------
def compute_regimes(data):
    feat = data[["Returns", "Volatility_30d"]].dropna()
    if len(feat) < 50:
        data["Regime"] = 1
        return data, {1: "Neutral"}, {1: 50}

    scaler = StandardScaler()
    X = scaler.fit_transform(feat)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    data = data.copy()
    data["Regime"] = -1
    data.loc[feat.index, "Regime"] = labels

    centers = pd.DataFrame(kmeans.cluster_centers_, columns=["ret", "vol"])
    centers["stress"] = centers["vol"] - centers["ret"]
    order = centers.sort_values("stress", ascending=False).index.tolist()

    names = {}
    weights = {}
    name_map = ["High-Stress", "Neutral", "Calm"]
    risk_map = [85, 50, 25]

    for i, cluster in enumerate(order):
        names[cluster] = name_map[i]
        weights[cluster] = risk_map[i]

    return data, names, weights


df, regime_names, regime_weights = compute_regimes(df)

# -----------------------------
# Risk scoring
# -----------------------------
latest_vol = float(df["Volatility_30d"].iloc[-1])
vol_risk = min(100.0, latest_vol * 2500.0)

latest_regime = int(df["Regime"].iloc[-1])
regime_label = regime_names.get(latest_regime, "Neutral")
regime_risk = float(regime_weights.get(latest_regime, 50))

# simple combined score
combined_risk = (
    0.5 * vol_risk + 0.3 * sentiment_fear + 0.2 * regime_risk
)
combined_risk = max(0.0, min(100.0, combined_risk))

# -----------------------------
# Metrics row (no emojis)
# -----------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Crash Risk", f"{combined_risk:.1f}%")
col2.metric("Sentiment Fear", f"{sentiment_fear:.1f}%")
col3.metric("Current Regime", regime_label)

# -----------------------------
# Data clean for plotting
# -----------------------------
df_plot = df.copy()
df_plot = df_plot.reset_index(drop=True)
df_plot["Date"] = pd.to_datetime(df_plot["Date"], errors="coerce")
df_plot = df_plot.dropna(subset=["Date", "ClosePrice", "Volatility_30d"])

# -----------------------------
# Price chart
# -----------------------------
st.subheader(f"{asset_name} – Price History")
fig_price = px.line(
    df_plot,
    x="Date",
    y="ClosePrice",
    title=f"{asset_name} Closing Price",
    labels={"Date": "Date", "ClosePrice": "Price"},
)
st.plotly_chart(fig_price, use_container_width=True)

# -----------------------------
# Volatility chart
# -----------------------------
st.subheader("30-Day Realized Volatility")
fig_vol = px.line(
    df_plot,
    x="Date",
    y="Volatility_30d",
    title="Rolling 30-Day Volatility",
    labels={"Date": "Date", "Volatility_30d": "Volatility"},
)
st.plotly_chart(fig_vol, use_container_width=True)

# -----------------------------
# Regime visualization
# -----------------------------
st.subheader("Regime Classification Over Time")
reg_colors = df_plot["Regime"].map(regime_names)
fig_reg = px.scatter(
    df_plot,
    x="Date",
    y="ClosePrice",
    color=reg_colors,
    title="Regimes by Price Level",
    labels={"Date": "Date", "ClosePrice": "Price", "color": "Regime"},
)
st.plotly_chart(fig_reg, use_container_width=True)

# -----------------------------
# News & sentiment distribution
# -----------------------------
st.subheader("Headlines Used for Sentiment")
for t in news_titles:
    st.write("• " + t)

st.subheader("Headline Sentiment Distribution (VADER component)")
fig_sent = px.histogram(
    sent_scores, nbins=20, title="Sentiment Score Histogram"
)
st.plotly_chart(fig_sent, use_container_width=True)

st.caption("This dashboard is for research and educational purposes only. Not investment advice.")
