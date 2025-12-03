import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ============================
# NLTK VADER (always available)
# ============================
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

vader = SentimentIntensityAnalyzer()

# ============================
# Optional FinBERT (mini) setup
# ============================
finbert_model = None
finbert_tokenizer = None

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

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
    If FinBERT or transformers/torch are not available, returns None.
    """
    if finbert_model is None or finbert_tokenizer is None:
        return None
    try:
        if len(texts) == 0:
            return None

        texts = texts[:20]  # keep it light
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

        scores = []
        # FinBERT labels: 0=negative, 1=neutral, 2=positive
        for p in probs:
            neg, neu, pos = p
            scores.append(float(pos - neg))  # in [-1,1]

        return float(np.mean(scores))
    except Exception:
        return None


# ============================
# Streamlit page config
# ============================
st.set_page_config(page_title="AI Market Crash Warning System", layout="wide")

st.title("AI Market Crash Warning System")
st.caption(
    "Research dashboard combining volatility analysis, regime classification, and multi-source sentiment."
)

# ============================
# Asset universe
# ============================
ASSETS = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "S&P 500 (^GSPC)": "^GSPC",
    "Nasdaq 100 (^NDX)": "^NDX",
    "Gold (GC=F)": "GC=F",
}

asset_name = st.sidebar.selectbox("Select asset", list(ASSETS.keys()))
symbol = ASSETS[asset_name]

# ============================
# News helpers
# ============================

def fetch_cryptopanic_news():
    """CryptoPanic: general crypto news."""
    try:
        url = "https://cryptopanic.com/api/v1/posts/?auth_token=bbf69ca77e536fa8d3&public=true"
        r = requests.get(url, timeout=5)
        data = r.json()
        if "results" in data and isinstance(data["results"], list):
            return [item.get("title", "Untitled") for item in data["results"]]
    except Exception:
        pass
    return []


def fetch_reddit_titles(subreddit):
    """Best-effort Reddit fetch, with fallback to empty list."""
    try:
        url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=20"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=5)
        data = r.json()
        titles = [
            child["data"]["title"]
            for child in data.get("data", {}).get("children", [])
            if "data" in child and "title" in child["data"]
        ]
        return titles
    except Exception:
        return []


def load_news_for_symbol(sym: str):
    """
    Asset-specific headlines:
      - Crypto: CryptoPanic + crypto subreddits
      - Indices/Gold: stocks/gold subreddits
    """
    headlines = []

    # Crypto assets
    if sym.endswith("-USD"):
        headlines.extend(fetch_cryptopanic_news())
        if sym == "BTC-USD":
            headlines.extend(fetch_reddit_titles("Bitcoin"))
        elif sym == "ETH-USD":
            headlines.extend(fetch_reddit_titles("ethereum"))
        else:
            headlines.extend(fetch_reddit_titles("CryptoCurrency"))
    else:
        # Traditional assets
        if sym in ("^GSPC", "^NDX"):
            headlines.extend(fetch_reddit_titles("stocks"))
            headlines.extend(fetch_reddit_titles("wallstreetbets"))
        elif sym == "GC=F":
            headlines.extend(fetch_reddit_titles("Gold"))

    # Fallback generic headlines if everything fails
    if len(headlines) == 0:
        headlines = [
            "Markets trade in a narrow range with muted volatility",
            "Analysts describe sentiment as cautious but stable",
            "No major macroeconomic surprises reported today",
        ]

    # De-duplicate while preserving order
    headlines = list(dict.fromkeys(headlines))
    return headlines


news_titles = load_news_for_symbol(symbol)

# ============================
# Sentiment calculation
# ============================

def vader_sentiment(titles):
    """Returns (average_score, list_of_scores) using VADER."""
    scores = []
    for t in titles:
        try:
            scores.append(vader.polarity_scores(t)["compound"])
        except Exception:
            scores.append(0.0)
    if len(scores) == 0:
        return 0.0, []
    return float(np.mean(scores)), scores


vader_avg, vader_scores = vader_sentiment(news_titles)
finbert_avg = finbert_sentiment_score(news_titles)

# Combine VADER + FinBERT, if available
if finbert_avg is not None:
    combined_sentiment = 0.7 * finbert_avg + 0.3 * vader_avg
else:
    combined_sentiment = vader_avg

# Map sentiment [-1,1] → fear [0,100]: more negative = more fear
sentiment_fear = (1.0 - combined_sentiment) * 50.0
sentiment_fear = float(max(0.0, min(100.0, sentiment_fear)))

# ============================
# Price & volatility data
# ============================

@st.cache_data
def load_price_data(sym: str) -> pd.DataFrame:
    df = yf.download(sym, period="3y", interval="1d")

    # Fallback if API fails
    if df is None or df.empty:
        fb = pd.DataFrame(
            {
                "Date": pd.date_range(end=pd.Timestamp.today(), periods=5),
                "ClosePrice": [100, 102, 101, 103, 105],
            }
        )
        fb["Returns"] = fb["ClosePrice"].pct_change()
        fb["Volatility_30d"] = fb["Returns"].rolling(30).std()
        return fb.dropna().reset_index(drop=True)

    df["Date"] = df.index

    # Guarantee ClosePrice exists
    rename_map = {}
    if "Close" in df.columns:
        rename_map["Close"] = "ClosePrice"
    if "Adj Close" in df.columns and "ClosePrice" not in rename_map:
        rename_map["Adj Close"] = "ClosePrice"

    df = df.rename(columns=rename_map)

    if "ClosePrice" not in df.columns:
        # Try MultiIndex columns like ('Close','BTC-USD')
        for col in df.columns:
            if isinstance(col, tuple) and "Close" in str(col[0]):
                df["ClosePrice"] = df[col]
                break

    if "ClosePrice" not in df.columns:
        # Emergency fallback: use first numeric column
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df["ClosePrice"] = df[col]
                break

    # Final safety: if still missing, stop cleanly
    if "ClosePrice" not in df.columns:
        fb = pd.DataFrame(
            {
                "Date": pd.date_range(end=pd.Timestamp.today(), periods=5),
                "ClosePrice": [100, 102, 101, 103, 105],
            }
        )
        fb["Returns"] = fb["ClosePrice"].pct_change()
        fb["Volatility_30d"] = fb["Returns"].rolling(30).std()
        return fb.dropna().reset_index(drop=True)

    df["Returns"] = df["ClosePrice"].pct_change()
    df["Volatility_30d"] = df["Returns"].rolling(30).std()
    df = df.dropna(subset=["ClosePrice"]).reset_index(drop=True)

    return df


df = load_price_data(symbol)

if df.empty:
    st.error("Price data could not be loaded for this asset.")
    st.stop()

# ============================
# Regime detection via KMeans
# ============================

def compute_regimes(data: pd.DataFrame):
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
    label_map = ["High-Stress", "Neutral", "Calm"]
    risk_map = [85, 50, 25]

    for i, cluster in enumerate(order):
        names[cluster] = label_map[i]
        weights[cluster] = risk_map[i]

    return data, names, weights


df, regime_names, regime_weights = compute_regimes(df)

# ============================
# Risk scores
# ============================

latest_vol = float(df["Volatility_30d"].iloc[-1])
vol_risk = min(100.0, latest_vol * 2500.0)

latest_regime = int(df["Regime"].iloc[-1])
regime_label = regime_names.get(latest_regime, "Neutral")
regime_risk = float(regime_weights.get(latest_regime, 50))

combined_risk = 0.5 * vol_risk + 0.3 * sentiment_fear + 0.2 * regime_risk
combined_risk = float(max(0.0, min(100.0, combined_risk)))

# ============================
# Top metrics (no emojis)
# ============================

col1, col2, col3 = st.columns(3)
col1.metric("Crash Risk", f"{combined_risk:.1f}%")
col2.metric("Sentiment Fear", f"{sentiment_fear:.1f}%")
col3.metric("Current Regime", regime_label)

# ============================
# Clean data for plotting
# ============================

df_plot = df.copy()
df_plot = df_plot.reset_index(drop=True)
df_plot["Date"] = pd.to_datetime(df_plot["Date"], errors="coerce")

# Ensure numeric
for col in df_plot.columns:
    if col not in ["Date"]:
        df_plot[col] = pd.to_numeric(df_plot[col], errors="coerce")

df_plot = df_plot.dropna(subset=["Date", "ClosePrice", "Volatility_30d"])

# ============================
# Price chart
# ============================

st.subheader(f"{asset_name} – Price History")
fig_price = px.line(
    df_plot,
    x="Date",
    y="ClosePrice",
    title=f"{asset_name} Closing Price",
    labels={"Date": "Date", "ClosePrice": "Price"},
)
st.plotly_chart(fig_price, use_container_width=True)

# ============================
# Volatility chart
# ============================

st.subheader("30-Day Realized Volatility")
fig_vol = px.line(
    df_plot,
    x="Date",
    y="Volatility_30d",
    title="Rolling 30-Day Volatility",
    labels={"Date": "Date", "Volatility_30d": "Volatility"},
)
st.plotly_chart(fig_vol, use_container_width=True)

# ============================
# Regime visualization
# ============================

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

# ============================
# News & sentiment distribution
# ============================

st.subheader("Headlines Used for Sentiment")
for t in news_titles:
    st.write("• " + t)

st.subheader("Headline Sentiment Distribution (VADER component)")
if len(vader_scores) > 0:
    fig_sent = px.histogram(
        vader_scores,
        nbins=20,
        title="Sentiment Score Histogram",
        labels={"value": "VADER compound score"},
    )
    st.plotly_chart(fig_sent, use_container_width=True)
else:
    st.info("No sentiment scores available to plot.")

st.caption("This dashboard is for research and educational purposes only. Not investment advice.")
