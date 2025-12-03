import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ======================================
#               VADER SETUP
# ======================================
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    try:
        nltk.download("vader_lexicon")
    except Exception:
        pass

# Try to load VADER normally; if it fails, use dummy analyzer
try:
    vader = SentimentIntensityAnalyzer()
except Exception:
    class DummyAnalyzer:
        def polarity_scores(self, _):
            return {"compound": 0.0}
    vader = DummyAnalyzer()


# ======================================
#           STREAMLIT CONFIG
# ======================================
st.set_page_config(page_title="AI Market Crash Warning System", layout="wide")

st.title("AI Market Crash Warning System")
st.caption("Dashboard combining volatility, regime detection, and sentiment.")


# ======================================
#               ASSET LIST
# ======================================
ASSETS = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "S&P 500 (^GSPC)": "^GSPC",
    "Nasdaq 100 (^NDX)": "^NDX",
    "Gold (GC=F)": "GC=F",
}

asset_name = st.sidebar.selectbox("Select Asset", list(ASSETS.keys()))
symbol = ASSETS[asset_name]


# ======================================
#               NEWS SOURCES
# ======================================

def fetch_cryptopanic():
    try:
        url = "https://cryptopanic.com/api/v1/posts/?auth_token=bbf69ca77e536fa8d3&public=true"
        r = requests.get(url, timeout=5).json()
        return [p.get("title", "Untitled") for p in r.get("results", [])]
    except:
        return []


def fetch_reddit(sub):
    try:
        url = f"https://www.reddit.com/r/{sub}/hot.json?limit=20"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=5).json()
        return [i["data"]["title"] for i in r.get("data", {}).get("children", [])]
    except:
        return []


@st.cache_data
def load_news(sym):
    titles = []

    if sym.endswith("-USD"):  # Crypto
        titles += fetch_cryptopanic()
        if sym == "BTC-USD":
            titles += fetch_reddit("Bitcoin")
        elif sym == "ETH-USD":
            titles += fetch_reddit("ethereum")
        titles += fetch_reddit("CryptoCurrency")
    else:  # Stocks / indices / gold
        if sym in ["^GSPC", "^NDX"]:
            titles += fetch_reddit("stocks")
            titles += fetch_reddit("wallstreetbets")
        if sym == "GC=F":
            titles += fetch_reddit("Gold")

    if len(titles) == 0:
        titles = [
            "Markets stable today",
            "No major volatility reported",
            "Analysts expect sideways movement",
        ]

    return list(dict.fromkeys(titles))  # Remove duplicates


news_titles = load_news(symbol)


# ======================================
#               SENTIMENT
# ======================================
def vader_sentiment(titles):
    scores = []
    for t in titles:
        try:
            scores.append(vader.polarity_scores(t)["compound"])
        except:
            scores.append(0)
    if len(scores) == 0:
        return 0.0, []
    return np.mean(scores), scores


sentiment_avg, vader_scores = vader_sentiment(news_titles)
sentiment_fear = (1 - sentiment_avg) * 50
sentiment_fear = float(max(0, min(100, sentiment_fear)))


# ======================================
#               PRICE DATA
# ======================================
@st.cache_data
def load_price(sym):
    df = yf.download(sym, period="3y", interval="1d")

    # Fallback data if Yahoo fails
    if df is None or df.empty:
        fb = pd.DataFrame({
            "Date": pd.date_range(end=pd.Timestamp.today(), periods=180),
            "ClosePrice": np.linspace(100, 120, 180)
        })
        fb["Returns"] = fb["ClosePrice"].pct_change()
        fb["Volatility_30d"] = fb["Returns"].rolling(30).std()
        return fb.dropna().reset_index(drop=True)

    df["Date"] = df.index

    # Try multiple ways to extract ClosePrice safely:
    if "Close" in df.columns:
        df = df.rename(columns={"Close": "ClosePrice"})
    elif "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "ClosePrice"})
    else:
        # Look for first numeric column
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df["ClosePrice"] = df[col]
                break

    # If still missing → fallback synthetic
    if "ClosePrice" not in df.columns:
        fb = pd.DataFrame({
            "Date": pd.date_range(end=pd.Timestamp.today(), periods=180),
            "ClosePrice": np.linspace(100, 120, 180)
        })
        fb["Returns"] = fb["ClosePrice"].pct_change()
        fb["Volatility_30d"] = fb["Returns"].rolling(30).std()
        return fb.dropna().reset_index(drop=True)

    df["Returns"] = df["ClosePrice"].pct_change()
    df["Volatility_30d"] = df["Returns"].rolling(30).std()

    df = df.dropna(subset=["Returns", "Volatility_30d"])
    df = df.reset_index(drop=True)
    return df


df = load_price(symbol)


# ======================================
#           REGIME DETECTION
# ======================================
def detect_regimes(data):
    feat = data[["Returns", "Volatility_30d"]]
    if len(feat) < 60:
        data["Regime"] = 1
        return data, {1: "Neutral"}, {1: 50}

    scaler = StandardScaler()
    X = scaler.fit_transform(feat)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    data["Regime"] = labels
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=["ret", "vol"])
    centers["stress"] = centers["vol"] - centers["ret"]
    order = centers.sort_values("stress", ascending=False).index.tolist()

    names = {}
    risks = {}
    label_map = ["High-Stress", "Neutral", "Calm"]
    risk_map = [85, 50, 25]

    for i, cluster in enumerate(order):
        names[cluster] = label_map[i]
        risks[cluster] = risk_map[i]

    return data, names, risks


df, regime_names, risk_weights = detect_regimes(df)

latest_reg = df["Regime"].iloc[-1]
regime_label = regime_names[latest_reg]
regime_risk = risk_weights[latest_reg]

vol_risk = float(min(100, df["Volatility_30d"].iloc[-1] * 2500))
combined_risk = 0.5 * vol_risk + 0.3 * sentiment_fear + 0.2 * regime_risk


# ======================================
#                METRICS
# ======================================
col1, col2, col3 = st.columns(3)
col1.metric("Crash Risk", f"{combined_risk:.1f}%")
col2.metric("Sentiment Fear", f"{sentiment_fear:.1f}%")
col3.metric("Current Regime", regime_label)


# ======================================
#                PLOTS
# ======================================
df_plot = df.copy()

st.subheader("Price History")
fig_price = px.line(df_plot, x="Date", y="ClosePrice")
st.plotly_chart(fig_price, use_container_width=True)

st.subheader("30-Day Volatility")
fig_vol = px.line(df_plot, x="Date", y="Volatility_30d")
st.plotly_chart(fig_vol, use_container_width=True)

st.subheader("Regimes vs Price")
fig_reg = px.scatter(
    df_plot, x="Date", y="ClosePrice",
    color=df_plot["Regime"].map(regime_names)
)
st.plotly_chart(fig_reg, use_container_width=True)


# ======================================
#               SENTIMENT HISTOGRAM
# ======================================
st.subheader("Headline Sentiment Distribution")

if len(vader_scores) > 0:
    fig_sent = px.histogram(
        vader_scores,
        nbins=20,
        title="VADER Sentiment Histogram",
        labels={"value": "VADER compound score"}
    )
    st.plotly_chart(fig_sent, use_container_width=True)
else:
    st.info("No sentiment data available.")

st.subheader("Headlines Used")
for t in news_titles:
    st.write("• " + t)


st.caption("Research dashboard only. Not financial advice.")
