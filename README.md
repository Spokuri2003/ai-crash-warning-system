# AI Early-Warning System for Market Stress

This project provides a lightweight, real-time demonstration of an early-warning signal for market stress.  
The system combines Bitcoin price volatility with sentiment extracted from live crypto news to generate a simple crash-risk estimate.  
The focus is on feature engineering, data acquisition, and real-time analytics—not long-horizon backtesting.

---

## Overview

The application integrates two primary components:

1. **Market Volatility**  
   - Uses 30-day realized volatility derived from BTC-USD daily returns.  
   - Acts as a proxy for market uncertainty and risk.

2. **News Sentiment Analysis**  
   - Fetches current crypto news headlines via a public API.  
   - Applies VADER (NLP) to compute headline-level sentiment scores.  
   - Aggregates sentiment into a "fear index."

A combined score is generated to approximate near-term market stress conditions.  
The project is structured for clarity, extensibility, and public deployment using Streamlit.

---

## Features

- Real-time BTC-USD price and volatility
- Headline-level sentiment scoring
- Aggregated sentiment-based “fear” metric
- Combined volatility + sentiment risk indicator
- Interactive charts (Plotly)
- Live news feed table
- Automated model interpretation (“low / moderate / high stress”)
- Fully deployable on Streamlit Cloud

---

## Tech Stack

- **Python**
- **Streamlit** (web application)
- **yfinance** (market data)
- **Pandas / NumPy** (feature engineering)
- **VADER Sentiment** (NLP)
- **Plotly** (interactive visualization)
- **Requests** (API integration)

---

## Running the App Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Repository Structure

```
ai-crash-warning-system/
│
├── app.py               # Streamlit app
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

---

## Future Work

The current version is intentionally minimal. Potential next steps include:

- Multi-asset integration (S&P 500, NASDAQ, Gold, VIX)
- FinBERT or LLM-based sentiment modeling
- Regime classification (HMM/K-Means)
- Correlation breakdown and contagion indicators
- Feature importance and ML-based crash probability models
- Multi-page Streamlit interface (stress dashboard, asset panels)
- Automated alerts (email / webhook)

---

## Author

**Swathi Pokuri**  
Data Science & Quantitative Analysis  
GitHub: https://github.com/Spokuri2003

